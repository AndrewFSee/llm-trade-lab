"""FAISS-backed semantic retrieval over the hypothesis ledger.

In Phase 1 the LLM consults this retriever before generating new hypotheses,
to ground its output in similar past hypotheses + their realized outcomes.

Type-aware via post-filter: when `filter_type` is given, the retriever
oversamples (k * 5) candidates from FAISS and post-filters by
hypothesis.type, returning the top k that match.

Index persistence:
  - {index_dir}/hypotheses.faiss     binary FAISS index
  - {index_dir}/hypothesis_ids.json   parallel id list (FAISS row -> ledger id)

Embedding model: BAAI/bge-small-en-v1.5 (384-dim, ~134MB, ~100 emb/s on CPU).
Cosine similarity via normalized inner product on IndexFlatIP.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Literal

import faiss
import numpy as np

from llm_trade_lab.memory.ledger import Ledger
from llm_trade_lab.schema.hypothesis import Hypothesis

DEFAULT_INDEX_DIR = Path("data/faiss")
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"

EncodeFn = Callable[[list[str]], np.ndarray]


def _hypothesis_text(h: Hypothesis) -> str:
    """Canonical string representation of a hypothesis for embedding."""
    parts = [h.name, h.thesis_text]
    if h.universe:
        parts.append("Universe: " + ", ".join(h.universe))
    if h.type == "event_driven":
        ev = h.trigger_event
        parts.append(f"Event: {ev.event_type} from {ev.source} on {ev.event_date}")
        bens = ", ".join(f"{b.ticker}({b.mechanism})" for b in h.beneficiaries)
        if bens:
            parts.append(f"Beneficiaries: {bens}")
    return "\n".join(parts)


class HypothesisRetriever:
    def __init__(
        self,
        ledger: Ledger,
        *,
        index_dir: Path = DEFAULT_INDEX_DIR,
        model_name: str = DEFAULT_MODEL_NAME,
        encode_fn: EncodeFn | None = None,
    ):
        self.ledger = ledger
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "hypotheses.faiss"
        self.ids_path = self.index_dir / "hypothesis_ids.json"

        if encode_fn is not None:
            self._encode = encode_fn
            probe = encode_fn(["probe"])
            self.dim = int(probe.shape[1])
        else:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
            # Newer sentence-transformers renamed this to get_embedding_dimension;
            # fall back to the old name for older installs.
            get_dim = getattr(
                self._model, "get_embedding_dimension", None
            ) or self._model.get_sentence_embedding_dimension
            self.dim = int(get_dim())

            def _encode(texts: list[str]) -> np.ndarray:
                return self._model.encode(
                    texts, normalize_embeddings=True, convert_to_numpy=True
                ).astype(np.float32)

            self._encode = _encode

        self.hypothesis_ids: list[str] = []
        if self.index_path.exists() and self.ids_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.hypothesis_ids = json.loads(self.ids_path.read_text(encoding="utf-8"))
            if self.index.d != self.dim:
                raise ValueError(
                    f"Existing FAISS index dim={self.index.d} does not match "
                    f"current encoder dim={self.dim}. Delete {self.index_dir} to rebuild."
                )
        else:
            self.index = faiss.IndexFlatIP(self.dim)

    # ----- write ------------------------------------------------------------

    def add_hypothesis(self, hypothesis_id: str, hypothesis: Hypothesis) -> None:
        """Embed and add a hypothesis. Idempotent: skips if id already indexed."""
        if hypothesis_id in self.hypothesis_ids:
            return
        emb = self._encode([_hypothesis_text(hypothesis)])
        self.index.add(emb)
        self.hypothesis_ids.append(hypothesis_id)

    def reindex_from_ledger(self) -> int:
        """Rebuild the FAISS index from all hypotheses currently in the ledger.

        Useful after a schema change or if the on-disk index is lost. Returns
        the number of hypotheses indexed.
        """
        self.index = faiss.IndexFlatIP(self.dim)
        self.hypothesis_ids = []
        # Naive: scan all ids by iterating the SQLite table.
        with self.ledger._conn() as conn:
            rows = conn.execute("SELECT id FROM hypothesis ORDER BY created_at").fetchall()
        ids = [r[0] for r in rows]
        if not ids:
            return 0
        hypotheses = [self.ledger.get_hypothesis(hid) for hid in ids]
        texts = [_hypothesis_text(h) for h in hypotheses if h is not None]
        valid_ids = [hid for hid, h in zip(ids, hypotheses) if h is not None]
        if not texts:
            return 0
        embs = self._encode(texts)
        self.index.add(embs)
        self.hypothesis_ids = valid_ids
        return len(valid_ids)

    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        self.ids_path.write_text(
            json.dumps(self.hypothesis_ids), encoding="utf-8"
        )

    # ----- read -------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        filter_type: Literal["statistical", "event_driven"] | None = None,
    ) -> list[tuple[str, float, Hypothesis]]:
        """Return top-k similar hypotheses as (id, cosine_similarity, hypothesis).

        With `filter_type`, oversamples 5x from FAISS and post-filters by
        hypothesis.type to return k matches of the requested type.
        """
        if self.index.ntotal == 0:
            return []
        emb = self._encode([query])
        oversample = k * 5 if filter_type else k
        oversample = min(oversample, self.index.ntotal)
        scores, indices = self.index.search(emb, oversample)

        out: list[tuple[str, float, Hypothesis]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.hypothesis_ids):
                continue
            hid = self.hypothesis_ids[idx]
            h = self.ledger.get_hypothesis(hid)
            if h is None:
                continue
            if filter_type and h.type != filter_type:
                continue
            out.append((hid, float(score), h))
            if len(out) >= k:
                break
        return out

    @property
    def size(self) -> int:
        return int(self.index.ntotal)
