"""Search Federal Register for recent FDA rules — pharma catalyst examples."""
from __future__ import annotations

from dotenv import load_dotenv

from llm_trade_lab.data.federal_register_ingest import search_documents


def main() -> None:
    load_dotenv()
    # FDA's drug approvals and patent extensions are filed as Notices, not Rules.
    docs = search_documents(
        since="2026-01-01",
        until="2026-04-25",
        agencies=["food-and-drug-administration"],
        per_page=10,
    )
    print(f"Recent FDA Federal Register docs ({len(docs)}):")
    for d in docs:
        agencies = ", ".join(d.agencies)[:30]
        topics = ", ".join(d.topics[:3])[:40]
        print(
            f"  {d.publication_date}  [{d.document_type[:14]:<14s}]  "
            f"({agencies:<30s})  {d.title[:80]}"
        )
        if topics:
            print(f"    topics: {topics}")


if __name__ == "__main__":
    main()
