"""Reddit ingest tests."""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from llm_trade_lab.data import reddit_ingest


def test_get_reddit_raises_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("REDDIT_USER_AGENT", raising=False)
    with pytest.raises(RuntimeError, match="REDDIT_CLIENT_ID"):
        reddit_ingest._get_reddit()


def test_fetch_validates_sort() -> None:
    with pytest.raises(ValueError, match="sort"):
        reddit_ingest.fetch_subreddit_posts("stocks", sort="bogus")


def test_row_to_post_round_trip() -> None:
    p = reddit_ingest.RedditPost(
        id="abc",
        subreddit="stocks",
        title="t",
        selftext="s",
        author="u",
        created_utc=datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc),
        score=10,
        upvote_ratio=0.9,
        num_comments=3,
        flair="DD",
        permalink="https://reddit.com/r/stocks/comments/abc",
        url="https://reddit.com/r/stocks/comments/abc",
    )
    d = p.to_dict()
    assert d["created_utc"] == "2025-06-01T12:00:00+00:00"
    p2 = reddit_ingest._row_to_post(d)
    assert p2 == p


@pytest.mark.integration
@pytest.mark.skipif(
    not (os.environ.get("REDDIT_CLIENT_ID") and os.environ.get("REDDIT_CLIENT_SECRET")),
    reason="Reddit credentials not set",
)
def test_fetch_stocks_hot_integration(tmp_path) -> None:
    posts = reddit_ingest.fetch_subreddit_posts(
        "stocks", sort="hot", limit=3, cache_dir=tmp_path
    )
    assert len(posts) >= 1
    assert all(p.subreddit.lower() == "stocks" for p in posts)
