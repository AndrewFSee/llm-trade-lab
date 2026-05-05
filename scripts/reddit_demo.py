"""Pull hot posts from r/stocks for a quick sentiment look."""
from __future__ import annotations

from dotenv import load_dotenv

from llm_trade_lab.data.reddit_ingest import fetch_subreddit_posts


def main() -> None:
    load_dotenv()
    posts = fetch_subreddit_posts("stocks", sort="hot", limit=15)
    print(f"r/stocks hot ({len(posts)}):")
    for p in posts:
        flair = f"[{p.flair[:14]}]" if p.flair else "[-]"
        print(
            f"  {p.created_utc.date()}  {flair:<16s}  "
            f"score={p.score:>5d}  comments={p.num_comments:>4d}  "
            f"{p.title[:90]}"
        )


if __name__ == "__main__":
    main()
