"""List recent 119th-Congress bills, then deep-fetch one to show structured detail.

Requires CONGRESS_GOV_API_KEY in .env.
"""
from __future__ import annotations

from dotenv import load_dotenv

from llm_trade_lab.data.congress_ingest import get_bill_detail, list_recent_bills


def main() -> None:
    load_dotenv()

    print("Listing 10 most recently updated 119th Congress bills...")
    bills = list_recent_bills(congress=119, limit=10)
    for b in bills:
        policy = b.policy_area or "-"
        print(
            f"  {b.bill_id:18s}  {b.latest_action_date}  [{policy[:24]:<24s}]  {b.title[:80]}"
        )

    if not bills:
        print("No bills returned.")
        return

    target = bills[0]
    print(
        f"\nDeep-fetching {target.bill_id} ({target.title[:60]}...)"
    )
    detail = get_bill_detail(target.congress, target.bill_type, target.number)
    sponsors_summary = ", ".join(
        f"{s.get('firstName', '')} {s.get('lastName', '')} ({s.get('party', '?')})".strip()
        for s in detail.sponsors[:3]
    )
    print(f"  sponsors: {sponsors_summary or '-'}")
    print(f"  cosponsors_count: {detail.cosponsors_count}")
    print(f"  actions_count: {detail.actions_count}")
    print(
        f"  text_version: {detail.text_version_type} ({detail.text_version_date}); "
        f"length={len(detail.text)} chars"
    )
    if detail.text:
        excerpt = detail.text[:500].replace("\n", " ")
        print(f"  text excerpt: {excerpt}...")


if __name__ == "__main__":
    main()
