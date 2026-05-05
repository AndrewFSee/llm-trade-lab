"""Run a few example texts through the entity resolver."""
from __future__ import annotations

from llm_trade_lab.data.entity_resolver import load_default

EXAMPLES = [
    "Farm bill provides $5B in fertilizer subsidies to support domestic potash production.",
    "FDA approves new GLP-1 indication for cardiovascular risk reduction.",
    "Defense authorization act increases funding for nuclear submarine production.",
    "Solar tax credits extended through 2032.",
    "New tariffs on Chinese semiconductor imports announced.",
    "Treasury sanctions Russian oil majors and refining infrastructure.",
    "SAFE Banking Act advances; cannabis rescheduling to Schedule III.",
    "FOMC discusses regional bank capital requirements amid rate uncertainty.",
    "SEC approves first spot bitcoin ETF for retail platforms.",
    "EPA finalizes new waste management rules for landfill emissions.",
    "Section 232 steel and aluminum tariffs reinstated on Chinese imports.",
    "FCC announces 5G spectrum auction; new wireless broadband expansion.",
    "DOE proposes critical minerals strategy targeting rare earths and copper.",
    "FDA finalizes tobacco rule restricting flavored vaping products.",
    "Medicare Advantage payment rate revisions proposed by CMS.",
    "Congressional hearing on social media content moderation and Section 230.",
    "AI infrastructure tax credit included in new appropriations bill.",
]


def main() -> None:
    r = load_default()
    print(f"Loaded {len(r.known_themes())} themes: {', '.join(r.known_themes())}")
    print()
    for text in EXAMPLES:
        themes = r.matched_themes(text)
        beneficiaries = r.resolve(text)
        print(f'> "{text}"')
        if not themes:
            print("  (no themes matched)")
            continue
        print(f"  themes: {', '.join(themes)}")
        for b in sorted(beneficiaries, key=lambda x: -x.confidence):
            print(
                f"    {b.ticker:<5s} (conf={b.confidence:.2f}, theme={b.matched_theme:<14s})  "
                f"{b.name} -- {b.mechanism}"
            )
        print()


if __name__ == "__main__":
    main()
