#!/usr/bin/env python3
"""Validate scoring: confirm sparse skill fails, MEDDIC-trained skill passes.

Run: python validate_scoring.py
"""
import asyncio, os, sys
import anthropic

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
if not API_KEY:
    print("ERROR: ANTHROPIC_API_KEY not set"); sys.exit(1)

FORMAT_KEYWORDS = [
    "**metrics**", "**economic buyer**", "**decision criteria**",
    "**decision process**", "**identify pain**", "**champion**", "deal verdict",
]

def score(review: str, bug: str | None, keywords: list) -> dict:
    rv = review.lower()
    fmt = sum(1 for kw in FORMAT_KEYWORDS if kw in rv) / len(FORMAT_KEYWORDS)
    if bug is None:
        ok = "qualified" in rv and "not qualified" not in rv
    else:
        ok = any(kw in rv for kw in keywords)
    combined = round(0.6 * fmt + 0.4 * (1.0 if ok else 0.0), 2)
    return {"passed": combined >= 0.65, "format": round(fmt, 2), "ok": ok, "combined": combined}

SPARSE = "# Sales Opportunity Qualifier\n\nReview sales opportunity notes and assess whether the deal should advance.\n"

MEDDIC = """# Sales Opportunity Qualifier

Qualify every opportunity using the MEDDIC framework. Score all 6 components.

## MEDDIC Framework

**Metrics**: Quantified business impact — what is the $ value of the problem?
**Economic Buyer**: Who controls the budget and can sign the contract? Have you met them?
**Decision Criteria**: What criteria will the prospect use to choose a vendor?
**Decision Process**: What are the steps, approvals, and timeline to purchase?
**Identify Pain**: What specific business pain drives urgency? Is the status quo sustainable?
**Champion**: Who inside the account is selling on your behalf internally?

## Required Output Format

Score each component: STRONG / WEAK / MISSING, then give a Deal Verdict.

## Qualification
**Metrics**: [STRONG/WEAK/MISSING] — [evidence from notes]
**Economic Buyer**: [STRONG/WEAK/MISSING] — [evidence]
**Decision Criteria**: [STRONG/WEAK/MISSING] — [evidence]
**Decision Process**: [STRONG/WEAK/MISSING] — [evidence]
**Identify Pain**: [STRONG/WEAK/MISSING] — [evidence]
**Champion**: [STRONG/WEAK/MISSING] — [evidence]

**Deal Verdict**: QUALIFIED | NOT QUALIFIED | NEEDS DISCOVERY
**Risk Flags**: [specific gaps to close before advancing stage]

## Red Flags
- Economic Buyer not met → do not advance past Discovery
- Decision Process undefined → always NEEDS DISCOVERY
- No Champion → high churn risk; identify or create one
- Metrics unquantified → revisit pain quantification before Proposal
- Decision Process has board/committee vote > 3 months away → flag as timeline risk
"""

TASK = {
    "title": "Acme Corp — Database Migration",
    "code": """Account: Acme Corp (2,400 employees, manufacturing)
Contact: Mark T., IT Director
Notes: Team is losing ~$500k/year in manual data reconciliation.
Mark is engaged and wants to move fast. No budget conversation yet.
Have not met anyone in Finance or the C-suite.
Next step: technical deep-dive with Mark's team next week.""",
    "bug": "Economic Buyer not identified",
    "keywords": ["economic buyer", "not identified", "missing", "not qualified", "needs discovery"],
}

async def test(skill: str, label: str):
    client = anthropic.AsyncAnthropic(api_key=API_KEY)
    msg = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        system=skill,
        messages=[{"role": "user", "content":
            f"Please qualify this sales opportunity:\n\nDeal: {TASK['title']}\n\n{TASK['code']}"}],
    )
    review = msg.content[0].text
    result = score(review, TASK["bug"], TASK["keywords"])
    print(f"\n{'='*60}\nSKILL: {label}\n{'='*60}")
    print(f"ASSESSMENT:\n{review[:700]}")
    print(f"\nSCORE: format={result['format']} | gap_found={result['ok']} | combined={result['combined']} | PASSED={result['passed']}")
    return result

async def main():
    r1 = await test(SPARSE, "SPARSE (initial — no MEDDIC)")
    r2 = await test(MEDDIC, "MEDDIC-trained (after optimization)")
    print(f"\n{'='*60}")
    print(f"IMPROVEMENT: {r1['combined']} → {r2['combined']}  (delta: +{round(r2['combined']-r1['combined'],2)})")
    print(f"PASS: {r1['passed']} → {r2['passed']}")
    if not r1['passed'] and r2['passed']:
        print("Demo will show meaningful improvement: YES")
    else:
        print("WARNING: check scoring — gap smaller than expected")

asyncio.run(main())
