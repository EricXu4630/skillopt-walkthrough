#!/usr/bin/env python3
"""Quick validation: confirm initial skill fails and optimized skill passes.

Run: python validate_scoring.py
"""
import asyncio, os, sys
import anthropic

# Fix Windows console encoding for emoji in LLM responses
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
if not API_KEY:
    print("ERROR: ANTHROPIC_API_KEY not set")
    sys.exit(1)

FORMAT_KEYWORDS = [
    "## summary", "**critical**", "**improvements**",
    "## conclusion", "request changes", "**verdict**",
]

def score_task(review: str, bug: str | None, keywords: list) -> dict:
    rv = review.lower()
    format_hits = sum(1 for kw in FORMAT_KEYWORDS if kw in rv)
    format_score = format_hits / len(FORMAT_KEYWORDS)
    bug_caught = ("approved" in rv) if bug is None else any(kw in rv for kw in keywords)
    combined = 0.6 * format_score + 0.4 * (1.0 if bug_caught else 0.0)
    return {"passed": combined >= 0.65, "format": round(format_score, 2),
            "bug": bug_caught, "combined": round(combined, 2), "format_hits": format_hits}

SPARSE_SKILL = "# Code Reviewer\n\nReview the provided code and report any issues.\n"

OPTIMIZED_SKILL = """# Code Reviewer

Review the provided code and report any issues.

## Required Output Format

Always structure your review EXACTLY as follows — no deviation:

```
## Summary
[One paragraph: what does this code do and what is your overall assessment]

## Findings
**Critical**: [List critical bugs, security vulnerabilities, data loss risks — or "None"]
**Improvements**: [List code quality suggestions, performance notes — or "None"]

## Conclusion
**Verdict**: Request Changes | Approved
```

## Security Checklist
- SQL queries: f-string or % interpolation → use parameterized queries
- Secrets: hardcoded API keys/passwords → use environment variables
- Shell calls: shell=True with user input → use list form subprocess
- File handles: open() without with → use context manager (with open(...) as f)
- Concurrency: += on shared state without lock → use threading.Lock or atomic type
- Path input: string concatenation with user path → use os.path.realpath + prefix check
"""

TASK = {
    "title": "auth.py — user login query",
    "code": """def login(user, pwd):
    q = f"SELECT * FROM users WHERE name='{user}' AND pass='{pwd}'"
    return db.execute(q)""",
    "bug": "SQL injection",
    "keywords": ["sql injection", "parameterized", "injection"],
}

async def test_skill(skill: str, label: str):
    client = anthropic.AsyncAnthropic(api_key=API_KEY)
    msg = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        system=skill,
        messages=[{"role": "user", "content":
            f"Please review this code change:\n\nFile: {TASK['title']}\n\n```python\n{TASK['code']}\n```"}],
    )
    review = msg.content[0].text
    result = score_task(review, TASK["bug"], TASK["keywords"])
    print(f"\n{'='*60}")
    print(f"SKILL: {label}")
    print(f"{'='*60}")
    print(f"REVIEW:\n{review[:600]}")
    print(f"\nSCORE: format={result['format']} ({result['format_hits']}/6 keywords) | bug={result['bug']} | combined={result['combined']} | PASSED={result['passed']}")
    return result

async def main():
    print("Testing scoring with sparse vs optimized skill...")
    r1 = await test_skill(SPARSE_SKILL, "SPARSE (initial)")
    r2 = await test_skill(OPTIMIZED_SKILL, "OPTIMIZED (after training)")
    print(f"\n{'='*60}")
    print(f"RESULT: {r1['combined']} → {r2['combined']} (delta: +{round(r2['combined']-r1['combined'],2)})")
    print(f"PASS:   {r1['passed']} → {r2['passed']}")
    print("Demo will show meaningful improvement: YES" if not r1['passed'] and r2['passed'] else
          "WARNING: Check scoring — sparse may already pass")

asyncio.run(main())
