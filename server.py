#!/usr/bin/env python3
"""SkillOpt live training server.

Implements the actual SkillOpt training loop (rollout → reflect → select → gate)
applied to a code review task, streaming events via SSE so the browser can
visualize each step in real time.

Usage:
    ANTHROPIC_API_KEY=sk-... python server.py
    Then open skillopt_demo.html in a browser.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from typing import AsyncGenerator

import anthropic
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
if not API_KEY:
    print("ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr)

# ── Training tasks (code review PRs) ─────────────────────────────────────────

TRAIN_TASKS = [
    {
        "id": "auth",
        "title": "auth.py — user login query",
        "sev": "CRIT",
        "code": """def login(user, pwd):
    q = f"SELECT * FROM users WHERE name='{user}' AND pass='{pwd}'"
    return db.execute(q)""",
        "bug": "SQL injection via f-string interpolation",
        "keywords": ["sql injection", "parameterized", "injection", "f-string", "format string", "user input"],
    },
    {
        "id": "stripe",
        "title": "stripe.py — payment API call",
        "sev": "CRIT",
        "code": """STRIPE_KEY = "sk_live_4xK2mLpQr8nZ..."
requests.post(url, headers={"Authorization": f"Bearer {STRIPE_KEY}"})""",
        "bug": "hardcoded API key leaks in version control",
        "keywords": ["hardcoded", "secret", "api key", "env", "version control", "credentials"],
    },
    {
        "id": "config",
        "title": "config.py — JSON file reader",
        "sev": "CRIT",
        "code": """def read_config(path):
    f = open(path, 'r')
    data = json.load(f)
    return data""",
        "bug": "resource leak — file handle not closed on exception",
        "keywords": ["resource leak", "with statement", "context manager", "file handle", "close"],
    },
    {
        "id": "metrics",
        "title": "metrics.py — request counter",
        "sev": "CRIT",
        "code": """request_count = 0

def handle_request():
    global request_count
    request_count += 1  # called from multiple threads""",
        "bug": "race condition — += is not atomic under concurrent access",
        "keywords": ["race condition", "thread", "lock", "atomic", "concurrent", "thread-safe"],
    },
    {
        "id": "ping",
        "title": "utils.py — host ping check",
        "sev": "CRIT",
        "code": """def ping(host):
    cmd = f"ping -c 1 {host}"
    return subprocess.run(cmd, shell=True, capture_output=True)""",
        "bug": "command injection — unsanitized input passed to shell",
        "keywords": ["command injection", "shell=true", "shell injection", "arbitrary command", "subprocess"],
    },
    {
        "id": "last_n",
        "title": "list_utils.py — last N items",
        "sev": "MOD",
        "code": """def last_n(items, n):
    return items[len(items)-n : len(items)+1]""",
        "bug": "off-by-one — slice end is len(items)+1, returns n+1 items",
        "keywords": ["off-by-one", "len(items)+1", "slice", "n+1", "returns n+1"],
    },
    {
        "id": "divide",
        "title": "math.py — safe division",
        "sev": "OK",
        "code": """def safe_divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b""",
        "bug": None,
        "keywords": ["no issue", "looks correct", "lgtm", "no bug", "correct", "fine"],
    },
    {
        "id": "profile",
        "title": "users.py — profile name lookup",
        "sev": "MOD",
        "code": """def get_display_name(user):
    return user['profile']['name'].strip()""",
        "bug": "KeyError — no guard if 'profile' or 'name' key is absent",
        "keywords": ["keyerror", "key error", ".get(", "missing key", "key guard", "absent"],
    },
]

VAL_TASKS = [
    {
        "id": "timing",
        "title": "auth.py — token comparison",
        "sev": "CRIT",
        "code": """def verify_token(user_token, stored_token):
    return user_token == stored_token""",
        "bug": "timing attack — == leaks token info via response time",
        "keywords": ["timing", "compare_digest", "hmac", "constant-time", "timing attack", "side-channel"],
    },
    {
        "id": "ssrf",
        "title": "proxy.py — URL fetcher",
        "sev": "CRIT",
        "code": """def fetch_url(user_input_url):
    return requests.get(user_input_url).text""",
        "bug": "SSRF — user can target internal services",
        "keywords": ["ssrf", "server-side request", "internal", "localhost", "url validation", "request forgery"],
    },
    {
        "id": "traversal",
        "title": "files.py — file reader",
        "sev": "CRIT",
        "code": """def read_file(filename):
    path = "/data/files/" + filename
    with open(path) as f:
        return f.read()""",
        "bug": "path traversal — ../../../etc/passwd",
        "keywords": ["path traversal", "../", "directory traversal", "sanitize", "realpath", "arbitrary file"],
    },
    {
        "id": "db_leak",
        "title": "db.py — connection pool",
        "sev": "MOD",
        "code": """def get_connection():
    conn = psycopg2.connect(DATABASE_URL)
    return conn  # caller must close""",
        "bug": "resource leak — connection not closed if caller forgets",
        "keywords": ["resource leak", "context manager", "with", "close", "finally", "connection"],
    },
]

# ── Skill document ────────────────────────────────────────────────────────────

INITIAL_SKILL = """# Code Reviewer

Review submitted code for bugs, security vulnerabilities, and correctness issues.

## Approach
- Read the full code carefully before commenting
- Prioritize critical issues (security, data loss) over style
- Be specific: name the exact problem and suggest how to fix it
- If the code is correct, say so clearly
"""

AGENT_SYSTEM_TEMPLATE = "{skill}"

OPTIMIZER_SYSTEM = """You are an expert code review skill optimizer. You receive a skill.md document used by a code review agent, plus a set of failed reviews (cases where the agent missed real bugs).

Your job: propose targeted edits to improve the skill document so the agent will catch these patterns in the future.

Output a JSON object with EXACTLY this structure:
{
  "analysis": "2-3 sentences identifying the pattern across these failures",
  "edits": [
    {"op": "append", "text": "## Section Title\\n- Specific actionable rule: pattern → fix"},
    {"op": "append", "text": "- Another rule on a new line"}
  ]
}

Rules for good edits:
- Write SPECIFIC patterns with SPECIFIC fixes (e.g. "f-string in SQL query → parameterized query")
- Each edit.text is a markdown block to append to the skill doc
- Group related rules under a section header (## Security, ## Resources, etc.)
- Maximum L edits (specified in the prompt)
- Focus on the common pattern across all failures, not edge cases
"""


# ── Scoring ───────────────────────────────────────────────────────────────────

def score(review: str, task: dict) -> bool:
    """Keyword-match scoring: did the review catch the bug?"""
    if task["bug"] is None:
        return True  # No bug — always pass (we don't penalize false negatives here)
    review_lower = review.lower()
    return any(kw in review_lower for kw in task["keywords"])


# ── Agent + optimizer calls ───────────────────────────────────────────────────

async def run_agent(skill: str, task: dict) -> dict:
    """Run the frozen agent on one task. Returns the trace result."""
    client = anthropic.AsyncAnthropic(api_key=API_KEY)

    prompt = (
        f"Review this code for bugs or security issues:\n\n"
        f"File: {task['title']}\n\n"
        f"```python\n{task['code']}\n```\n\n"
        f"What issues do you see, if any? Be specific."
    )

    try:
        msg = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=350,
            system=skill,
            messages=[{"role": "user", "content": prompt}],
        )
        review = msg.content[0].text
    except Exception as e:
        review = f"[agent error: {e}]"

    passed = score(review, task)
    return {
        "id": task["id"],
        "title": task["title"],
        "sev": task["sev"],
        "code": task["code"],
        "bug": task["bug"],
        "review": review,
        "passed": passed,
    }


async def stream_optimizer(skill: str, failures: list[dict], lr: int) -> AsyncGenerator[dict, None]:
    """Stream the optimizer's analysis. Yields token events then a final result event."""
    client = anthropic.AsyncAnthropic(api_key=API_KEY)

    failure_text = ""
    for i, f in enumerate(failures, 1):
        failure_text += f"\n### Failure {i}: {f['title']}\n"
        failure_text += f"Bug present: {f['bug']}\n"
        failure_text += f"Code:\n```python\n{f['code']}\n```\n"
        truncated = f['review'][:500] + ("..." if len(f['review']) > 500 else "")
        failure_text += f"Agent's review (missed the bug):\n{truncated}\n"

    user = (
        f"## Current Skill\n```\n{skill}\n```\n\n"
        f"## Failed Reviews (propose at most L={lr} edits)\n"
        f"{failure_text}"
    )

    response_text = ""
    try:
        async with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            system=OPTIMIZER_SYSTEM,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            async for text in stream.text_stream:
                response_text += text
                yield {"type": "reflect_token", "token": text}
    except Exception as e:
        yield {"type": "reflect_token", "token": f"\n[optimizer error: {e}]"}

    # Parse the JSON result
    try:
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            parsed = json.loads(json_match.group())
            analysis = parsed.get("analysis", "")
            edits = parsed.get("edits", [])
        else:
            analysis = response_text[:400]
            edits = []
    except Exception:
        analysis = response_text[:400]
        edits = []

    yield {"type": "reflect_result", "analysis": analysis, "edits": edits}


def apply_edits(skill: str, edits: list[dict]) -> str:
    """Apply patch edits to the skill document."""
    result = skill.rstrip()
    for edit in edits:
        text = edit.get("text", "").strip()
        if text:
            result += "\n\n" + text
    return result + "\n"


# ── Main training loop ────────────────────────────────────────────────────────

async def training_loop(lr: int, epochs: int) -> AsyncGenerator[str, None]:
    """The actual SkillOpt loop. Yields SSE-formatted strings."""

    def emit(data: dict) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    if not API_KEY:
        yield emit({"type": "error", "message": "ANTHROPIC_API_KEY not set. Set it and restart the server."})
        return

    skill = INITIAL_SKILL
    baseline_score = 0.0
    score_history: list[float] = []

    yield emit({"type": "start", "skill": skill, "epochs": epochs, "lr": lr,
                "n_train": len(TRAIN_TASKS), "n_val": len(VAL_TASKS)})
    await asyncio.sleep(0)

    for epoch in range(epochs):
        yield emit({"type": "epoch_start", "epoch": epoch, "skill": skill})
        await asyncio.sleep(0)

        # ── 1. ROLLOUT ────────────────────────────────────────────────────────
        yield emit({"type": "phase", "phase": "ROLLOUT", "epoch": epoch,
                    "desc": f"Running agent on {len(TRAIN_TASKS)} training PRs"})
        await asyncio.sleep(0)

        traces: list[dict] = []
        for task in TRAIN_TASKS:
            yield emit({"type": "task_start", "task_id": task["id"], "title": task["title"]})
            await asyncio.sleep(0)

            result = await run_agent(skill, task)
            traces.append(result)

            yield emit({"type": "task_result", **result})
            await asyncio.sleep(0.05)

        passes = sum(1 for t in traces if t["passed"])
        train_score = passes / len(TRAIN_TASKS)
        yield emit({"type": "rollout_done", "epoch": epoch, "score": train_score,
                    "passes": passes, "total": len(TRAIN_TASKS)})
        await asyncio.sleep(0)

        # ── 2. REFLECT ────────────────────────────────────────────────────────
        failures = [t for t in traces if not t["passed"]]
        edits: list[dict] = []
        analysis = ""

        if not failures:
            yield emit({"type": "phase", "phase": "REFLECT", "epoch": epoch,
                        "desc": "No failures — nothing to reflect on", "skip": True})
            yield emit({"type": "reflect_result", "analysis": "Perfect training score — no edits needed.",
                        "edits": []})
        else:
            yield emit({"type": "phase", "phase": "REFLECT", "epoch": epoch,
                        "desc": f"Optimizer analyzing {len(failures)} failures",
                        "n_failures": len(failures)})
            await asyncio.sleep(0)

            async for event in stream_optimizer(skill, failures, lr):
                if event["type"] == "reflect_result":
                    analysis = event["analysis"]
                    edits = event["edits"]
                yield emit(event)
                await asyncio.sleep(0)

        # ── 3. SELECT ─────────────────────────────────────────────────────────
        selected = edits[:lr]
        candidate = apply_edits(skill, selected)

        yield emit({"type": "phase", "phase": "SELECT", "epoch": epoch,
                    "desc": f"Selecting top {lr} edits (learning rate = {lr})"})
        yield emit({"type": "select_result", "selected": selected,
                    "total_proposed": len(edits), "lr": lr, "candidate_skill": candidate})
        await asyncio.sleep(0)

        # ── 4. GATE ───────────────────────────────────────────────────────────
        yield emit({"type": "phase", "phase": "GATE", "epoch": epoch,
                    "desc": f"Validating on {len(VAL_TASKS)} held-out tasks"})
        await asyncio.sleep(0)

        val_traces: list[dict] = []
        for task in VAL_TASKS:
            result = await run_agent(candidate, task)
            val_traces.append(result)
            yield emit({"type": "val_task", **result})
            await asyncio.sleep(0.05)

        val_passes = sum(1 for t in val_traces if t["passed"])
        val_score = val_passes / len(VAL_TASKS)
        accepted = val_score > baseline_score or (val_score == baseline_score and epoch == 0)

        if accepted:
            skill = candidate
            baseline_score = val_score

        score_history.append(val_score)

        yield emit({
            "type": "gate_result",
            "epoch": epoch,
            "val_score": val_score,
            "baseline": baseline_score,
            "accepted": accepted,
            "delta": val_score - (score_history[-2] if len(score_history) >= 2 else 0.0),
            "new_skill": skill if accepted else None,
        })
        yield emit({"type": "epoch_done", "epoch": epoch, "train_score": train_score,
                    "val_score": val_score, "skill": skill})
        await asyncio.sleep(0)

    yield emit({"type": "done", "final_skill": skill, "final_score": baseline_score,
                "score_history": score_history})


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/train")
async def train(lr: int = 3, epochs: int = 4):
    async def generate():
        async for chunk in training_loop(lr, epochs):
            yield chunk

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {"ok": True, "key_set": bool(API_KEY)}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("SkillOpt Server")
    print("=" * 40)
    print(f"API key: {'set (ok)' if API_KEY else 'NOT SET - set ANTHROPIC_API_KEY'}")
    print(f"Open skillopt_demo.html in your browser")
    print("=" * 40)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
