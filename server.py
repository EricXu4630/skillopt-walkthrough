#!/usr/bin/env python3
"""SkillOpt live training server — MEDDIC Sales Qualification domain.

Trains a skill.md document to apply the MEDDIC sales qualification framework
to CRM opportunity notes. The sparse initial skill gives generic deal opinions;
the optimizer discovers MEDDIC and teaches it to the agent over training epochs.

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

# ── Training tasks (CRM opportunity notes) ────────────────────────────────────
# Each task is a realistic deal snapshot from a sales CRM.
# The "bug" field describes the MEDDIC gap the agent must surface.
# Keywords are the specific terms a MEDDIC-trained agent would use.

TRAIN_TASKS = [
    {
        "id": "acme_db",
        "title": "Acme Corp — Database Migration",
        "sev": "CRIT",
        "code": """Account: Acme Corp (2,400 employees, manufacturing)
Stage: Discovery
Contact: Mark T., IT Director
Notes: Team is losing ~$500k/year in manual data reconciliation across 3 legacy systems.
Mark is very engaged and wants to move fast. He mentioned "the business is frustrated."
No budget conversation yet. Have not met anyone in Finance or the C-suite.
Next step: technical deep-dive with Mark's team next week.""",
        "bug": "Economic Buyer not identified — only speaking to IT Director, no Finance or C-suite contact",
        "keywords": ["economic buyer", "not identified", "missing", "not qualified", "needs discovery", "c-suite", "finance"],
    },
    {
        "id": "retailco_analytics",
        "title": "RetailCo — Analytics Platform",
        "sev": "OK",
        "code": """Account: RetailCo (8,000 employees, retail)
Stage: Evaluation
Contact: Sarah Chen, VP Analytics (internal champion); CFO Patricia M. (economic buyer)
Notes: CFO personally requested the evaluation after seeing competitor case study.
Quantified pain: losing 3% market share due to 2-week reporting lag = $4.2M/year.
Decision criteria: Salesforce integration (must-have), SOC2 certification, < 6-week implementation.
Decision process: CFO + CTO sign off, legal review, then procurement. Target: end of Q3.
Sarah is actively selling internally, blocked 2 competing solutions already.
Budget: $850k approved in current fiscal year.""",
        "bug": None,
        "keywords": ["qualified", "strong", "champion", "economic buyer"],
    },
    {
        "id": "healthtech_compliance",
        "title": "HealthTech — Compliance Platform",
        "sev": "CRIT",
        "code": """Account: HealthTech Inc. (300 employees, digital health startup)
Stage: Discovery
Contact: James L., Engineering Manager
Notes: HIPAA compliance audit deadline in 6 months, currently failing 4 controls.
Real urgency — regulators gave them a formal notice. James says "we have to fix this."
Budget: James thinks it's around $80-100k but hasn't confirmed with anyone.
Decision: James mentioned a "steering committee" but couldn't name who's on it.
Champion: James seems supportive but worried about job security if this goes wrong.
Have only done one call so far.""",
        "bug": "Economic Buyer not confirmed, Decision Process opaque, Champion is low-credibility",
        "keywords": ["economic buyer", "decision process", "champion", "not qualified", "needs discovery", "missing"],
    },
    {
        "id": "finserv_risk",
        "title": "FinServ Capital — Risk Platform",
        "sev": "OK",
        "code": """Account: FinServ Capital (1,200 employees, financial services)
Stage: Proposal
Contact: Tom W., Risk Manager (champion); David K., CRO (economic buyer)
Notes: CRO David confirmed $3M annual cost of manual risk reviews, wants 70% reduction.
Decision criteria: SOC2 + ISO 27001 required, must integrate with Bloomberg Terminal.
Decision process: CRO approves, then legal + procurement (2-week SLA each). Board notified.
Timeline: Must deploy before Q1 earnings cycle. 3 competing vendors evaluated.
Tom actively coaching us: shared competitor pricing, introduced us to legal team proactively.
Budget: $1.2M allocated, approved in last board meeting.""",
        "bug": None,
        "keywords": ["qualified", "strong", "economic buyer", "champion"],
    },
    {
        "id": "manufacturing_erp",
        "title": "GlobalMfg — ERP Upgrade",
        "sev": "CRIT",
        "code": """Account: GlobalMfg (5,000 employees, industrial manufacturing)
Stage: Discovery
Contact: Linda P., Operations Manager; Kevin S., IT Manager
Notes: Production scheduling delays costing roughly $1M per quarter per Linda's estimate.
Both Linda and Kevin are very enthusiastic. Lots of good conversations.
Kevin mentioned the COO "would eventually need to sign off" but hasn't set up that meeting.
Decision: Not clear — Kevin thinks IT decides, Linda thinks Ops decides. Neither confirmed.
Competition: We're the first vendor they've spoken to.
Next step: Linda wants to do a pilot in one plant.""",
        "bug": "Economic Buyer not identified, Decision Process undefined — two contacts with conflicting views on ownership",
        "keywords": ["economic buyer", "decision process", "missing", "not qualified", "needs discovery", "conflicting"],
    },
    {
        "id": "startup_hr",
        "title": "StartupXYZ — HR Platform",
        "sev": "MOD",
        "code": """Account: StartupXYZ (90 employees, SaaS)
Stage: Discovery
Contact: Maya R., CEO
Notes: CEO handles HR directly, describes onboarding as "a mess." Wants to automate it.
No dollar figure on the problem — "it's just a pain, we're growing fast."
CEO is the decision maker and budget holder (confirmed). She can sign contracts up to $50k.
No other vendors being evaluated. Timeline: "sometime this year."
Strong interest but no urgency driver beyond general growth.""",
        "bug": "Metrics are unquantified — no dollar value on the problem; Identify Pain is vague, no urgency",
        "keywords": ["metrics", "weak", "missing", "unquantified", "needs discovery", "pain", "urgency"],
    },
    {
        "id": "legalfirm_docs",
        "title": "Meridian Legal — Document Review",
        "sev": "OK",
        "code": """Account: Meridian Legal LLP (180 attorneys, law firm)
Stage: Proposal
Contact: Emily F., IT Director (champion); Managing Partners (economic buyers, committee of 3)
Notes: Paralegals averaging 22 hrs/week on document review @ $95/hr fully-loaded = $430k/year.
Partners committee approves all tech purchases > $25k (confirmed, met with two of three).
Decision criteria: ABA ethics compliance, on-premise deployment option, 99.9% uptime SLA.
Decision process: IT Director recommends, Partners vote (monthly meeting), then contract.
Emily has prepared a business case document and shared our proposal with all three partners.
No competitive bake-off — we're the preferred vendor. Timeline: next Partners meeting in 6 weeks.""",
        "bug": None,
        "keywords": ["qualified", "strong", "champion", "economic buyer"],
    },
    {
        "id": "edtech_lms",
        "title": "Riverside USD — Learning Management",
        "sev": "MOD",
        "code": """Account: Riverside Unified School District (52,000 students)
Stage: Evaluation
Contact: Dr. Chen, Superintendent (economic buyer, budget owner)
Notes: Current LMS contract expiring. 40% of teachers report system is "unusable."
Pain: Teachers spending 5 hrs/week on workarounds = $2.8M in lost instructional time.
Superintendent has budget authority and is the internal champion.
Decision criteria: Accessibility (WCAG 2.1 AA), Google Classroom integration, offline mode.
Decision process: Superintendent recommends → School Board vote (quorum required).
Next School Board meeting with tech on agenda: 9 months away.
Budget: $1.4M approved in next fiscal year (starts in 4 months).""",
        "bug": "Decision Process has a 9-month board vote delay — critical timeline risk not flagged",
        "keywords": ["decision process", "timeline", "risk", "board", "delay", "9 months", "needs discovery"],
    },
]

VAL_TASKS = [
    {
        "id": "saas_security",
        "title": "CloudSec Inc. — Security Platform",
        "sev": "OK",
        "code": """Account: CloudSec Inc. (650 employees, B2B SaaS)
Stage: Proposal
Contact: Rachel M., CISO (economic buyer + champion)
Notes: Recent breach cost $2.1M in remediation + reputational damage. Board mandated fix.
CISO controls security budget ($3M annually), is the decision maker and internal advocate.
Decision criteria: SOC2 Type II, zero-trust architecture, API-first integration.
Decision process: CISO approves, CFO countersigns > $500k. Legal reviews all contracts.
Timeline: Must be deployed before next annual audit in 5 months. Competing against 2 vendors.
Budget: $800k pre-approved by board resolution.""",
        "bug": None,
        "keywords": ["qualified", "strong", "economic buyer", "champion"],
    },
    {
        "id": "telecom_network",
        "title": "NexTel — Network Management",
        "sev": "CRIT",
        "code": """Account: NexTel Communications (12,000 employees, telecom)
Stage: Discovery
Contact: Bob K., Network Architect; Sandra P., VP Engineering
Notes: Network outages costing $800k/month in SLA penalties. Bob has been tracking metrics carefully.
Sandra is engaged and sees the value. She mentioned the CTO "has final say on big purchases."
Bob would be the primary user and says he could be an internal advocate.
Decision: Sandra thinks CTO decides, but wasn't sure about procurement involvement.
Budget: Sandra believes there's budget but "we'd have to make the case to CTO."
Have not met the CTO. No formal evaluation criteria established.""",
        "bug": "Economic Buyer not identified (CTO unmet), Decision Process unclear, Champion is mid-level only",
        "keywords": ["economic buyer", "champion", "decision process", "missing", "not qualified"],
    },
    {
        "id": "insurance_claims",
        "title": "Pacific Insurance — Claims Processing",
        "sev": "CRIT",
        "code": """Account: Pacific Insurance Group (3,400 employees, insurance)
Stage: Evaluation
Contact: Diane L., Claims Director; Peter H., VP Operations
Notes: Manual claims processing taking 14 days avg vs. industry benchmark of 5 days.
Cost of delay: $3.5M/year in customer churn + regulatory exposure.
Peter confirmed there's budget in next year's OpEx. "Diane owns this decision."
Decision criteria: Must integrate with Guidewire ClaimCenter, state DOI compliance.
Decision process: Diane recommends, Peter approves — both in the room. Legal reviews.
Diane has a competing initiative (process re-engineering) that could solve same problem.
No explicit champion identified beyond Diane who is also the key contact.""",
        "bug": "Competing internal initiative (process re-engineering) is a disqualifying risk; Champion ambiguous",
        "keywords": ["competing", "risk", "champion", "initiative", "needs discovery"],
    },
    {
        "id": "consulting_analytics",
        "title": "Apex Consulting — Analytics Suite",
        "sev": "OK",
        "code": """Account: Apex Consulting (420 employees, management consulting)
Stage: Proposal
Contact: Marcus T., Managing Director (economic buyer); Priya S., Head of Analytics (champion)
Notes: Consultants spending 30% of billable time on data prep = $5.2M in unbillable hours/year.
Marcus controls the budget ($2M/year tech budget), approved evaluation, attended 2 demos.
Priya built the business case and presented to the leadership team. Fully committed.
Decision criteria: Python/R integration, on-prem option, multi-tenant client data isolation.
Decision process: Marcus approves, COO notified (not a blocker), contracts through legal.
Timeline: Wants to deploy before Q1 client engagements (8 weeks). Only vendor evaluated.
Budget: $640k identified from decommissioned legacy tools.""",
        "bug": None,
        "keywords": ["qualified", "strong", "champion", "economic buyer"],
    },
]

# ── Skill document ────────────────────────────────────────────────────────────
# Deliberately sparse: no MEDDIC framework, no output structure, no scoring rubric.
# The optimizer's job is to discover and teach the MEDDIC framework over epochs.

INITIAL_SKILL = """# Sales Opportunity Qualifier

Review sales opportunity notes and assess whether the deal should advance to the next stage.
"""

# ── Scoring ───────────────────────────────────────────────────────────────────
# Two-component score:
#   FORMAT (60%): must score all 6 MEDDIC components with the required headers
#   DEAL ASSESSMENT (40%): must correctly identify whether deal is qualified or at-risk
#
# Without MEDDIC in the skill, Haiku gives generic opinions ("this looks promising").
# With MEDDIC in the skill, it produces a structured 6-component scorecard.

FORMAT_KEYWORDS = [
    "**metrics**",
    "**economic buyer**",
    "**decision criteria**",
    "**decision process**",
    "**identify pain**",
    "**champion**",
    "deal verdict",
]

def score_task(review: str, task: dict) -> dict:
    rv = review.lower()

    format_hits = sum(1 for kw in FORMAT_KEYWORDS if kw in rv)
    format_score = format_hits / len(FORMAT_KEYWORDS)

    if task["bug"] is None:
        # Good deal: must say "qualified" and not flag false risks
        assessment_correct = "qualified" in rv and "not qualified" not in rv
        assessment_desc = "correctly identifies deal as QUALIFIED"
    else:
        # Deal has a MEDDIC gap: must surface it using the right keywords
        assessment_correct = any(kw in rv for kw in task["keywords"])
        assessment_desc = f"surfaces risk: {task['bug']}"

    combined = 0.6 * format_score + 0.4 * (1.0 if assessment_correct else 0.0)
    passed = combined >= 0.65

    fail_reasons = []
    if not assessment_correct:
        if task["bug"] is None:
            fail_reasons.append("missed verdict: should identify this deal as QUALIFIED")
        else:
            fail_reasons.append(f"missed MEDDIC gap: {task['bug']}")
    if format_score < 0.5:
        fail_reasons.append(
            "format non-compliant: must score all 6 MEDDIC components "
            "(**Metrics** / **Economic Buyer** / **Decision Criteria** / "
            "**Decision Process** / **Identify Pain** / **Champion**) and include Deal Verdict"
        )

    return {
        "passed": passed,
        "format_score": round(format_score, 2),
        "assessment_correct": assessment_correct,
        "combined": round(combined, 2),
        "fail_reasons": fail_reasons,
    }


# ── Optimizer prompt ──────────────────────────────────────────────────────────

OPTIMIZER_SYSTEM = """You are an expert sales methodology skill optimizer. You receive a skill.md document used by a sales qualification agent, plus a set of failed assessments with explicit failure reasons.

Your job: propose targeted edits to the skill document so the agent correctly qualifies deals.

There are two failure types:
1. "format non-compliant" — the agent did not score all 6 MEDDIC components. Add the MEDDIC framework and required output format.
2. "missed MEDDIC gap" — the agent gave a generic opinion and missed a specific deal risk. Add explicit detection rules for that gap type.

The MEDDIC framework (for reference when proposing edits):
- Metrics: Quantified business impact ($value of the problem)
- Economic Buyer: Who controls the budget and can sign the contract
- Decision Criteria: What criteria will they use to choose a vendor
- Decision Process: The steps, approvals, and timeline to purchase
- Identify Pain: The specific business pain and what drives urgency
- Champion: Who is selling internally on your behalf

Required output format for opportunity assessments:
  ## Qualification
  **Metrics**: [STRONG/WEAK/MISSING] — [evidence]
  **Economic Buyer**: [STRONG/WEAK/MISSING] — [evidence]
  **Decision Criteria**: [STRONG/WEAK/MISSING] — [evidence]
  **Decision Process**: [STRONG/WEAK/MISSING] — [evidence]
  **Identify Pain**: [STRONG/WEAK/MISSING] — [evidence]
  **Champion**: [STRONG/WEAK/MISSING] — [evidence]

  **Deal Verdict**: QUALIFIED | NOT QUALIFIED | NEEDS DISCOVERY
  **Risk Flags**: [specific gaps to close before advancing stage]

Output a JSON object:
{
  "analysis": "2-3 sentences identifying the dominant failure pattern and what edit fixes it",
  "edits": [
    {"op": "append", "text": "## Section\\n- Specific rule or full format template"}
  ]
}

Rules:
- If most failures are format-related, the top edit must add the full MEDDIC framework and output format template
- If most failures miss specific gaps (e.g. Economic Buyer unmet), add explicit detection rules
- Be specific and actionable — vague guidance doesn't change agent behavior
- Maximum L edits (specified in prompt); prioritize highest-impact changes first
"""


# ── Agent + optimizer calls ───────────────────────────────────────────────────

async def run_agent(skill: str, task: dict) -> dict:
    client = anthropic.AsyncAnthropic(api_key=API_KEY)

    prompt = (
        f"Please qualify this sales opportunity:\n\n"
        f"Deal: {task['title']}\n\n"
        f"{task['code']}"
    )

    try:
        msg = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            system=skill,
            messages=[{"role": "user", "content": prompt}],
        )
        review = msg.content[0].text
    except Exception as e:
        review = f"[agent error: {e}]"

    result = score_task(review, task)

    return {
        "id": task["id"],
        "title": task["title"],
        "sev": task["sev"],
        "code": task["code"],
        "bug": task["bug"],
        "review": review,
        "passed": result["passed"],
        "format_score": result["format_score"],
        "bug_caught": result["assessment_correct"],
        "combined": result["combined"],
        "fail_reasons": result["fail_reasons"],
    }


async def stream_optimizer(skill: str, failures: list[dict], lr: int) -> AsyncGenerator[dict, None]:
    client = anthropic.AsyncAnthropic(api_key=API_KEY)

    failure_text = ""
    for i, f in enumerate(failures, 1):
        failure_text += f"\n### Failure {i}: {f['title']}\n"
        failure_text += f"MEDDIC gap: {f['bug'] or 'None (correct deal, missed qualified verdict)'}\n"
        failure_text += f"Failure reason: {'; '.join(f['fail_reasons'])}\n"
        failure_text += f"Opportunity notes:\n{f['code']}\n"
        truncated = f['review'][:500] + ("..." if len(f['review']) > 500 else "")
        failure_text += f"Agent's assessment:\n{truncated}\n"

    user = (
        f"## Current Skill\n```\n{skill}\n```\n\n"
        f"## Failed Assessments (propose at most L={lr} edits)\n"
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
    result = skill.rstrip()
    for edit in edits:
        text = edit.get("text", "").strip()
        if text:
            result += "\n\n" + text
    return result + "\n"


# ── Main training loop ────────────────────────────────────────────────────────

async def training_loop(lr: int, epochs: int) -> AsyncGenerator[str, None]:

    def emit(data: dict) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    if not API_KEY:
        yield emit({"type": "error", "message": "ANTHROPIC_API_KEY not set."})
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
                    "desc": f"Qualifying {len(TRAIN_TASKS)} pipeline opportunities"})
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
            yield emit({"type": "reflect_result", "analysis": "Perfect training score — no edits needed.", "edits": []})
        else:
            yield emit({"type": "phase", "phase": "REFLECT", "epoch": epoch,
                        "desc": f"Optimizer analyzing {len(failures)} failed qualifications",
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
                    "desc": f"Validating on {len(VAL_TASKS)} held-out opportunities"})
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


if __name__ == "__main__":
    import uvicorn
    print("SkillOpt Server — MEDDIC Sales Qualification")
    print("=" * 40)
    print(f"API key: {'set (ok)' if API_KEY else 'NOT SET'}")
    print("Open skillopt_demo.html in your browser")
    print("=" * 40)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
