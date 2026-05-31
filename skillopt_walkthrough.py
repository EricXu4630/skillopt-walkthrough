#!/usr/bin/env python3
"""
SkillOpt Walkthrough — Minimal Annotated Implementation
========================================================
Trains a skill document for a frozen LLM agent on arithmetic word problems.

The core insight: instead of updating model *weights*, SkillOpt updates
the *instructions* the model reads (a Markdown file called skill.md).
The frozen model never changes. Only the text it reads improves.

    Deep Learning          SkillOpt
    ─────────────          ────────
    model weights    →     skill.md
    forward pass     →     rollout  (agent runs tasks)
    loss function    →     score    (pass/fail per task)
    backprop         →     reflect  (LLM analyzes failures → edit patches)
    gradient clip    →     select   (limit edits per step = "learning rate")
    SGD step         →     update   (apply edits to skill.md)
    validation set   →     gate     (reject edits that hurt val score)

Run (no API key needed):
    python skillopt_walkthrough.py

Run with real Claude API calls:
    ANTHROPIC_API_KEY=sk-... python skillopt_walkthrough.py --live
"""

import os, sys, json, textwrap, time
from dataclasses import dataclass
from typing import Optional

# ─── CONFIG ──────────────────────────────────────────────────────────────────

LEARNING_RATE = 3       # max edits allowed per step  (= gradient clipping)
NUM_EPOCHS    = 3       # how many training epochs to run
LIVE_MODE     = "--live" in sys.argv

# ─── DATA TYPES ──────────────────────────────────────────────────────────────

@dataclass
class Trace:
    """One agent run: question → answer → score."""
    task_id:  str
    question: str
    answer:   str     # what the agent actually said
    expected: str     # ground truth
    score:    float   # 1.0 = pass, 0.0 = fail

    @property
    def passed(self) -> bool:
        return self.score >= 1.0


@dataclass
class Edit:
    """One proposed change to the skill document."""
    op:     str   # "append" | "replace" | "delete"
    text:   str   # new text to add / replacement text
    target: str = ""  # for replace/delete: the text to find and change


# ─── SKILL DOCUMENT ──────────────────────────────────────────────────────────

class SkillDoc:
    """
    The skill document is the 'trainable state' in SkillOpt.

    It's a plain Markdown string that gets injected into the agent's
    system prompt before every task. Training never touches the model
    weights — it only edits this text.

    Think of it as a living instruction manual that gets smarter each epoch.
    """

    def __init__(self, text: str = ""):
        self.text = text

    def apply(self, edit: Edit) -> "SkillDoc":
        """Return a new SkillDoc with one edit applied (immutable)."""
        s = self.text
        if edit.op == "append":
            s = s.rstrip() + "\n" + edit.text
        elif edit.op == "replace" and edit.target in s:
            s = s.replace(edit.target, edit.text, 1)
        elif edit.op == "delete" and edit.target in s:
            s = s.replace(edit.target, "", 1)
        return SkillDoc(s)

    def apply_all(self, edits: list[Edit]) -> "SkillDoc":
        """Apply multiple edits sequentially."""
        doc = self
        for e in edits:
            doc = doc.apply(e)
        return doc

    def __str__(self):
        return self.text


# ─── MOCK DATA ───────────────────────────────────────────────────────────────
# Pre-baked results for mock mode (no API key needed).
# These simulate 3 epochs of training on arithmetic word problems.
# Each epoch: skill improves → score improves → gate accepts.

MOCK_TRAINING_TASKS = [
    {"id": "t1",  "question": "What is 15% of 200?",                          "answer": "30"},
    {"id": "t2",  "question": "If a shirt costs $40 and is 25% off, what is the sale price?", "answer": "30"},
    {"id": "t3",  "question": "A car travels 60 miles per hour for 2.5 hours. How far does it go?", "answer": "150"},
    {"id": "t4",  "question": "What is three-quarters of 48?",                 "answer": "36"},
    {"id": "t5",  "question": "If you split $72 evenly among 8 people, how much does each get?", "answer": "9"},
    {"id": "t6",  "question": "A recipe needs 2/3 cup of sugar. If you triple the recipe, how much sugar?", "answer": "2"},
    {"id": "t7",  "question": "What is 8 squared minus 4 cubed?",              "answer": "0"},
    {"id": "t8",  "question": "A temperature drops from 5°C to -12°C. How many degrees did it drop?", "answer": "17"},
]

MOCK_VAL_TASKS = [
    {"id": "v1", "question": "What is 30% of 90?",                            "answer": "27"},
    {"id": "v2", "question": "If a price increases by 20% from $50, what is the new price?", "answer": "60"},
    {"id": "v3", "question": "What is 5 squared plus 3 cubed?",               "answer": "52"},
    {"id": "v4", "question": "A temperature changes from -8°C to 3°C. What is the change?", "answer": "11"},
]

# Simulated traces per epoch (what the agent would produce)
MOCK_TRACES = {
    0: [  # Epoch 0: bare skill, lots of failures
        Trace("t1", "What is 15% of 200?",                 "15",   "30",  0.0),  # forgot to apply %
        Trace("t2", "If a shirt costs $40, 25% off...",    "10",   "30",  0.0),  # calculated discount, not price
        Trace("t3", "Car travels 60mph for 2.5 hours...",  "150",  "150", 1.0),  # easy multiplication, correct
        Trace("t4", "What is three-quarters of 48?",       "36",   "36",  1.0),  # fraction, correct
        Trace("t5", "Split $72 among 8 people...",         "9",    "9",   1.0),  # division, correct
        Trace("t6", "Recipe needs 2/3 cup, tripled...",    "6/3",  "2",   0.0),  # gave fraction instead of simplified number
        Trace("t7", "What is 8² minus 4³?",                "0",    "0",   1.0),  # powers, correct
        Trace("t8", "Temp drops from 5°C to -12°C...",    "-7",   "17",  0.0),  # subtracted instead of finding absolute difference
    ],
    1: [  # Epoch 1: percentage + fraction rules added
        Trace("t1", "What is 15% of 200?",                 "30",   "30",  1.0),  # percentage fixed
        Trace("t2", "If a shirt costs $40, 25% off...",    "30",   "30",  1.0),  # sale price fixed
        Trace("t3", "Car travels 60mph for 2.5 hours...",  "150",  "150", 1.0),  # still correct
        Trace("t4", "What is three-quarters of 48?",       "36",   "36",  1.0),  # still correct
        Trace("t5", "Split $72 among 8 people...",         "9",    "9",   1.0),  # still correct
        Trace("t6", "Recipe needs 2/3 cup, tripled...",    "2",    "2",   1.0),  # fraction simplified correctly
        Trace("t7", "What is 8² minus 4³?",                "0",    "0",   1.0),  # still correct
        Trace("t8", "Temp drops from 5°C to -12°C...",    "-7",   "17",  0.0),  # still wrong: sign direction
    ],
    2: [  # Epoch 2: signed difference rule added
        Trace("t1", "What is 15% of 200?",                 "30",   "30",  1.0),
        Trace("t2", "If a shirt costs $40, 25% off...",    "30",   "30",  1.0),
        Trace("t3", "Car travels 60mph for 2.5 hours...",  "150",  "150", 1.0),
        Trace("t4", "What is three-quarters of 48?",       "36",   "36",  1.0),
        Trace("t5", "Split $72 among 8 people...",         "9",    "9",   1.0),
        Trace("t6", "Recipe needs 2/3 cup, tripled...",    "2",    "2",   1.0),
        Trace("t7", "What is 8² minus 4³?",                "0",    "0",   1.0),
        Trace("t8", "Temp drops from 5°C to -12°C...",    "17",   "17",  1.0),  # signed diff fixed
    ],
}

# Simulated optimizer analysis per epoch
MOCK_ANALYSIS = {
    0: (
        "3 of 5 failures share a pattern: percentage problems.\n"
        "  • t1: computed 15 (the %) instead of 15% × 200 = 30\n"
        "  • t2: computed the discount ($10) instead of the final price ($30)\n"
        "  • t6: left fraction unsimplified (6/3 instead of 2)\n"
        "  • t8: signed-number subtraction (5 - (-12) = 17, not -7)\n\n"
        "Two distinct patterns. Proposing edits for both."
    ),
    1: (
        "Only 1 failure remains: signed-number problems.\n"
        "  • t8: 'drops from 5°C to -12°C' — agent computed 5 - 12 = -7\n"
        "         instead of |5 - (-12)| = 17\n\n"
        "The percentage and fraction rules from epoch 0 are working.\n"
        "Proposing 1 edit to handle signed differences."
    ),
    2: (
        "All 8 training tasks passed. No failures to analyze.\n"
        "No edits proposed. Proceeding to gate."
    ),
}

# Simulated proposed edits per epoch
MOCK_EDITS = {
    0: [
        Edit("append", "- Percentage: 'X% of Y' means (X/100) × Y. Always multiply, don't just return X."),
        Edit("append", "- Sale price after discount: final_price = original × (1 - discount%). Return the final price, not the discount amount."),
        Edit("append", "- Always simplify fractions to decimals or whole numbers in your final answer."),
    ],
    1: [
        Edit("append", "- Temperature change / signed difference: 'drops from A to B' means |A - B|. If B is negative, A - B = A + |B|. Always give a positive number for 'how many degrees changed'."),
    ],
    2: [],  # no edits needed — all tasks passed
}

# Simulated gate results per epoch (accepted, new_val_score)
MOCK_GATE = {
    0: (True,  0.50),   # baseline was 0.25, candidate is 0.50 → accept
    1: (True,  0.75),   # baseline was 0.50, candidate is 0.75 → accept
    2: (True,  1.00),   # baseline was 0.75, candidate is 1.00 → accept, done
}


# ─── LIVE MODE: LLM CALLS ────────────────────────────────────────────────────

def call_claude(prompt: str) -> str:
    """Call Claude API. Used only in --live mode."""
    try:
        import anthropic
    except ImportError:
        print("\n[ERROR] anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text


def live_rollout(skill: SkillDoc, tasks: list) -> list[Trace]:
    traces = []
    for task in tasks:
        system = f"You are an arithmetic solver.\n\n{skill}\n\nAnswer with only the number. No units, no explanation."
        response = call_claude(f"{system}\n\nQuestion: {task['question']}")
        answer = response.strip().split()[0].replace(",", "")
        expected = str(task["answer"])
        score = 1.0 if answer == expected else 0.0
        traces.append(Trace(task["id"], task["question"], answer, expected, score))
        time.sleep(0.3)  # rate limit
    return traces


def live_reflect(traces: list[Trace], skill: SkillDoc) -> tuple[str, list[Edit]]:
    failures = [t for t in traces if not t.passed]
    if not failures:
        return "All tasks passed — no edits needed.", []

    failure_block = "\n".join(
        f"Q: {t.question}\nAgent answered: {t.answer}\nCorrect: {t.expected}"
        for t in failures
    )
    prompt = f"""You are optimizing a skill document for an arithmetic solver.

Current skill:
{skill}

Failed tasks:
{failure_block}

Find the most common failure pattern. Propose up to {LEARNING_RATE} edits to fix it.
Output only valid JSON (no markdown fences):
{{
  "analysis": "one or two sentences describing the dominant failure pattern",
  "edits": [
    {{"op": "append", "text": "new rule to add to the bottom of the skill"}}
  ]
}}"""

    raw = call_claude(prompt)
    try:
        data = json.loads(raw)
        edits = [Edit(op=e["op"], text=e["text"], target=e.get("target","")) for e in data.get("edits", [])]
        return data.get("analysis", ""), edits
    except Exception:
        return f"(could not parse optimizer response)\n{raw}", []


# ─── THE FOUR STEPS ──────────────────────────────────────────────────────────

def rollout(skill: SkillDoc, tasks: list, epoch: int) -> list[Trace]:
    """
    STEP 1 — ROLLOUT
    ════════════════
    The frozen agent executes tasks using the current skill document.
    The skill is prepended to the system prompt — the model never changes.

    Returns: list of Trace (question, agent answer, score)
    """
    if LIVE_MODE:
        return live_rollout(skill, tasks)
    else:
        return MOCK_TRACES[epoch]


def reflect(traces: list[Trace], skill: SkillDoc, epoch: int) -> tuple[str, list[Edit]]:
    """
    STEP 2 — REFLECT
    ════════════════
    Optimizer LLM reads failure traces and proposes edit patches.

    This replaces backpropagation. Instead of a gradient vector,
    we get a list of Edit operations (append/replace/delete lines).

    The optimizer is a *separate* LLM call — the 'target' model that
    ran the tasks is frozen and unchanged.
    """
    if LIVE_MODE:
        return live_reflect(traces, skill)
    else:
        return MOCK_ANALYSIS[epoch], MOCK_EDITS[epoch]


def select_edits(edits: list[Edit]) -> tuple[list[Edit], list[Edit]]:
    """
    STEP 3 — SELECT
    ════════════════
    Apply at most LEARNING_RATE edits per step.

    This is gradient clipping: prevents the skill from changing too
    drastically in a single update. Rejected edits are saved as
    negative feedback context for the next epoch's optimizer.
    """
    return edits[:LEARNING_RATE], edits[LEARNING_RATE:]


def gate(candidate: SkillDoc, baseline_score: float, val_tasks: list, epoch: int) -> tuple[bool, float]:
    """
    STEP 4 — GATE
    ════════════════
    Validate candidate skill on held-out tasks.
    Accept if score improved. Revert if it regressed.

    This is the validation set / early stopping equivalent.
    It prevents the optimizer from accidentally making things worse.
    """
    if LIVE_MODE:
        val_traces = rollout(candidate, val_tasks, epoch)
        new_score = sum(t.score for t in val_traces) / len(val_traces)
        return new_score >= baseline_score, new_score
    else:
        return MOCK_GATE[epoch]


# ─── DISPLAY HELPERS ─────────────────────────────────────────────────────────

def hr(char="─", width=70):
    print(char * width)

def section(title: str):
    print()
    hr("═")
    print(f"  {title}")
    hr("═")

def step(num: int, name: str, analogy: str):
    print()
    hr()
    print(f"  STEP {num}: {name}  ·  (DL analogy: {analogy})")
    hr()

def print_skill(skill: SkillDoc, label="Current skill.md"):
    print(f"\n  ┌─ {label} {'─'*(50-len(label))}┐")
    for line in str(skill).strip().split("\n"):
        print(f"  │  {line}")
    print("  └" + "─"*53 + "┘")

def print_traces(traces: list[Trace]):
    passes = sum(1 for t in traces if t.passed)
    print(f"\n  Results: {passes}/{len(traces)} passed  ({passes/len(traces)*100:.0f}%)\n")
    for t in traces:
        icon = "✓" if t.passed else "✗"
        status = "PASS" if t.passed else "FAIL"
        print(f"  {icon} [{status}]  {t.question[:55]:<55}")
        if not t.passed:
            print(f"           Agent: '{t.answer}'   Expected: '{t.expected}'")

def print_edits(edits: list[Edit], rejected: list[Edit]):
    if not edits and not rejected:
        print("\n  No edits proposed.")
        return
    print(f"\n  Proposed edits (learning rate = {LEARNING_RATE}):\n")
    for i, e in enumerate(edits):
        prefix = "  ✓ APPLY " if i < LEARNING_RATE else "  ✗ SKIP  "
        print(f"  {prefix} [{e.op.upper()}]  {e.text[:65]}")
    for e in rejected:
        print(f"  ✗ SKIP   [{e.op.upper()}]  {e.text[:65]}  ← exceeds learning rate")


# ─── MAIN TRAINING LOOP ──────────────────────────────────────────────────────

def train():
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║           SkillOpt Walkthrough — Arithmetic Task                ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    print("  Goal: improve a skill.md so a frozen LLM solves arithmetic")
    print("  problems correctly — without ever changing the model weights.")
    print()
    print(f"  Mode:          {'LIVE (real Claude API)' if LIVE_MODE else 'MOCK (pre-baked data, no API needed)'}")
    print(f"  Learning rate: {LEARNING_RATE} edits/step")
    print(f"  Epochs:        {NUM_EPOCHS}")

    # ── Initial skill ────────────────────────────────────────────────────────
    skill = SkillDoc(
        "# Arithmetic Solver\n\n"
        "Answer arithmetic questions.\n"
        "Give only the final number — no units, no explanation."
    )

    baseline_score = 0.0
    rejected_memory: list[Edit] = []  # negative feedback for optimizer
    best_skill = skill
    best_score = 0.0

    training_tasks  = MOCK_TRAINING_TASKS
    validation_tasks = MOCK_VAL_TASKS

    # ── Epoch loop ───────────────────────────────────────────────────────────
    for epoch in range(NUM_EPOCHS):
        section(f"EPOCH {epoch}")
        print_skill(skill)

        # ── STEP 1: ROLLOUT ──────────────────────────────────────────────────
        step(1, "ROLLOUT", "forward pass")
        print("\n  Running agent on training tasks with current skill...\n")
        if LIVE_MODE: print("  (calling Claude API...)\n")

        traces = rollout(skill, training_tasks, epoch)
        print_traces(traces)

        train_score = sum(t.score for t in traces) / len(traces)

        # ── STEP 2: REFLECT ──────────────────────────────────────────────────
        step(2, "REFLECT", "backpropagation → text gradients")
        print("\n  Optimizer analyzes failure traces...\n")
        if LIVE_MODE: print("  (calling optimizer LLM...)\n")

        analysis, proposed_edits = reflect(traces, skill, epoch)

        print("  Analysis:")
        for line in analysis.strip().split("\n"):
            print(f"    {line}")

        # ── STEP 3: SELECT ───────────────────────────────────────────────────
        step(3, "SELECT", "gradient clipping")

        # Prepend rejected memory as negative context (real SkillOpt does this)
        if rejected_memory:
            print(f"\n  (Optimizer also sees {len(rejected_memory)} previously rejected edit(s) as negative feedback)")

        selected, newly_rejected = select_edits(proposed_edits)
        rejected_memory.extend(newly_rejected)

        print_edits(selected, newly_rejected)

        # ── Apply edits to get candidate skill ───────────────────────────────
        candidate = skill.apply_all(selected)

        if selected:
            print("\n  Updated skill.md:")
            print_skill(candidate, "candidate skill.md")

        # ── STEP 4: GATE ─────────────────────────────────────────────────────
        step(4, "GATE", "validation set / early stopping")
        print("\n  Validating candidate on held-out tasks...\n")
        if LIVE_MODE: print("  (calling Claude API on validation set...)\n")

        accepted, new_val_score = gate(candidate, baseline_score, validation_tasks, epoch)

        print(f"  Baseline val score : {baseline_score*100:.0f}%")
        print(f"  Candidate val score: {new_val_score*100:.0f}%")
        print()

        if accepted:
            print("  ✓ GATE ACCEPTED — skill improved. Keeping update.")
            skill = candidate
            baseline_score = new_val_score
            if new_val_score > best_score:
                best_score = new_val_score
                best_skill = skill
        else:
            print("  ✗ GATE REJECTED — skill did not improve. Reverting.")

    # ── Final summary ────────────────────────────────────────────────────────
    section("TRAINING COMPLETE")
    print()
    print(f"  Best validation score: {best_score*100:.0f}%")
    print()
    print("  Exporting best_skill.md...")
    print_skill(best_skill, "best_skill.md  ← deploy this")

    with open("best_skill.md", "w") as f:
        f.write(str(best_skill))

    print()
    print("  ┌──────────────────────────────────────────────────────────┐")
    print("  │  best_skill.md saved.                                    │")
    print("  │  Inject this into any agent's system prompt — no        │")
    print("  │  retraining needed, even on a different model.           │")
    print("  └──────────────────────────────────────────────────────────┘")
    print()


if __name__ == "__main__":
    train()
