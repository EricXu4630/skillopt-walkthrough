# SkillOpt Walkthrough

A minimal, heavily-annotated implementation of the [SkillOpt](https://microsoft.github.io/SkillOpt/) training loop — built to make the concepts as clear as possible.

**Core idea:** Instead of training model weights, SkillOpt trains a Markdown file (`skill.md`) that the frozen agent reads before every task. No GPU needed. Works with any LLM.

---

## The Deep Learning Analogy

| Deep Learning | SkillOpt |
|---|---|
| Model weights | `skill.md` |
| Forward pass | **Rollout** — agent runs tasks with current skill |
| Loss function | Task score (pass / fail) |
| Backpropagation | **Reflect** — optimizer LLM reads failures → edit patches |
| Gradient clipping | **Select** — limit edits per step (`learning_rate`) |
| SGD step | **Update** — apply edits to `skill.md` |
| Validation / early stopping | **Gate** — reject edits that hurt held-out score |

---

## Files

| File | What it is |
|---|---|
| `skillopt_walkthrough.py` | Minimal Python implementation, step-by-step with comments |
| `skillopt_demo.html` | Interactive browser demo — click through 4 epochs, toggle good/poor dataset |
| `best_skill.md` | Output after training (generated when you run the script) |

---

## Run it

**Mock mode** (no API key — uses pre-baked data, shows the full loop):
```bash
python skillopt_walkthrough.py
```

**Live mode** (real Claude API calls):
```bash
pip install anthropic
ANTHROPIC_API_KEY=sk-... python skillopt_walkthrough.py --live
```

**Interactive demo** — open `skillopt_demo.html` in any browser. No server needed.

---

## What you'll see (mock mode output)

```
EPOCH 0
  skill.md: 2 lines (bare minimum)

  STEP 1: ROLLOUT   — agent runs training tasks
  Results: 3/8 passed (38%)
  ✗ [FAIL]  What is 15% of 200?          Agent: '15'   Expected: '30'
  ✗ [FAIL]  Temp drops from 5°C to -12°C  Agent: '-7'   Expected: '17'
  ...

  STEP 2: REFLECT   — optimizer analyzes failures
  Analysis: 3 failures share a pattern: percentage computation...

  STEP 3: SELECT    — apply up to 3 edits (learning rate)
  ✓ APPLY  [APPEND]  Percentage: 'X% of Y' means (X/100) × Y...
  ✓ APPLY  [APPEND]  Sale price after discount: final = original × (1 - %)...

  STEP 4: GATE      — validate on held-out tasks
  Baseline: 25%   Candidate: 50%
  ✓ GATE ACCEPTED — skill improved. Keeping update.

EPOCH 1
  skill.md: 6 lines (percentage + fraction rules added, shown in green in demo)
  ...

TRAINING COMPLETE
  Best validation score: 100%
  best_skill.md saved.
```

---

## The good vs. poor dataset contrast

The interactive demo (`skillopt_demo.html`) lets you toggle between:

- **Diverse dataset** — traces cover many query types → optimizer finds clear patterns → score improves each epoch (28% → 54% → 71% → 85%)
- **Narrow dataset** — traces cover only simple queries → optimizer can't learn what it doesn't see → score plateaus quickly (28% → 35% → 37% → 38%)

**Key insight:** the optimizer can only write rules for failure patterns it observes. Dataset coverage is the real bottleneck, not the optimizer's intelligence.

---

## The four steps in ~20 lines

```python
skill = SkillDoc("# My Agent\nBe helpful.")
baseline_score = 0.0

for epoch in range(NUM_EPOCHS):
    # 1. ROLLOUT — run tasks, collect traces
    traces = rollout(skill, training_tasks)

    # 2. REFLECT — optimizer reads failures, proposes edits
    edits = reflect(failures_from(traces), skill)

    # 3. SELECT — gradient clipping: apply at most N edits
    selected = edits[:LEARNING_RATE]
    candidate = skill.apply_all(selected)

    # 4. GATE — accept only if validation improved
    new_score = evaluate(candidate, val_tasks)
    if new_score >= baseline_score:
        skill = candidate
        baseline_score = new_score

save("best_skill.md", skill)
```

---

## Transfer

The exported `best_skill.md` works zero-shot on models it was never trained with.
Inject it into any agent's system prompt — no retraining required.

---

*Based on [microsoft/SkillOpt](https://github.com/microsoft/SkillOpt) — "Executive Strategy for Self-Evolving Agent Skills"*
