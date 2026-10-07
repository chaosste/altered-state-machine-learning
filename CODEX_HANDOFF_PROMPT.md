# Prompt for a New Codex Chat

Copy the text below into a new Codex desktop chat opened on `/Users/stephenbeale/Projects/Altered State Machine Learning`.

---

You are taking over the Altered State Machine Learning project.

Workspace:
`/Users/stephenbeale/Projects/Altered State Machine Learning`

GitHub:
`https://github.com/chaosste/altered-state-machine-learning`

Start by reading, in order:

1. `PROJECT_STATUS.md`
2. `program.md`
3. `README.md`
4. `eval.py`
5. `train.py`

Then verify the current branch, remote, working tree, and the current `evidence-response-v3` result files under `logs/seed*-arm4/`.

Project outline:

- The agent maintains an explicit belief about a hidden partner model and has a learned action policy.
- `BeliefUpdateScore` is the only automatic selection score.
- Arm 4 lowers prior precision from 1.5 to 0.5 and raises belief prediction-error gain from 0.35 to 1.0.
- The evaluator distinguishes genuine ignored evidence (movement below 0.02 after strong evidence) from belief that moved but remained uncertain.
- Only genuine ignored evidence participates in score deductions and keep/discard decisions.
- Keep/discard rules live only in `eval.py`; do not recreate them in runners or user interfaces.
- Experiment outputs and earlier scoring snapshots are intentionally tracked in `logs/`.

Current evidence:

- Eleven Arm 4 seed comparisons exist.
- Arm 4 is kept for seeds 7, 29, 37, 41, 43, 47, and 53.
- It is discarded for seeds 11 and 17 because score gain is insufficient, and for seeds 23 and 31 because perseveration worsens too much.
- Mean score gain across the 11 seeds is about 0.0805.
- No saved policy gives the appropriate `undetermined` response on the no-access scenario.
- There is not yet a documented rule for combining several seed-level decisions into one overall promotion decision.

Your first task is read-only: summarize the project, verify the figures in `PROJECT_STATUS.md` against the committed logs, identify any discrepancy, and propose a clear multi-seed promotion rule without changing code or results. Distinguish facts from recommendations. After presenting that review, wait for direction before changing evaluation rules or training code.

When execution is requested later:

- Write any evaluation-rule change in `program.md` before changing code.
- Preserve prior result JSON before re-evaluating checkpoints.
- Use the same seeds and episode counts for paired comparisons.
- Keep progress and final reports in plain language.

---
