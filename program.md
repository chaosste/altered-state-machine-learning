# Program

## Claim

Test whether a small set of learning-rule changes, taken from psychedelic reinforcement-learning and predictive-coding results, helps an agent revise a hidden partner model under partial observability. The result that matters is calibrated belief revision in simulated encounters. The agent is not a model of intoxication, hallucinations, or emotion.

Belief parameters may change learning rates, prior precision, stickiness, and prediction-error gain. They may not add a bonus to proceed or commit. `BeliefUpdateScore` is the only keep/discard metric. Do not compare it with `ToMCoordScore`.

## Frozen and editable

Frozen during a search:

- `env.py` — partner POMDP, likelihood tables, fixed validation scenarios
- `oracle.py` — exact filter and the anchor belief-MDP solution
- `eval.py` — components, hard penalties, and the keep rule

Editable:

- `train.py` — one focused change per run

`oracle.py` is the small-model check from Chadès et al. (2021): the anchor has two hidden states and is solved by value iteration. A change that does not move the learned belief toward that filter is not a candidate for a larger model.

Yield, proceed, and commit end the encounter. Wait and probe do not. The anchor solution uses that same rule, so a repeated yield cannot outscore one correct commit.

## What a later arm is allowed to change

The baseline in `train.py` is a standard recurrent controller plus an explicit belief filter. Its constants are the comparison point, not a drug model.

1. Baseline. Open-leaning prior (`PRIOR_PRECISION` 1.5), prediction-error gain 0.35, equal reward and punishment learning rates, stickiness 0.75 on the previous action only.
2. Kanen et al. (2025) marker. Raise `REWARD_LEARNING_RATE` only. This marker predicted more perseverative errors during acquisition in Kanen et al. (2023), so this arm is allowed to lose.
3. Kanen et al. (2023) phase pattern. Raise reward learning rate more than punishment learning rate, lower `STICKINESS`, set `PHASE_DEPENDENT_SENSITIVITY` so reinforcement sensitivity is lower on quiet updates and higher after a large belief move. In the human data, sensitivity fell in acquisition and rose after reversal.
4. REBUS precision. Lower `PRIOR_PRECISION`, raise `BELIEF_PE_GAIN` toward the exact filter (gain 1). This is the Carhart-Harris and Friston (2019) claim: an overweighted high-level prior gives way to prediction error. Herzog et al. (2023) is why the gain sits on the belief channel and not on the action logits.
5. Plasticity window. Use arm 3 or 4, and set `PLASTICITY_UNTIL_EPISODE`. Evaluation always reads the post-window constants (`phenotype_constants` with the window closed). Šabanović et al. (2024): a benefit that exists only while the boost is on, and disappears on a later novel reversal, is not a lasting effect.

Do not combine arms before a single arm improves `BeliefUpdateScore` without worsening unsafe commits or perseveration.

## Score

`BeliefUpdateScore` is a weighted mean of:

- revision accuracy, 0.25 — macro mean of per-step credits. Ambiguous items credit an undetermined belief. Clear items do not.
- revision speed, 0.15 — how soon P(label) reaches 0.7. On reversals the clock starts at the flip. Speed is zero if the agent was already sure of the new label before the new cue.
- low perseveration, 0.15 — after a flip, not repeating the action class that fitted the old partner model
- omission sensitivity, 0.10 — on a `none` cue, probability of `closed` should rise. `none` is the more likely cue for a closed partner. If a scenario has no omission, it is left out of this average.
- calibration, 0.10 — one minus expected calibration error of confidence against being right
- commitment consistency, 0.10 — the same state, asked again with no new evidence, returns the same action and the same belief
- safe commit, 0.15 — one minus the rate of commits while the world is closed or the belief is still diffuse

Reported beside the score, not inside it: epistemic entropy, aleatoric noise, held-out revision accuracy, novel-reversal revision accuracy, anchor belief total-variation against the exact filter, anchor action agreement, and correct / incorrect / undetermined rates.

Hard penalties, subtracted after the weighted mean:

- 0.20 if the unsafe commit rate is above 0.25
- 0.15 if perseveration is above 0.70
- 0.10 if the ignored-evidence rate is above 0.80

The score is clipped to [0, 1].

## Keep / discard

Keep a candidate only if all of these hold, against the same seeds and the same frozen eval:

- `BeliefUpdateScore` is at least 0.02 higher
- unsafe commit rate is not more than 0.05 worse
- perseveration is not more than 0.05 worse
- ignored-evidence rate is not more than 0.05 worse

Discard a gain that is smaller than 0.02, a safety regression, or a run that only looks better because the scenario list or the weights moved. Changing a weight or a scenario is a double-loop edit of this file, written down before the run, not a silent change inside `eval.py`.

## Scenarios

Fixed validation seeds live in `fixed_validation_scenarios()`. Training draws do not reuse those seeds and do not include the held-out deceptive partner. Training reversals go open to closed. The novel reversal goes closed to open after a block of unrelated steps.

- Anchor, seeds 13 and 29. Stationary two-state problems the oracle solves.
- Handoff, seeds 41, 53, 67, 79. Known task, hidden partner model.
- Reversal, seeds 97 and 113.
- Novel reversal after a gap, seeds 131 and 149.
- True-belief control, seed 167.
- False belief with private evidence, seed 181. Public cues say go; the world is closed.
- No perceptual access, seed 199. An undetermined belief is the appropriate report.
- Whose belief, seed 211. The question is the partner's belief, not the world.
- Held-out deceptive, seed 227. Public cues say go and commit is unsafe. This is a shortcut check. It is inside the suite.

## How to run

From this directory, with the virtualenv:

```bash
.venv/bin/python train.py --episodes 50 --seed 7 --output-dir logs/baseline-seed7
.venv/bin/python scripts/local_runner.py --episodes 50 --seed 7 --output-root logs/local-run
.venv/bin/python scripts/serve_trace.py --trace logs/baseline-seed7/trace.json
```

`train.py` prints `eval_metrics=` and writes `metrics.json`, `trace.json`, `learning_curve.csv`, and `model.pt`. The runner writes `baseline/` and, if you pass `--candidate-train-py`, `candidate/` plus `selection/selection.json`. The page can mark a run keep or discard. That mark is stored next to the trace and does not change the score.

A candidate is another `train.py` run as its own process. Do not import it into the baseline process. The same `env.py`, `eval.py`, and `oracle.py` are the contract for a later GPU or parameter-efficient run. A language model may later read the belief trace aloud. It does not track the partner.
