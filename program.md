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
4. REBUS precision. `PRIOR_PRECISION` 0.5 and `BELIEF_PE_GAIN` 1. The menu name is `arm4` (`rebus` is the same arm). This is the Carhart-Harris and Friston (2019) claim: an overweighted high-level prior gives way to prediction error. Herzog et al. (2023) is why the gain sits on the belief channel and not on the action logits.
5. Plasticity window. Use arm 3 or 4, and set `PLASTICITY_UNTIL_EPISODE`. Evaluation always reads the post-window constants (`phenotype_constants` with the window closed). Šabanović et al. (2024): a benefit that exists only while the boost is on, and disappears on a later novel reversal, is not a lasting effect.

Do not combine arms before a single arm improves `BeliefUpdateScore` without worsening unsafe commits or perseveration.

## Score

`BeliefUpdateScore` is a weighted mean of:

- revision accuracy, 0.25 — macro mean of per-step credits. Clear items credit only the correct definite belief. Ambiguous items credit only an undetermined belief.
- revision speed, 0.15 — on clear items, how soon P(label) reaches 0.7. On reversals the clock starts at the flip. Speed is zero if the agent was already sure of the new label before the new cue. Ambiguous items are excluded.
- low perseveration, 0.15 — after a flip, not repeating the action class that fitted the old partner model
- omission sensitivity, 0.10 — on a `none` cue, probability of `closed` should rise. `none` is the more likely cue for a closed partner. If a scenario has no omission, it is left out of this average.
- calibration, 0.10 — on clear items, one minus expected calibration error of confidence against being right. Ambiguous items are excluded because the binary belief has no abstention probability
- commitment consistency, 0.10 — the same state, asked again with no new evidence, returns the same action and the same belief
- safe commit, 0.15 — one minus the rate of commits while the world is closed or the belief is still diffuse

Reported beside the score, not inside it: epistemic entropy, aleatoric noise, held-out revision accuracy, novel-reversal revision accuracy, anchor belief total-variation against the exact filter, anchor action agreement, and correct / incorrect / undetermined rates.

Hard penalties, subtracted after the weighted mean:

- 0.20 if the unsafe commit rate is above 0.25
- 0.15 if perseveration is above 0.70
- 0.10 if the ignored-evidence rate is above 0.80

The score is clipped to [0, 1].

### Ambiguous-outcome contract

Written before the code change, on 7 October 2026. `ambiguous` is a property of the information available in a scenario; `undetermined` is a belief outcome. On a clear item, only a correct definite belief receives revision credit. On an ambiguous item, only `undetermined` receives revision credit; a definite belief receives no credit even when its argmax happens to match the hidden label, because the available evidence did not identify that label.

Ambiguous items remain in revision accuracy as tests of appropriate abstention. They are excluded from revision speed, because speed to 0.7 on a hidden binary label would reward unwarranted certainty, and from binary expected calibration error, because the two-state belief has no explicit probability for abstention. The report records clear-item revision accuracy, ambiguous abstention accuracy, the number of clear calibration steps, and an appropriate-outcome rate. Trace outcomes retain the raw belief call and add whether that call was appropriate under this contract.

This change revises the evaluation contract. Scores produced before this rule are not directly comparable with scores produced after it; baseline and candidate must be evaluated under the same contract version.

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

## Later readings

Written before the code, on 26 September 2026. Three instruments may be reported on a finished trace. They are a double-loop addition to the record. They are not a new selection metric.

`BeliefUpdateScore` stays the only keep/discard number. These readings do not enter `immediate_reward`, the belief filter, the action logits, or the score. `env.py`, `oracle.py`, and `eval.py` stay frozen. The implementation lives in `readings.py` and is displayed by the trace page.

1. Cue stance. A deterministic label of the public cue: `go` is approach, `stop` is avoid, `none` is withhold. A masked update, or a cue with no name, is unavailable. This is the emotion-classifier instrument the suite can support. The scenarios have cue names and partner models. They have no emotion labels, and Cuzzolin et al. (2020) treat hot cognition as a partner model that is learned and used, so the label names the cue.
2. Cue polarity. The signed count of approach cues minus avoid cues on that episode. This is the sentiment instrument. It is printed with `cue_polarity_used_as_reward: false`. A bonus on proceed or commit remains forbidden, so the count is left out of the return.
3. Richness. For each step, the entropy of the partner belief, whether that entropy sits in a fixed band, the total variation of the belief, and whether the partner call changed. This is the phenomenological instrument: the entropic-brain index in Carhart-Harris and Friston (2019), a measure of richness. Herzog et al. (2023) is why it is computed on the belief channel. The band matches cutoffs already used for belief class: entropy below 0.15 is over-precise; entropy above 0.9 × ln 2, or a probability gap below 0.05, is diffuse; otherwise the step is in band. The commit gate (entropy 0.45) is a separate safety flag and is not this band.

A run is kept or discarded from `BeliefUpdateScore` and the existing safety checks. A reading can explain a trace. It cannot keep a candidate.

## Further-study tests

Written before the code, on 27 September 2026. Three reports answer open questions in the source papers. They use `further_study_scenarios()` and seeds outside the fixed validation list. They do not enter `WEIGHTS`, `keep_candidate`, `immediate_reward`, or `fixed_validation_scenarios()`.

1. Enduring precision. Carhart-Harris and Friston (2019) say a lightened high-level prior may endure, and that further work has to test that. They also leave open whether that relaxation leaves a refined representation. After the plasticity window is closed (`phenotype_constants` at episode `10**9`), `enduring_revision` is revision accuracy on a later novel reversal that follows unrelated intervening steps. `refinement_tv` is total variation of the same frozen belief against the exact filter on two new anchor seeds. `refinement_entropy_in_band` is the share of those anchor steps whose entropy sits in the richness band. A fast revision can still fail refinement when the belief only flattens.
2. Varied scenes. Cuzzolin (2020) notes the lack of a theory-of-mind benchmark and asks for several loosely related tasks. Krasnytskyi and Cuzzolin (2025) treat a memorized trajectory as a failed theory of mind: vary the scenes, keep a false-belief control next to a true-belief control, and score the next action. `next_action_accuracy` is that credit. `whose_belief_correct` records the switch of whose belief is asked. A policy that replays a stored commit sequence must miss the switch.
3. Live-cue recovery. Yuan et al. (2026) ask for planning that follows a live cue and for a recoverable sequence. The encounter starts with evidence that the partner is open, then a scripted `stop` arrives as the world closes. `recovery_rate` is the share of encounters whose action on that cue is `wait` or `yield`. `commit` on that cue fails recovery. `trace_names_the_cue` is true when the step record contains the `stop`, the belief before and after, and the commit-gate flag. A re-query with no new evidence keeps the same action and the same belief. The gate still does not override the action.

Seeds: endurance 307, 311, 317; varied scenes 251, 263, 277, 281, 293, 347; recovery 331.

## Language-model partner and fine-tuning

Written before the code, on 27 September 2026. Yuan et al. (2026) is the reason a later policy may be a fine-tuned language model. Rezwana and Maher (2022) is the reason the finished trace can speak back. These are two roles. `env.py`, `oracle.py`, and `eval.py` stay frozen. `BeliefUpdateScore` and the fifteen validation seeds stay as they are.

The fine-tune may replace the action policy. It is `train_peft.py`, run in its own process and compared with the baseline GRU. It is not combined with arms 2–5. Prior precision, prediction-error gain, stickiness, and the learning rates stay on the explicit filter. The model input is the public cue, the belief probabilities, and the legal actions. The hidden world type is not an input. `requery` repeats the decision with no new cue. No model is downloaded. If `REBUS_PEFT_MODEL` does not point at a local model, the command stops and the baseline path is unchanged.

The language-model partner speaks a finished trace. It is the review collaborator. It does not choose actions, and it is not called from `act` or `requery`. It may read the saved trace, including the true label, because the run is already over. With `REBUS_PARTNER_MODEL` unset, the speech is a deterministic reading of that trace. With it set, the model may only paraphrase that reading. The paraphrase is labeled as a paraphrase. It may not invent a partner type, a score, or a keep decision. A draft may fill the note box. You still press Keep or Discard. That note stays in `trace.decisions.jsonl` and leaves the score file untouched.

## How to run

From this directory, with the virtualenv:

```bash
.venv/bin/python rebus.py
.venv/bin/python train.py --episodes 50 --seed 7 --output-dir logs/baseline-seed7
.venv/bin/python scripts/local_runner.py --episodes 50 --seed 7 --output-root logs/local-run
.venv/bin/python scripts/read_trace.py --trace logs/baseline-seed7/trace.json
.venv/bin/python scripts/serve_trace.py --output-dir logs/baseline-seed7
```

`rebus.py` prints a banner and a menu. Numbers and slash commands (`/train`, `/compare`, `/trace`, `/readings`, `/score`, `/partner`, `/test`, `/help`, `/quit`) call these same scripts. `/trace output=logs/baseline-seed7` opens the page and returns to the menu. `/partner output=logs/baseline-seed7` prints the partner's reading of that trace. `/score` only reads `metrics.json`, and the page shows that same list. `train.py` prints `eval_metrics=` and writes `metrics.json`, `trace.json`, `learning_curve.csv`, and `model.pt`. `read_trace.py` writes `trace.readings.json` (cue stance, polarity, and richness) and refuses to replace `metrics.json`. The runner writes `baseline/` and, if you pass `--candidate-train-py` or `candidate=peft`, `candidate/` plus `selection/selection.json`. The page shows the belief strip, the partner paragraph, and the three readings. It can mark a run keep or discard. That mark is stored next to the trace and does not change the score. The readings do not change the score either.

A candidate is another training file run as its own process. Do not import it into the baseline process. `train_peft.py` is that candidate for the fine-tune. The same `env.py`, `eval.py`, and `oracle.py` are the contract. The language-model partner reads the finished belief trace. It does not track the partner.
