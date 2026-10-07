# Altered State Machine Learning

A small, local-first experiment. It asks whether a few learning-rule changes, taken from psychedelic reinforcement-learning and predictive-coding results, help an agent revise a hidden partner model under partial observability.

The result that matters is calibrated belief revision in simulated encounters. The agent is a decision model with an inspectable belief. It is not a model of intoxication, hallucinations, or emotion.

The scientific contract lives in [`program.md`](program.md). This file is the map: how to run the first loop, what the words mean, and what belongs in a public repository.

## Status

The frozen world, the exact small solution, the score, the baseline trainer, the tests, and the belief-trace page are in place. The current evaluator is `evidence-response-v3`: ignored evidence means almost no movement after strong evidence, while belief that moved but remained uncertain is reported separately.

Eleven Arm 4 seed comparisons are committed under `logs/`. Seven pass every per-seed keep rule and four are discarded. See [`PROJECT_STATUS.md`](PROJECT_STATUS.md) for the current results, open questions, and continuation workflow. [`CODEX_HANDOFF_PROMPT.md`](CODEX_HANDOFF_PROMPT.md) contains a ready-to-paste prompt for a new Codex desktop chat.

Cue stance, cue polarity, and richness can be read off a finished trace. They are reports. `BeliefUpdateScore` remains the only keep/discard metric. The boundary is written in [`program.md`](program.md) under Later readings.

Online runs and GPU fine-tuning are later steps. They must use this same `env.py`, `oracle.py`, and `eval.py`. A language model may later read the belief trace aloud. It does not track the partner.

## Requirements

- Python 3.11 or newer
- The packages in [`requirements.txt`](requirements.txt): NumPy and PyTorch

CPU is enough for the starter loop. `--device` on `train.py` accepts `cpu`, `mps`, or `cuda` when you have one.

## Setup

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -t .
```

If you keep using the path directly, the same commands are `.venv/bin/python` and `.venv/bin/pip`.

## Starter workflow

Do these in order. One hypothesis, one change, then a comparison against the baseline at the same seed.

1. Run the tests in the setup block. They lock the scenario seeds, check the exact filter, and check that a policy which trusts the latest public cue fails the held-out partner.
2. Train the baseline and leave `train.py` untouched:

   ```bash
   python train.py --episodes 50 --seed 7 --output-dir logs/baseline-seed7
   ```

3. Read `logs/baseline-seed7/metrics.json`. The selection field is `BeliefUpdateScore`. The other fields say why.
4. Write the readings beside the trace:

   ```bash
   python scripts/read_trace.py --trace logs/baseline-seed7/trace.json
   ```

   That writes `logs/baseline-seed7/trace.readings.json`. Cue stance names each public cue. Cue polarity is the signed count of those stances. Richness is the entropy band of the partner belief. The file is a report. It leaves `metrics.json` alone.
5. Open the trace:

   ```bash
   python scripts/serve_trace.py \
     --trace logs/baseline-seed7/trace.json \
     --metrics logs/baseline-seed7/metrics.json
   ```

   The page is at `http://127.0.0.1:8765/`. Each strip shows the cue, the belief before and after, epistemic and aleatoric uncertainty, the action, whether the commit gate would have blocked it, the cue stance, and the richness band. The scenario line shows cue polarity. `BeliefUpdateScore` stays the selection metric.
6. Write one sentence in a note: which single constant in `train.py` you expect to move, and which component should move with it. The allowed sequence is in [`program.md`](program.md).
7. Choose one pre-registered arm. From the menu, `candidate=arm4` is the prior-precision edit (prior precision 0.5, prediction-error gain 1). `/arms` lists the others. A hand-written copy of `train.py` is still accepted as a path. Do not edit `env.py`, `oracle.py`, or `eval.py`.
8. Compare at the same seed:

   ```bash
   python scripts/local_runner.py \
     --episodes 50 \
     --seed 7 \
     --output-root logs/seed7-arm2 \
     --candidate-train-py candidates/arm2_reward_lr.py
   ```

   The runner starts a separate process for the candidate, so its constants cannot leak into the baseline. It writes `selection/selection.json` with the keep/discard decision, every rule and threshold, both ignored-evidence rates, both moved-but-uncertain rates, and explicit rejection reasons.

   From the terminal menu, compare the same candidate across several seeds with one command:

   ```text
   /compare episodes=50 seeds=7,11,17,23,29 candidate=arm4 output=logs
   ```

   Batch output uses one comparison directory per seed, such as `logs/seed7-arm4/` and `logs/seed11-arm4/`. Seeds are run serially. A failure stops the batch and reports which outputs completed.
9. On the trace page, mark the run keep or discard and write the reason. That mark is stored beside the trace, in a `.decisions.jsonl` file. It does not change `BeliefUpdateScore`.
10. Keep the candidate only when `selection.json` says `keep`. A higher reward learning rate that speeds acquisition and increases perseveration does not win.
11. Repeat from step 6. Combine arms only after a single arm has already passed the keep rule.

Changing a weight or a scenario is a separate kind of edit. Write it in [`program.md`](program.md) before the run. That is the double loop. Do not hide it inside `eval.py`.

## Later readings

Three instruments can be read off a finished trace. They explain a run. They do not select one.

- **Cue stance.** `go` is approach, `stop` is avoid, `none` is withhold. A masked update is unavailable.
- **Cue polarity.** Approach cues minus avoid cues for that episode. The report sets `cue_polarity_used_as_reward` to false. The count stays out of the return.
- **Richness.** Entropy of the partner belief, whether that entropy sits in the band, how far the belief moved, and whether the partner call changed. Entropy below 0.15 is over-precise. Entropy above 0.9 × ln 2, or a probability gap below 0.05, is diffuse. Otherwise the step is in band.

```bash
python scripts/read_trace.py --trace logs/baseline-seed7/trace.json
```

The trace page shows the same three readings. `BeliefUpdateScore` remains the only keep/discard metric. The boundary is the Later readings section of [`program.md`](program.md).

## Further-study tests

Three reports sit beside the score. They use [`further_study_scenarios()`](env.py) and seeds outside the fixed validation list. They do not change `BeliefUpdateScore` or the keep rule.

- **Enduring revision.** Revision accuracy on a novel reversal that follows unrelated intervening steps, read after the plasticity window has closed.
- **Refinement.** Total variation of that same belief against the exact filter on two extra anchor problems, plus the share of those steps whose entropy is inside the richness band. A fast revision can still be a flat belief.
- **Varied scenes.** Several questions about one partner model: the world, the partner's belief, and the next action. A true-belief item is paired with each false-belief item. Credit is for the next action. A replay of a stored commit sequence misses the switch of whose belief is asked.
- **Recovery.** After evidence that the partner is open, a scripted `stop` arrives as the world closes. `wait` or `yield` recovers. `commit` on that cue does not. The trace names the `stop`, the belief move, and the commit gate.

The full statement is the Further-study tests section of [`program.md`](program.md).

## Commands

| Command | What it does |
| --- | --- |
| `python rebus.py` | Prints the ASML banner, then a menu. Commands run with `.venv` when that folder is present |
| `python train.py --episodes 50 --seed 7 --output-dir logs/baseline-seed7` | Trains the baseline, evaluates the fixed suite, prints `eval_metrics=` |
| `python scripts/local_runner.py --episodes 50 --seed 7 --output-root logs/local-run` | Trains the baseline only |
| `python scripts/local_runner.py ... --candidate-train-py PATH` | Trains baseline and candidate, then applies the keep rule |
| `python scripts/serve_trace.py --output-dir logs/baseline-seed7` | Opens the belief-trace page for that run. `--port` defaults to 8765. `--trace` still accepts a trace file |
| `python scripts/read_trace.py --trace PATH` | Writes cue stance, polarity, and richness beside the trace. It refuses to replace `metrics.json` |
| `python -m unittest discover -s tests -t .` | Runs the integrity tests |

`train.py` defaults are 200 episodes, seed 7, device `cpu`, and a required `--output-dir`. The runner defaults to 50 episodes and seed 7.

The menu and the page share one run directory. `/train output=logs/baseline-seed7` writes `metrics.json` and `trace.json` there. `/score metrics=logs/baseline-seed7/metrics.json` and the page print that score as the same list. `/trace output=logs/baseline-seed7` opens the page and returns to the menu, so you can train again while the page stays up. Refresh the page after a new train of that directory. `/quit` closes a page the menu opened. A direct `serve_trace.py` process stays in the foreground in that terminal. Two pages cannot share port 8765. A comparison directory has no trace at its root: open `output=logs/local-run/baseline` or `output=logs/local-run/candidate`. The Keep and Discard buttons write a note beside the trace. The automatic decision is `selection/selection.json` from `/compare`; a discard includes the failed criteria and measured rejection reasons.

## Repository map

| Path | Role during a search |
| --- | --- |
| [`PROJECT_STATUS.md`](PROJECT_STATUS.md) | Current evaluation version, multi-seed results, open decisions, and continuation workflow |
| [`CODEX_HANDOFF_PROMPT.md`](CODEX_HANDOFF_PROMPT.md) | Ready-to-paste context for a new Codex desktop chat |
| [`program.md`](program.md) | Claim, score, keep rule, allowed arms, Later readings, and further-study tests |
| [`rebus.py`](rebus.py) | Terminal banner, numbered menu, and slash commands |
| [`env.py`](env.py) | Frozen partner POMDP, the fixed validation scenarios, and `further_study_scenarios()` |
| [`oracle.py`](oracle.py) | Frozen exact filter and the anchor belief-MDP solution |
| [`eval.py`](eval.py) | Frozen score, components, keep rule, and canonical selection report |
| [`train.py`](train.py) | The only file a search edits |
| [`scripts/local_runner.py`](scripts/local_runner.py) | Baseline versus candidate, in separate processes |
| [`scripts/serve_trace.py`](scripts/serve_trace.py) | Local belief-trace page |
| [`readings.py`](readings.py) | Cue stance, polarity, and richness. Reports only |
| [`scripts/read_trace.py`](scripts/read_trace.py) | Writes `trace.readings.json` from a finished trace |
| [`interface/trace_page.py`](interface/trace_page.py) | Page renderer and the keep/discard note |
| [`tests/`](tests) | Seed lock, oracle checks, memorizer failure, trainer smoke, and readings staying off the score |
| [`academic_basis/`](academic_basis) | Source papers, grouped by question. The trainer does not import them |
| [`logs/`](logs/) | Committed metrics, traces, checkpoints, learning curves, selection reports, and earlier scoring snapshots |

## What a run writes

Inside the output directory:

- `metrics.json` — `BeliefUpdateScore`, the components, the seed, the episode count, and the constants used at evaluation
- `trace.json` — cue, belief, uncertainty, action, and commit gate for every validation episode
- `trace.readings.json` — optional report from `scripts/read_trace.py`: cue stance, polarity, and richness. It is not a score
- `learning_curve.csv` — return and belief match by training episode
- `model.pt` — network weights and the phenotype constants

The runner adds `baseline/`, optional `candidate/`, and `selection/selection.json`.

Evaluation of a plasticity-window arm uses the constants after the window has closed. A gain that exists only while the boost is on, and disappears on the later novel reversal, is not a lasting effect.

## Rules that keep a result comparable

- Compare runs that share the evaluation code, the scenario list, and the seed.
- Belief parameters may change learning rates, prior precision, stickiness, and prediction-error gain.
- They may not add a bonus to proceed or commit.
- `BeliefUpdateScore` is the only keep/discard metric. Do not compare it with `ToMCoordScore` from the earlier coordination benchmark.
- A reading can explain a trace. It cannot keep a candidate.
- The anchor has two hidden states and is solved by value iteration. A change that does not move the learned belief toward that filter is not a candidate for a larger model.
- Yield, proceed, and commit end the encounter. Wait and probe do not. A repeated yield cannot outscore one correct commit.

The numeric weights, hard penalties, and keep thresholds are specified in [`program.md`](program.md) and implemented in [`eval.py`](eval.py). If those two disagree, `eval.py` is what the runner executes, and the disagreement should be fixed in the open.

## Tests

```bash
python -m unittest discover -s tests -t .
```

The suite checks that likelihood rows sum to one, that scripted rollouts repeat, that a `go` cue from a uniform prior matches the hand-computed posterior, that the anchor policy commits when the partner is almost surely open and yields when it is almost surely closed, that a cue-memorizer fails the held-out partner, that a confirmation flip breaks commitment consistency, that cue stance, polarity, and richness leave the reward and `BeliefUpdateScore` unchanged, that the terminal menu's `/score` and `/readings` commands do the same, and that the three further-study reports leave the fifteen validation seeds and the keep rule unchanged.

## Before the GitHub repository

Include the source, `program.md`, `requirements.txt`, and this file. The
reference papers in `academic_basis/` remain local research material and are
not included in the public repository because their redistribution rights are
not assumed.

Leave out:

- `.venv/`
- `logs/`
- `__pycache__/` and `*.pyc`
- `graphify-out/`
- `*.decisions.jsonl` notes from local reviews

The source repository includes an MIT [`LICENSE`](LICENSE). Papers under `academic_basis/` remain subject to their publishers' terms.

## Glossary

### Actions and observations

**Wait.** Stay in the encounter for one step and take the ordinary noisy cue. The episode continues.

**Probe.** Pay a small extra cost to make the next cue sharper. This is the surveillance action in a POMDP: spend a step to reduce state uncertainty.

**Yield.** End the encounter by giving way. This is the appropriate ending when the partner model is closed.

**Proceed.** End the encounter with a milder commitment than commit. The payoff is smaller when the partner is open, and the penalty is smaller when the partner is closed.

**Commit.** End the encounter with the high-stakes act. Safe when the world is open and the belief is sharp enough. Unsafe when the world is closed, or when the belief is still diffuse.

**Cue.** The public observation on a step: `go`, `stop`, or `none`.

**None / omission.** A missing event. In this likelihood, `none` is more common when the partner model is closed, so a `none` cue should raise the probability of closed.

**Open.** The partner model in which commit is appropriate.

**Closed.** The partner model in which commit is unsafe and yield is appropriate.

**Clarity.** A flag on the observation that follows a probe. It tells the controller the cue was drawn from the sharper table.

### The decision problem

**POMDP.** A partially observable Markov decision process. The agent chooses a sequence of actions to maximize reward, but it sees noisy cues rather than the hidden state. Chadès et al. (2021) use this tuple for decisions under state uncertainty and model uncertainty. This suite is the small discrete case of that idea.

**Partial observability.** The partner model is hidden. The agent must act on cues, and on a private channel when the scenario provides one.

**Hidden state.** Here, the partner model: open or closed. In a false-belief item, the partner's belief and the world can differ.

**Belief.** A probability distribution over the partner model. The baseline stores it as logits and turns it into probabilities with a softmax.

**Belief update.** The rule that moves the belief after a cue. With prediction-error gain 1 and a uniform start, it is exact Bayes on the public likelihood.

**Prior.** The belief at the start of an episode, before cues. The baseline prior leans open.

**Prior precision.** How strongly that starting belief is held. `PRIOR_PRECISION` 1.5 is an overweighted open prior. Lowering it is the REBUS-style edit: the high-level belief becomes easier to revise.

**Prediction error.** The surprise of the cue under each partner model, implemented as the log likelihood of that cue.

**Prediction-error gain.** `BELIEF_PE_GAIN`. It scales how far one cue moves the belief. The baseline uses 0.35. The exact filter uses 1.

**Exact filter.** The Bayes update with gain 1 and a uniform prior, using the same likelihood tables as the environment. Evaluation compares the agent's belief with this filter.

**Belief MDP.** The fully observed problem whose state is the belief. On the anchor, value iteration solves it on a grid over the probability of open.

**Anchor.** The two stationary scenarios (seeds 13 and 29) that the exact solution covers. Other families are scored, and they are not claimed to be solved exactly.

**Oracle.** The exact filter plus the anchor action from value iteration. "Oracle" here means the solved small model, which is the reference a later larger model has to approach.

**Value iteration.** The backward pass that computes the best action for each belief and each number of steps left on the anchor.

**Policy.** The mapping from the current observation and belief to an action. The baseline policy is a GRU plus linear heads. The belief itself is the explicit filter, so the numbers that matter can be read without opening the network.

**Stickiness.** A bonus on the previous action's logit. It is choice repetition, the Kanen stimulus-stickiness parameter. It is not a bonus for proceed or commit. The baseline value is 0.75.

**Reinforcement sensitivity.** How sharply the action logits are scaled before the action is chosen. In the Kanen models this is the explore/exploit parameter. The baseline keeps it constant. A later arm may lower it on quiet updates and raise it after a large belief move.

**Reward learning rate and punishment learning rate.** Separate weights on the value loss after positive and negative immediate rewards. They are equal in the baseline. Raising only the reward rate is the Kanen 2025 marker, and that arm is allowed to lose: in the 2023 human data, a higher reward learning rate during acquisition predicted more perseverative errors after reversal.

**Phenotype constants.** The named numbers at the top of `train.py` that implement those learning rules. Evaluation reads them after any plasticity window has closed.

**Plasticity window.** A training span during which the edited rates are on. After `PLASTICITY_UNTIL_EPISODE`, training and evaluation use the consolidated constants. This follows Šabanović et al. (2024): a benefit has to show up later, on a novel reversal, after the boost has ended.

### Scenarios

**Handoff.** A structured encounter. The task is known and the partner model is hidden. This is the simple human–robot case: read intent, then yield or commit.

**Reversal.** The partner model flips during the episode. Training reversals go open to closed. The score clock for speed starts at the flip.

**Novel reversal.** A flip in the other direction, closed to open, after a block of unrelated steps that are not evidence. The agent cannot pass this item by memorizing the training direction.

**Intervening trials.** Those unrelated steps. Belief updates are masked, so the gap is not a string of fake cues.

**False belief.** Public cues follow what the partner believes, which can disagree with the world. A private channel can reveal the world.

**True-belief control.** The same family with public cues and private evidence in agreement. It checks that the item is solvable when there is no conflict.

**Perceptual access.** Whether the agent can see the private channel. With access missing, the public cue is not evidence about the world. An undetermined belief is the appropriate report.

**Whose-belief.** The question is what the partner believes, not what is true of the world. Private evidence is about the world, so the filter ignores it for this question.

**Held-out partner.** A deceptive case used only at test. Public cues say go, and commit is unsafe. Training never draws it. A policy that copies the latest cue fails it.

**Held-out.** Reserved for test. The point, from Krasnytskyi and Cuzzolin (2025), is that success on trained patterns can be memorization.

### Score words

**BeliefUpdateScore.** The only number used to keep or discard a change. It is a weighted mean of revision accuracy, revision speed, low perseveration, omission sensitivity, calibration, commitment consistency, and safe commit, minus hard penalties, clipped to the range 0 to 1.

**Revision accuracy.** On each scored step, a clear item credits only the correct definite belief. An ambiguous item credits only an undetermined belief; a definite guess receives no credit even when it happens to match the hidden label.

**Revision speed.** On clear items, how soon the probability of the true label reaches 0.7. On a reversal, speed is zero if the agent was already sure of the new label before the new cue. Ambiguous items are excluded because speed toward a hidden binary label would reward unwarranted certainty.

**Perseveration.** After a flip, repeating the action class that fitted the old partner model. This is the Kanen failure mode.

**Omission sensitivity.** On a `none` cue, whether the probability of closed rose.

**Epistemic uncertainty.** Uncertainty about which partner model is true. Reported as the entropy of the belief.

**Aleatoric uncertainty.** Noise in the cue itself, reported as one minus the probability of the cue that was actually seen. It is printed beside the score and is not folded into the belief.

**Calibration.** On clear items, whether confidence matches how often the belief is right. The component is one minus the expected calibration error. Ambiguous items are excluded because the two-state belief has no explicit abstention probability.

**Expected calibration error.** A binned gap between average confidence and average accuracy.

**Commitment consistency.** The same situation is asked again with a confirmation flag and no new evidence. The action and the reported belief should stay put. This follows Solaki et al. (2025).

**Undetermined.** A belief that is too flat to call, or whose two probabilities are nearly tied. On an ambiguous item it is the only credited belief outcome. On a clear item it receives no credit.

**Unsafe commit.** A commit while the world is closed, or while belief entropy is still above the commit threshold.

**Commit gate.** A flag on the trace when a commit would be blocked because the belief is too diffuse or the probability of open is too low. The environment still carries out the action, so the score can count it. The page shows the gate. It does not secretly fix the action.

**Ignored evidence.** After a strong cue, belief movement is below 0.02. This retains the score deduction and the maximum permitted increase used by the keep/discard decision.

**Moved but uncertain.** After a strong cue, belief movement is at least 0.02 but entropy remains above 0.9 × ln 2. This is reported separately and does not cause a score deduction or automatic discard. Clear scenarios can still lose revision accuracy or speed when uncertainty remains unresolved.

**Hard penalty.** A subtraction applied after the weighted mean when unsafe commits, perseveration, or ignored evidence cross a fixed high threshold. The thresholds are in [`program.md`](program.md).

**Keep / discard.** The comparison rule. The score must rise by at least 0.02, and unsafe commits, perseveration, and ignored evidence must not worsen by more than 0.05. The trace-page buttons record a human note. They do not apply this rule.

**Total variation.** Half the sum of absolute differences between two distributions. Anchor belief total variation is the distance between the learned belief and the exact filter.

**Anchor action agreement.** How often the chosen action matches the anchor solution at the exact filter's belief.

**Correct, incorrect, undetermined rates.** The share of scenarios whose final belief falls in each class. These rates are reported beside the score.

### Experiment words

**Baseline.** The untouched `train.py`: open-leaning prior, low prediction-error gain, equal reward and punishment learning rates, stickiness on the previous action only.

**Arm.** One pre-registered edit. Arm 2 raises the reward learning rate only. Arm 3 is the Kanen phase pattern. Arm 4 lowers prior precision and raises prediction-error gain on the belief channel. Arm 5 is a plasticity window around arm 3 or 4.

**Frozen file.** `env.py`, `oracle.py`, and `eval.py` during a search. Editing them changes the test, so the new number is not comparable.

**Variant.** The `VARIANT` string stored in `metrics.json`. Name it after the file you ran.

**Seed.** The integer that fixes training draws and, separately, the seeds inside each validation scenario. Compare equal seeds.

**Double loop.** A change to the question itself: a weight, a scenario, or a penalty. Write it in `program.md` first. The phrase is from Argyris and Schön, as used in the OntoOmnia note, and here it only means that kind of recorded edit.

**Single loop.** An edit inside `train.py` that tries to do better on the current score.

**Shortcut.** A policy that uses a surface cue, such as "go means commit," and fails when that cue is a lie. The held-out partner is the check.

**REBUS.** Relaxed beliefs under psychedelics (Carhart-Harris and Friston, 2019). In this code it means lowering the precision of a high-level prior so prediction error can revise it, with the gain on the belief channel. It does not mean raising action entropy without a limit.

**Hot cognition.** In Cuzzolin et al. (2020), thinking that has to track someone else's changing state and use that state to choose. Here that is the partner model. The cue-stance reading names the public cue. The partner model remains the thing the score tracks.

**Cue stance.** A label of the public cue on a finished step: approach for `go`, avoid for `stop`, withhold for `none`, unavailable when the update is masked. This is the emotion-classifier instrument in the current suite. The scenarios carry cue names, and the label is computed after the run.

**Cue polarity.** The signed count of approach cues minus avoid cues in one episode. This is the sentiment instrument. The report sets `cue_polarity_used_as_reward` to false, and the count stays out of `immediate_reward`.

**Richness.** The entropic-brain index on the belief channel: entropy, an entropy band, how far the belief moved, and whether the partner call changed. Entropy below 0.15 is over-precise. Entropy above 0.9 × ln 2, or a probability gap below 0.05, is diffuse. Otherwise the step is in band. The commit gate at entropy 0.45 is a separate safety flag.

**Enduring revision.** Revision accuracy on a novel reversal after a block of unrelated steps, using the constants from a closed plasticity window. It is reported beside the score.

**Refinement.** How close that post-window belief stays to the exact filter, and whether its entropy remains inside the richness band. Flattening the belief fails refinement.

**Varied scenes.** A small theory-of-mind set outside the fifteen fixed scenarios. The questions share a partner model and ask about the world, the partner's belief, or the next action. False-belief items have true-belief pairs.

**Recovery.** The action taken when a live `stop` contradicts an open partner. `wait` and `yield` recover. `commit` does not. The belief trace has to name that cue.

**Theory of mind.** Attributing a belief to the partner. The whose-belief item asks for that attribution. The score is still belief revision, not a claim that the network has a mind.

## Reading

The papers are grouped under [`academic_basis/`](academic_basis):

- `human_brain_models/` — what the learning-rule changes are allowed to be
- `machine_learning_behaviour_tests/` — how the tests are specified, including the POMDP primer
- `applied_uses/` — handoff, reversal, and an inspectable decision as the practical target
- `ethics_human_ai_communication/` — the trace page: the run states its belief, and a person can answer keep or discard

[`program.md`](program.md) is the short version of how those papers constrain the code.
