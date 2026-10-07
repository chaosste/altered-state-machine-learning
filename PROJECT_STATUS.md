# Altered State Machine Learning — Project Status

Updated 7 October 2026.

## Repository identity

- Local project: `/Users/stephenbeale/Projects/Altered State Machine Learning`
- GitHub: `https://github.com/chaosste/altered-state-machine-learning`
- Branch: `main`
- Terminal prompt: `asml>`
- Python entry point: `.venv/bin/python rebus.py`

## Purpose

This project tests whether a small set of learning-rule changes helps an agent revise a hidden partner model under partial observability. `BeliefUpdateScore` is the only score used by the automatic keep/discard decision. The environment, exact filter, fixed evaluation scenarios, score components, and keep rules are documented in `program.md`.

## Current evaluation rules

Current version: `evidence-response-v3`.

- On clear questions, only the correct definite belief receives revision credit.
- On the no-access ambiguous question, only `undetermined` receives revision credit.
- The ambiguous question is excluded from revision speed and binary calibration.
- `ignored_evidence_rate` now means strong evidence followed by belief movement below `0.02`.
- `moved_but_uncertain_rate` means belief moved by at least `0.02` after strong evidence but remained above the uncertainty limit.
- `moved_but_uncertain_rate` is reported and shown in traces. It does not cause a score deduction or automatic discard.

The keep decision requires all four conditions:

1. `BeliefUpdateScore` improves by at least `0.02`.
2. Unsafe commit rate increases by no more than `0.05`.
3. Perseveration increases by no more than `0.05`.
4. Ignored-evidence rate increases by no more than `0.05`.

`eval.py` owns the measurements, thresholds, decision, and rejection reasons. Runners only train and serialize its report.

## Current Arm 4 results

Arm 4 lowers prior precision from `1.5` to `0.5` and raises belief prediction-error gain from `0.35` to `1.0`.

| Seed | Baseline score | Arm 4 score | Gain | Decision | Reason when discarded |
| ---: | ---: | ---: | ---: | :---: | --- |
| 7 | 0.6332 | 0.7393 | +0.1061 | Keep | |
| 11 | 0.6656 | 0.6734 | +0.0078 | Discard | Score gain below 0.02 |
| 17 | 0.6289 | 0.6179 | -0.0110 | Discard | Score decreased |
| 23 | 0.6564 | 0.7393 | +0.0829 | Discard | Perseveration increased by 0.2083 |
| 29 | 0.6332 | 0.7393 | +0.1061 | Keep | |
| 31 | 0.6606 | 0.7647 | +0.1041 | Discard | Perseveration increased by 0.0833 |
| 37 | 0.6332 | 0.7393 | +0.1061 | Keep | |
| 41 | 0.6332 | 0.7393 | +0.1061 | Keep | |
| 43 | 0.6332 | 0.7393 | +0.1061 | Keep | |
| 47 | 0.6342 | 0.6991 | +0.0649 | Keep | |
| 53 | 0.6332 | 0.7393 | +0.1061 | Keep | |

Summary across 11 seeds:

- Kept: 7 seeds (`7, 29, 37, 41, 43, 47, 53`)
- Discarded: 4 seeds (`11, 17, 23, 31`)
- Mean baseline score: `0.6405`
- Mean Arm 4 score: `0.7209`
- Mean score gain: `0.0805`
- Mean ignored-evidence rate: `0.2996 → 0.2703`
- Mean moved-but-uncertain rate: `0.0510 → 0.2157`

The project does not yet define one decision that combines several training seeds. Each seed currently receives its own keep/discard decision. Choosing and documenting the multi-seed promotion rule is the next important decision.

All saved policies currently have `ambiguous_abstention_accuracy: 0`: none returned `undetermined` on the no-access scenario.

## Working commands

Start the terminal interface:

```bash
cd "/Users/stephenbeale/Projects/Altered State Machine Learning"
.venv/bin/python rebus.py
```

Compare Arm 4 across several seeds:

```text
/compare episodes=50 seeds=7,11,17,23,29 candidate=arm4 output=logs
```

Open a candidate trace:

```text
/trace output=logs/seed7-arm4/candidate
```

Show a saved score:

```text
/score metrics=logs/seed7-arm4/candidate/metrics.json
```

## Repository rules for continuing the work

- Read `program.md` before changing evaluation meaning.
- Record a score or scenario change in `program.md` before editing `eval.py` or the fixed scenarios.
- Compare baseline and candidate using the same seed, episode count, fixed scenarios, and evaluation version.
- Do not reconstruct keep/discard logic outside `eval.py`.
- Preserve old evaluation JSON before rewriting current results.
- Run logs, checkpoints, traces, selection reports, and decision notes are intentionally committed.

## Suggested next workflow

1. Confirm the 11-seed results from `logs/seed*-arm4/selection/selection.json`.
2. Define a multi-seed promotion rule before using the 11 runs to make one overall Arm 4 decision.
3. Examine the two remaining failure types separately: insufficient score improvement (`11`, `17`) and excess perseveration (`23`, `31`).
4. Decide whether the no-access failure requires a new candidate or should remain a reported limitation.
5. Pre-register the next candidate in `program.md`, run it against the same seed set, and compare it with both baseline and Arm 4.
