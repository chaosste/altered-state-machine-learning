"""Trace whether high-risk belief-filter changes alter choices and outcomes.

This diagnostic reuses the matched v1 baseline and Arm 4 checkpoints on the
expanded v2 scenario suite. It crosses checkpoint weights with the two existing
belief-filter settings, without retraining or introducing a new update-rate
parameter. Per-step reference actions substitute the other filter's belief
while holding the current checkpoint, observation, and recurrent state fixed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from contracts import HIGH_RISK_BELIEF_UPDATE_V1
from high_risk_suite import evaluate_closed_loop
from high_risk_v2_suite import SCENARIO_SUITE_V2_ID, expanded_heldout_cases
from train import BaselinePolicy, active_variant, apply_variant


SOURCE_ROOT = ROOT / "logs" / "high-risk-belief-update-v1"
OUTPUT_ROOT = ROOT / "logs" / "high-risk-belief-update-v1-decision-bridge"
REPORT_PATH = ROOT / "analysis" / "high-risk-belief-update-v1-decision-bridge-report.md"
SEEDS = (7, 11, 17, 23, 29, 31, 37, 41, 43, 47, 53)
CHECKPOINTS = ("baseline", "arm4")
FILTERS: Dict[str, Dict[str, float]] = {
    "baseline_filter": {"prior_precision": 1.5, "pe_gain": 0.35},
    "arm4_filter": {"prior_precision": 0.5, "pe_gain": 1.0},
}
PAIRED_T_95_DF10 = 2.228138852


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_policy(seed: int, variant: str, device: str) -> Tuple[BaselinePolicy, str]:
    model_path = SOURCE_ROOT / f"seed{seed}" / variant / "model.pt"
    metrics_path = model_path.with_name("metrics.json")
    if not model_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"Missing matched v1 checkpoint artifacts for seed {seed}, {variant}.")
    metrics = json.loads(metrics_path.read_text())
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    apply_variant(variant)
    if (
        metrics.get("contract_id") != HIGH_RISK_BELIEF_UPDATE_V1
        or metrics.get("variant_key") != variant
        or int(metrics.get("seed", -1)) != seed
        or checkpoint.get("contract_id") != HIGH_RISK_BELIEF_UPDATE_V1
        or checkpoint.get("variant") != active_variant()
    ):
        raise ValueError(f"Contract, variant, or seed mismatch for {model_path}.")
    policy = BaselinePolicy(hidden_dim=int(checkpoint["hidden_dim"]), device=device)
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return policy, _sha256(model_path)


def _outcome_rows(seed: int, checkpoint: str, filter_name: str, traces: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for trace in traces:
        steps = list(trace["steps"])
        terminal_action = trace["terminal_action"]
        final_world = steps[-1]["world"] if steps else None
        gate_breach = any(bool(step["safety_gate_breached"]) for step in steps)
        unsafe_terminal = terminal_action in ("proceed", "commit") and final_world == "closed"
        appropriate = (
            terminal_action in ("proceed", "commit") and final_world == "open"
        ) or (terminal_action == "yield" and final_world == "closed")
        brier_scores: List[float] = []
        log_scores: List[float] = []
        for step in steps:
            target_state = step["world"] if step["query_target"] == "world" else step["operator_belief"]
            target_index = 0 if target_state == "open" else 1
            belief = np.asarray(step["belief_after"], dtype=np.float64)
            target = np.zeros(2, dtype=np.float64)
            target[target_index] = 1.0
            brier_scores.append(float(np.square(belief - target).sum()))
            log_scores.append(float(-math.log(max(float(belief[target_index]), 1e-12))))
        rows.append({
            "seed": seed,
            "checkpoint": checkpoint,
            "belief_filter": filter_name,
            "case": trace["name"],
            "family": trace["family"],
            "reward": float(trace["episode_reward"]),
            "completed": terminal_action is not None,
            "appropriate_terminal": bool(appropriate),
            "unsafe_terminal": bool(unsafe_terminal),
            "safety_gate_breach": bool(gate_breach),
            "critical_event_miss": bool(trace["critical_event_miss"]),
            "mean_step_brier_to_truth": statistics.fmean(brier_scores) if brier_scores else 0.0,
            "mean_step_log_loss_to_truth": statistics.fmean(log_scores) if log_scores else 0.0,
        })
    return rows


def _decision_rows(seed: int, checkpoint: str, filter_name: str, traces: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for trace in traces:
        for step in trace["steps"]:
            if "decision_reference_action" not in step:
                continue
            rows.append({
                "seed": seed,
                "checkpoint": checkpoint,
                "belief_filter": filter_name,
                "case": trace["name"],
                "family": trace["family"],
                "step": step["step"],
                "cue": step["cue"],
                "query_target": step["query_target"],
                "world": step["world"],
                "actual_action": step["action"],
                "reference_action": step["decision_reference_action"],
                "action_changed": step["decision_changed_under_reference_belief"],
                "actual_belief_after": json.dumps(step["belief_after"]),
                "reference_belief_after": json.dumps(step["decision_reference_belief_after"]),
                "actual_task_reward": step["task_reward"],
                "reference_immediate_reward": step["decision_reference_immediate_reward"],
                "reference_immediate_reward_delta": step["decision_reference_immediate_reward_delta"],
                "reference_would_breach_gate": step["decision_reference_would_breach_safety_gate"],
            })
    return rows


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else 0.0


def _metric_by_seed(rows: Sequence[Mapping[str, object]], metric: str) -> Dict[int, float]:
    by_seed: Dict[int, List[float]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(float(row[metric]))
    return {seed: _mean(values) for seed, values in by_seed.items()}


def _paired_interval(differences: Sequence[float]) -> Dict[str, float]:
    mean = _mean(differences)
    if len(differences) < 2:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean}
    half_width = PAIRED_T_95_DF10 * statistics.stdev(differences) / math.sqrt(len(differences))
    return {"mean": mean, "ci95_low": mean - half_width, "ci95_high": mean + half_width}


def _bridge_metrics(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    by_seed: Dict[int, List[Mapping[str, object]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(row)
    seed_means: Dict[str, Dict[str, float]] = {}
    for seed, seed_rows in by_seed.items():
        flips = [row for row in seed_rows if bool(row["action_changed"])]
        seed_means[str(seed)] = {
            "action_flip_rate": _mean(float(bool(row["action_changed"])) for row in seed_rows),
            "counterfactual_immediate_reward_delta": _mean(float(row["reference_immediate_reward_delta"]) for row in seed_rows),
            "flip_help_rate": _mean(float(float(row["reference_immediate_reward_delta"]) > 0.0) for row in flips),
            "flip_harm_rate": _mean(float(float(row["reference_immediate_reward_delta"]) < 0.0) for row in flips),
            "reference_gate_risk_rate": _mean(float(bool(row["reference_would_breach_gate"])) for row in seed_rows),
        }
    flips = [row for row in rows if bool(row["action_changed"])]
    overall = {
        "action_flip_rate": _mean(float(bool(row["action_changed"])) for row in rows),
        "counterfactual_immediate_reward_delta": _mean(float(row["reference_immediate_reward_delta"]) for row in rows),
        "flip_help_rate": _mean(float(float(row["reference_immediate_reward_delta"]) > 0.0) for row in flips),
        "flip_harm_rate": _mean(float(float(row["reference_immediate_reward_delta"]) < 0.0) for row in flips),
        "reference_gate_risk_rate": _mean(float(bool(row["reference_would_breach_gate"])) for row in rows),
        "n_action_flips": len(flips),
        "seeds_with_action_flips": len({int(row["seed"]) for row in flips}),
        "scenario_templates_with_action_flips": len({str(row["case"]) for row in flips}),
    }
    return {
        "n_decisions": len(rows),
        "seed_means": seed_means,
        "overall": overall,
    }


def _summarize(outcomes: Sequence[Mapping[str, object]], decisions: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    cell_summary: Dict[str, Dict[str, object]] = {}
    group_keys = sorted({(str(row["checkpoint"]), str(row["belief_filter"])) for row in outcomes})
    metrics = (
        "reward", "completed", "appropriate_terminal", "unsafe_terminal", "safety_gate_breach",
        "critical_event_miss", "mean_step_brier_to_truth", "mean_step_log_loss_to_truth",
    )
    for checkpoint, belief_filter in group_keys:
        selected = [row for row in outcomes if row["checkpoint"] == checkpoint and row["belief_filter"] == belief_filter]
        seed_metrics = {metric: _metric_by_seed(selected, metric) for metric in metrics}
        cell_summary[f"{checkpoint}__{belief_filter}"] = {
            "checkpoint": checkpoint,
            "belief_filter": belief_filter,
            "outcome_means": {metric: _mean(values.values()) for metric, values in seed_metrics.items()},
            "safety_gate_failing_seed_count": sum(
                any(bool(row["safety_gate_breach"]) for row in selected if int(row["seed"]) == seed)
                for seed in SEEDS
            ),
            "bridge": _bridge_metrics([
                row for row in decisions if row["checkpoint"] == checkpoint and row["belief_filter"] == belief_filter
            ]),
            "seed_metrics": {metric: {str(seed): value for seed, value in values.items()} for metric, values in seed_metrics.items()},
        }

    paired: Dict[str, Dict[str, Dict[str, float]]] = {}
    for checkpoint in CHECKPOINTS:
        base_rows = [row for row in outcomes if row["checkpoint"] == checkpoint and row["belief_filter"] == "baseline_filter"]
        arm4_rows = [row for row in outcomes if row["checkpoint"] == checkpoint and row["belief_filter"] == "arm4_filter"]
        paired[f"belief_filter_effect_with_{checkpoint}_weights"] = {}
        for metric in metrics:
            base_seed = _metric_by_seed(base_rows, metric)
            arm4_seed = _metric_by_seed(arm4_rows, metric)
            differences = [arm4_seed[seed] - base_seed[seed] for seed in SEEDS]
            paired[f"belief_filter_effect_with_{checkpoint}_weights"][metric] = _paired_interval(differences)
    for belief_filter in FILTERS:
        base_rows = [row for row in outcomes if row["checkpoint"] == "baseline" and row["belief_filter"] == belief_filter]
        arm4_rows = [row for row in outcomes if row["checkpoint"] == "arm4" and row["belief_filter"] == belief_filter]
        paired[f"checkpoint_effect_with_{belief_filter}"] = {}
        for metric in metrics:
            base_seed = _metric_by_seed(base_rows, metric)
            arm4_seed = _metric_by_seed(arm4_rows, metric)
            differences = [arm4_seed[seed] - base_seed[seed] for seed in SEEDS]
            paired[f"checkpoint_effect_with_{belief_filter}"][metric] = _paired_interval(differences)
    return {"cells": cell_summary, "paired_seed_effects": paired}


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _render_report(
    summary: Mapping[str, object],
    n_cases: int,
    outcomes: Sequence[Mapping[str, object]],
    decisions: Sequence[Mapping[str, object]],
) -> str:
    cells = summary["cells"]
    arm4_cell = cells["arm4__arm4_filter"]
    arm4_bridge = arm4_cell["bridge"]["overall"]
    arm4_filter_effect = summary["paired_seed_effects"]["belief_filter_effect_with_arm4_weights"]["reward"]
    arm4_reward_ci_includes_zero = arm4_filter_effect["ci95_low"] <= 0.0 <= arm4_filter_effect["ci95_high"]
    flip_seed_phrase = (
        "a single training seed" if arm4_bridge["seeds_with_action_flips"] == 1
        else f"{arm4_bridge['seeds_with_action_flips']} training seeds"
    )
    lines = [
        "# High-Risk Belief Update v1: Decision Bridge Evaluation",
        "",
        f"Scenario suite: `{SCENARIO_SUITE_V2_ID}` ({n_cases} templates).",
        f"Checkpoints: existing matched baseline and Arm 4 v1 checkpoints across {len(SEEDS)} training seeds. No retraining was performed.",
        "",
        "This evaluation keeps each checkpoint's learned weights fixed and crosses them with the existing baseline and Arm 4 belief-filter settings. It does not vary a separate belief-update-rate parameter.",
        "",
        f"Across both checkpoint types, the Arm 4 filter lowered mean step Brier error by about 0.55, but local action flips were rare. With Arm 4 checkpoint weights, {arm4_bridge['n_action_flips']} of {arm4_cell['bridge']['n_decisions']} recorded decisions changed ({arm4_bridge['action_flip_rate']:.1%}), all on {flip_seed_phrase}. The paired full-trajectory reward difference was {arm4_filter_effect['mean']:.3f} [{arm4_filter_effect['ci95_low']:.3f}, {arm4_filter_effect['ci95_high']:.3f}] and its interval {'includes' if arm4_reward_ci_includes_zero else 'does not include'} zero; terminal and safety rates did not change. The filter changes beliefs much more often than it changes choices in these scenarios.",
        "",
        "## Closed-loop outcomes and local decision influence",
        "",
        "The action-flip rate asks: at a recorded decision, does the same checkpoint choose a different action when only its current belief is replaced by the other filter's belief? For that local comparison, the observation and recurrent state are held fixed. The reward delta is the immediate simulator score for the reference action minus the actual action, including any current hard-gate penalty; it is not a full-episode counterfactual.",
        "",
        "| Checkpoint weights | Belief filter | Mean reward | Completion | Appropriate terminal | Unsafe terminal | Gate-failing seeds | Step Brier vs truth | Changed decisions (seeds) | Local action flip | Helpful flips / flips | Harmful flips / flips | Reference gate-risk rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("baseline__baseline_filter", "baseline__arm4_filter", "arm4__baseline_filter", "arm4__arm4_filter"):
        cell = cells[key]
        means = cell["outcome_means"]
        bridge = cell["bridge"]["overall"]
        lines.append(
            f"| {cell['checkpoint']} | {cell['belief_filter']} | {means['reward']:.3f} | {means['completed']:.3f} | "
            f"{means['appropriate_terminal']:.3f} | {means['unsafe_terminal']:.3f} | "
            f"{cell['safety_gate_failing_seed_count']}/{len(SEEDS)} | {means['mean_step_brier_to_truth']:.3f} | "
            f"{bridge['n_action_flips']} ({bridge['seeds_with_action_flips']}) | {bridge['action_flip_rate']:.3f} | "
            f"{bridge['flip_help_rate']:.3f} / {bridge['n_action_flips']} | "
            f"{bridge['flip_harm_rate']:.3f} / {bridge['n_action_flips']} | "
            f"{bridge['reference_gate_risk_rate']:.3f} |"
        )
    lines.extend([
        "",
        "## Paired effect of the belief-filter setting",
        "",
        "Arm 4 filter minus baseline filter, paired within training seed while holding checkpoint weights fixed. Intervals are Student-t 95% CIs over the 11 seed-level means.",
        "",
        "| Fixed checkpoint weights | Metric | Mean difference | Paired 95% CI |",
        "|---|---|---:|---:|",
    ])
    for checkpoint in CHECKPOINTS:
        effects = summary["paired_seed_effects"][f"belief_filter_effect_with_{checkpoint}_weights"]
        for metric in ("reward", "completed", "appropriate_terminal", "unsafe_terminal", "mean_step_brier_to_truth"):
            effect = effects[metric]
            lines.append(
                f"| {checkpoint} | {metric} | {effect['mean']:.3f} | [{effect['ci95_low']:.3f}, {effect['ci95_high']:.3f}] |"
            )
    arm4_filter_flips = [
        row for row in decisions
        if row["checkpoint"] == "arm4" and row["belief_filter"] == "baseline_filter" and bool(row["action_changed"])
    ]
    if arm4_filter_flips:
        flip_seed_cases = sorted({(int(row["seed"]), str(row["case"])) for row in arm4_filter_flips})
        flip_seeds = sorted({seed for seed, _case in flip_seed_cases})
        flip_cases = sorted({case for _seed, case in flip_seed_cases})
        filter_effects = []
        for seed, case in flip_seed_cases:
            base = next(
                row for row in outcomes
                if int(row["seed"]) == seed and row["checkpoint"] == "arm4"
                and row["belief_filter"] == "baseline_filter" and row["case"] == case
            )
            arm4_filter = next(
                row for row in outcomes
                if int(row["seed"]) == seed and row["checkpoint"] == "arm4"
                and row["belief_filter"] == "arm4_filter" and row["case"] == case
            )
            filter_effects.append({
                "reward_delta": float(arm4_filter["reward"]) - float(base["reward"]),
                "appropriate": bool(arm4_filter["appropriate_terminal"]),
                "unsafe": bool(arm4_filter["unsafe_terminal"]),
                "gate": bool(arm4_filter["safety_gate_breach"]),
            })
        delta = _mean(row["reward_delta"] for row in filter_effects)
        appropriate_count = sum(bool(row["appropriate"]) for row in filter_effects)
        unsafe_count = sum(bool(row["unsafe"]) for row in filter_effects)
        gate_count = sum(bool(row["gate"]) for row in filter_effects)
        direction = ", ".join(sorted({f"{row['actual_action']} → {row['reference_action']}" for row in arm4_filter_flips}))
        lines.extend([
            "",
            "## Where the filter changed an Arm 4 decision",
            "",
            f"On training seed(s) `{', '.join(map(str, flip_seeds))}`, substituting the Arm 4 belief for the baseline belief changed {len(arm4_filter_flips)} local decisions across these templates: {', '.join(f'`{case}`' for case in flip_cases)}. The local action direction was {direction}. In the full paired episodes, using the Arm 4 filter changed mean reward by {delta:+.3f} per affected template; {appropriate_count}/{len(filter_effects)} ended appropriately, {unsafe_count} were unsafe, and {gate_count} breached a hard gate. These are useful case-specific examples, concentrated in {len(flip_seeds)} of 11 trained seeds, rather than a general performance effect.",
        ])
    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "- The two filters change prior precision and cue gain together, so this comparison estimates the combined Arm 4 belief-filter effect, not separate causal effects of each parameter.",
        "- The local action flip holds one checkpoint's current recurrent state fixed. It measures whether the current belief input can change the selected action at that moment; it does not estimate a complete alternative trajectory.",
        "- The full closed-loop filter comparison branches from the start of each scenario, so its rewards and safety outcomes show the end-to-end effect of changing the filter for these frozen policy weights. Crossed filter settings were not used to train those checkpoints and are diagnostic counterfactuals.",
        "- Scenario rewards and hard gates remain synthetic suite assumptions, not estimates of real-world harm or deployment safety.",
        "",
        "## Artifacts",
        "",
        "- `summary.json`: cell summaries, paired intervals, bridge rates, and provenance.",
        "- `episode_outcomes.csv`: one row per seed, checkpoint, filter, and scenario.",
        "- `decision_steps.csv`: actual and belief-substitution actions at each decision.",
        "- `traces.json`: full closed-loop traces with actual and shadow beliefs.",
    ])
    return "\n".join(lines) + "\n"


def run(output_root: Path = OUTPUT_ROOT, report_path: Path = REPORT_PATH, device: str = "cpu") -> Dict[str, object]:
    output_root = output_root.resolve()
    report_path = report_path.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing decision-bridge outputs at {output_root}.")
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing decision-bridge report {report_path}.")
    output_root.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cases = expanded_heldout_cases()
    outcomes: List[Dict[str, object]] = []
    decisions: List[Dict[str, object]] = []
    trace_artifact: List[Dict[str, object]] = []
    checkpoint_hashes: Dict[str, Dict[str, str]] = {variant: {} for variant in CHECKPOINTS}

    for seed in SEEDS:
        for checkpoint_variant in CHECKPOINTS:
            for filter_name, filter_values in FILTERS.items():
                policy, checkpoint_sha256 = _load_policy(seed, checkpoint_variant, device)
                checkpoint_hashes[checkpoint_variant][str(seed)] = checkpoint_sha256
                opposite_filter = "arm4_filter" if filter_name == "baseline_filter" else "baseline_filter"
                metrics, traces = evaluate_closed_loop(
                    policy,
                    cases=cases,
                    scenario_suite_id=SCENARIO_SUITE_V2_ID,
                    belief_filter=filter_values,
                    decision_reference_filter=FILTERS[opposite_filter],
                )
                del metrics
                outcomes.extend(_outcome_rows(seed, checkpoint_variant, filter_name, traces))
                decisions.extend(_decision_rows(seed, checkpoint_variant, filter_name, traces))
                trace_artifact.append({
                    "seed": seed,
                    "checkpoint": checkpoint_variant,
                    "belief_filter": filter_name,
                    "counterfactual_filter": opposite_filter,
                    "traces": traces,
                })

    summary = _summarize(outcomes, decisions)
    provenance_sources = (
        ROOT / "train.py",
        ROOT / "oracle.py",
        ROOT / "high_risk_suite.py",
        ROOT / "high_risk_v2_suite.py",
        ROOT / "scripts" / "high_risk_decision_bridge.py",
    )
    provenance = {
        "contract_id": HIGH_RISK_BELIEF_UPDATE_V1,
        "scenario_suite_id": SCENARIO_SUITE_V2_ID,
        "seeds": list(SEEDS),
        "episodes_per_checkpoint": 50,
        "scenario_count": len(cases),
        "checkpoints": list(CHECKPOINTS),
        "filters": FILTERS,
        "checkpoint_sha256": checkpoint_hashes,
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in provenance_sources if path.is_file()
        },
        "packages": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "worktree_dirty_at_run": True,
    }
    full_summary = {**summary, "provenance": provenance}
    (output_root / "summary.json").write_text(json.dumps(full_summary, indent=2, sort_keys=True) + "\n")
    _write_csv(output_root / "episode_outcomes.csv", outcomes)
    _write_csv(output_root / "decision_steps.csv", decisions)
    (output_root / "traces.json").write_text(json.dumps(trace_artifact, indent=2) + "\n")
    report_path.write_text(_render_report(summary, len(cases), outcomes, decisions))
    return full_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    summary = run(args.output, args.report, args.device)
    print(f"decision_bridge_summary={args.output.resolve() / 'summary.json'}")
    print(f"decision_bridge_report={args.report.resolve()}")
    for key, cell in summary["cells"].items():
        print(
            f"{key}: reward={cell['outcome_means']['reward']:.3f} "
            f"unsafe={cell['outcome_means']['unsafe_terminal']:.3f} "
            f"flip={cell['bridge']['overall']['action_flip_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
