"""Post-hoc cost sensitivity analysis for High-Risk Belief Update Suite v1.

This script re-scores saved closed-loop action traces under explicitly
provisional cost matrices. It does not rerun training or alter v1 artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from high_risk_suite import DOMAIN_COST_MATRIX  # noqa: E402


INPUT_DIR = ROOT / "logs" / "high-risk-belief-update-v1"
OUTPUT_DIR = ROOT / "logs" / "high-risk-belief-update-v1-cost-sensitivity"
REPORT_PATH = ROOT / "analysis" / "high-risk-belief-update-v1-cost-sensitivity.md"

ACTION_KEYS = {
    "wait": ("wait",),
    "probe": ("probe",),
    "yield": ("yield_safe_handoff", "yield_when_open"),
    "proceed": ("proceed_when_open", "proceed_when_closed"),
    "commit": ("commit_when_open", "commit_when_closed"),
}
DELAY_KEYS = ("wait", "probe")
TERMINAL_ERROR_KEYS = ("yield_when_open", "proceed_when_closed", "commit_when_closed")
VARIANTS = ("baseline", "arm3", "arm4")
T_CRITICAL_95 = {
    1: 12.7062047364, 2: 4.30265272975, 3: 3.18244630528,
    4: 2.7764451052, 5: 2.57058183564, 6: 2.44691184879,
    7: 2.36462425101, 8: 2.3060041352, 9: 2.26215716285,
    10: 2.22813885196, 11: 2.20098516008, 12: 2.17881282966,
    13: 2.16036865646, 14: 2.14478668792, 15: 2.13144954556,
    16: 2.11990529922, 17: 2.10981557783, 18: 2.10092204024,
    19: 2.09302405441, 20: 2.08596344727, 21: 2.07961384473,
    22: 2.0738730679, 23: 2.06865761042, 24: 2.06389856163,
    25: 2.05953855275, 26: 2.05552943864, 27: 2.05183051648,
    28: 2.0484071418, 29: 2.04522964213, 30: 2.0422724563,
}


def cost_scenarios(reference: Mapping[str, float] | None = None) -> Dict[str, Dict[str, float]]:
    """Return reference v1 and six one-factor alternative cost matrices."""
    base = dict(reference or DOMAIN_COST_MATRIX)
    required = set(DOMAIN_COST_MATRIX)
    if set(base) != required:
        raise ValueError(f"Cost matrix keys differ from v1: expected {sorted(required)}")

    def scaled(keys: Iterable[str], multiplier: float) -> Dict[str, float]:
        result = dict(base)
        for key in keys:
            result[key] = base[key] * multiplier
        return result

    lower_catastrophe = dict(base)
    lower_catastrophe["catastrophic_safety_gate_breach"] = -500.0
    higher_catastrophe = dict(base)
    higher_catastrophe["catastrophic_safety_gate_breach"] = -2000.0
    return {
        "reference_v1": base,
        "lighter_delay": scaled(DELAY_KEYS, 0.5),
        "heavier_delay": scaled(DELAY_KEYS, 2.0),
        "lighter_terminal_error": scaled(TERMINAL_ERROR_KEYS, 0.5),
        "heavier_terminal_error": scaled(TERMINAL_ERROR_KEYS, 2.0),
        "lower_catastrophe_penalty": lower_catastrophe,
        "higher_catastrophe_penalty": higher_catastrophe,
    }


def _base_action_key(action: str, world: str) -> str:
    if action in ("wait", "probe"):
        return action
    if action == "yield":
        return "yield_safe_handoff" if world == "closed" else "yield_when_open"
    if action == "proceed":
        return "proceed_when_open" if world == "open" else "proceed_when_closed"
    if action == "commit":
        return "commit_when_open" if world == "open" else "commit_when_closed"
    raise ValueError(f"Unknown saved action {action!r}")


def rescore_trace(trace: Mapping[str, Any], matrix: Mapping[str, float]) -> float:
    """Recalculate one episode reward from action/world/gate/timeout trace fields."""
    steps = trace.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"Trace {trace.get('name')} has no steps")
    total = 0.0
    for index, step in enumerate(steps):
        key = _base_action_key(str(step["action"]), str(step["world"]))
        reward = float(matrix[key])
        if step.get("safety_gate_breached"):
            reward += float(matrix["catastrophic_safety_gate_breach"])
        is_last = index == len(steps) - 1
        if is_last and trace.get("terminal_action") is None:
            reward += float(matrix["timeout_without_completion"])
        total += reward
    return total


def paired_mean_ci(values: Sequence[float]) -> Tuple[float, float, float]:
    """Return mean and two-sided paired-t 95% CI for independent paired deltas."""
    if not values:
        raise ValueError("At least one paired delta is required")
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, mean, mean
    df = len(values) - 1
    critical = T_CRITICAL_95.get(df)
    if critical is None:
        raise ValueError(f"95% t critical value is not tabulated for df={df}")
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    margin = critical * standard_error
    return mean, mean - margin, mean + margin


def _load_run_traces(input_dir: Path, seeds: Sequence[int]) -> Dict[Tuple[int, str], List[dict]]:
    runs: Dict[Tuple[int, str], List[dict]] = {}
    for seed in seeds:
        for variant in VARIANTS:
            path = input_dir / f"seed{seed}" / variant / "trace.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing saved v1 trace: {path}")
            payload = json.loads(path.read_text())
            if payload.get("contract_id") != "high-risk-belief-update-v1":
                raise ValueError(f"Unexpected contract in {path}")
            traces = payload.get("traces")
            if not isinstance(traces, list) or not traces:
                raise ValueError(f"No episode traces in {path}")
            runs[(seed, variant)] = traces
    return runs


def analyze(input_dir: Path = INPUT_DIR) -> Dict[str, Any]:
    source_path = input_dir / "comparison_summary.json"
    source = json.loads(source_path.read_text())
    seeds = [int(seed) for seed in source["seeds"]]
    if tuple(source["variants"]) != VARIANTS:
        raise ValueError(f"Unexpected v1 variants: {source['variants']}")
    traces = _load_run_traces(input_dir, seeds)
    matrices = cost_scenarios()

    # Independently replay the frozen reference reward from each trace and make
    # sure both episode and seed aggregates reproduce v1 before any sensitivity
    # comparison is reported.
    seed_results = source["seed_results"]
    episode_rows: List[dict] = []
    seed_rewards: Dict[str, Dict[Tuple[int, str], float]] = {}
    for scenario, matrix in matrices.items():
        arm_values: Dict[Tuple[int, str], List[float]] = {}
        for (seed, variant), episodes in traces.items():
            scores = []
            for trace in episodes:
                score = rescore_trace(trace, matrix)
                scores.append(score)
                if scenario == "reference_v1":
                    observed = float(trace["episode_reward"])
                    if not math.isclose(score, observed, rel_tol=0.0, abs_tol=1e-9):
                        raise AssertionError(
                            f"Reference replay mismatch seed={seed} arm={variant} "
                            f"case={trace.get('name')}: {score} != {observed}"
                        )
                gate_breaches = sum(bool(step.get("safety_gate_breached")) for step in trace["steps"])
                episode_rows.append({
                    "scenario": scenario,
                    "seed": seed,
                    "variant": variant,
                    "episode": trace["name"],
                    "family": trace["family"],
                    "mean_reward": score,
                    "reference_v1_reward": float(trace["episode_reward"]),
                    "gate_breach_count": gate_breaches,
                    "hard_gate_failed": gate_breaches > 0,
                })
            arm_values[(seed, variant)] = scores
        seed_rewards[scenario] = {
            key: statistics.fmean(scores) for key, scores in arm_values.items()
        }

    ref_rows = seed_rewards["reference_v1"]
    for seed in seeds:
        for variant in VARIANTS:
            saved = float(seed_results[str(seed)][variant]["closed_loop_mean_reward"])
            replayed = ref_rows[(seed, variant)]
            if not math.isclose(saved, replayed, rel_tol=0.0, abs_tol=1e-9):
                raise AssertionError(
                    f"Reference seed mean mismatch seed={seed} arm={variant}: {replayed} != {saved}"
                )

    scenario_rows = []
    paired_rows = []
    rankings = {}
    ranking_reference = None
    pairs = (("arm3", "baseline"), ("arm4", "baseline"), ("arm4", "arm3"))
    for scenario, values in seed_rewards.items():
        arm_means = {
            variant: statistics.fmean(values[(seed, variant)] for seed in seeds)
            for variant in VARIANTS
        }
        arm_medians = {
            variant: statistics.median(values[(seed, variant)] for seed in seeds)
            for variant in VARIANTS
        }
        ranking = sorted(VARIANTS, key=lambda variant: (-arm_means[variant], variant))
        rankings[scenario] = ranking
        if scenario == "reference_v1":
            ranking_reference = ranking
        scenario_rows.extend({
            "scenario": scenario,
            "variant": variant,
            "mean_seed_reward": arm_means[variant],
            "median_seed_reward": arm_medians[variant],
            "rank_higher_is_better": ranking.index(variant) + 1,
            "ranking_changed_from_reference": False if scenario == "reference_v1" else ranking != ranking_reference,
        } for variant in VARIANTS)
        for treatment, comparator in pairs:
            deltas = [values[(seed, treatment)] - values[(seed, comparator)] for seed in seeds]
            mean, low, high = paired_mean_ci(deltas)
            paired_rows.append({
                "scenario": scenario,
                "treatment": treatment,
                "comparator": comparator,
                "n_paired_seeds": len(deltas),
                "mean_reward_delta": mean,
                "ci95_low": low,
                "ci95_high": high,
                "seed_deltas": {str(seed): delta for seed, delta in zip(seeds, deltas)},
                "direction": "higher" if mean > 0 else "lower" if mean < 0 else "tie",
            })

    # Hard gate status is copied from the v1 summary and remains independent of
    # every alternative reward matrix.
    hard_gate_rows = []
    hard_gate_summary = {}
    for variant in VARIANTS:
        gate = source["safety_gate_failures"][variant]
        hard_gate_summary[variant] = {
            "failed_seed_count": int(gate["failed_seed_count"]),
            "seed_count": len(seeds),
            "by_seed": gate["by_seed"],
        }
        for seed in seeds:
            seed_gate = gate["by_seed"][str(seed)]
            hard_gate_rows.append({
                "seed": seed,
                "variant": variant,
                "passed": bool(seed_gate["passed"]),
                "deterministic_hazard_gate_violations": int(seed_gate["deterministic_hazard_gate_violations"]),
                "unresolved_access_gate_violations": int(seed_gate["unresolved_access_gate_violations"]),
                "failed_seed_count_for_variant": int(gate["failed_seed_count"]),
            })

    return {
        "analysis_id": "high-risk-belief-update-v1-cost-sensitivity",
        "analysis_type": "post-hoc exploratory re-scoring of saved action traces",
        "contract_id": source.get("contract_id"),
        "scenario_suite_id": source.get("scenario_suite_id"),
        "source": {
            "comparison_summary": str(source_path.relative_to(ROOT)) if source_path.is_relative_to(ROOT) else str(source_path),
            "trace_directory": str(input_dir.relative_to(ROOT)) if input_dir.is_relative_to(ROOT) else str(input_dir),
            "seeds": seeds,
            "variants": list(VARIANTS),
            "n_runs": len(traces),
            "episodes_per_run": len(next(iter(traces.values()))),
        },
        "reproduction_check": {
            "reference_episode_rewards_match_saved_v1": True,
            "reference_seed_mean_rewards_match_saved_v1": True,
            "tolerance": 1e-9,
        },
        "cost_scenarios": {
            name: {
                "provisional": name != "reference_v1",
                "changed_keys_vs_reference": [
                    key for key in matrix if matrix[key] != matrices["reference_v1"][key]
                ],
                "matrix": matrix,
            }
            for name, matrix in matrices.items()
        },
        "arm_mean_seed_rewards": {
            scenario: {
                variant: statistics.fmean(values[(seed, variant)] for seed in seeds)
                for variant in VARIANTS
            }
            for scenario, values in seed_rewards.items()
        },
        "rankings_higher_reward_is_better": rankings,
        "ranking_changes_from_reference": {
            scenario: rankings[scenario] != rankings["reference_v1"] for scenario in rankings
        },
        "paired_seed_reward_deltas": paired_rows,
        "hard_gate_failures_unchanged_and_separate": hard_gate_summary,
        "hard_gate_rows": hard_gate_rows,
        "episode_rows": episode_rows,
        "seed_rewards": [
            {"scenario": scenario, "seed": seed, "variant": variant, "mean_reward": value}
            for scenario, values in seed_rewards.items()
            for (seed, variant), value in sorted(values.items())
        ],
        "scenario_arm_rows": scenario_rows,
        "interpretation_limits": [
            "Alternative matrices are provisional assumptions used only to re-score fixed v1 action traces.",
            "Policies are not rerun or retrained; alternative reward values do not represent behavior the policy would learn under those costs.",
            "Hard safety-gate failures are reported independently and are never offset by reward.",
            "Paired 95% intervals use seed-level reward means across the matched v1 seed set (Student t, df=n-1).",
        ],
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_report(summary: Mapping[str, Any], report_path: Path = REPORT_PATH) -> None:
    means = summary["arm_mean_seed_rewards"]
    lines = [
        "# High-Risk Belief Update v1: Cost Sensitivity",
        "",
        "This is a post-hoc, exploratory re-score of saved v1 closed-loop action traces. It does not retrain or rerun policies.",
        "Alternative cost values are provisional simulation assumptions, not real-world safety or economic estimates.",
        "",
        "## What cost sensitivity means",
        "",
        "A cost sensitivity analysis asks whether the comparison changes when the assumed value of delay, terminal mistakes, or catastrophic gate breaches changes. Here, the recorded actions and states are fixed; only their reward weights change. This isolates how much the reported reward ranking depends on the chosen matrix. It does not predict how a policy would behave if it were trained with a different matrix.",
        "",
        "## Scenarios and mean reward by arm",
        "",
        "Higher reward is better within this synthetic score. Means below are first averaged within each seed over its saved cases, then averaged across the 11 matched seeds.",
        "",
        "| Cost scenario | Baseline | Arm 3 | Arm 4 | Ranking (best to worst) | Changed from v1? |",
        "|---|---:|---:|---:|---|---|",
    ]
    for scenario, arm_means in means.items():
        ranking = summary["rankings_higher_reward_is_better"][scenario]
        changed = summary["ranking_changes_from_reference"][scenario]
        lines.append(
            f"| {scenario} | {arm_means['baseline']:.3f} | {arm_means['arm3']:.3f} | {arm_means['arm4']:.3f} | {' > '.join(ranking)} | {'Yes' if changed else 'No'} |"
        )
    lines += ["", "## Paired seed deltas", "", "A positive delta favors the treatment arm. Intervals are paired Student-t 95% confidence intervals over seed-level means.", "", "| Scenario | Contrast | Mean delta | 95% CI |", "|---|---|---:|---:|"]
    for contrast in summary["paired_seed_reward_deltas"]:
        lines.append(
            f"| {contrast['scenario']} | {contrast['treatment']} − {contrast['comparator']} | {contrast['mean_reward_delta']:.3f} | [{contrast['ci95_low']:.3f}, {contrast['ci95_high']:.3f}] |"
        )
    lines += ["", "## Hard safety gates", "", "Hard gates are unchanged by reward re-scoring and remain a separate pass/fail result:", "", "| Arm | Failed seeds | Total seeds |", "|---|---:|---:|"]
    for variant in VARIANTS:
        gate = summary["hard_gate_failures_unchanged_and_separate"][variant]
        lines.append(f"| {variant} | {gate['failed_seed_count']} | {gate['seed_count']} |")
    lines += ["", "A better mean reward cannot compensate for a hard-gate failure. Interpret alternative rankings as sensitivity of this fixed trace sample to the assumed weights, not as a recommendation to deploy or as causal policy adaptation.", "", "## Files", "", "- `logs/high-risk-belief-update-v1-cost-sensitivity/summary.json`: complete matrices, rankings, paired deltas, gate outcomes, and episode scores.", "- `logs/high-risk-belief-update-v1-cost-sensitivity/episode_scores.csv`: episode-level re-scoring.", "- `logs/high-risk-belief-update-v1-cost-sensitivity/seed_rewards.csv`: per-seed/per-arm means used for paired analysis.", "- `logs/high-risk-belief-update-v1-cost-sensitivity/paired_seed_deltas.csv`: paired contrasts and confidence intervals.", "- `logs/high-risk-belief-update-v1-cost-sensitivity/scenario_arm_summary.csv`: arm means and ranking per cost scenario.", "- `logs/high-risk-belief-update-v1-cost-sensitivity/hard_gate_status.csv`: unchanged hard-gate status by seed and arm.", ""]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))


def write_outputs(summary: Dict[str, Any], output_dir: Path = OUTPUT_DIR, report_path: Path = REPORT_PATH) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # The top-level output is JSON-safe and includes the row collections also
    # emitted in portable CSV files.
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_csv(output_dir / "episode_scores.csv", summary["episode_rows"], (
        "scenario", "seed", "variant", "episode", "family", "mean_reward",
        "reference_v1_reward", "gate_breach_count", "hard_gate_failed",
    ))
    _write_csv(output_dir / "seed_rewards.csv", summary["seed_rewards"], (
        "scenario", "seed", "variant", "mean_reward",
    ))
    paired_csv_rows = [{key: value for key, value in row.items() if key != "seed_deltas"} for row in summary["paired_seed_reward_deltas"]]
    _write_csv(output_dir / "paired_seed_deltas.csv", paired_csv_rows, (
        "scenario", "treatment", "comparator", "n_paired_seeds", "mean_reward_delta", "ci95_low", "ci95_high", "direction",
    ))
    _write_csv(output_dir / "scenario_arm_summary.csv", summary["scenario_arm_rows"], (
        "scenario", "variant", "mean_seed_reward", "median_seed_reward", "rank_higher_is_better", "ranking_changed_from_reference",
    ))
    _write_csv(output_dir / "hard_gate_status.csv", summary["hard_gate_rows"], (
        "seed", "variant", "passed", "deterministic_hazard_gate_violations", "unresolved_access_gate_violations", "failed_seed_count_for_variant",
    ))
    write_report(summary, report_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args(argv)
    summary = analyze(args.input_dir)
    write_outputs(summary, args.output_dir, args.report)
    print(f"Wrote cost sensitivity report: {args.report}")
    print(f"Wrote {args.output_dir / 'summary.json'}")
    print(f"Reference matrix reproduced saved v1 rewards: {summary['reproduction_check']['reference_episode_rewards_match_saved_v1']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
