"""Matched baseline/Arm 3/Arm 4 runner for high-risk-belief-update-v1."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contracts import HIGH_RISK_BELIEF_UPDATE_V1
from high_risk_suite import DOMAIN_COST_MATRIX, SCENARIO_SUITE_ID, scenario_assumptions
from train import (
    active_variant_key,
    apply_variant,
    phenotype_constants,
    load_training_schedule,
    scenario_schedule_hash,
    scenario_schedule_payload,
    train,
    training_schedule_for_seed,
)


DEFAULT_SEEDS = (7, 11, 17, 23, 29, 31, 37, 41, 43, 47, 53)
REQUIRED_VARIANTS = ("baseline", "arm3", "arm4")
REPORT_PATH = ROOT / "analysis" / "arm3-arm4-high-risk-v1-report.md"
CONTRAST_METRICS = {
    "native_prior_initial_brier": (
        "belief_replay", "native_prior", "initial_prior_calibration", "mean_brier_error_before_evidence"
    ),
    "common_prior_initial_brier": (
        "belief_replay", "common_prior", "initial_prior_calibration", "mean_brier_error_before_evidence"
    ),
    "native_prior_stepwise_brier_to_exact": (
        "belief_replay", "native_prior", "within_episode_updates", "stepwise_posterior_brier_error"
    ),
    "common_prior_stepwise_brier_to_exact": (
        "belief_replay", "common_prior", "within_episode_updates", "stepwise_posterior_brier_error"
    ),
    "common_prior_brier_ground_truth": (
        "belief_replay", "common_prior", "within_episode_updates", "brier_score_against_ground_truth"
    ),
    "common_prior_log_score_ground_truth": (
        "belief_replay", "common_prior", "within_episode_updates", "log_score_against_ground_truth"
    ),
    "common_prior_world_attribution_accuracy": (
        "belief_replay", "common_prior", "within_episode_updates", "world_belief_attribution_accuracy"
    ),
    "common_prior_operator_attribution_accuracy": (
        "belief_replay", "common_prior", "within_episode_updates", "operator_belief_attribution_accuracy"
    ),
    "common_prior_revision_latency_steps": (
        "belief_replay", "common_prior", "within_episode_updates", "revision_latency_steps_after_decisive_change"
    ),
    "common_prior_appropriate_abstention_rate": (
        "belief_replay", "common_prior", "within_episode_updates", "appropriate_abstention_rate_when_unresolved"
    ),
    "closed_loop_mean_reward": ("closed_loop", "mean_reward"),
    "closed_loop_completion_rate": ("closed_loop", "completion_rate"),
    "closed_loop_appropriate_terminal_rate": ("closed_loop", "appropriate_terminal_action"),
    "closed_loop_inappropriate_terminal_rate": ("closed_loop", "inappropriate_terminal_action"),
    "closed_loop_unsafe_terminal_rate": ("closed_loop", "unsafe_terminal_action"),
    "closed_loop_critical_event_miss_rate": ("closed_loop", "critical_event_miss"),
    "closed_loop_probe_count": ("closed_loop", "probe_count"),
    "closed_loop_delay_steps": ("closed_loop", "delay_steps"),
}


def _get(payload: Mapping[str, object], path: Sequence[str]) -> float:
    value: object = payload
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(".".join(path))
        value = value[key]
    return float(value)


def _t_critical_975(df: int) -> float:
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
        19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
        25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    return table.get(df, 1.96)


def _paired_summary(differences: Sequence[float], confidence_interval: bool = True) -> Dict[str, object]:
    values = [float(value) for value in differences]
    mean = statistics.mean(values)
    median = statistics.median(values)
    if confidence_interval and len(values) > 1:
        margin = _t_critical_975(len(values) - 1) * statistics.stdev(values) / math.sqrt(len(values))
    else:
        margin = 0.0
    return {
        "n_seeds": len(values),
        "mean_difference": mean,
        "median_difference": median,
        "paired_ci95_mean": [mean - margin, mean + margin] if confidence_interval else None,
        "confidence_interval_status": "seed_paired_student_t" if confidence_interval else "not_applicable_deterministic_replay",
        "per_seed_difference": values,
    }


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_provenance() -> Dict[str, object]:
    tracked_sources = [
        ROOT / "contracts.py", ROOT / "train.py", ROOT / "env.py", ROOT / "oracle.py",
        ROOT / "high_risk_suite.py", ROOT / "scripts" / "high_risk_compare.py",
        ROOT / "eval.py", ROOT / "program.md",
    ]
    research_sources = sorted((ROOT / "academic_basis").rglob("*.pdf"))
    research_sources.extend([
        ROOT / "analysis" / "conditions_3_4_6_modal.ipynb",
        ROOT / "analysis" / "conditions_3_4_6_modal_embedded.ipynb",
        ROOT / "analysis" / "conditions_3_4_6_logs.zip",
    ])
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unavailable", True
    packages = {"python": platform.python_version()}
    for package, label in (("numpy", "numpy"), ("torch", "torch")):
        try:
            packages[label] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[label] = "unavailable"
    return {
        "git_revision": revision,
        "worktree_dirty_at_run": dirty,
        "packages": packages,
        "source_sha256": {str(path.relative_to(ROOT)): _source_hash(path) for path in tracked_sources if path.is_file()},
        "research_source_sha256": {str(path.relative_to(ROOT)): _source_hash(path) for path in research_sources if path.is_file()},
    }


def _profile_means(run: Mapping[str, object]) -> Dict[str, float]:
    return {name: _get(run, path) for name, path in CONTRAST_METRICS.items()}


def build_comparison_summary(
    results: Mapping[int, Mapping[str, Mapping[str, object]]],
    episodes: int,
    seeds: Sequence[int],
    provenance: Mapping[str, object],
    device: str,
) -> Dict[str, object]:
    seed_results: Dict[str, object] = {}
    for seed in seeds:
        seed_results[str(seed)] = {
            variant: _profile_means(results[seed][variant])
            for variant in REQUIRED_VARIANTS
        }

    variant_summary: Dict[str, object] = {}
    for variant in REQUIRED_VARIANTS:
        metrics_by_name = {
            name: [_get(results[seed][variant], path) for seed in seeds]
            for name, path in CONTRAST_METRICS.items()
        }
        variant_summary[variant] = {
            name: {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "per_seed": {str(seed): value for seed, value in zip(seeds, values)},
            }
            for name, values in metrics_by_name.items()
        }

    pairings = (("arm3", "baseline"), ("arm4", "baseline"), ("arm4", "arm3"))
    contrasts: Dict[str, object] = {}
    for first, second in pairings:
        pair_name = f"{first}_minus_{second}"
        contrasts[pair_name] = {}
        for name, path in CONTRAST_METRICS.items():
            deltas = [_get(results[seed][first], path) - _get(results[seed][second], path) for seed in seeds]
            fixed_replay = name.startswith(("native_prior", "common_prior"))
            contrasts[pair_name][name] = _paired_summary(deltas, confidence_interval=not fixed_replay)

    family_summary: Dict[str, object] = {}
    replay_family_summary: Dict[str, object] = {}
    replay_uncertainty_summary: Dict[str, object] = {}
    evidence_response_summary: Dict[str, object] = {}
    information_availability_summary: Dict[str, object] = {}
    prior_alignment_summary: Dict[str, object] = {}
    for variant in REQUIRED_VARIANTS:
        by_family: Dict[str, List[Mapping[str, object]]] = {}
        for seed in seeds:
            families = results[seed][variant]["closed_loop"]["by_scenario_family"]
            assert isinstance(families, Mapping)
            for family, values in families.items():
                by_family.setdefault(str(family), []).append(values)
        family_summary[variant] = {
            family: {
                field: statistics.mean(float(row[field]) for row in family_rows)
                for field in (
                    "mean_reward", "completion", "appropriate_terminal_action",
                    "inappropriate_terminal_action", "unsafe_terminal_action",
                    "critical_event_miss", "probe_count", "delay_steps", "safe_handoff",
                    "no_terminal_action", "deterministic_hazard_gate_violations",
                    "unresolved_access_gate_violations",
                )
            }
            for family, family_rows in by_family.items()
        }
        replay_family_summary[variant] = {}
        replay_uncertainty_summary[variant] = {}
        evidence_response_summary[variant] = {}
        information_availability_summary[variant] = {}
        prior_alignment_summary[variant] = {}
        for prior_mode in ("native_prior", "common_prior"):
            family_values: Dict[str, List[Mapping[str, object]]] = {}
            uncertainty_values: Dict[str, List[Mapping[str, object]]] = {}
            evidence_values: Dict[str, List[Mapping[str, object]]] = {}
            availability_values: Dict[str, List[Mapping[str, object]]] = {}
            alignment_values: Dict[str, List[Mapping[str, object]]] = {}
            for seed in seeds:
                replay = results[seed][variant]["belief_replay"][prior_mode]
                for family, metrics in replay["calibration_by_scenario_family"].items():
                    family_values.setdefault(str(family), []).append(metrics)
                for uncertainty, metrics in replay["calibration_by_uncertainty_type"].items():
                    uncertainty_values.setdefault(str(uncertainty), []).append(metrics)
                for category, metrics in replay["update_response_by_evidence"].items():
                    evidence_values.setdefault(str(category), []).append(metrics)
                for availability, metrics in replay["response_by_information_availability"].items():
                    availability_values.setdefault(str(availability), []).append(metrics)
                for alignment, metrics in replay["initial_prior_calibration"]["by_truth_alignment"].items():
                    alignment_values.setdefault(str(alignment), []).append(metrics)
            replay_family_summary[variant][prior_mode] = {
                family: {
                    field: statistics.mean(float(row[field]) for row in rows)
                    for field in ("brier", "log_score", "posterior_tv_error", "state_accuracy", "calibration_gap", "appropriate_abstention_rate", "n_steps")
                }
                for family, rows in family_values.items()
            }
            replay_uncertainty_summary[variant][prior_mode] = {
                kind: {
                    field: statistics.mean(float(row[field]) for row in rows)
                    for field in ("brier", "log_score", "posterior_tv_error", "state_accuracy", "calibration_gap", "appropriate_abstention_rate", "n_steps")
                }
                for kind, rows in uncertainty_values.items()
            }
            evidence_response_summary[variant][prior_mode] = {
                category: {
                    field: statistics.mean(float(row[field]) for row in rows)
                    for field in ("mean_direction_toward_truth", "mean_update_magnitude_tv", "n_steps")
                }
                for category, rows in evidence_values.items()
            }
            information_availability_summary[variant][prior_mode] = {
                name: {
                    field: statistics.mean(float(row[field]) for row in rows)
                    for field in ("mean_direction_toward_truth", "mean_update_magnitude_tv", "mean_posterior_tv_error_to_exact", "appropriate_abstention_rate", "n_steps")
                }
                for name, rows in availability_values.items()
            }
            prior_alignment_summary[variant][prior_mode] = {
                alignment: {
                    field: statistics.mean(float(row[field]) for row in rows)
                    for field in ("mean_brier_error_before_evidence", "mean_log_score_before_evidence", "n_cases")
                }
                for alignment, rows in alignment_values.items()
            }

    failures = {}
    for variant in REQUIRED_VARIANTS:
        by_seed = {}
        for seed in seeds:
            task = results[seed][variant]["closed_loop"]
            by_seed[str(seed)] = {
                "deterministic_hazard_gate_violations": int(task["deterministic_hazard_gate_violations"]),
                "unresolved_access_gate_violations": int(task["unresolved_access_gate_violations"]),
                "passed": bool(task["safety_gates_pass"]),
            }
        failures[variant] = {
            "failed_seed_count": sum(not item["passed"] for item in by_seed.values()),
            "by_seed": by_seed,
        }

    schedule_hashes = {
        str(seed): sorted({str(results[seed][variant]["training_schedule_sha256"]) for variant in REQUIRED_VARIANTS})
        for seed in seeds
    }
    if any(len(hashes) != 1 for hashes in schedule_hashes.values()):
        raise ValueError("Matched variants did not use identical training schedule hashes.")
    return {
        "contract_id": HIGH_RISK_BELIEF_UPDATE_V1,
        "scenario_suite_id": SCENARIO_SUITE_ID,
        "training_episodes": episodes,
        "device": device,
        "seeds": list(seeds),
        "variants": list(REQUIRED_VARIANTS),
        "training_schedule_sha256_by_seed": {seed: values[0] for seed, values in schedule_hashes.items()},
        "seed_results": seed_results,
        "variant_means_and_medians": variant_summary,
        "paired_contrasts": contrasts,
        "closed_loop_family_summary": family_summary,
        "belief_calibration_by_family": replay_family_summary,
        "belief_calibration_by_uncertainty_type": replay_uncertainty_summary,
        "belief_update_response_by_evidence": evidence_response_summary,
        "belief_response_by_information_availability": information_availability_summary,
        "initial_prior_calibration_by_truth_alignment": prior_alignment_summary,
        "safety_gate_failures": failures,
        "provenance": dict(provenance),
    }


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}" if value < 0 else f"{value:.{digits}f}"


def render_report(summary: Mapping[str, object], results: Mapping[int, Mapping[str, Mapping[str, object]]]) -> str:
    seeds = [int(seed) for seed in summary["seeds"]]
    seed_results = summary["seed_results"]
    lines = [
        "# Arm 3 vs Arm 4: High-Risk Belief Update Suite v1",
        "",
        f"Contract: `{HIGH_RISK_BELIEF_UPDATE_V1}`  ",
        f"Scenario suite: `{SCENARIO_SUITE_ID}`  ",
        f"Training: {summary['training_episodes']} episodes per variant; seeds `{','.join(map(str, seeds))}`; device `{summary['device']}`; shared pre-generated schedule per seed.",
        "",
        f"This is an initial {len(seeds)}-seed computational comparison. It is not broad deployment evidence, and none of the simulated domain mappings, rewards, or action consequences are validated for real-world use.",
        "",
        "## What the suite measures",
        "",
        "Controlled replay feeds the same cue sequence and previous-action sequence to every arm. Native-prior results include each arm's configured prior; common-prior results use the baseline prior in all arms, isolating evidence-response differences. Initial-prior calibration is shown separately from within-episode updates. Exact posteriors use the declared cue source/mapping, prior, and marked transition model; a reversal transition applies a symmetric 0.5 switch before that step's cue, resetting the reference posterior to uniform without revealing the new state.",
        "",
        "Closed-loop results let each trained policy act in the same hidden scenario templates. Completion means a terminal decision before timeout; appropriateness and safety are reported separately. The cost matrix is frozen in `high_risk_suite.py`: "
        + "; ".join(f"{key}={value:g}" for key, value in DOMAIN_COST_MATRIX.items()) + ". A gate breach is reported separately and cannot be offset by average reward.",
        "",
        "## Arm-level means and medians",
        "",
        "Each value is first averaged within seed, then summarized across the matched seed cohort.",
        "",
        "| Arm | Native initial-prior Brier | Common initial-prior Brier | Native replay Brier vs exact | Common replay Brier vs exact | Common-prior Brier vs truth | Task reward | Completion | Appropriate terminals | Inappropriate terminals | Unsafe terminals | Probe count | Delay steps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    fields = [
        "native_prior_initial_brier", "common_prior_initial_brier",
        "native_prior_stepwise_brier_to_exact", "common_prior_stepwise_brier_to_exact",
        "common_prior_brier_ground_truth", "closed_loop_mean_reward", "closed_loop_completion_rate",
        "closed_loop_appropriate_terminal_rate", "closed_loop_inappropriate_terminal_rate",
        "closed_loop_unsafe_terminal_rate", "closed_loop_probe_count", "closed_loop_delay_steps",
    ]
    for variant in REQUIRED_VARIANTS:
        aggregate = summary["variant_means_and_medians"][variant]
        cells = [variant]
        for field in fields:
            row = aggregate[field]
            cells.append(f"{_fmt(float(row['mean']))} / {_fmt(float(row['median']))}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "Cells show mean / median. Initial-prior error is calculated before evidence. Replay Brier-to-exact is the squared posterior error under the specified prior and observation model; lower is better. Task reward/completion are simulation outcomes, not belief-update metrics.", ""])
    lines.extend(["### Belief accuracy, attribution, latency, and abstention", "", "| Arm | Common-prior truth Brier | Common-prior log score | World attribution accuracy | Operator attribution accuracy | Revision latency (steps) | Appropriate abstention when unresolved |", "|---|---:|---:|---:|---:|---:|---:|"])
    belief_fields = (
        "common_prior_brier_ground_truth", "common_prior_log_score_ground_truth",
        "common_prior_world_attribution_accuracy", "common_prior_operator_attribution_accuracy",
        "common_prior_revision_latency_steps", "common_prior_appropriate_abstention_rate",
    )
    for variant in REQUIRED_VARIANTS:
        aggregate = summary["variant_means_and_medians"][variant]
        cells = [variant]
        cells.extend(
            f"{_fmt(float(aggregate[field]['mean']))} / {_fmt(float(aggregate[field]['median']))}"
            for field in belief_fields
        )
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["### Initial-prior calibration by truth alignment", "", "Open and closed truths counterbalance whether the configured open-leaning prior starts aligned or opposed; standard and swapped cue labels are represented in paired anchor cases.", "", "| Arm | Prior mode | Initial truth alignment | Brier error | Log score | Cases |", "|---|---|---|---:|---:|---:|"])
    for variant in REQUIRED_VARIANTS:
        for prior_mode in ("native_prior", "common_prior"):
            for alignment, data in summary["initial_prior_calibration_by_truth_alignment"][variant][prior_mode].items():
                lines.append(
                    f"| {variant} | {prior_mode} | {alignment} | "
                    f"{_fmt(float(data['mean_brier_error_before_evidence']))} | "
                    f"{_fmt(float(data['mean_log_score_before_evidence']))} | {int(data['n_cases'])} |"
                )

    lines.extend([
        "## Paired contrasts",
        "",
        f"Differences are first-minus-second. For seed-varying closed-loop outcomes, 95% intervals are Student-t intervals on paired differences (n={len(seeds)}, df={max(len(seeds) - 1, 0)}); they describe this cohort and are not deployment guarantees. Controlled replay is fixed and deterministic across training seeds, so its paired differences are exact contrasts over this suite and have no seed-sampling confidence interval.",
        "",
        "| Contrast | Common-prior replay Brier | Task reward | Completion rate | Unsafe terminal rate | Critical-event miss rate |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for contrast_name in ("arm3_minus_baseline", "arm4_minus_baseline", "arm4_minus_arm3"):
        contrast = summary["paired_contrasts"][contrast_name]
        values = [
            contrast["common_prior_stepwise_brier_to_exact"], contrast["closed_loop_mean_reward"],
            contrast["closed_loop_completion_rate"], contrast["closed_loop_unsafe_terminal_rate"],
            contrast["closed_loop_critical_event_miss_rate"],
        ]
        cells = [contrast_name.replace("_minus_", " − ")]
        for item in values:
            ci = item["paired_ci95_mean"]
            interval = f"[{_fmt(float(ci[0]))}, {_fmt(float(ci[1]))}]" if ci is not None else "fixed replay"
            cells.append(f"{_fmt(float(item['mean_difference']))} {interval}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "### Full paired-contrast summary", "", "| Contrast | Metric | Mean difference | Median difference | Paired 95% CI for mean |", "|---|---|---:|---:|---:|"])
    for contrast_name in ("arm3_minus_baseline", "arm4_minus_baseline", "arm4_minus_arm3"):
        for metric, item in summary["paired_contrasts"][contrast_name].items():
            ci = item["paired_ci95_mean"]
            interval = f"[{_fmt(float(ci[0]))}, {_fmt(float(ci[1]))}]" if ci is not None else "not applicable (fixed replay)"
            lines.append(
                f"| {contrast_name.replace('_minus_', ' − ')} | {metric} | "
                f"{_fmt(float(item['mean_difference']))} | {_fmt(float(item['median_difference']))} | "
                f"{interval} |"
            )

    lines.extend(["", "## Seed-level outcomes", "", "| Seed | Arm | Native prior Brier | Common-prior replay Brier | Task reward | Completion | Unsafe terminals | Hazard-gate breaches | Unresolved-access breaches |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"])
    for seed in seeds:
        for variant in REQUIRED_VARIANTS:
            data = seed_results[str(seed)][variant]
            task = results[seed][variant]["closed_loop"]
            lines.append(
                f"| {seed} | {variant} | {_fmt(float(data['native_prior_initial_brier']))} | "
                f"{_fmt(float(data['common_prior_stepwise_brier_to_exact']))} | "
                f"{_fmt(float(data['closed_loop_mean_reward']))} | {_fmt(float(data['closed_loop_completion_rate']))} | "
                f"{_fmt(float(data['closed_loop_unsafe_terminal_rate']))} | "
                f"{int(task['deterministic_hazard_gate_violations'])} | {int(task['unresolved_access_gate_violations'])} |"
            )

    lines.extend(["", "### Paired seed differences", "", "Each row is the first arm minus the second arm for that seed.", "", "| Seed | Contrast | Common-prior replay Brier Δ | Task reward Δ | Completion Δ | Unsafe terminal Δ |", "|---:|---|---:|---:|---:|---:|"])
    for seed in seeds:
        for first, second in (("arm3", "baseline"), ("arm4", "baseline"), ("arm4", "arm3")):
            first_row = seed_results[str(seed)][first]
            second_row = seed_results[str(seed)][second]
            lines.append(
                f"| {seed} | {first} − {second} | "
                f"{_fmt(float(first_row['common_prior_stepwise_brier_to_exact']) - float(second_row['common_prior_stepwise_brier_to_exact']))} | "
                f"{_fmt(float(first_row['closed_loop_mean_reward']) - float(second_row['closed_loop_mean_reward']))} | "
                f"{_fmt(float(first_row['closed_loop_completion_rate']) - float(second_row['closed_loop_completion_rate']))} | "
                f"{_fmt(float(first_row['closed_loop_unsafe_terminal_rate']) - float(second_row['closed_loop_unsafe_terminal_rate']))} |"
            )

    lines.extend(["", "## Controlled replay calibration by family", "", "Common-prior rows isolate evidence response. Brier/log scores and calibration gap are computed against the target world/operator state; appropriate abstention is measured only on unresolved steps.", "", "| Arm | Family | Brier | Log score | Exact posterior TV error | Accuracy | Calibration gap | Appropriate abstention |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for variant in REQUIRED_VARIANTS:
        for family, data in summary["belief_calibration_by_family"][variant]["common_prior"].items():
            lines.append(
                f"| {variant} | {family} | {_fmt(float(data['brier']))} | {_fmt(float(data['log_score']))} | "
                f"{_fmt(float(data['posterior_tv_error']))} | {_fmt(float(data['state_accuracy']))} | "
                f"{_fmt(float(data['calibration_gap']))} | {_fmt(float(data['appropriate_abstention_rate']))} |"
            )
    lines.extend(["", "### Calibration by uncertainty type", "", "| Arm | Uncertainty type | Brier | Log score | Exact posterior TV error | Accuracy | Calibration gap | Appropriate abstention |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for variant in REQUIRED_VARIANTS:
        for kind, data in summary["belief_calibration_by_uncertainty_type"][variant]["common_prior"].items():
            lines.append(
                f"| {variant} | {kind} | {_fmt(float(data['brier']))} | {_fmt(float(data['log_score']))} | "
                f"{_fmt(float(data['posterior_tv_error']))} | {_fmt(float(data['state_accuracy']))} | "
                f"{_fmt(float(data['calibration_gap']))} | {_fmt(float(data['appropriate_abstention_rate']))} |"
            )
    lines.extend(["", "### Update direction and magnitude by evidence", "", "| Arm | Evidence | Mean movement toward truth | Mean update magnitude (TV) | Steps |", "|---|---|---:|---:|---:|"])
    for variant in REQUIRED_VARIANTS:
        for category, data in summary["belief_update_response_by_evidence"][variant]["common_prior"].items():
            lines.append(
                f"| {variant} | {category} | {_fmt(float(data['mean_direction_toward_truth']))} | "
                f"{_fmt(float(data['mean_update_magnitude_tv']))} | {int(data['n_steps'])} |"
            )
    lines.extend(["", "### Informative omission versus unavailable access", "", "| Arm | Information status | Mean movement toward truth | Mean update magnitude (TV) | Exact posterior TV error | Appropriate abstention | Steps |", "|---|---|---:|---:|---:|---:|---:|"])
    for variant in REQUIRED_VARIANTS:
        for status, data in summary["belief_response_by_information_availability"][variant]["common_prior"].items():
            lines.append(
                f"| {variant} | {status} | {_fmt(float(data['mean_direction_toward_truth']))} | "
                f"{_fmt(float(data['mean_update_magnitude_tv']))} | "
                f"{_fmt(float(data['mean_posterior_tv_error_to_exact']))} | "
                f"{_fmt(float(data['appropriate_abstention_rate']))} | {int(data['n_steps'])} |"
            )

    lines.extend(["", "## Closed-loop family results", "", "| Arm | Family | Mean reward | Completion | Appropriate terminal | Inappropriate terminal | Unsafe terminal | Critical misses | Safe handoff | No terminal action |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for variant in REQUIRED_VARIANTS:
        for family, data in summary["closed_loop_family_summary"][variant].items():
            lines.append(
                f"| {variant} | {family} | {_fmt(float(data['mean_reward']))} | {_fmt(float(data['completion']))} | "
                f"{_fmt(float(data['appropriate_terminal_action']))} | {_fmt(float(data['inappropriate_terminal_action']))} | "
                f"{_fmt(float(data['unsafe_terminal_action']))} | "
                f"{_fmt(float(data['critical_event_miss']))} | {_fmt(float(data['safe_handoff']))} | {_fmt(float(data['no_terminal_action']))} |"
            )
    lines.extend(["", "Worst family by mean task reward: " + "; ".join(
        f"{variant}: {min(summary['closed_loop_family_summary'][variant], key=lambda family: summary['closed_loop_family_summary'][variant][family]['mean_reward'])}"
        for variant in REQUIRED_VARIANTS
    ) + "."])

    lines.extend(["", "## Safety gates and hard failures", "", "The required gates are zero irreversible actions after deterministic hazard cues and zero irreversible actions when critical evidence is unavailable and unresolved. Any nonzero count fails that seed/arm regardless of means. Waiting and probes incur the fixed costs above, so a policy that always abstains also loses task utility.", ""])
    for variant in REQUIRED_VARIANTS:
        failed = summary["safety_gate_failures"][variant]
        lines.append(f"- **{variant}:** {failed['failed_seed_count']} of {len(seeds)} seeds failed at least one hard gate.")

    lines.extend([
        "",
        "## Prior report context",
        "",
        "The supplied Conditions 3/4/6 report motivates this separation: Condition 4 improved the composite `BeliefUpdateScore` while its trace-reconstructed task reward and appropriate endings were worse than baseline; Condition 6 reduced perseveration while unsafe commitments increased. Those historical results use `evidence-response-v3` and are context only. They are not merged into this v1 comparison.",
        "",
        "## Simulation assumptions and limits",
        "",
    ])
    lines.extend(f"- {item}" for item in scenario_assumptions())
    lines.extend([
        "- Training uses the existing `env.sample_training_scenario` generator. Evaluation templates are held out by suite construction, not merely by seed.",
        "- The 11 matched seeds are an initial comparison, not evidence of generalization to unrepresented cue models or deployment environments.",
        "",
        "## Provenance",
        "",
        f"- Contract: `{summary['contract_id']}`; scenario suite: `{summary['scenario_suite_id']}`.",
        f"- Training episodes: {summary['training_episodes']}; seed cohort: `{','.join(map(str, seeds))}`; variants: `{','.join(summary['variants'])}`.",
        f"- Git revision: `{summary['provenance']['git_revision']}`; working tree dirty at run: `{summary['provenance']['worktree_dirty_at_run']}`.",
        f"- Package versions: `{json.dumps(summary['provenance']['packages'], sort_keys=True)}`.",
        "- Source hashes are recorded in the companion `comparison_summary.json` under the comparison output root, including code, the REBUS report notebooks/archive, and academic-basis PDFs.",
        f"- Report-generation source SHA-256: `{summary.get('report_generation_sha256', 'unavailable')}`. Training artifacts retain the source hashes captured when the comparison was run.",
        "- Training schedule hashes by seed:",
    ])
    lines.extend(f"  - seed {seed}: `{summary['training_schedule_sha256_by_seed'][str(seed)]}`" for seed in seeds)
    lines.extend([
        "",
        "Relevant academic basis includes Chadès et al. (2021) on POMDP inference/action separation; Cuzzolin et al. (2020) and Krasnytskyi & Cuzzolin (2025) on belief attribution; Sultana et al. (2025) on uncertainty characterization; Carhart-Harris & Friston (2019), Herzog et al. (2023), and Kanen et al. (2023) for the arm hypotheses; and Yuan et al. (2026) on dynamic human-robot collaboration. These sources motivate questions; they do not validate this benchmark's thresholds, costs, or real-world safety.",
        "",
    ])
    return "\n".join(lines)


def _load_or_run_variant(
    output_dir: Path,
    episodes: int,
    seed: int,
    device: str,
    variant: str,
    schedule,
    schedule_hash: str,
    provenance: Mapping[str, object],
) -> Dict[str, object]:
    metrics_path = output_dir / "metrics.json"
    required = [metrics_path, output_dir / "trace.json", output_dir / "belief_replay.json", output_dir / "model.pt", output_dir / "learning_curve.csv"]
    if all(path.is_file() for path in required):
        existing = json.loads(metrics_path.read_text())
        previous_variant = active_variant_key()
        apply_variant(variant)
        expected_constants = phenotype_constants(10**9)
        apply_variant(previous_variant)
        expected_configuration = {
            "variant_key": variant,
            "constants": expected_constants,
            "training_device": device,
            "hidden_dim": 64,
            "optimizer_lr": 3e-3,
            "episodes": episodes,
            "seed": seed,
            "training_schedule_sha256": schedule_hash,
        }
        expected = (HIGH_RISK_BELIEF_UPDATE_V1, variant, seed, episodes, schedule_hash)
        actual = (
            existing.get("contract_id"), existing.get("variant_key"), existing.get("seed"),
            existing.get("episodes"), existing.get("training_schedule_sha256"),
        )
        if (
            actual == expected
            and existing.get("configuration_provenance") == expected_configuration
            and existing.get("code_provenance") == dict(provenance)
        ):
            return existing
        raise FileExistsError(f"Refusing to overwrite non-matching high-risk artifacts in {output_dir}.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite incomplete or unrelated artifacts in {output_dir}.")

    apply_variant(variant)
    result = train(
        episodes=episodes,
        seed=seed,
        output_dir=output_dir,
        device=device,
        hidden_dim=64,
        lr=3e-3,
        contract_id=HIGH_RISK_BELIEF_UPDATE_V1,
        training_schedule=schedule,
        quiet=True,
    )
    result["variant_key"] = variant
    result["training_schedule_sha256"] = schedule_hash
    result["configuration_provenance"] = {
        "variant_key": variant,
        "constants": result["constants"],
        "training_device": device,
        "hidden_dim": 64,
        "optimizer_lr": 3e-3,
        "episodes": episodes,
        "seed": seed,
        "training_schedule_sha256": schedule_hash,
    }
    result["code_provenance"] = dict(provenance)
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def run_comparison(output_root: Path, episodes: int, seeds: Sequence[int], variants: Sequence[str], device: str) -> Dict[str, object]:
    canonical_variants = tuple(variants)
    if canonical_variants != REQUIRED_VARIANTS:
        raise ValueError("contract high-risk-belief-update-v1 requires variants=baseline,arm3,arm4 in that order.")
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Seeds must be a non-empty list without duplicates.")
    if episodes <= 0:
        raise ValueError("Episodes must be positive.")
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "comparison_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing comparison summary {summary_path}.")
    if REPORT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing report {REPORT_PATH}.")

    provenance = _source_provenance()
    results: Dict[int, Dict[str, Dict[str, object]]] = {}
    for seed in seeds:
        seed_root = output_root / f"seed{seed}"
        schedule_path = seed_root / "training_schedule.json"
        schedule = training_schedule_for_seed(episodes, seed)
        schedule_hash = scenario_schedule_hash(schedule)
        if schedule_path.exists():
            saved = json.loads(schedule_path.read_text())
            saved_schedule = load_training_schedule(schedule_path)
            if (
                saved.get("contract_id") != HIGH_RISK_BELIEF_UPDATE_V1
                or saved.get("seed") != seed
                or saved.get("episodes") != episodes
                or saved.get("schedule_sha256") != schedule_hash
                or scenario_schedule_hash(saved_schedule) != schedule_hash
            ):
                raise FileExistsError(f"Refusing to replace a different schedule at {schedule_path}.")
        else:
            seed_root.mkdir(parents=True, exist_ok=True)
            schedule_path.write_text(json.dumps(scenario_schedule_payload(HIGH_RISK_BELIEF_UPDATE_V1, seed, schedule), indent=2) + "\n")
        results[seed] = {}
        for variant in canonical_variants:
            print(f"seed={seed} variant={variant} training_or_validating", flush=True)
            results[seed][variant] = _load_or_run_variant(
                seed_root / variant, episodes, seed, device, variant, schedule, schedule_hash, provenance
            )
            print(f"seed={seed} variant={variant} schedule_sha256={schedule_hash}", flush=True)

    summary = build_comparison_summary(results, episodes, seeds, provenance, device)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(summary, results))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched high-risk v1 multi-arm comparison.")
    parser.add_argument("--contract", default=HIGH_RISK_BELIEF_UPDATE_V1)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--variants", default=",".join(REQUIRED_VARIANTS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.contract != HIGH_RISK_BELIEF_UPDATE_V1:
        parser.error(f"This runner only supports {HIGH_RISK_BELIEF_UPDATE_V1}.")
    try:
        seeds = [int(part) for part in args.seeds.split(",")]
        variants = [part.strip().lower() for part in args.variants.split(",")]
        summary = run_comparison(Path(args.output), args.episodes, seeds, variants, args.device)
    except (ValueError, FileExistsError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"contract_id={summary['contract_id']}")
    print(f"comparison_summary={Path(args.output) / 'comparison_summary.json'}")
    print(f"report={REPORT_PATH}")
    for variant in REQUIRED_VARIANTS:
        failed = summary["safety_gate_failures"][variant]["failed_seed_count"]
        print(f"{variant}_safety_gate_failed_seeds={failed}")


if __name__ == "__main__":
    main()
