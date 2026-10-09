"""Evaluate saved v1 policies on a new held-out scenario-template expansion."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from contracts import HIGH_RISK_BELIEF_UPDATE_V1
from high_risk_suite import (
    BASELINE_PRIOR_PRECISION,
    SCENARIO_SUITE_ID as V1_SCENARIO_SUITE_ID,
    evaluate_belief_replay,
    evaluate_closed_loop,
    high_risk_cases,
)
from high_risk_v2_suite import SCENARIO_SUITE_V2_ID, additional_heldout_cases, expanded_heldout_cases
from scripts.high_risk_compare import (
    DEFAULT_SEEDS,
    REQUIRED_VARIANTS,
    build_comparison_summary,
    render_report,
)
from train import BaselinePolicy, apply_variant


V1_OUTPUT = ROOT / "logs" / "high-risk-belief-update-v1"
OUTPUT_ROOT = ROOT / "logs" / "high-risk-belief-update-v1-heldout-v2"
REPORT_PATH = ROOT / "analysis" / "high-risk-belief-update-v1-heldout-v2-report.md"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _provenance() -> Dict[str, object]:
    sources = [
        ROOT / "contracts.py", ROOT / "train.py", ROOT / "env.py", ROOT / "oracle.py",
        ROOT / "high_risk_suite.py", ROOT / "high_risk_v2_suite.py",
        ROOT / "scripts" / "high_risk_heldout_v2.py",
    ]
    research = [
        ROOT / "analysis" / "arm3-arm4-high-risk-v1-report.md",
        ROOT / "logs" / "high-risk-belief-update-v1" / "comparison_summary.json",
    ]
    research.extend(sorted((ROOT / "academic_basis").rglob("*.pdf")))
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
    for package in ("numpy", "torch"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "unavailable"
    return {
        "git_revision": revision,
        "worktree_dirty_at_run": dirty,
        "packages": packages,
        "source_sha256": {str(path.relative_to(ROOT)): _file_sha256(path) for path in sources if path.is_file()},
        "research_source_sha256": {str(path.relative_to(ROOT)): _file_sha256(path) for path in research if path.is_file()},
    }


def _load_policy(seed: int, variant: str, source_metrics: Mapping[str, object], device: str):
    source_dir = V1_OUTPUT / f"seed{seed}" / variant
    checkpoint_path = source_dir / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("contract_id") != HIGH_RISK_BELIEF_UPDATE_V1:
        raise ValueError(f"Unexpected training contract in {checkpoint_path}.")
    if source_metrics.get("contract_id") != HIGH_RISK_BELIEF_UPDATE_V1:
        raise ValueError(f"Unexpected metrics contract at {source_dir / 'metrics.json'}.")
    if source_metrics.get("variant_key") != variant or source_metrics.get("seed") != seed:
        raise ValueError(f"Training metadata mismatch in {source_dir / 'metrics.json'}.")
    apply_variant(variant)
    policy = BaselinePolicy(hidden_dim=int(checkpoint["hidden_dim"]), device=device)
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return policy, _file_sha256(checkpoint_path)


def _metric_projection(metrics: Mapping[str, object], track: str) -> Dict[str, object]:
    projected = dict(metrics)
    if track == "combined":
        projected["belief_replay"] = metrics["belief_replay"]["combined"]
        projected["closed_loop"] = metrics["closed_loop"]["combined"]
    else:
        projected["belief_replay"] = metrics["belief_replay"]["added_templates_only"]
        projected["closed_loop"] = metrics["closed_loop"]["added_templates_only"]
    return projected


def render_heldout_report(
    combined_summary: Mapping[str, object],
    added_summary: Mapping[str, object],
) -> str:
    lines = [
        "# Held-out scenario expansion for High-Risk Belief Update v1",
        "",
        f"Contract: `{HIGH_RISK_BELIEF_UPDATE_V1}`  ",
        f"Expanded scenario suite: `{SCENARIO_SUITE_V2_ID}`  ",
        f"Original suite: `{V1_SCENARIO_SUITE_ID}` with {len(high_risk_cases())} cases; added {len(additional_heldout_cases())} new templates; combined suite has {len(expanded_heldout_cases())} cases.",
        f"Training: 50 episodes per arm on the original v1 schedule; device {combined_summary['device']}. Existing saved baseline, Arm 3, and Arm 4 checkpoints were reused; no retraining was performed.",
        "",
        "This is a post-training held-out-template extension. The new cases alter evidence order, access recovery, operator/world attribution, stale reports, label mappings, and reversal patterns. The domain consequences and cost matrix remain v1 simulation assumptions.",
        "",
        "## Combined original plus added templates",
        "",
        "| Arm | Common replay Brier vs exact | Common replay Brier vs truth | Mean task reward | Completion | Appropriate terminal | Inappropriate terminal | Unsafe terminal | Gate-failing seeds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in REQUIRED_VARIANTS:
        p = combined_summary["variant_means_and_medians"][variant]
        lines.append(
            f"| {variant} | {p['common_prior_stepwise_brier_to_exact']['mean']:.3f} | "
            f"{p['common_prior_brier_ground_truth']['mean']:.3f} | "
            f"{p['closed_loop_mean_reward']['mean']:.3f} | "
            f"{p['closed_loop_completion_rate']['mean']:.3f} | "
            f"{p['closed_loop_appropriate_terminal_rate']['mean']:.3f} | "
            f"{p['closed_loop_inappropriate_terminal_rate']['mean']:.3f} | "
            f"{p['closed_loop_unsafe_terminal_rate']['mean']:.3f} | "
            f"{combined_summary['safety_gate_failures'][variant]['failed_seed_count']}/11 |"
        )
    lines.extend([
        "",
        "## Added templates only",
        "",
        "| Arm | Common replay Brier vs exact | Common replay Brier vs truth | Mean task reward | Completion | Appropriate terminal | Inappropriate terminal | Unsafe terminal | Gate-failing seeds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for variant in REQUIRED_VARIANTS:
        p = added_summary["variant_means_and_medians"][variant]
        lines.append(
            f"| {variant} | {p['common_prior_stepwise_brier_to_exact']['mean']:.3f} | "
            f"{p['common_prior_brier_ground_truth']['mean']:.3f} | "
            f"{p['closed_loop_mean_reward']['mean']:.3f} | "
            f"{p['closed_loop_completion_rate']['mean']:.3f} | "
            f"{p['closed_loop_appropriate_terminal_rate']['mean']:.3f} | "
            f"{p['closed_loop_inappropriate_terminal_rate']['mean']:.3f} | "
            f"{p['closed_loop_unsafe_terminal_rate']['mean']:.3f} | "
            f"{added_summary['safety_gate_failures'][variant]['failed_seed_count']}/11 |"
        )
    lines.extend([
        "",
        "## Added-template family outcomes",
        "",
        "| Arm | Family | Mean reward | Completion | Appropriate terminal | Inappropriate terminal | Unsafe terminal | Critical misses | Safe handoff | No terminal action |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for variant in REQUIRED_VARIANTS:
        for family, row in added_summary["closed_loop_family_summary"][variant].items():
            lines.append(
                f"| {variant} | {family} | {row['mean_reward']:.3f} | {row['completion']:.3f} | "
                f"{row['appropriate_terminal_action']:.3f} | {row['inappropriate_terminal_action']:.3f} | "
                f"{row['unsafe_terminal_action']:.3f} | {row['critical_event_miss']:.3f} | "
                f"{row['safe_handoff']:.3f} | {row['no_terminal_action']:.3f} |"
            )
    lines.extend([
        "",
        "## Paired differences on added templates",
        "",
        "Intervals use the 11 paired saved training seeds. Replay contrasts are deterministic on the fixed scripted histories and are reported as exact differences without a seed confidence interval.",
        "",
        "| Contrast | Outcome | Mean difference | Median difference | Paired 95% CI |",
        "|---|---|---:|---:|---:|",
    ])
    for contrast in ("arm3_minus_baseline", "arm4_minus_baseline", "arm4_minus_arm3"):
        for metric in (
            "common_prior_stepwise_brier_to_exact",
            "closed_loop_mean_reward",
            "closed_loop_completion_rate",
            "closed_loop_appropriate_terminal_rate",
            "closed_loop_inappropriate_terminal_rate",
            "closed_loop_unsafe_terminal_rate",
            "closed_loop_critical_event_miss_rate",
        ):
            row = added_summary["paired_contrasts"][contrast][metric]
            ci = row["paired_ci95_mean"]
            interval = "fixed replay" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            lines.append(
                f"| {contrast.replace('_minus_', ' − ')} | {metric} | "
                f"{row['mean_difference']:.3f} | {row['median_difference']:.3f} | {interval} |"
            )
    lines.extend([
        "",
        "Paired task-outcome intervals use the 11 existing training seeds and describe model-to-model variability over this fixed added-template set. Controlled replay is deterministic and has no seed-sampling confidence interval. Any hard-gate breach remains a failure independent of average reward.",
        "",
        "## Worst added-template family",
        "",
        "| Arm | Lowest-reward family | Mean reward | Unsafe terminal rate | Critical miss rate |",
        "|---|---|---:|---:|---:|",
    ])
    for variant in REQUIRED_VARIANTS:
        families = added_summary["closed_loop_family_summary"][variant]
        worst = min(families, key=lambda name: families[name]["mean_reward"])
        row = families[worst]
        lines.append(
            f"| {variant} | {worst} | {row['mean_reward']:.3f} | "
            f"{row['unsafe_terminal_action']:.3f} | {row['critical_event_miss']:.3f} |"
        )
    lines.extend([
        "",
        "## Provenance and limits",
        "",
        f"- Original v1 comparison root: `logs/high-risk-belief-update-v1/`; schedule hashes and training metrics remain unchanged.",
        f"- New outputs: `logs/high-risk-belief-update-v1-heldout-v2/`; canonical contract ID `{HIGH_RISK_BELIEF_UPDATE_V1}` and scenario suite ID `{SCENARIO_SUITE_V2_ID}`.",
        f"- Source/package provenance is recorded in the new comparison summary. The v1 trained models were evaluated as saved.",
        "- Robot, route, communication, source-reliability, safety, and reward mappings are synthetic assumptions. Results do not validate real systems.",
        "- This expands scenario-template coverage but retains the original 11 trained seed models; it is not an independent retraining cohort.",
        "",
    ])
    return "\n".join(lines)


def run_expansion(output_root: Path = OUTPUT_ROOT, seeds: Sequence[int] = DEFAULT_SEEDS) -> Dict[str, object]:
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing held-out v2 outputs in {output_root}.")
    if REPORT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing report {REPORT_PATH}.")
    if tuple(seeds) != tuple(DEFAULT_SEEDS):
        raise ValueError("This expansion is paired to the existing 11-seed cohort.")
    if not V1_OUTPUT.is_dir():
        raise FileNotFoundError(V1_OUTPUT)

    devices = {
        json.loads((V1_OUTPUT / f"seed{seed}" / variant / "metrics.json").read_text())["training_device"]
        for seed in seeds
        for variant in REQUIRED_VARIANTS
    }
    if len(devices) != 1:
        raise ValueError(f"The held-out expansion requires one matched training device, found {sorted(devices)}.")
    evaluation_device = devices.pop()

    output_root.mkdir(parents=True, exist_ok=True)
    provenance = _provenance()
    base_cases = high_risk_cases()
    added_cases = additional_heldout_cases()
    all_cases = [*base_cases, *added_cases]
    combined_results: Dict[int, Dict[str, Dict[str, object]]] = {seed: {} for seed in seeds}
    added_results: Dict[int, Dict[str, Dict[str, object]]] = {seed: {} for seed in seeds}

    for seed in seeds:
        for variant in REQUIRED_VARIANTS:
            source_dir = V1_OUTPUT / f"seed{seed}" / variant
            source_metrics = json.loads((source_dir / "metrics.json").read_text())
            device = str(source_metrics["training_device"])
            policy, checkpoint_sha = _load_policy(seed, variant, source_metrics, device)
            print(f"seed={seed} variant={variant} evaluating={SCENARIO_SUITE_V2_ID}", flush=True)
            with torch.no_grad():
                combined_replay, combined_replay_trace = evaluate_belief_replay(
                    variant, source_metrics["constants"], BASELINE_PRIOR_PRECISION,
                    cases=all_cases, scenario_suite_id=SCENARIO_SUITE_V2_ID,
                )
                added_replay, added_replay_trace = evaluate_belief_replay(
                    variant, source_metrics["constants"], BASELINE_PRIOR_PRECISION,
                    cases=added_cases, scenario_suite_id=SCENARIO_SUITE_V2_ID,
                )
                combined_task, combined_trace = evaluate_closed_loop(
                    policy, cases=all_cases, scenario_suite_id=SCENARIO_SUITE_V2_ID,
                )
                added_task, added_trace = evaluate_closed_loop(
                    policy, cases=added_cases, scenario_suite_id=SCENARIO_SUITE_V2_ID,
                )

            run_dir = output_root / f"seed{seed}" / variant
            run_dir.mkdir(parents=True, exist_ok=True)
            common = {
                "contract_id": HIGH_RISK_BELIEF_UPDATE_V1,
                "scenario_suite_id": SCENARIO_SUITE_V2_ID,
                "training_contract_id": HIGH_RISK_BELIEF_UPDATE_V1,
                "training_scenario_suite_id": V1_SCENARIO_SUITE_ID,
                "variant": source_metrics["variant"],
                "variant_key": variant,
                "seed": seed,
                "episodes": 50,
                "training_device": device,
                "evaluation_device": device,
                "training_schedule_sha256": source_metrics["training_schedule_sha256"],
                "constants": source_metrics["constants"],
                "training_checkpoint_sha256": checkpoint_sha,
                "training_metrics_path": str((source_dir / "metrics.json").relative_to(ROOT)),
                "n_original_templates": len(base_cases),
                "n_added_templates": len(added_cases),
                "n_combined_templates": len(all_cases),
                "belief_replay": {"combined": combined_replay, "added_templates_only": added_replay},
                "closed_loop": {"combined": combined_task, "added_templates_only": added_task},
                "configuration_provenance": source_metrics["configuration_provenance"],
                "code_provenance": provenance,
            }
            metrics_path = run_dir / "metrics.json"
            metrics_path.write_text(json.dumps(common, indent=2, sort_keys=True) + "\n")
            (run_dir / "trace.json").write_text(json.dumps({
                "contract_id": HIGH_RISK_BELIEF_UPDATE_V1,
                "scenario_suite_id": SCENARIO_SUITE_V2_ID,
                "traces": combined_trace,
            }, indent=2) + "\n")
            (run_dir / "added_templates_trace.json").write_text(json.dumps({
                "contract_id": HIGH_RISK_BELIEF_UPDATE_V1,
                "scenario_suite_id": SCENARIO_SUITE_V2_ID,
                "traces": added_trace,
            }, indent=2) + "\n")
            (run_dir / "belief_replay.json").write_text(json.dumps({
                "contract_id": HIGH_RISK_BELIEF_UPDATE_V1,
                "scenario_suite_id": SCENARIO_SUITE_V2_ID,
                "combined": combined_replay_trace,
                "added_templates_only": added_replay_trace,
            }, indent=2) + "\n")

            combined_results[seed][variant] = {**common, "training_schedule_sha256": source_metrics["training_schedule_sha256"]}
            combined_results[seed][variant]["belief_replay"] = combined_replay
            combined_results[seed][variant]["closed_loop"] = combined_task
            added_results[seed][variant] = {**common, "training_schedule_sha256": source_metrics["training_schedule_sha256"]}
            added_results[seed][variant]["belief_replay"] = added_replay
            added_results[seed][variant]["closed_loop"] = added_task

    combined_summary = build_comparison_summary(combined_results, 50, seeds, provenance, evaluation_device)
    combined_summary.update({
        "contract_id": HIGH_RISK_BELIEF_UPDATE_V1,
        "scenario_suite_id": SCENARIO_SUITE_V2_ID,
        "training_scenario_suite_id": V1_SCENARIO_SUITE_ID,
        "retrained": False,
        "n_original_templates": len(base_cases),
        "n_added_templates": len(added_cases),
        "n_combined_templates": len(all_cases),
        "training_episodes_per_variant": 50,
    })
    added_summary = build_comparison_summary(added_results, 50, seeds, provenance, evaluation_device)
    added_summary.update({
        "contract_id": HIGH_RISK_BELIEF_UPDATE_V1,
        "scenario_suite_id": f"{SCENARIO_SUITE_V2_ID}-added-only",
        "training_scenario_suite_id": V1_SCENARIO_SUITE_ID,
        "retrained": False,
        "n_added_templates": len(added_cases),
        "training_episodes_per_variant": 50,
    })
    final_summary = {
        **combined_summary,
        "added_templates_only": added_summary,
    }
    (output_root / "comparison_summary.json").write_text(json.dumps(final_summary, indent=2, sort_keys=True) + "\n")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_heldout_report(combined_summary, added_summary))
    return final_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate v1 trained models on an expanded held-out template suite.")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    try:
        summary = run_expansion(args.output)
    except (FileExistsError, FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"contract_id={summary['contract_id']}")
    print(f"scenario_suite_id={summary['scenario_suite_id']}")
    print(f"comparison_summary={args.output / 'comparison_summary.json'}")
    print(f"report={REPORT_PATH}")


if __name__ == "__main__":
    main()
