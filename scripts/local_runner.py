"""Train the baseline, optionally a candidate train.py, and write the keep/discard note.

The candidate runs in its own process so its constants cannot leak into the baseline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import (  # noqa: E402
    KEEP_MAX_IGNORED_WORSEN,
    KEEP_MAX_PERSEVERATION_WORSEN,
    KEEP_MAX_UNSAFE_WORSEN,
    KEEP_MIN_SCORE_DELTA,
    keep_candidate,
)


def build_selection(
    baseline: dict[str, object],
    candidate: dict[str, object],
    seed: int,
    episodes: int,
) -> dict[str, object]:
    """Explain the frozen keep rule alongside its boolean decision."""
    score_gain = float(candidate["BeliefUpdateScore"]) - float(baseline["BeliefUpdateScore"])
    unsafe_change = float(candidate["unsafe_commit_rate"]) - float(baseline["unsafe_commit_rate"])
    perseveration_change = float(candidate["perseveration"]) - float(baseline["perseveration"])
    ignored_change = float(candidate["ignored_evidence_rate"]) - float(baseline["ignored_evidence_rate"])

    criteria = {
        "minimum_score_gain": {
            "metric": "BeliefUpdateScore",
            "actual_change": score_gain,
            "required_minimum_change": KEEP_MIN_SCORE_DELTA,
            "passed": score_gain + 1e-9 >= KEEP_MIN_SCORE_DELTA,
        },
        "maximum_unsafe_commit_rate_increase": {
            "metric": "unsafe_commit_rate",
            "actual_change": unsafe_change,
            "maximum_allowed_increase": KEEP_MAX_UNSAFE_WORSEN,
            "passed": unsafe_change <= KEEP_MAX_UNSAFE_WORSEN,
        },
        "maximum_perseveration_increase": {
            "metric": "perseveration",
            "actual_change": perseveration_change,
            "maximum_allowed_increase": KEEP_MAX_PERSEVERATION_WORSEN,
            "passed": perseveration_change <= KEEP_MAX_PERSEVERATION_WORSEN,
        },
        "maximum_ignored_evidence_rate_increase": {
            "metric": "ignored_evidence_rate",
            "actual_change": ignored_change,
            "maximum_allowed_increase": KEEP_MAX_IGNORED_WORSEN,
            "passed": ignored_change <= KEEP_MAX_IGNORED_WORSEN,
        },
    }

    rejection_reasons = []
    score_gate = criteria["minimum_score_gain"]
    if not score_gate["passed"]:
        rejection_reasons.append(
            f"BeliefUpdateScore gain {score_gain:.6f} is below the required {KEEP_MIN_SCORE_DELTA:.6f}."
        )
    for key in (
        "maximum_unsafe_commit_rate_increase",
        "maximum_perseveration_increase",
        "maximum_ignored_evidence_rate_increase",
    ):
        gate = criteria[key]
        if not gate["passed"]:
            rejection_reasons.append(
                f"{gate['metric']} increased by {float(gate['actual_change']):.6f}, "
                f"above the allowed {float(gate['maximum_allowed_increase']):.6f}."
            )

    kept = keep_candidate(baseline, candidate)
    if kept != all(bool(gate["passed"]) for gate in criteria.values()):
        raise RuntimeError("Selection explanation disagrees with the frozen keep rule.")

    return {
        "keep": kept,
        "decision": "keep" if kept else "discard",
        "rejection_reasons": rejection_reasons,
        "criteria": criteria,
        "baseline_variant": baseline.get("variant"),
        "candidate_variant": candidate.get("variant"),
        "baseline_evaluation_contract_version": baseline.get("evaluation_contract_version"),
        "candidate_evaluation_contract_version": candidate.get("evaluation_contract_version"),
        "baseline_BeliefUpdateScore": baseline["BeliefUpdateScore"],
        "candidate_BeliefUpdateScore": candidate["BeliefUpdateScore"],
        "score_gain": score_gain,
        "baseline_unsafe_commit_rate": baseline["unsafe_commit_rate"],
        "candidate_unsafe_commit_rate": candidate["unsafe_commit_rate"],
        "baseline_perseveration": baseline["perseveration"],
        "candidate_perseveration": candidate["perseveration"],
        "baseline_ignored_evidence_rate": baseline["ignored_evidence_rate"],
        "candidate_ignored_evidence_rate": candidate["ignored_evidence_rate"],
        "seed": seed,
        "episodes": episodes,
    }


def run_train(
    train_py: Path,
    episodes: int,
    seed: int,
    output_dir: Path,
    device: str,
    variant: str = "baseline",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            sys.executable,
            str(train_py),
            "--episodes",
            str(episodes),
            "--seed",
            str(seed),
            "--output-dir",
            str(output_dir),
            "--device",
            device,
            "--variant",
            variant,
        ],
        cwd=str(ROOT),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Local baseline-vs-candidate gate.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--baseline-train-py", type=str, default=str(ROOT / "train.py"))
    parser.add_argument("--candidate-train-py", type=str, default="")
    parser.add_argument("--candidate-variant", type=str, default="")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    baseline_dir = output_root / "baseline"
    run_train(Path(args.baseline_train_py), args.episodes, args.seed, baseline_dir, args.device, "baseline")
    if not args.candidate_train_py and not args.candidate_variant:
        print(f"baseline_metrics={baseline_dir / 'metrics.json'}")
        return

    candidate_dir = output_root / "candidate"
    candidate_py = Path(args.candidate_train_py) if args.candidate_train_py else Path(args.baseline_train_py)
    run_train(
        candidate_py,
        args.episodes,
        args.seed,
        candidate_dir,
        args.device,
        args.candidate_variant or "baseline",
    )
    baseline = json.loads((baseline_dir / "metrics.json").read_text())
    candidate = json.loads((candidate_dir / "metrics.json").read_text())
    selection = build_selection(baseline, candidate, args.seed, args.episodes)
    selection_dir = output_root / "selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    (selection_dir / "selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    print(f"selection={selection_dir / 'selection.json'}")
    print(f"keep={selection['keep']}")
    for reason in selection["rejection_reasons"]:
        print(f"reason={reason}")


if __name__ == "__main__":
    main()
