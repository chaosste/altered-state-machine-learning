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

from eval import keep_candidate  # noqa: E402


def run_train(train_py: Path, episodes: int, seed: int, output_dir: Path, device: str) -> None:
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
    args = parser.parse_args()

    output_root = Path(args.output_root)
    baseline_dir = output_root / "baseline"
    run_train(Path(args.baseline_train_py), args.episodes, args.seed, baseline_dir, args.device)
    if not args.candidate_train_py:
        print(f"baseline_metrics={baseline_dir / 'metrics.json'}")
        return

    candidate_dir = output_root / "candidate"
    run_train(Path(args.candidate_train_py), args.episodes, args.seed, candidate_dir, args.device)
    baseline = json.loads((baseline_dir / "metrics.json").read_text())
    candidate = json.loads((candidate_dir / "metrics.json").read_text())
    kept = keep_candidate(baseline, candidate)
    selection = {
        "keep": kept,
        "baseline_variant": baseline.get("variant"),
        "candidate_variant": candidate.get("variant"),
        "baseline_BeliefUpdateScore": baseline["BeliefUpdateScore"],
        "candidate_BeliefUpdateScore": candidate["BeliefUpdateScore"],
        "baseline_unsafe_commit_rate": baseline["unsafe_commit_rate"],
        "candidate_unsafe_commit_rate": candidate["unsafe_commit_rate"],
        "baseline_perseveration": baseline["perseveration"],
        "candidate_perseveration": candidate["perseveration"],
        "seed": args.seed,
        "episodes": args.episodes,
    }
    selection_dir = output_root / "selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    (selection_dir / "selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    print(f"selection={selection_dir / 'selection.json'}")
    print(f"keep={kept}")


if __name__ == "__main__":
    main()
