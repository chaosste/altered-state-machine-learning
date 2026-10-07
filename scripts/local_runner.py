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

from eval import selection_report  # noqa: E402


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
    selection = selection_report(baseline, candidate, seed=args.seed, episodes=args.episodes)
    selection_dir = output_root / "selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    (selection_dir / "selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    print(f"selection={selection_dir / 'selection.json'}")
    print(f"keep={selection['keep']}")
    for reason in selection["rejection_reasons"]:
        print(f"reason={reason}")


if __name__ == "__main__":
    main()
