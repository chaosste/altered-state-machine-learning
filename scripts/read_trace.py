"""Write cue stance, polarity, and richness beside a finished trace.

The output is a report. It does not rewrite metrics.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interface.trace_page import load_traces
from readings import read_traces


def write_readings(trace_path: Path, output_path: Path) -> Path:
    if output_path.name == "metrics.json":
        raise ValueError("readings must not replace metrics.json")
    report = read_traces(load_traces(trace_path))
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Report cue stance, polarity, and richness for a trace.")
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.trace.with_suffix(".readings.json")
    path = write_readings(args.trace, output)
    print(f"readings={path}", flush=True)


if __name__ == "__main__":
    main()
