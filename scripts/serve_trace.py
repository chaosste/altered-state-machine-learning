"""Serve a belief-trace page. Keep/discard notes do not change BeliefUpdateScore."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interface.trace_page import make_server, run_files  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Show a saved belief trace.")
    parser.add_argument("--output-dir", type=str, default="", help="Directory written by train.py")
    parser.add_argument("--trace", type=str, default="")
    parser.add_argument("--metrics", type=str, default="")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    metrics_path = Path(args.metrics) if args.metrics else None
    if args.trace:
        trace_path = Path(args.trace)
        if not trace_path.is_file():
            print(f"No trace at {trace_path}.", file=sys.stderr)
            sys.exit(1)
    elif args.output_dir:
        trace_path, found_metrics, error = run_files(Path(args.output_dir))
        if error or trace_path is None:
            print(error or f"No trace at {args.output_dir}.", file=sys.stderr)
            sys.exit(1)
        if metrics_path is None:
            metrics_path = found_metrics
    else:
        parser.error("Pass --output-dir or --trace.")
    try:
        server = make_server(trace_path, port=args.port, metrics_path=metrics_path)
    except OSError as exc:
        print(f"Port {args.port} is already in use ({exc}).", file=sys.stderr)
        sys.exit(1)
    host, port = server.server_address
    print(f"trace_page=http://{host}:{port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
