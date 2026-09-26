"""Serve a belief-trace page. Keep/discard notes do not change BeliefUpdateScore."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interface.trace_page import make_server  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Show a saved belief trace.")
    parser.add_argument("--trace", type=str, required=True)
    parser.add_argument("--metrics", type=str, default="")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    score = None
    if args.metrics:
        score = float(json.loads(Path(args.metrics).read_text())["BeliefUpdateScore"])
    server = make_server(Path(args.trace), score=score, port=args.port)
    host, port = server.server_address
    print(f"trace_page=http://{host}:{port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
