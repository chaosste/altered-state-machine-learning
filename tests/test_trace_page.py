"""The trace page records a human mark and leaves the score file alone."""

from __future__ import annotations

import json
import threading
import unittest
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory

from interface.trace_page import (
    append_decision,
    decisions_path,
    load_traces,
    make_server,
    render_trace_page,
    run_files,
)


class TracePageTests(unittest.TestCase):
    def test_mark_does_not_rewrite_metrics(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "trace.json"
            metrics = root / "metrics.json"
            trace.write_text(
                json.dumps(
                    [
                        {
                            "name": "anchor_open",
                            "blurb": "Stationary open partner.",
                            "outcome": "correct",
                            "steps": [
                                {
                                    "cue": "go",
                                    "action": "wait",
                                    "requery_action": "wait",
                                    "belief_after": [0.7, 0.3],
                                    "exact_belief": [0.88, 0.12],
                                    "epistemic_entropy": 0.61,
                                    "aleatoric": 0.25,
                                    "label": "open",
                                    "gate_would_block": False,
                                }
                            ],
                        }
                    ]
                )
            )
            metrics.write_text(json.dumps({"BeliefUpdateScore": 0.5}) + "\n")
            before = metrics.read_text()
            page = render_trace_page(load_traces(trace), score=0.5)
            self.assertIn("anchor_open", page)
            self.assertIn("- BeliefUpdateScore: 0.5", page)
            self.assertIn("BeliefUpdateScore is the only keep/discard metric", page)
            self.assertIn("does not change the score", page)
            self.assertIn("stance approach", page)
            self.assertIn("Cue polarity +1", page)
            self.assertIn("richness in band", page)
            self.assertIn("The score ignores them", page)
            append_decision(trace, "discard", "perseveration rose", "anchor_open")
            self.assertEqual(metrics.read_text(), before)
            saved = decisions_path(trace).read_text()
            self.assertIn("discard", saved)
            self.assertNotIn("BeliefUpdateScore", saved)

    def test_http_mark_round_trip(self) -> None:
        with TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.json"
            trace.write_text(json.dumps([{"name": "handoff_closed", "blurb": "", "outcome": "incorrect", "steps": []}]))
            server = make_server(trace, score=None, port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            host, port = server.server_address
            try:
                page = urllib.request.urlopen(f"http://{host}:{port}/").read().decode()
                self.assertIn("handoff_closed", page)
                request = urllib.request.Request(
                    f"http://{host}:{port}/decision",
                    data=json.dumps({"mark": "keep", "note": "worth another seed", "scenario": "handoff_closed"}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                payload = json.loads(urllib.request.urlopen(request).read().decode())
                self.assertTrue(payload["ok"])
                self.assertTrue(payload["score_unchanged"])
                self.assertIn("keep", decisions_path(trace).read_text())
            finally:
                server.shutdown()
                server.server_close()

    def test_page_rereads_metrics_and_compare_root_names_each_side(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "trace.json"
            metrics = root / "metrics.json"
            trace.write_text(json.dumps([{"name": "anchor_open", "blurb": "", "outcome": "", "steps": []}]))
            metrics.write_text(json.dumps({"BeliefUpdateScore": 0.5, "variant": "baseline"}))
            server = make_server(trace, port=0, metrics_path=metrics)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            host, port = server.server_address
            try:
                first = urllib.request.urlopen(f"http://{host}:{port}/").read().decode()
                self.assertIn("- BeliefUpdateScore: 0.5", first)
                self.assertIn("- variant: baseline", first)
                metrics.write_text(json.dumps({"BeliefUpdateScore": 0.8, "variant": "arm4_rebus"}))
                second = urllib.request.urlopen(f"http://{host}:{port}/").read().decode()
                self.assertIn("- BeliefUpdateScore: 0.8", second)
                self.assertIn("- variant: arm4_rebus", second)
            finally:
                server.shutdown()
                server.server_close()
            compare = root / "compare"
            (compare / "baseline").mkdir(parents=True)
            (compare / "candidate").mkdir()
            (compare / "baseline" / "trace.json").write_text("[]")
            (compare / "candidate" / "trace.json").write_text("[]")
            found, found_metrics, error = run_files(compare)
            self.assertIsNone(found)
            self.assertIsNone(found_metrics)
            self.assertIn(f"output={compare / 'baseline'}", error)
            self.assertIn(f"output={compare / 'candidate'}", error)


if __name__ == "__main__":
    unittest.main()
