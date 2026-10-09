from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from contracts import EVIDENCE_RESPONSE_V3
from scripts.local_runner import _ensure_contract_metadata


class RunnerContractTests(unittest.TestCase):
    def test_legacy_candidate_outputs_receive_canonical_contract_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            metrics = {"evaluation_contract_version": EVIDENCE_RESPONSE_V3, "BeliefUpdateScore": 0.5}
            (output / "metrics.json").write_text(json.dumps(metrics))
            (output / "trace.json").write_text(json.dumps([{"name": "anchor", "steps": []}]))
            persisted = _ensure_contract_metadata(output)
            trace = json.loads((output / "trace.json").read_text())
            self.assertEqual(persisted["contract_id"], EVIDENCE_RESPONSE_V3)
            self.assertEqual(trace["contract_id"], EVIDENCE_RESPONSE_V3)
            self.assertEqual(trace["traces"][0]["name"], "anchor")


if __name__ == "__main__":
    unittest.main()

