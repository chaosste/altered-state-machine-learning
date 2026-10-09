"""The terminal menu prints a banner and dispatches slash commands."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rebus import banner, dispatch, format_metrics_list, menu, present_command_output, project_python


class TuiTests(unittest.TestCase):
    def test_banner_names_asml_and_fits_the_terminal(self) -> None:
        art = banner()
        self.assertIn("ALTERED STATE MACHINE LEARNING", art)
        for line in art.splitlines():
            self.assertLessEqual(len(line), 80)

    def test_menu_lists_the_slash_commands(self) -> None:
        text = menu()
        for command in ("/train", "/compare", "/trace", "/readings", "/score", "/test", "/help", "/quit", "/contracts"):
            self.assertIn(command, text)
        self.assertIn("1", text)
        self.assertIn("8", text)

    def test_help_and_unknown_commands_show_the_menu(self) -> None:
        help_text = dispatch("/help").text
        self.assertIn("/readings", help_text)
        self.assertFalse(dispatch("/help").quit)
        unknown = dispatch("/nope")
        self.assertIn("/train", unknown.text)
        self.assertFalse(unknown.quit)
        numbered = dispatch("7")
        self.assertEqual(numbered.text, help_text)

    def test_quit_leaves_without_text(self) -> None:
        outcome = dispatch("/quit")
        self.assertTrue(outcome.quit)
        self.assertEqual(outcome.text, "")

    def test_score_is_read_only(self) -> None:
        with TemporaryDirectory() as tmp:
            metrics = Path(tmp) / "metrics.json"
            metrics.write_text(json.dumps({"BeliefUpdateScore": 0.5, "revision_accuracy": 0.25}) + "\n")
            before = metrics.read_bytes()
            outcome = dispatch(f"/score metrics={metrics}", ask=None)
            self.assertIn("- BeliefUpdateScore: 0.5", outcome.text)
            self.assertIn("- revision_accuracy: 0.25", outcome.text)
            self.assertEqual(metrics.read_bytes(), before)

    def test_readings_leave_metrics_untouched(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "trace.json"
            metrics = root / "metrics.json"
            trace.write_text(
                json.dumps(
                    [
                        {
                            "name": "anchor_open",
                            "steps": [
                                {
                                    "cue": "go",
                                    "masked": False,
                                    "belief_before": [0.6, 0.4],
                                    "belief_after": [0.7, 0.3],
                                    "epistemic_entropy": 0.4,
                                }
                            ],
                        }
                    ]
                )
            )
            metrics.write_text(json.dumps({"BeliefUpdateScore": 0.5}) + "\n")
            before = metrics.read_bytes()
            outcome = dispatch(f"/readings trace={trace}", ask=None)
            self.assertIn("Done.", outcome.text)
            self.assertEqual(metrics.read_bytes(), before)
            report = json.loads(trace.with_suffix(".readings.json").read_text())
            self.assertEqual(report["scenarios"][0]["cue_stance"], ["approach"])
            self.assertFalse(report["changes_belief_update_score"])

    def test_train_command_is_a_subprocess_spec(self) -> None:
        seen = []

        def runner(cmd):
            seen.append(cmd)
            return 0

        outcome = dispatch("/train episodes=50 seed=7 output=logs/baseline-seed7", ask=None, runner=runner)
        self.assertEqual(outcome.text, "Done.")
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0], project_python())
        self.assertTrue(seen[0][0].endswith(".venv/bin/python"))
        self.assertTrue(seen[0][1].endswith("train.py"))
        self.assertIn("--episodes", seen[0])
        self.assertIn("50", seen[0])
        self.assertNotIn("--candidate-train-py", seen[0])

    def test_compare_arm4_selects_the_rebus_variant(self) -> None:
        seen = []

        def runner(cmd):
            seen.append(cmd)
            return 0

        outcome = dispatch(
            "/compare episodes=50 seed=7 output=logs/seed7-arm4 candidate=rebus",
            ask=None,
            runner=runner,
        )
        self.assertEqual(outcome.text, "Done.")
        self.assertIn("--candidate-variant", seen[0])
        self.assertEqual(seen[0][seen[0].index("--candidate-variant") + 1], "arm4")
        self.assertNotIn("--candidate-train-py", seen[0])
        self.assertIn("Prior precision 0.5", dispatch("/arms").text)

    def test_contract_listing_and_matched_v1_compare_dispatch(self) -> None:
        self.assertIn("evidence-response-v3", dispatch("/contracts").text)
        self.assertIn("high-risk-belief-update-v1", dispatch("/contracts").text)
        seen = []

        def runner(cmd):
            seen.append(cmd)
            return 0

        outcome = dispatch(
            "/compare contract=high-risk-belief-update-v1 variants=baseline,arm3,arm4 episodes=50 "
            "seeds=7,11 output=logs/high-risk-belief-update-v1",
            ask=None,
            runner=runner,
        )
        self.assertEqual(outcome.text, "Done.")
        self.assertTrue(seen[0][1].endswith("high_risk_compare.py"))
        self.assertIn("--contract", seen[0])
        self.assertEqual(seen[0][seen[0].index("--variants") + 1], "baseline,arm3,arm4")
        self.assertEqual(seen[0][seen[0].index("--seeds") + 1], "7,11")

    def test_compare_rejects_both_selector_forms_and_contract_mismatch(self) -> None:
        called = []
        outcome = dispatch(
            "/compare candidate=arm4 variants=baseline,arm3,arm4",
            ask=None,
            runner=lambda cmd: called.append(cmd) or 0,
        )
        self.assertIn("do not provide both", outcome.text)
        self.assertEqual(called, [])
        outcome = dispatch(
            "/compare variants=baseline,arm3,arm4",
            ask=None,
            runner=lambda cmd: called.append(cmd) or 0,
        )
        self.assertIn("reserved", outcome.text)
        with TemporaryDirectory() as tmp:
            metrics = Path(tmp) / "metrics.json"
            metrics.write_text(json.dumps({"contract_id": "evidence-response-v3", "BeliefUpdateScore": 0.5}))
            outcome = dispatch(
                f"/score metrics={metrics} contract=high-risk-belief-update-v1",
                ask=None,
            )
            self.assertIn("does not match artifact contract evidence-response-v3", outcome.text)

    def test_trace_rejects_an_explicit_mismatched_contract(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "trace.json").write_text(json.dumps({"contract_id": "evidence-response-v3", "traces": []}))
            (root / "metrics.json").write_text(json.dumps({"contract_id": "evidence-response-v3", "BeliefUpdateScore": 0.5}))
            outcome = dispatch(
                f"/trace output={root} contract=high-risk-belief-update-v1",
                ask=None,
                runner=lambda _cmd: 0,
            )
            self.assertIn("does not match artifact contract evidence-response-v3", outcome.text)

    def test_trace_uses_the_train_output_directory(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "trace.json").write_text("[]")
            seen = []

            def runner(cmd):
                seen.append(cmd)
                return 0

            outcome = dispatch(f"/trace output={root}", ask=None, runner=runner)
            self.assertEqual(outcome.text, "Done.")
            self.assertIn("--output-dir", seen[0])
            self.assertEqual(seen[0][seen[0].index("--output-dir") + 1], str(root))
            self.assertNotIn("--trace", seen[0])

    def test_eval_metrics_print_as_a_list(self) -> None:
        raw = (
            'eval_metrics={"BeliefUpdateScore": 0.626, "revision_accuracy": 0.449, '
            '"constants": {"pe_gain": 0.35, "reward_lr": 0.15}}\n'
            "saved_checkpoint=logs/baseline-seed7/model.pt\n"
        )
        text = present_command_output(raw)
        self.assertNotIn("eval_metrics=", text)
        self.assertIn("- BeliefUpdateScore: 0.626", text)
        self.assertIn("- revision_accuracy: 0.449", text)
        self.assertIn("- constants", text)
        self.assertIn("  - pe_gain: 0.35", text)
        self.assertIn("saved_checkpoint=logs/baseline-seed7/model.pt", text)
        self.assertGreater(text.count("\n"), 3)
        listed = format_metrics_list({"BeliefUpdateScore": 0.5, "seed": 7})
        self.assertTrue(listed.startswith("- BeliefUpdateScore: 0.5"))
        self.assertIn("- seed: 7", listed)


if __name__ == "__main__":
    unittest.main()
