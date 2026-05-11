"""Small checks for train/test benchmark artifact reporting.

This intentionally avoids pytest because the deepgate environment may not have
pytest installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts import benchmark_itc99_gate


class TrainTestReportingTests(unittest.TestCase):
    def test_reporting_helpers_write_provenance_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_path = root / "benchmark_report.json"
            csv_path = root / "benchmark_report.csv"
            manifest_path = root / "run_manifest.json"
            notion_path = root / "notion_result_summary.md"
            report = {
                "created_at": "2026-05-09T00:00:00+00:00",
                "command": ["python", "-m", "scripts.benchmark_itc99_gate", "--model", "m.pt"],
                "run_id": "unit",
                "model": "m.pt",
                "bench": "data/bench/ITC99/b17.bench",
                "fault_list": "data/bench/ITC99/b17_gate_10pct_faults.json",
                "candidate_count": 8,
                "ai_attempts": 2,
                "candidate_seed_base": 20260504,
                "enable_ai_propagation": True,
                "max_backtracks": 5000,
                "total": 2,
                "succeeded": 1,
                "failed": 1,
                "coverage": 0.5,
                "coverage_target": 1.0,
                "passed_coverage_target": False,
                "backtrack_target": True,
                "passed_backtrack_target": True,
                "compare_classic": True,
                "classic_timeout": 30.0,
                "classic_succeeded": 2,
                "ai_backtracks_total": 1,
                "classic_backtracks_total": 3,
                "ai_backtracks_on_success": 1,
                "classic_backtracks_on_ai_success": 3,
                "ai_backtrack_ratio_on_success": 1 / 3,
                "ai_less_backtracks_count": 1,
                "total_time_s": 0.2,
                "classic_time_s": 0.3,
                "baseline_comparison": {
                    "label": "baseline",
                    "source": "docs/checkpoint_compatibility_summary.md",
                    "coverage": 0.25,
                    "observed": 0.5,
                    "delta": 0.25,
                    "decision_comparable": True,
                    "comparison_note": "same configured benchmark scope",
                },
                "artifact_paths": {
                    "json": str(json_path),
                    "csv": str(csv_path),
                    "notion_summary": str(notion_path),
                },
                "per_fault": [
                    {
                        "fault_index": 0,
                        "gate_id": 10,
                        "fault_val": 1,
                        "ok": True,
                        "attempts_used": 1,
                        "ai_backtracks": 1,
                        "classic_ok": True,
                        "classic_backtracks": 3,
                        "ai_less_backtracks": True,
                        "ai_error": None,
                        "time_s": 0.1,
                        "classic_time_s": 0.2,
                    },
                    {
                        "fault_index": 1,
                        "gate_id": 11,
                        "fault_val": 0,
                        "ok": False,
                        "attempts_used": 2,
                        "ai_backtracks": 0,
                        "classic_ok": True,
                        "classic_backtracks": 0,
                        "ai_less_backtracks": None,
                        "ai_error": "no-fallback miss",
                        "time_s": 0.1,
                        "classic_time_s": 0.1,
                    },
                ],
            }
            args = argparse.Namespace(
                model="m.pt",
                fault_list="faults.json",
                device="auto",
                full=False,
                limit_faults=2,
                candidate_count=8,
                ai_attempts=2,
                candidate_seed_base=20260504,
                enable_ai_propagation=True,
                max_backtracks=5000,
                compare_classic=True,
                classic_timeout=30.0,
                coverage_target=1.0,
                backtrack_target=True,
                run_id="unit",
                baseline_label="baseline",
                baseline_coverage=0.25,
                baseline_source="docs/checkpoint_compatibility_summary.md",
            )

            benchmark_itc99_gate._write_json(str(json_path), report)
            benchmark_itc99_gate._write_csv(str(csv_path), report["per_fault"])
            manifest = benchmark_itc99_gate._build_manifest(args, [str(json_path), str(csv_path)])
            benchmark_itc99_gate._write_json(str(manifest_path), manifest)
            benchmark_itc99_gate._write_notion_summary(
                str(notion_path),
                report,
                str(manifest_path),
            )

            payload = json.loads(json_path.read_text())
            self.assertEqual(payload["coverage"], 0.5)
            self.assertFalse(payload["passed_coverage_target"])
            self.assertEqual(payload["baseline_comparison"]["delta"], 0.25)

            with csv_path.open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["ok"], "True")

            manifest_payload = json.loads(manifest_path.read_text())
            self.assertEqual(manifest_payload["baseline"]["coverage"], 0.25)
            self.assertIn("coverage_target", manifest_payload["inputs"])
            self.assertIn("ai_attempts", manifest_payload["inputs"])
            self.assertIn("enable_ai_propagation", manifest_payload["inputs"])

            notion_text = notion_path.read_text()
            self.assertIn("Experiment Log", notion_text)
            self.assertIn("Coverage target", notion_text)
            self.assertIn("Backtrack target", notion_text)


if __name__ == "__main__":
    unittest.main()
