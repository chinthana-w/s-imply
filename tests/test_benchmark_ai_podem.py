import unittest
from unittest.mock import patch

from src.atpg.benchmark_ai_podem import run_benchmark
from src.atpg.podem import BACKTRACK_LIMIT
from src.util.struct import Fault, LogicValue


class TestBenchmarkAiPodem(unittest.TestCase):
    @patch("src.atpg.benchmark_ai_podem.get_statistics")
    @patch("src.atpg.benchmark_ai_podem.podem")
    @patch("src.atpg.benchmark_ai_podem.initialize")
    @patch("src.atpg.benchmark_ai_podem.reset_gates")
    @patch("src.atpg.benchmark_ai_podem.get_all_faults")
    @patch("src.atpg.benchmark_ai_podem.parse_bench_file")
    def test_vanilla_benchmark_counts_only_success_status(
        self,
        mock_parse_bench_file,
        mock_get_all_faults,
        mock_reset_gates,
        mock_initialize,
        mock_podem,
        mock_get_statistics,
    ):
        """Benchmark coverage must not count truthy failure status codes as detections."""
        mock_parse_bench_file.return_value = ([None], 0)
        mock_get_all_faults.return_value = [Fault(1, LogicValue.D)]
        mock_podem.return_value = BACKTRACK_LIMIT
        mock_get_statistics.return_value = {
            "backtrack_count": 0,
            "backtrace_count": 0,
        }

        result = run_benchmark("dummy.bench", "dummy_model.pt", mode="vanilla")

        self.assertEqual(result["faults"], 1)
        self.assertEqual(result["detected"], 0)
        self.assertEqual(result["coverage"], 0)
        mock_reset_gates.assert_called_once()
        mock_initialize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
