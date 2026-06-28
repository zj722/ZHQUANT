import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from zhquant.strategy_batch import run_strategy_directory


def make_market(close_values: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(close_values), freq="D")
    close = pd.Series(close_values, index=dates, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000.0,
        },
        index=dates,
    )


class StrategyBatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.strategy_dir = self.tmp / "strategies"
        self.strategy_dir.mkdir()
        root = Path(__file__).parents[1]
        shutil.copy(root / "strategies" / "simple_sma_pullback.json", self.strategy_dir / "sma_a.json")
        shutil.copy(root / "strategies" / "simple_sma_pullback.json", self.strategy_dir / "sma_b.json")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_run_strategy_directory_writes_summary_and_details(self) -> None:
        market_data = {
            "AAA": make_market([10, 11, 12, 13, 14, 15, 14, 13, 14, 15, 16, 17]),
            "BBB": make_market([20, 19, 20, 21, 22, 23, 22, 21, 22, 23, 24, 25]),
        }

        run = run_strategy_directory(
            strategy_path=self.strategy_dir,
            tickers=["AAA", "BBB"],
            market_data=market_data,
            initial_cash=10_000,
            output_root=self.tmp / "runs",
            period_label="test",
        )

        self.assertEqual(len(run.summary), 2)
        self.assertTrue((run.output_dir / "summary.csv").exists())
        self.assertTrue((run.output_dir / "details.json").exists())
        self.assertIn("avg_score", run.summary.columns)
        self.assertIn("strategies", run.details)


if __name__ == "__main__":
    unittest.main()

