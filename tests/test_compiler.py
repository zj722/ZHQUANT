import json
import unittest
from pathlib import Path

import pandas as pd

from zhquant.compiler import compile_strategy


def make_ohlcv(close_start: float, volume_last: float = 200.0) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    close = pd.Series([close_start + i for i in range(30)], index=dates)
    volume = pd.Series([100.0] * 29 + [volume_last], index=dates)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


class CompilerTest(unittest.TestCase):
    def test_basic_momentum_compiles_to_signal_matrices(self) -> None:
        strategy_path = Path(__file__).parents[1] / "strategies" / "basic_momentum.json"
        strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
        symbols = strategy["universe"]["symbols"] + ["QQQ"]
        market_data = {symbol: make_ohlcv(100.0) for symbol in symbols}

        compiled = compile_strategy(strategy, market_data)

        self.assertEqual(list(compiled.entry.columns), strategy["universe"]["symbols"])
        self.assertEqual(compiled.entry.shape, compiled.exit.shape)
        self.assertTrue(compiled.entry.iloc[-1].all())
        self.assertFalse(compiled.exit.iloc[-1].any())
        self.assertEqual(len(compiled.stateful_exit_rules), 2)
        self.assertIn("QQQ", compiled.data_dependencies)


if __name__ == "__main__":
    unittest.main()

