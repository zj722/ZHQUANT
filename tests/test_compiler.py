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

    def test_shifted_rolling_max_supports_prior_high_breakout(self) -> None:
        strategy = {
            "name": "breakout_shift_test",
            "timeframe": "1d",
            "universe": {"type": "static_list", "symbols": ["AAA"]},
            "entry": {
                "op": ">",
                "left": {"indicator": "close"},
                "right": {"indicator": "rolling_max", "source": "high", "window": 5, "shift": 1},
            },
            "exit": {
                "op": ">=",
                "left": {"field": "holding_days"},
                "right": 5,
            },
            "risk": {
                "max_position_pct": 1.0,
                "max_positions": 1,
                "execution": "next_open",
                "slippage_bps": 0,
                "commission_bps": 0,
            },
        }
        market = make_ohlcv(10.0)
        market.loc[market.index[-1], "close"] = market["high"].iloc[-6:-1].max() + 1
        market.loc[market.index[-1], "high"] = market.loc[market.index[-1], "close"] + 0.5

        compiled = compile_strategy(strategy, {"AAA": market})

        self.assertTrue(compiled.entry.iloc[-1]["AAA"])

    def test_rolling_count_counts_boolean_conditions(self) -> None:
        strategy = {
            "name": "distribution_count_test",
            "timeframe": "1d",
            "universe": {"type": "static_list", "symbols": ["AAA"]},
            "entry": {
                "op": ">=",
                "left": {
                    "indicator": "rolling_count",
                    "window": 5,
                    "condition": {
                        "op": "<",
                        "left": {"indicator": "close"},
                        "right": {"indicator": "close", "shift": 1},
                    },
                },
                "right": 3,
            },
            "exit": {
                "op": ">=",
                "left": {"field": "holding_days"},
                "right": 5,
            },
            "risk": {
                "max_position_pct": 1.0,
                "max_positions": 1,
                "execution": "next_open",
                "slippage_bps": 0,
                "commission_bps": 0,
            },
        }
        market_data = {"AAA": make_ohlcv(30.0)}
        market_data["AAA"]["close"] = [30, 31, 30, 29, 30, 29, 28, 27, 28, 29] + list(range(30, 50))

        compiled = compile_strategy(strategy, market_data)

        self.assertTrue(compiled.entry.iloc[7]["AAA"])

    def test_named_diagnostics_compile_to_signal_matrices(self) -> None:
        strategy = {
            "name": "diagnostics_test",
            "timeframe": "1d",
            "universe": {"type": "static_list", "symbols": ["AAA"]},
            "entry": {
                "op": ">",
                "left": {"indicator": "close"},
                "right": {"indicator": "sma", "source": "close", "window": 2},
            },
            "exit": {
                "op": ">=",
                "left": {"field": "holding_days"},
                "right": 5,
            },
            "diagnostics": {
                "stock_trend_ok": {
                    "op": ">",
                    "left": {"indicator": "close"},
                    "right": {"indicator": "sma", "source": "close", "window": 2},
                }
            },
            "risk": {
                "max_position_pct": 1.0,
                "max_positions": 1,
                "execution": "next_open",
                "slippage_bps": 0,
                "commission_bps": 0,
            },
        }
        market = make_ohlcv(10.0)
        market.loc[market.index[-1], "close"] = 80.0
        market_data = {"AAA": market}

        compiled = compile_strategy(strategy, market_data)

        self.assertIn("stock_trend_ok", compiled.diagnostics)
        self.assertEqual(compiled.diagnostics["stock_trend_ok"].shape, compiled.entry.shape)

    def test_rolling_quantile_compiles_return_threshold(self) -> None:
        strategy = {
            "name": "rolling_quantile_test",
            "timeframe": "1d",
            "universe": {"type": "static_list", "symbols": ["AAA"]},
            "entry": {
                "op": ">",
                "left": {"indicator": "return", "source": "close", "window": 5},
                "right": {
                    "indicator": "rolling_quantile",
                    "source": "close",
                    "return_window": 5,
                    "window": 10,
                    "quantile": 0.9,
                    "shift": 1,
                },
            },
            "exit": {
                "op": ">=",
                "left": {"field": "holding_days"},
                "right": 5,
            },
            "risk": {
                "max_position_pct": 1.0,
                "max_positions": 1,
                "execution": "next_open",
                "slippage_bps": 0,
                "commission_bps": 0,
            },
        }
        market = make_ohlcv(10.0)
        market.loc[market.index[-1], "close"] = 80.0
        market_data = {"AAA": market}

        compiled = compile_strategy(strategy, market_data)

        self.assertEqual(compiled.entry.shape, (30, 1))
        self.assertTrue(compiled.entry.iloc[-1]["AAA"])


if __name__ == "__main__":
    unittest.main()
