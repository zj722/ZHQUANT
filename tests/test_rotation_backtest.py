import unittest

import numpy as np
import pandas as pd

from zhquant.dsl_validator import StrategyDSLValidator
from zhquant.rotation_backtest import run_rotation_backtest


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


class RotationBacktestTest(unittest.TestCase):
    def test_rotation_strategy_validates(self) -> None:
        strategy = {
            "name": "rotation_validate_test",
            "timeframe": "1d",
            "universe": {"type": "static_list", "symbols": ["AAA", "BBB"]},
            "rotation": {
                "score": {"indicator": "return", "source": "close", "window": 126},
                "top_n": 1,
                "rebalance": "monthly",
                "require_positive_score": True,
            },
            "risk": {"execution": "next_open", "slippage_bps": 5, "commission_bps": 1},
        }

        StrategyDSLValidator().validate(strategy)

    def test_monthly_momentum_rotation_selects_top_symbol(self) -> None:
        periods = 220
        strategy = {
            "name": "rotation_test",
            "timeframe": "1d",
            "universe": {"type": "static_list", "symbols": ["AAA", "BBB", "CCC"]},
            "rotation": {
                "score": {"indicator": "return", "source": "close", "window": 126},
                "top_n": 1,
                "rebalance": "monthly",
                "require_positive_score": False,
            },
            "risk": {"execution": "next_open", "slippage_bps": 0, "commission_bps": 0},
        }
        market_data = {
            "AAA": make_market(np.linspace(100, 300, periods).tolist()),
            "BBB": make_market(np.linspace(100, 130, periods).tolist()),
            "CCC": make_market(np.linspace(100, 80, periods).tolist()),
        }

        result = run_rotation_backtest(strategy, market_data, initial_cash=10_000)

        self.assertGreater(result.metrics["final_equity"], 10_000)
        self.assertIn("total_return", result.metrics)
        self.assertIn("total_return", result.benchmark_metrics)
        self.assertFalse(result.weight_log.empty)
        self.assertFalse(result.order_log.empty)
        self.assertFalse(result.new_entry_plan.empty)
        self.assertIn("status", result.new_entry_plan.columns)
        self.assertIn(result.new_entry_plan.iloc[0]["status"], {"BUY_NOW", "WAIT_PULLBACK", "TOO_EXTENDED", "INVALID_TREND", "INSUFFICIENT_DATA"})
        self.assertIn("AAA", result.order_log["symbol"].tolist())
        self.assertLessEqual(result.weight_log.sum(axis=1).max(), 1.01)

    def test_require_positive_score_can_hold_cash(self) -> None:
        periods = 180
        strategy = {
            "name": "rotation_cash_test",
            "timeframe": "1d",
            "universe": {"type": "static_list", "symbols": ["AAA", "BBB"]},
            "rotation": {
                "score": {"indicator": "return", "source": "close", "window": 60},
                "top_n": 1,
                "rebalance": "monthly",
                "require_positive_score": True,
            },
            "risk": {"execution": "next_open", "slippage_bps": 0, "commission_bps": 0},
        }
        market_data = {
            "AAA": make_market(np.linspace(100, 80, periods).tolist()),
            "BBB": make_market(np.linspace(100, 70, periods).tolist()),
        }

        result = run_rotation_backtest(strategy, market_data, initial_cash=10_000)

        self.assertTrue(result.order_log.empty)
        self.assertAlmostEqual(result.metrics["final_equity"], 10_000)
        self.assertEqual(result.metrics["exposure_time"], 0)

    def test_daily_stop_loss_exits_before_next_rebalance(self) -> None:
        periods = 180
        strategy = {
            "name": "rotation_stop_loss_test",
            "timeframe": "1d",
            "universe": {"type": "static_list", "symbols": ["AAA", "BBB"]},
            "rotation": {
                "score": {"indicator": "return", "source": "close", "window": 60},
                "top_n": 1,
                "rebalance": "monthly",
                "require_positive_score": False,
            },
            "risk": {
                "execution": "next_open",
                "slippage_bps": 0,
                "commission_bps": 0,
                "stop_loss_pct": -0.05,
            },
        }
        aaa = np.concatenate(
            [
                np.linspace(100, 170, 110),
                np.linspace(170, 120, 10),
                np.repeat(120, periods - 120),
            ]
        )
        market_data = {
            "AAA": make_market(aaa.tolist()),
            "BBB": make_market(np.linspace(100, 101, periods).tolist()),
        }

        result = run_rotation_backtest(strategy, market_data, initial_cash=10_000, liquidate_end=False)

        self.assertIn("risk_stop_loss", result.order_log["reason"].tolist())
        stop_order = result.order_log[result.order_log["reason"] == "risk_stop_loss"].iloc[0]
        next_rebalance_orders = result.order_log[result.order_log["date"] > stop_order["date"]]
        self.assertTrue(next_rebalance_orders.empty or (next_rebalance_orders.iloc[0]["date"] - stop_order["date"]).days >= 1)


if __name__ == "__main__":
    unittest.main()
