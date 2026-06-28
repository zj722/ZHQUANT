import unittest

import pandas as pd

from zhquant.backtest import run_backtest


def make_market(close_values: list[float], volume_values: list[float] | None = None) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(close_values), freq="D")
    close = pd.Series(close_values, index=dates, dtype=float)
    volume = pd.Series(volume_values or [1_000_000.0] * len(close_values), index=dates, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


def base_strategy(exit_rule: dict, risk_extra: dict | None = None, single_entry: bool = False) -> dict:
    risk = {
        "max_position_pct": 0.5,
        "max_positions": 1,
        "execution": "next_open",
        "slippage_bps": 10,
        "commission_bps": 5,
    }
    if risk_extra:
        risk.update(risk_extra)

    entry = {
        "op": ">",
        "left": {"indicator": "close"},
        "right": {"indicator": "sma", "source": "close", "window": 2},
    }
    if single_entry:
        entry = {
            "all": [
                entry,
                {
                    "op": ">",
                    "left": {"indicator": "volume"},
                    "right": {
                        "op": "*",
                        "left": {"indicator": "sma", "source": "volume", "window": 2},
                        "right": 1.5,
                    },
                },
            ]
        }

    return {
        "name": "test_strategy",
        "timeframe": "1d",
        "universe": {"type": "static_list", "symbols": ["AAA"]},
        "entry": entry,
        "exit": exit_rule,
        "risk": risk,
    }


class BacktestTest(unittest.TestCase):
    def test_holding_days_exit_produces_trade_log_and_metrics(self) -> None:
        strategy = base_strategy(
            {
                "op": ">=",
                "left": {"field": "holding_days"},
                "right": 3,
            },
            single_entry=True,
        )
        volume = [100, 100, 400, 100, 100, 100, 100, 100]
        market_data = {"AAA": make_market([10, 11, 12, 13, 14, 15, 16, 17], volume_values=volume)}

        result = run_backtest(strategy, market_data, initial_cash=10_000)

        self.assertEqual(len(result.trade_log), 1)
        trade = result.trade_log.iloc[0]
        self.assertEqual(trade["exit_reason"], "holding_days")
        self.assertEqual(trade["holding_days"], 4)
        self.assertGreater(trade["net_pnl"], 0)
        self.assertGreater(result.metrics["final_equity"], 10_000)
        self.assertEqual(result.metrics["trade_count"], 1)

    def test_stop_loss_risk_exit(self) -> None:
        strategy = base_strategy(
            {
                "op": ">=",
                "left": {"field": "holding_days"},
                "right": 20,
            },
            risk_extra={"stop_loss_pct": -0.05},
        )
        market_data = {"AAA": make_market([10, 11, 12, 10, 9, 8, 8, 8])}

        result = run_backtest(strategy, market_data, initial_cash=10_000)

        self.assertFalse(result.trade_log.empty)
        self.assertEqual(result.trade_log.iloc[0]["exit_reason"], "stop_loss")
        self.assertLess(result.trade_log.iloc[0]["net_return"], 0)
        self.assertLessEqual(result.metrics["max_drawdown"], 0)

    def test_take_profit_risk_exit(self) -> None:
        strategy = base_strategy(
            {
                "op": ">=",
                "left": {"field": "holding_days"},
                "right": 20,
            },
            risk_extra={"take_profit_pct": 0.1},
        )
        market_data = {"AAA": make_market([10, 11, 12, 14, 15, 16, 17, 18])}

        result = run_backtest(strategy, market_data, initial_cash=10_000)

        self.assertFalse(result.trade_log.empty)
        self.assertEqual(result.trade_log.iloc[0]["exit_reason"], "take_profit")
        self.assertGreater(result.trade_log.iloc[0]["net_return"], 0)


if __name__ == "__main__":
    unittest.main()
