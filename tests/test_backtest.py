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
        self.assertIn("total_return", result.benchmark_metrics)
        self.assertIn(result.score["verdict"], {"PASS", "FAIL"})
        self.assertIn("exposure_time", result.metrics)
        self.assertIn("sortino", result.metrics)

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

    def test_current_action_buy_when_latest_entry_signal_is_true(self) -> None:
        strategy = base_strategy(
            {
                "op": ">=",
                "left": {"field": "holding_days"},
                "right": 20,
            }
        )
        market_data = {"AAA": make_market([10, 9, 11])}

        result = run_backtest(strategy, market_data, initial_cash=10_000)

        action = result.current_actions.iloc[0]
        self.assertEqual(action["symbol"], "AAA")
        self.assertEqual(action["action"], "BUY")
        self.assertEqual(action["reason"], "entry_signal")
        self.assertFalse(action["in_position"])
        self.assertEqual(action["execution"], "next_open")

    def test_current_action_sell_when_latest_exit_signal_is_true(self) -> None:
        strategy = base_strategy(
            {
                "op": ">=",
                "left": {"field": "holding_days"},
                "right": 1,
            }
        )
        market_data = {"AAA": make_market([10, 11, 12, 13])}

        result = run_backtest(strategy, market_data, initial_cash=10_000)

        action = result.current_actions.iloc[0]
        self.assertEqual(action["symbol"], "AAA")
        self.assertEqual(action["action"], "SELL")
        self.assertEqual(action["reason"], "holding_days")
        self.assertTrue(action["in_position"])

    def test_stateful_exit_can_use_entry_price_and_atr_indicator(self) -> None:
        strategy = base_strategy(
            {
                "op": "<",
                "left": {"indicator": "close"},
                "right": {
                    "op": "-",
                    "left": {"field": "entry_price"},
                    "right": {
                        "op": "*",
                        "left": {"indicator": "atr", "window": 3},
                        "right": 0.5,
                    },
                },
            },
            risk_extra={"slippage_bps": 0, "commission_bps": 0},
        )
        market_data = {"AAA": make_market([10, 11, 12, 13, 8, 8, 8])}

        result = run_backtest(strategy, market_data, initial_cash=10_000)

        self.assertFalse(result.trade_log.empty)
        self.assertEqual(result.trade_log.iloc[0]["exit_reason"], "stateful_exit")
        self.assertLess(result.trade_log.iloc[0]["net_return"], 0)

    def test_add_entry_pyramids_into_existing_position(self) -> None:
        strategy = base_strategy(
            {
                "op": ">=",
                "left": {"field": "holding_days"},
                "right": 3,
            },
            risk_extra={"max_additions": 1, "add_position_pct": 0.25, "slippage_bps": 0, "commission_bps": 0},
        )
        strategy["add_entry"] = {
            "op": ">",
            "left": {"indicator": "close"},
            "right": {"indicator": "sma", "source": "close", "window": 2},
        }
        market_data = {"AAA": make_market([10, 11, 12, 13, 14, 15])}

        result = run_backtest(strategy, market_data, initial_cash=10_000)

        self.assertFalse(result.trade_log.empty)
        self.assertEqual(result.trade_log.iloc[0]["add_count"], 1)
        self.assertGreater(result.trade_log.iloc[0]["shares"], 10_000 * 0.5 / 12)
        self.assertEqual(result.order_log["action"].tolist(), ["BUY", "ADD", "SELL"])

    def test_reduce_exit_sells_part_of_existing_position(self) -> None:
        strategy = base_strategy(
            {
                "op": ">=",
                "left": {"field": "holding_days"},
                "right": 10,
            },
            risk_extra={"reduce_position_pct": 0.5, "slippage_bps": 0, "commission_bps": 0},
        )
        strategy["reduce_exit"] = {
            "op": "==",
            "left": {"indicator": "close"},
            "right": 13,
        }
        market_data = {"AAA": make_market([10, 11, 12, 13, 14, 15])}

        result = run_backtest(strategy, market_data, initial_cash=10_000)

        self.assertIn("REDUCE", result.order_log["action"].tolist())
        reduced_trade = result.trade_log[result.trade_log["exit_reason"] == "reduce_exit"].iloc[0]
        self.assertAlmostEqual(reduced_trade["shares"], result.order_log.iloc[0]["shares"] * 0.5)


if __name__ == "__main__":
    unittest.main()
