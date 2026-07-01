import unittest

import pandas as pd

from zhquant.backtest import BacktestResult
from zhquant.compiler import CompiledSignals
from zhquant.reporting import format_backtest_report, format_diagnostic_report


class ReportingTest(unittest.TestCase):
    def test_report_includes_money_fields(self) -> None:
        index = pd.date_range("2024-01-01", periods=2)
        result = BacktestResult(
            strategy_name="demo",
            equity_curve=pd.DataFrame({"equity": [1000, 1100]}, index=index),
            order_log=pd.DataFrame(
                [
                    {
                        "symbol": "AAPL",
                        "date": index[0],
                        "action": "BUY",
                        "price": 100.0,
                        "shares": 10.0,
                        "notional": 1000.0,
                        "commission": 1.0,
                        "reason": "entry_signal",
                    }
                ]
            ),
            trade_log=pd.DataFrame(
                [
                    {
                        "symbol": "AAPL",
                        "entry_date": index[0],
                        "exit_date": index[1],
                        "entry_price": 100.0,
                        "exit_price": 110.0,
                        "net_pnl": 100.0,
                        "net_return": 0.1,
                        "holding_days": 1,
                        "exit_reason": "force_close",
                    }
                ]
            ),
            metrics={
                "initial_cash": 1000.0,
                "final_equity": 1100.0,
                "net_profit": 100.0,
                "total_return": 0.1,
                "max_drawdown": -0.02,
                "sharpe": 1.2,
                "gross_profit": 100.0,
                "gross_loss": 0.0,
                "total_commission": 1.0,
                "avg_pnl_per_trade": 100.0,
                "trade_count": 1,
                "winning_trades": 1,
                "losing_trades": 0,
                "win_rate": 1.0,
                "avg_return_per_trade": 0.1,
                "best_trade": 0.1,
                "worst_trade": 0.1,
                "trades_per_year": 12.0,
            },
            compiled=CompiledSignals(
                strategy_name="demo",
                entry=pd.DataFrame(),
                exit=pd.DataFrame(),
                stateful_exit_rules=(),
                data_dependencies=(),
                diagnostics={},
            ),
            current_actions=pd.DataFrame(
                [
                    {
                        "symbol": "AAPL",
                        "signal_date": index[1],
                        "action": "BUY",
                        "reason": "entry_signal",
                        "in_position": False,
                        "entry_signal": True,
                        "exit_signal": False,
                        "execution": "next_open",
                    }
                ]
            ),
            diagnostic_log=pd.DataFrame(
                [
                    {"symbol": "AAPL", "signal_date": index[0], "market_ok": True, "final_buy_signal": False},
                    {"symbol": "AAPL", "signal_date": index[1], "market_ok": True, "final_buy_signal": True},
                ]
            ),
            diagnostic_pass_rates={"market_ok": 1.0, "final_buy_signal": 0.5},
        )

        report = format_backtest_report(result, "AAPL", "strategy.json", "1mo")

        self.assertIn("Net P/L: $+100.00", report)
        self.assertIn("Gross Profit From Winning Trades: $100.00", report)
        self.assertIn("Gross Loss From Losing Trades: $0.00", report)
        self.assertIn("Current Action", report)
        self.assertIn("BUY", report)

        diagnostics = format_diagnostic_report(result)

        self.assertIn("market_ok: 100.00%", diagnostics)
        self.assertIn("final_buy_signal: 50.00%", diagnostics)
        self.assertIn("Daily Conditions", diagnostics)


if __name__ == "__main__":
    unittest.main()
