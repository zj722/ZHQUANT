from __future__ import annotations

from pathlib import Path

import pandas as pd

from .backtest import BacktestResult


def format_backtest_report(
    result: BacktestResult,
    ticker: str,
    strategy_path: str | Path,
    period_label: str,
    max_trades: int = 10,
) -> str:
    metrics = result.metrics
    lines = [
        "ZHQUANT Backtest Result",
        "=" * 24,
        f"Strategy: {result.strategy_name}",
        f"Strategy file: {strategy_path}",
        f"Ticker: {ticker.upper()}",
        f"Period: {period_label}",
        "",
        "Summary",
        "-" * 7,
        f"Initial Cash: {_money(metrics.get('initial_cash'))}",
        f"Final Equity: {_money(metrics.get('final_equity'))}",
        f"Net P/L: {_signed_money(metrics.get('net_profit'))}",
        f"Total Return: {_pct(metrics.get('total_return'))}",
        f"Max Drawdown: {_pct_plain(metrics.get('max_drawdown'))}",
        f"Sharpe: {_number(metrics.get('sharpe'))}",
        "",
        "Money Made / Lost",
        "-" * 17,
        f"Gross Profit From Winning Trades: {_money(metrics.get('gross_profit'))}",
        f"Gross Loss From Losing Trades: {_money(metrics.get('gross_loss'))}",
        f"Total Commission: {_money(metrics.get('total_commission'))}",
        f"Average P/L Per Trade: {_signed_money(metrics.get('avg_pnl_per_trade'))}",
        "",
        "Trades",
        "-" * 6,
        f"Trade Count: {metrics.get('trade_count', 0)}",
        f"Winning Trades: {metrics.get('winning_trades', 0)}",
        f"Losing Trades: {metrics.get('losing_trades', 0)}",
        f"Win Rate: {_pct_plain(metrics.get('win_rate'))}",
        f"Average Return / Trade: {_pct(metrics.get('avg_return_per_trade'))}",
        f"Best Trade: {_pct(metrics.get('best_trade'))}",
        f"Worst Trade: {_pct(metrics.get('worst_trade'))}",
        f"Trades / Year: {_number(metrics.get('trades_per_year'))}",
    ]

    if not result.trade_log.empty:
        lines.extend(["", f"Trade Log (last {max_trades})", "-" * 20])
        lines.append(_trade_log_table(result.trade_log.tail(max_trades)))
    else:
        lines.extend(["", "Trade Log", "-" * 9, "No completed trades in this period."])

    return "\n".join(lines)


def _trade_log_table(trade_log: pd.DataFrame) -> str:
    display = trade_log.copy()
    display["entry_date"] = pd.to_datetime(display["entry_date"]).dt.strftime("%Y-%m-%d")
    display["exit_date"] = pd.to_datetime(display["exit_date"]).dt.strftime("%Y-%m-%d")
    display["net_pnl"] = display["net_pnl"].map(lambda value: f"{value:+,.2f}")
    display["net_return"] = display["net_return"].map(lambda value: f"{value:+.2%}")
    display["entry_price"] = display["entry_price"].map(lambda value: f"{value:,.2f}")
    display["exit_price"] = display["exit_price"].map(lambda value: f"{value:,.2f}")
    columns = [
        "symbol",
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
        "net_pnl",
        "net_return",
        "holding_days",
        "exit_reason",
    ]
    return display.loc[:, columns].to_string(index=False)


def _money(value: object) -> str:
    if value is None:
        return "N/A"
    return f"${float(value):,.2f}"


def _signed_money(value: object) -> str:
    if value is None:
        return "N/A"
    return f"${float(value):+,.2f}"


def _pct(value: object) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):+.2%}"


def _pct_plain(value: object) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.2%}"


def _number(value: object) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):,.2f}"
