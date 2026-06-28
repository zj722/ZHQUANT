from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

from .backtest import BacktestError, run_backtest
from .data_loader import DataLoadError, download_yfinance_ohlcv
from .dsl_validator import DSLValidationError, StrategyDSLValidator
from .reporting import format_backtest_report, format_batch_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a ZHQUANT strategy backtest using yfinance data.")
    parser.add_argument("strategy", help="Path to strategy DSL JSON.")
    parser.add_argument("ticker", nargs="?", help="Ticker to backtest, for example AAPL or NVDA.")
    parser.add_argument("--tickers", help="Comma-separated tickers for batch mode, for example AAPL,MSFT,NVDA,MU,AMD.")
    parser.add_argument("--period", default="1mo", help="yfinance period, for example 1mo, 3mo, 1y. Ignored if --start is set.")
    parser.add_argument("--start", help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", help="End date YYYY-MM-DD.")
    parser.add_argument("--initial-cash", type=float, default=100_000.0, help="Initial cash for the backtest.")
    parser.add_argument("--show-trades", type=int, default=10, help="Number of latest trades to print.")
    args = parser.parse_args(argv)

    try:
        base_strategy = _load_strategy(Path(args.strategy))
        tickers = _parse_tickers(args.ticker, args.tickers)
        required_symbols = set()
        strategies_by_ticker: dict[str, dict[str, Any]] = {}
        for ticker in tickers:
            strategy = _override_universe(base_strategy, ticker)
            strategies_by_ticker[ticker] = strategy
            required_symbols.update(_required_symbols(strategy))
        market_data = download_yfinance_ohlcv(
            sorted(required_symbols),
            period=args.period,
            start=args.start,
            end=args.end,
            interval="1d",
        )
        results = {
            ticker: run_backtest(strategy, market_data, initial_cash=args.initial_cash)
            for ticker, strategy in strategies_by_ticker.items()
        }
    except (OSError, json.JSONDecodeError, DSLValidationError, DataLoadError, BacktestError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    period_label = f"{args.start or ''} to {args.end or ''}".strip() if args.start or args.end else args.period
    if len(results) > 1:
        rows = [_batch_row(ticker, result) for ticker, result in results.items()]
        print(format_batch_report(rows, strategy_path=args.strategy, period_label=period_label))
        return 0

    ticker = next(iter(results))
    result = results[ticker]
    print(
        format_backtest_report(
            result=result,
            ticker=ticker,
            strategy_path=args.strategy,
            period_label=period_label,
            max_trades=args.show_trades,
        )
    )
    return 0


def _parse_tickers(ticker: str | None, tickers: str | None) -> list[str]:
    values: list[str] = []
    if ticker:
        values.append(ticker)
    if tickers:
        values.extend(item.strip() for item in tickers.split(","))
    normalized = []
    seen = set()
    for value in values:
        symbol = value.strip().upper()
        if not symbol or symbol in seen:
            continue
        normalized.append(symbol)
        seen.add(symbol)
    if not normalized:
        raise DataLoadError("Provide a ticker positional argument or --tickers AAPL,MSFT")
    return normalized


def _batch_row(ticker: str, result: Any) -> dict[str, object]:
    metrics = result.metrics
    benchmark = result.benchmark_metrics or {}
    score = result.score or {}
    return {
        "ticker": ticker,
        "verdict": score.get("verdict"),
        "score": score.get("score"),
        "strategy_return": metrics.get("total_return"),
        "buy_hold_return": benchmark.get("total_return"),
        "excess_return": score.get("excess_return"),
        "max_drawdown": metrics.get("max_drawdown"),
        "sharpe": metrics.get("sharpe"),
        "exposure_time": metrics.get("exposure_time"),
        "trades": metrics.get("trade_count"),
        "win_rate": metrics.get("win_rate"),
        "net_profit": metrics.get("net_profit"),
        "gross_profit": metrics.get("gross_profit"),
        "gross_loss": metrics.get("gross_loss"),
    }


def _load_strategy(path: Path) -> dict[str, Any]:
    strategy = json.loads(path.read_text(encoding="utf-8"))
    StrategyDSLValidator().validate(strategy)
    return strategy


def _override_universe(strategy: dict[str, Any], ticker: str) -> dict[str, Any]:
    patched = copy.deepcopy(strategy)
    patched["universe"] = {"type": "static_list", "symbols": [ticker]}
    max_positions = patched.get("risk", {}).get("max_positions")
    if isinstance(max_positions, int) and max_positions > 1:
        patched["risk"]["max_positions"] = 1
    StrategyDSLValidator().validate(patched)
    return patched


def _required_symbols(strategy: dict[str, Any]) -> list[str]:
    symbols = set(strategy["universe"]["symbols"])
    symbols.update(_walk_symbol_refs(strategy))
    return sorted(symbol.upper() for symbol in symbols)


def _walk_symbol_refs(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "symbol" and isinstance(item, str):
                refs.add(item)
            else:
                refs.update(_walk_symbol_refs(item))
    elif isinstance(value, list):
        for item in value:
            refs.update(_walk_symbol_refs(item))
    return refs


if __name__ == "__main__":
    raise SystemExit(main())
