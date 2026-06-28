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
from .reporting import format_backtest_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a ZHQUANT strategy backtest using yfinance data.")
    parser.add_argument("strategy", help="Path to strategy DSL JSON.")
    parser.add_argument("ticker", help="Ticker to backtest, for example AAPL or NVDA.")
    parser.add_argument("--period", default="1mo", help="yfinance period, for example 1mo, 3mo, 1y. Ignored if --start is set.")
    parser.add_argument("--start", help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", help="End date YYYY-MM-DD.")
    parser.add_argument("--initial-cash", type=float, default=100_000.0, help="Initial cash for the backtest.")
    parser.add_argument("--show-trades", type=int, default=10, help="Number of latest trades to print.")
    args = parser.parse_args(argv)

    try:
        strategy = _load_strategy(Path(args.strategy))
        ticker = args.ticker.upper()
        strategy = _override_universe(strategy, ticker)
        required_symbols = _required_symbols(strategy)
        market_data = download_yfinance_ohlcv(
            required_symbols,
            period=args.period,
            start=args.start,
            end=args.end,
            interval="1d",
        )
        result = run_backtest(strategy, market_data, initial_cash=args.initial_cash)
    except (OSError, json.JSONDecodeError, DSLValidationError, DataLoadError, BacktestError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    period_label = f"{args.start or ''} to {args.end or ''}".strip() if args.start or args.end else args.period
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

