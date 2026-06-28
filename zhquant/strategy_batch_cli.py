from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .backtest import BacktestError
from .backtest_cli import _parse_tickers
from .data_loader import DataLoadError, download_yfinance_ohlcv
from .dsl_validator import DSLValidationError
from .reporting import format_strategy_batch_report
from .strategy_batch import discover_strategy_files, required_symbols_for_strategy_dir, run_strategy_directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run every strategy JSON in a directory across a ticker basket.")
    parser.add_argument("strategies", help="Strategy JSON file or directory containing strategy JSON files.")
    parser.add_argument("--tickers", required=True, help="Comma-separated tickers, for example AAPL,MSFT,NVDA,MU,AMD.")
    parser.add_argument("--period", default="1mo", help="yfinance period. Ignored if --start is set.")
    parser.add_argument("--start", help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", help="End date YYYY-MM-DD.")
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--output-root", default="results/runs", help="Directory where summary.csv and details.json are saved.")
    args = parser.parse_args(argv)

    try:
        tickers = _parse_tickers(None, args.tickers)
        strategy_files = discover_strategy_files(Path(args.strategies))
        required_symbols = required_symbols_for_strategy_dir(strategy_files, tickers)
        market_data = download_yfinance_ohlcv(
            required_symbols,
            period=args.period,
            start=args.start,
            end=args.end,
            interval="1d",
        )
        period_label = f"{args.start or ''} to {args.end or ''}".strip() if args.start or args.end else args.period
        run = run_strategy_directory(
            strategy_path=args.strategies,
            tickers=tickers,
            market_data=market_data,
            initial_cash=args.initial_cash,
            output_root=args.output_root,
            period_label=period_label,
        )
    except (OSError, json.JSONDecodeError, DSLValidationError, DataLoadError, BacktestError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(format_strategy_batch_report(run.summary, strategy_path=args.strategies, period_label=period_label, output_dir=run.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

