from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .backtest import run_backtest
from .backtest_cli import _load_strategy, _override_universe, _required_symbols, backtest_result_row


@dataclass(frozen=True)
class StrategyBatchRun:
    summary: pd.DataFrame
    details: dict[str, Any]
    output_dir: Path | None = None


def discover_strategy_files(path: str | Path) -> list[Path]:
    strategy_path = Path(path)
    if strategy_path.is_file():
        return [strategy_path]
    if not strategy_path.exists():
        raise FileNotFoundError(strategy_path)
    return sorted(strategy_path.rglob("*.json"))


def required_symbols_for_strategy_dir(strategy_files: list[Path], tickers: list[str]) -> list[str]:
    symbols: set[str] = set()
    for file_path in strategy_files:
        strategy = _load_strategy(file_path)
        for ticker in tickers:
            patched = _override_universe(strategy, ticker)
            symbols.update(_required_symbols(patched))
    return sorted(symbols)


def run_strategy_directory(
    strategy_path: str | Path,
    tickers: list[str],
    market_data: dict[str, pd.DataFrame],
    initial_cash: float = 100_000.0,
    output_root: str | Path | None = None,
    period_label: str = "",
) -> StrategyBatchRun:
    strategy_files = discover_strategy_files(strategy_path)
    if not strategy_files:
        raise ValueError(f"No strategy JSON files found in {strategy_path}")

    normalized_tickers = _normalize_tickers(tickers)
    summary_rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {
        "period": period_label,
        "initial_cash": initial_cash,
        "tickers": normalized_tickers,
        "strategies": {},
    }

    for file_path in strategy_files:
        strategy = _load_strategy(file_path)
        ticker_rows: list[dict[str, Any]] = []
        ticker_details: dict[str, Any] = {}

        for ticker in normalized_tickers:
            patched = _override_universe(strategy, ticker)
            result = run_backtest(patched, market_data, initial_cash=initial_cash)
            row = backtest_result_row(ticker, result)
            ticker_rows.append(row)
            ticker_details[ticker] = {
                "row": _json_safe(row),
                "metrics": _json_safe(result.metrics),
                "benchmark_metrics": _json_safe(result.benchmark_metrics or {}),
                "score": _json_safe(result.score or {}),
            }

        aggregate = aggregate_strategy_rows(file_path, strategy, ticker_rows)
        summary_rows.append(aggregate)
        details["strategies"][str(file_path)] = {
            "name": strategy["name"],
            "summary": _json_safe(aggregate),
            "tickers": ticker_details,
        }

    summary = pd.DataFrame(summary_rows).sort_values(["avg_score", "pass_rate"], ascending=False)
    output_dir = _write_outputs(summary, details, output_root) if output_root else None
    return StrategyBatchRun(summary=summary, details=details, output_dir=output_dir)


def aggregate_strategy_rows(file_path: Path, strategy: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    pass_rate = float((frame["verdict"] == "PASS").mean()) if not frame.empty else 0.0
    avg_score = _mean(frame, "score")
    avg_excess = _mean(frame, "excess_return")
    avg_return = _mean(frame, "strategy_return")
    avg_buy_hold = _mean(frame, "buy_hold_return")
    avg_drawdown = _mean(frame, "max_drawdown")
    avg_sharpe = _mean(frame, "sharpe")
    avg_exposure = _mean(frame, "exposure_time")
    total_trades = int(frame["trades"].fillna(0).sum()) if "trades" in frame else 0
    verdict = "PASS" if pass_rate >= 0.5 and (avg_score or 0) >= 60 and (avg_excess or 0) > 0 else "FAIL"

    return {
        "strategy_file": str(file_path),
        "strategy_name": strategy["name"],
        "verdict": verdict,
        "avg_score": avg_score,
        "pass_rate": pass_rate,
        "avg_strategy_return": avg_return,
        "avg_buy_hold_return": avg_buy_hold,
        "avg_excess_return": avg_excess,
        "avg_max_drawdown": avg_drawdown,
        "avg_sharpe": avg_sharpe,
        "avg_exposure_time": avg_exposure,
        "total_trades": total_trades,
        "tickers_tested": len(rows),
    }


def _write_outputs(summary: pd.DataFrame, details: dict[str, Any], output_root: str | Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    return output_dir


def _normalize_tickers(tickers: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        symbol = ticker.strip().upper()
        if symbol and symbol not in seen:
            normalized.append(symbol)
            seen.add(symbol)
    if not normalized:
        raise ValueError("At least one ticker is required")
    return normalized


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(series.mean()) if not series.empty else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and pd.isna(value):
        return None
    return value
