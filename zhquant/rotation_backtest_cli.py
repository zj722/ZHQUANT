from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .data_loader import DataLoadError, download_yfinance_ohlcv
from .rotation_backtest import RotationBacktestError, RotationBacktestResult, run_rotation_backtest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a ZHQUANT portfolio rotation backtest using yfinance data.")
    parser.add_argument("strategy", help="Path to a rotation strategy JSON.")
    parser.add_argument("--period", default="2y", help="yfinance period. Ignored if --start is set.")
    parser.add_argument("--start", help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", help="End date YYYY-MM-DD.")
    parser.add_argument("--initial-cash", type=float, default=100_000.0, help="Initial cash for the backtest.")
    parser.add_argument("--equity-csv", help="Write portfolio equity curve to CSV.")
    parser.add_argument("--weights-csv", help="Write daily portfolio weights to CSV.")
    parser.add_argument("--orders-csv", help="Write rebalance orders to CSV.")
    parser.add_argument("--show-orders", type=int, default=12, help="Number of latest orders to print.")
    parser.add_argument(
        "--live-action",
        action="store_true",
        help="Do not liquidate at the final date; print the latest rebalance action for the next open.",
    )
    parser.add_argument(
        "--new-entry",
        action="store_true",
        help="Print a zero-position new-entry plan using rotation candidates plus entry-quality gates.",
    )
    args = parser.parse_args(argv)

    try:
        strategy = _load_strategy(Path(args.strategy))
        symbols = [symbol.upper() for symbol in strategy["universe"]["symbols"]]
        market_data = download_yfinance_ohlcv(
            symbols,
            period=args.period,
            start=args.start,
            end=args.end,
            interval="1d",
        )
        result = run_rotation_backtest(
            strategy,
            market_data,
            initial_cash=args.initial_cash,
            liquidate_end=not (args.live_action or args.new_entry),
        )
        _write_optional_csv(result.equity_curve, args.equity_csv)
        _write_optional_csv(result.weight_log, args.weights_csv)
        _write_optional_csv(result.order_log, args.orders_csv, index=False)
    except (OSError, json.JSONDecodeError, DataLoadError, RotationBacktestError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    period_label = f"{args.start or ''} to {args.end or ''}".strip() if args.start or args.end else args.period
    print(
        _format_report(
            result,
            strategy_path=args.strategy,
            period_label=period_label,
            max_orders=args.show_orders,
            include_live_action=args.live_action,
            include_new_entry=args.new_entry,
        )
    )
    return 0


def _load_strategy(path: Path) -> dict[str, Any]:
    strategy = json.loads(path.read_text(encoding="utf-8"))
    _validate_rotation_strategy(strategy)
    return strategy


def _validate_rotation_strategy(strategy: Any) -> None:
    if not isinstance(strategy, dict):
        raise RotationBacktestError("Strategy must be a JSON object")
    for key in ("name", "universe", "rotation", "risk"):
        if key not in strategy:
            raise RotationBacktestError(f"Strategy is missing required key: {key}")
    if not isinstance(strategy["name"], str) or not strategy["name"].strip():
        raise RotationBacktestError("Strategy name must be a non-empty string")

    universe = strategy["universe"]
    if not isinstance(universe, dict) or universe.get("type") != "static_list":
        raise RotationBacktestError("universe.type must be static_list")
    symbols = universe.get("symbols")
    if not isinstance(symbols, list) or not symbols:
        raise RotationBacktestError("universe.symbols must be a non-empty list")
    normalized = []
    for symbol in symbols:
        if not isinstance(symbol, str) or not symbol.strip():
            raise RotationBacktestError("universe.symbols must contain ticker strings")
        normalized.append(symbol.strip().upper())
    if len(normalized) != len(set(normalized)):
        raise RotationBacktestError("universe.symbols contains duplicate tickers")

    rotation = strategy["rotation"]
    if not isinstance(rotation, dict):
        raise RotationBacktestError("rotation must be an object")
    score = rotation.get("score")
    if not isinstance(score, dict):
        raise RotationBacktestError("rotation.score must be an object")
    if score.get("indicator") != "return":
        raise RotationBacktestError("rotation.score.indicator must be return")
    if score.get("source", "close") != "close":
        raise RotationBacktestError("rotation.score.source must be close")
    if not isinstance(score.get("window"), int) or score["window"] < 2 or score["window"] > 252:
        raise RotationBacktestError("rotation.score.window must be an integer from 2 to 252")
    if not isinstance(rotation.get("top_n"), int) or rotation["top_n"] < 1 or rotation["top_n"] > len(normalized):
        raise RotationBacktestError("rotation.top_n must be between 1 and the universe size")
    if rotation.get("rebalance", "monthly") not in {"weekly", "monthly"}:
        raise RotationBacktestError("rotation.rebalance must be weekly or monthly")
    if "require_positive_score" in rotation and not isinstance(rotation["require_positive_score"], bool):
        raise RotationBacktestError("rotation.require_positive_score must be boolean")

    risk = strategy["risk"]
    if not isinstance(risk, dict):
        raise RotationBacktestError("risk must be an object")
    if risk.get("execution", "next_open") != "next_open":
        raise RotationBacktestError("rotation strategies currently support only risk.execution=next_open")
    for key in ("slippage_bps", "commission_bps"):
        if key in risk and (not isinstance(risk[key], int | float) or risk[key] < 0 or risk[key] > 1000):
            raise RotationBacktestError(f"risk.{key} must be a number from 0 to 1000")
    if "stop_loss_pct" in risk and (
        not isinstance(risk["stop_loss_pct"], int | float)
        or risk["stop_loss_pct"] >= 0
        or risk["stop_loss_pct"] <= -1
    ):
        raise RotationBacktestError("risk.stop_loss_pct must be between -1 and 0")
    if "ma200_exit" in risk and not isinstance(risk["ma200_exit"], bool):
        raise RotationBacktestError("risk.ma200_exit must be boolean")
    if "atr_trailing_stop" in risk:
        trailing = risk["atr_trailing_stop"]
        if not isinstance(trailing, dict):
            raise RotationBacktestError("risk.atr_trailing_stop must be an object")
        if "enabled" in trailing and not isinstance(trailing["enabled"], bool):
            raise RotationBacktestError("risk.atr_trailing_stop.enabled must be boolean")
        if "multiple" in trailing and (
            not isinstance(trailing["multiple"], int | float) or trailing["multiple"] <= 0 or trailing["multiple"] > 20
        ):
            raise RotationBacktestError("risk.atr_trailing_stop.multiple must be between 0 and 20")
        if "activation_profit_pct" in trailing and (
            not isinstance(trailing["activation_profit_pct"], int | float)
            or trailing["activation_profit_pct"] < 0
            or trailing["activation_profit_pct"] > 10
        ):
            raise RotationBacktestError("risk.atr_trailing_stop.activation_profit_pct must be between 0 and 10")


def _write_optional_csv(frame: pd.DataFrame, path: str | None, index: bool = True) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=index)


def _format_report(
    result: RotationBacktestResult,
    strategy_path: str,
    period_label: str,
    max_orders: int,
    include_live_action: bool,
    include_new_entry: bool,
) -> str:
    metrics = result.metrics
    benchmark = result.benchmark_metrics
    excess = metrics["total_return"] - benchmark["total_return"]
    relative_excess = None
    if benchmark["total_return"] != -1:
        relative_excess = (1.0 + metrics["total_return"]) / (1.0 + benchmark["total_return"]) - 1.0

    latest_weights = result.weight_log.iloc[-1].sort_values(ascending=False)
    active_weights = latest_weights[latest_weights > 0.001]
    if active_weights.empty:
        weight_text = "Cash / no active position"
    else:
        weight_text = "\n".join(f"{symbol}: {_pct(weight)}" for symbol, weight in active_weights.items())

    lines = [
        "ZHQUANT Rotation Backtest Result",
        "================================",
        f"Strategy: {result.strategy_name}",
        f"Strategy file: {strategy_path}",
        f"Period: {period_label}",
        "",
        "Summary",
        "-------",
        f"Initial Cash: {_money(metrics['initial_cash'])}",
        f"Final Equity: {_money(metrics['final_equity'])}",
        f"Net P/L: {_money(metrics['net_profit'])}",
        f"Total Return: {_pct(metrics['total_return'])}",
        f"Max Drawdown: {_pct(metrics['max_drawdown'])}",
        f"Sharpe: {_number(metrics['sharpe'])}",
        f"Exposure Time: {_pct(metrics['exposure_time'])}",
    ]

    if include_live_action:
        lines.extend(["", *_format_live_action(result.current_action)])
    if include_new_entry:
        lines.extend(["", *_format_new_entry_plan(result.new_entry_plan, metrics["initial_cash"])])

    lines.extend(
        [
            "",
            "Benchmark",
            "---------",
            f"Benchmark: {benchmark['name']}",
            f"Benchmark Return: {_pct(benchmark['total_return'])}",
            f"Benchmark Max Drawdown: {_pct(benchmark['max_drawdown'])}",
            f"Excess Return: {_pct(excess)}",
            f"Relative Excess Return: {_pct(relative_excess)}",
            "",
            "Latest Weights",
            "--------------",
            weight_text,
            "",
            "Orders",
            "------",
            f"Order Count: {len(result.order_log)}",
        ]
    )

    if result.order_log.empty:
        lines.append("No orders.")
    elif max_orders <= 0:
        lines.append("Order display suppressed.")
    else:
        display = result.order_log.tail(max_orders).copy()
        display["date"] = pd.to_datetime(display["date"]).dt.strftime("%Y-%m-%d")
        lines.append(display.to_string(index=False))
    return "\n".join(lines)


def _format_live_action(action: dict[str, Any]) -> list[str]:
    signal_date = pd.Timestamp(action["signal_date"]).strftime("%Y-%m-%d")
    target = pd.Series(action["target_weights"]).sort_values(ascending=False)
    changes = pd.Series(action["weight_changes"]).sort_values()

    active_target = target[target > 0.001]
    target_lines = ["Cash / no active position"] if active_target.empty else [
        f"{symbol}: {_pct(weight)}" for symbol, weight in active_target.items()
    ]

    material_changes = changes[changes.abs() >= 0.001]
    if material_changes.empty:
        change_lines = ["No material change."]
    else:
        change_lines = [f"{symbol}: {_pct(change)}" for symbol, change in material_changes.items()]

    return [
        "Current Action",
        "--------------",
        f"Signal Date: {signal_date}",
        f"Action: {action['action']}",
        f"Reason: {action['reason']}",
        f"Execution: {action['execution']}",
        "Target Weights:",
        *target_lines,
        "Weight Changes:",
        *change_lines,
    ]


def _format_new_entry_plan(plan: pd.DataFrame, account_value: float) -> list[str]:
    if plan.empty:
        return [
            "New Entry Plan",
            "--------------",
            "No rotation candidates have enough data for a new-entry plan.",
        ]

    display = plan.copy()
    display["signal_date"] = pd.to_datetime(display["signal_date"]).dt.strftime("%Y-%m-%d")
    display["rotation_signal_date"] = pd.to_datetime(display["rotation_signal_date"]).dt.strftime("%Y-%m-%d")
    display["target_weight"] = display["target_weight"].map(_pct)
    display["suggested_initial_weight"] = display["suggested_initial_weight"].map(_pct)
    display["suggested_notional"] = plan["suggested_initial_weight"].astype(float) * account_value
    display["suggested_notional"] = display["suggested_notional"].map(_money)
    display["score_126d"] = display["score_126d"].map(_pct)
    display["rsi14"] = display["rsi14"].map(_number)
    display["pullback_from_20d_high"] = display["pullback_from_20d_high"].map(_pct)
    columns = [
        "symbol",
        "rank",
        "status",
        "target_weight",
        "suggested_initial_weight",
        "suggested_notional",
        "score_126d",
        "rsi14",
        "pullback_from_20d_high",
        "reason",
    ]
    return [
        "New Entry Plan",
        "--------------",
        "For a zero-position account, use this plan instead of Weight Changes.",
        display.loc[:, columns].to_string(index=False),
    ]


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:+.2f}%"


def _number(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
