from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class RotationBacktestError(ValueError):
    """Raised when a rotation strategy cannot be backtested."""


@dataclass(frozen=True)
class RotationBacktestResult:
    strategy_name: str
    equity_curve: pd.DataFrame
    weight_log: pd.DataFrame
    order_log: pd.DataFrame
    metrics: dict[str, Any]
    benchmark_metrics: dict[str, Any]
    current_action: dict[str, Any]
    new_entry_plan: pd.DataFrame


def run_rotation_backtest(
    strategy: dict[str, Any],
    market_data: dict[str, pd.DataFrame],
    initial_cash: float = 100_000.0,
    liquidate_end: bool = True,
) -> RotationBacktestResult:
    if initial_cash <= 0:
        raise RotationBacktestError("initial_cash must be positive")

    symbols = [symbol.upper() for symbol in strategy["universe"]["symbols"]]
    missing = [symbol for symbol in symbols if symbol not in market_data]
    if missing:
        raise RotationBacktestError(f"Missing market data for symbols: {missing}")

    data = {symbol: market_data[symbol].sort_index().copy() for symbol in symbols}
    index = _shared_index(data, symbols)
    if index.empty:
        raise RotationBacktestError("Universe symbols do not share any dates")

    close = pd.DataFrame({symbol: data[symbol]["close"].reindex(index) for symbol in symbols}, index=index)
    open_ = pd.DataFrame({symbol: data[symbol]["open"].reindex(index) for symbol in symbols}, index=index)
    ma200 = pd.DataFrame(
        {symbol: data[symbol]["close"].reindex(index).rolling(200).mean() for symbol in symbols},
        index=index,
    )
    atr20 = pd.DataFrame(
        {symbol: _atr(data[symbol].reindex(index)).rolling(20).mean() for symbol in symbols},
        index=index,
    )

    rotation = strategy["rotation"]
    score = _score_matrix(rotation["score"], close)
    top_n = int(rotation["top_n"])
    rebalance = str(rotation.get("rebalance", "monthly"))
    require_positive_score = bool(rotation.get("require_positive_score", False))
    risk_settings = _risk_settings(strategy)
    slippage_rate = float(strategy["risk"].get("slippage_bps", 0)) / 10_000.0
    commission_rate = float(strategy["risk"].get("commission_bps", 0)) / 10_000.0

    cash = float(initial_cash)
    shares = pd.Series(0.0, index=symbols)
    entry_prices = pd.Series(np.nan, index=symbols)
    highest_closes = pd.Series(np.nan, index=symbols)
    equity_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    pending_weights = pd.Series(0.0, index=symbols)
    pending_risk_exits: dict[str, str] = {}

    rebalance_dates = _rebalance_dates(index, rebalance)

    for idx, date in enumerate(index):
        if idx > 0 and pending_risk_exits:
            before_shares = shares.copy()
            cash = _sell_symbols_at_open(
                date=date,
                symbols=symbols,
                open_prices=open_.loc[date],
                exit_reasons=pending_risk_exits,
                cash=cash,
                shares=shares,
                commission_rate=commission_rate,
                slippage_rate=slippage_rate,
                order_rows=order_rows,
            )
            _sync_position_state(before_shares, shares, open_.loc[date], entry_prices, highest_closes, slippage_rate)
            pending_risk_exits = {}

        if idx > 0 and _has_pending_rebalance(date, index, rebalance_dates):
            before_shares = shares.copy()
            cash = _rebalance_at_open(
                date=date,
                symbols=symbols,
                open_prices=open_.loc[date],
                target_weights=pending_weights,
                cash=cash,
                shares=shares,
                commission_rate=commission_rate,
                slippage_rate=slippage_rate,
                order_rows=order_rows,
            )
            _sync_position_state(before_shares, shares, open_.loc[date], entry_prices, highest_closes, slippage_rate)

        equity = _portfolio_value(cash, shares, close.loc[date])
        weights = _current_weights(shares, close.loc[date], equity)
        equity_rows.append({"date": date, "cash": cash, "equity": equity, "open_positions": int((shares > 0).sum())})
        weight_row = {"date": date}
        weight_row.update({symbol: float(weights[symbol]) for symbol in symbols})
        weight_rows.append(weight_row)

        _update_highest_closes(shares, close.loc[date], highest_closes)
        pending_risk_exits = _risk_exit_signals(
            shares=shares,
            close_prices=close.loc[date],
            entry_prices=entry_prices,
            highest_closes=highest_closes,
            ma200=ma200.loc[date],
            atr20=atr20.loc[date],
            risk_settings=risk_settings,
        )

        if date in rebalance_dates and idx < len(index) - 1:
            excluded = set(pending_risk_exits)
            if risk_settings["ma200_exit"]:
                excluded.update(symbol for symbol in symbols if pd.notna(ma200.loc[date, symbol]) and close.loc[date, symbol] < ma200.loc[date, symbol])
            pending_weights = _target_weights(
                score.loc[date],
                top_n=top_n,
                require_positive=require_positive_score,
                excluded_symbols=excluded,
            )

    current_action = _current_action(
        strategy=strategy,
        index=index,
        score=score,
        rebalance_dates=rebalance_dates,
        current_weights=pd.Series(weight_rows[-1]).drop(labels=["date"]),
        top_n=top_n,
        require_positive_score=require_positive_score,
    )
    new_entry_plan = _new_entry_plan(
        data=data,
        index=index,
        score=score,
        rebalance_dates=rebalance_dates,
        top_n=top_n,
        require_positive_score=require_positive_score,
    )

    if liquidate_end and shares.sum() > 0:
        final_date = index[-1]
        cash = _rebalance_at_open(
            date=final_date,
            symbols=symbols,
            open_prices=close.loc[final_date],
            target_weights=pd.Series(0.0, index=symbols),
            cash=cash,
            shares=shares,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            order_rows=order_rows,
            price_type="close",
        )
        equity_rows.append({"date": final_date, "cash": cash, "equity": cash, "open_positions": 0})

    equity_curve = pd.DataFrame(equity_rows).drop_duplicates("date", keep="last").set_index("date")
    weight_log = pd.DataFrame(weight_rows).drop_duplicates("date", keep="last").set_index("date")
    order_log = pd.DataFrame(order_rows)
    metrics = _metrics(equity_curve, initial_cash)
    benchmark_metrics = _equal_weight_benchmark(close, initial_cash, commission_rate, slippage_rate)

    return RotationBacktestResult(
        strategy_name=strategy["name"],
        equity_curve=equity_curve,
        weight_log=weight_log,
        order_log=order_log,
        metrics=metrics,
        benchmark_metrics=benchmark_metrics,
        current_action=current_action,
        new_entry_plan=new_entry_plan,
    )


def _shared_index(data: dict[str, pd.DataFrame], symbols: list[str]) -> pd.Index:
    index = data[symbols[0]].index
    for symbol in symbols[1:]:
        index = index.intersection(data[symbol].index)
    return index.sort_values()


def _score_matrix(score_spec: dict[str, Any], close: pd.DataFrame) -> pd.DataFrame:
    if score_spec.get("indicator") != "return":
        raise RotationBacktestError("rotation.score currently supports only indicator=return")
    window = int(score_spec["window"])
    return close.pct_change(window)


def _rebalance_dates(index: pd.Index, rebalance: str) -> set[pd.Timestamp]:
    dates = pd.Series(index=index, data=index)
    if rebalance == "monthly":
        markers = dates.dt.to_period("M") != dates.shift(1).dt.to_period("M")
    elif rebalance == "weekly":
        markers = dates.dt.to_period("W") != dates.shift(1).dt.to_period("W")
    else:
        raise RotationBacktestError("rotation.rebalance must be monthly or weekly")
    return set(pd.Index(dates[markers]).dropna())


def _has_pending_rebalance(date: pd.Timestamp, index: pd.Index, rebalance_dates: set[pd.Timestamp]) -> bool:
    loc = index.get_loc(date)
    if loc == 0:
        return False
    previous_date = index[loc - 1]
    return previous_date in rebalance_dates


def _current_action(
    strategy: dict[str, Any],
    index: pd.Index,
    score: pd.DataFrame,
    rebalance_dates: set[pd.Timestamp],
    current_weights: pd.Series,
    top_n: int,
    require_positive_score: bool,
) -> dict[str, Any]:
    signal_date = index[-1]
    current_weights = current_weights.astype(float)

    if signal_date not in rebalance_dates:
        return {
            "signal_date": signal_date,
            "action": "HOLD",
            "reason": "no_rebalance_today",
            "execution": "none",
            "current_weights": current_weights.to_dict(),
            "target_weights": current_weights.to_dict(),
            "weight_changes": {symbol: 0.0 for symbol in current_weights.index},
        }

    target_weights = _target_weights(score.loc[signal_date], top_n=top_n, require_positive=require_positive_score)
    target_weights = target_weights.reindex(current_weights.index).fillna(0.0)
    changes = target_weights - current_weights
    material_change = bool((changes.abs() >= 0.01).any())
    return {
        "signal_date": signal_date,
        "action": "REBALANCE_NEXT_OPEN" if material_change else "HOLD",
        "reason": "monthly_rebalance_due" if material_change else "target_weights_unchanged",
        "execution": str(strategy.get("risk", {}).get("execution", "next_open")),
        "current_weights": current_weights.to_dict(),
        "target_weights": target_weights.to_dict(),
        "weight_changes": changes.to_dict(),
    }


def _new_entry_plan(
    data: dict[str, pd.DataFrame],
    index: pd.Index,
    score: pd.DataFrame,
    rebalance_dates: set[pd.Timestamp],
    top_n: int,
    require_positive_score: bool,
) -> pd.DataFrame:
    signal_date = index[-1]
    target_date = _latest_rotation_signal_date(index, rebalance_dates)
    if target_date is None:
        return pd.DataFrame()

    target_weights = _target_weights(score.loc[target_date], top_n=top_n, require_positive=require_positive_score)
    ranked_scores = pd.to_numeric(score.loc[target_date], errors="coerce").dropna().sort_values(ascending=False)
    if require_positive_score:
        ranked_scores = ranked_scores[ranked_scores > 0]

    rows = []
    for rank, symbol in enumerate(ranked_scores.head(top_n).index, start=1):
        frame = data[symbol].reindex(index)
        indicators = _entry_indicators(frame)
        row = _new_entry_row(
            symbol=symbol,
            rank=rank,
            signal_date=signal_date,
            target_date=target_date,
            target_weight=float(target_weights.get(symbol, 0.0)),
            score=float(ranked_scores[symbol]),
            indicators=indicators,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _latest_rotation_signal_date(index: pd.Index, rebalance_dates: set[pd.Timestamp]) -> pd.Timestamp | None:
    signal_date = index[-1]
    candidates = [date for date in rebalance_dates if date <= signal_date]
    if not candidates:
        return None
    return max(candidates)


def _entry_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    volume = frame["volume"]
    previous_close = close.shift(1)

    result = pd.DataFrame(index=frame.index)
    result["close"] = close
    result["previous_high"] = high.shift(1)
    result["ema20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    result["ma50"] = close.rolling(50).mean()
    result["ma200"] = close.rolling(200).mean()
    result["atr20"] = _atr(frame).rolling(20).mean()
    result["rsi14"] = _rsi(close, 14)
    result["return_5d"] = close.pct_change(5)
    result["return_126d"] = close.pct_change(126)
    result["return_5d_q90"] = result["return_5d"].rolling(252, min_periods=60).quantile(0.90).shift(1)
    result["pullback_from_20d_high"] = close / high.rolling(20).max() - 1.0
    result["volume_5d_avg"] = volume.rolling(5).mean()
    result["volume_20d_avg"] = volume.rolling(20).mean()
    result["low"] = low
    result["volume"] = volume
    result["previous_close"] = previous_close
    return result


def _atr(frame: pd.DataFrame) -> pd.Series:
    high = frame["high"]
    low = frame["low"]
    previous_close = frame["close"].shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100.0)
    rsi = rsi.mask((loss == 0) & (gain == 0), 50.0)
    return rsi


def _new_entry_row(
    symbol: str,
    rank: int,
    signal_date: pd.Timestamp,
    target_date: pd.Timestamp,
    target_weight: float,
    score: float,
    indicators: pd.DataFrame,
) -> dict[str, Any]:
    latest = indicators.loc[signal_date]
    required = ["close", "ema20", "ma50", "ma200", "atr20", "rsi14", "return_126d", "volume_20d_avg"]
    if latest[required].isna().any():
        return {
            "symbol": symbol,
            "rank": rank,
            "signal_date": signal_date,
            "rotation_signal_date": target_date,
            "status": "INSUFFICIENT_DATA",
            "new_entry_ok": False,
            "target_weight": target_weight,
            "suggested_initial_weight": 0.0,
            "score_126d": score,
            "close": _nan_safe(latest["close"]),
            "rsi14": _nan_safe(latest["rsi14"]),
            "pullback_from_20d_high": _nan_safe(latest["pullback_from_20d_high"]),
            "reason": "not enough indicator history for new-entry gate",
        }

    close = float(latest["close"])
    ema20 = float(latest["ema20"])
    ma50 = float(latest["ma50"])
    ma200 = float(latest["ma200"])
    atr20 = float(latest["atr20"])
    rsi14 = float(latest["rsi14"])
    return_5d = float(latest["return_5d"]) if pd.notna(latest["return_5d"]) else np.nan
    return_5d_q90 = float(latest["return_5d_q90"]) if pd.notna(latest["return_5d_q90"]) else np.nan
    pullback = float(latest["pullback_from_20d_high"])
    volume_5d = float(latest["volume_5d_avg"])
    volume_20d = float(latest["volume_20d_avg"])

    trend_ok = close > ma50 and close > ma200 and float(latest["return_126d"]) > 0
    extended_reasons = []
    if rsi14 > 72:
        extended_reasons.append("RSI14 > 72")
    if close > 1.12 * ema20:
        extended_reasons.append("close > 1.12 * EMA20")
    if close > 1.25 * ma50:
        extended_reasons.append("close > 1.25 * MA50")
    if close > ma50 + 4 * atr20:
        extended_reasons.append("close > MA50 + 4 * ATR20")
    if pd.notna(return_5d_q90) and return_5d > return_5d_q90:
        extended_reasons.append("5d return > 252d 90th percentile")

    volume_pullback_ok = volume_5d <= volume_20d
    healthy_entry = (
        close > ma50
        and 35 <= rsi14 <= 68
        and close <= 1.08 * ema20
        and -0.12 <= pullback <= -0.02
        and volume_pullback_ok
    )
    no_heavy_down_volume = not (
        close < float(latest["previous_close"])
        and float(latest["volume"]) > 1.2 * volume_20d
    )
    ema20_reclaim = (
        float(latest["low"]) <= ema20
        and close >= ema20
        and close > float(latest["previous_high"])
        and no_heavy_down_volume
    )

    if not trend_ok:
        status = "INVALID_TREND"
        new_entry_ok = False
        suggested_weight = 0.0
        reason = "candidate is in rotation list but trend gate failed"
    elif extended_reasons:
        status = "TOO_EXTENDED"
        new_entry_ok = False
        suggested_weight = 0.0
        reason = "; ".join(extended_reasons)
    elif healthy_entry:
        status = "BUY_NOW"
        new_entry_ok = True
        suggested_weight = target_weight * 0.5
        reason = "healthy pullback entry: trend valid, not extended, pullback/volume/RSI acceptable"
    elif ema20_reclaim:
        status = "BUY_NOW"
        new_entry_ok = True
        suggested_weight = target_weight * 0.5
        reason = "EMA20 reclaim entry: trend valid, not extended, price reclaimed EMA20"
    else:
        status = "WAIT_PULLBACK"
        new_entry_ok = False
        suggested_weight = 0.0
        reason = "trend valid but no healthy new-entry trigger; wait for pullback or EMA20 reclaim"

    return {
        "symbol": symbol,
        "rank": rank,
        "signal_date": signal_date,
        "rotation_signal_date": target_date,
        "status": status,
        "new_entry_ok": new_entry_ok,
        "target_weight": target_weight,
        "suggested_initial_weight": suggested_weight,
        "score_126d": score,
        "close": close,
        "rsi14": rsi14,
        "pullback_from_20d_high": pullback,
        "reason": reason,
    }


def _nan_safe(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _target_weights(
    scores: pd.Series,
    top_n: int,
    require_positive: bool,
    excluded_symbols: set[str] | None = None,
) -> pd.Series:
    values = pd.to_numeric(scores, errors="coerce").dropna().sort_values(ascending=False)
    if excluded_symbols:
        values = values.drop(labels=list(excluded_symbols), errors="ignore")
    if require_positive:
        values = values[values > 0]
    target = pd.Series(0.0, index=scores.index)
    if values.empty:
        return target
    selected = values.head(top_n).index
    target.loc[selected] = 1.0 / len(selected)
    return target


def _risk_settings(strategy: dict[str, Any]) -> dict[str, Any]:
    risk = strategy.get("risk", {})
    trailing = risk.get("atr_trailing_stop", {})
    if not isinstance(trailing, dict):
        trailing = {}
    return {
        "stop_loss_pct": risk.get("stop_loss_pct"),
        "ma200_exit": bool(risk.get("ma200_exit", False)),
        "atr_trailing_stop_enabled": bool(trailing.get("enabled", False)),
        "atr_trailing_stop_multiple": float(trailing.get("multiple", 4.0)),
        "atr_trailing_stop_activation_profit_pct": float(trailing.get("activation_profit_pct", 0.10)),
    }


def _sell_symbols_at_open(
    date: pd.Timestamp,
    symbols: list[str],
    open_prices: pd.Series,
    exit_reasons: dict[str, str],
    cash: float,
    shares: pd.Series,
    commission_rate: float,
    slippage_rate: float,
    order_rows: list[dict[str, Any]],
) -> float:
    for symbol in symbols:
        if symbol not in exit_reasons or shares[symbol] <= 0:
            continue
        raw_price = float(open_prices[symbol])
        if raw_price <= 0 or np.isnan(raw_price):
            continue
        fill_price = raw_price * (1.0 - slippage_rate)
        sell_shares = float(shares[symbol])
        sell_value = sell_shares * fill_price
        commission = sell_value * commission_rate
        shares[symbol] = 0.0
        cash += sell_value - commission
        order_rows.append(
            {
                "symbol": symbol,
                "date": date,
                "action": "SELL",
                "price": fill_price,
                "shares": sell_shares,
                "notional": sell_value,
                "commission": commission,
                "reason": exit_reasons[symbol],
            }
        )
    return float(cash)


def _sync_position_state(
    before_shares: pd.Series,
    shares: pd.Series,
    open_prices: pd.Series,
    entry_prices: pd.Series,
    highest_closes: pd.Series,
    slippage_rate: float,
) -> None:
    for symbol in shares.index:
        before = float(before_shares[symbol])
        after = float(shares[symbol])
        if after <= 1e-12:
            shares[symbol] = 0.0
            entry_prices[symbol] = np.nan
            highest_closes[symbol] = np.nan
            continue
        if after > before:
            add_shares = after - max(before, 0.0)
            fill_price = float(open_prices[symbol]) * (1.0 + slippage_rate)
            if before <= 1e-12 or pd.isna(entry_prices[symbol]):
                entry_prices[symbol] = fill_price
            else:
                entry_prices[symbol] = ((before * entry_prices[symbol]) + (add_shares * fill_price)) / after
            if pd.isna(highest_closes[symbol]):
                highest_closes[symbol] = fill_price


def _update_highest_closes(shares: pd.Series, close_prices: pd.Series, highest_closes: pd.Series) -> None:
    for symbol in shares.index:
        if shares[symbol] <= 0:
            continue
        close = float(close_prices[symbol])
        if pd.isna(highest_closes[symbol]):
            highest_closes[symbol] = close
        else:
            highest_closes[symbol] = max(float(highest_closes[symbol]), close)


def _risk_exit_signals(
    shares: pd.Series,
    close_prices: pd.Series,
    entry_prices: pd.Series,
    highest_closes: pd.Series,
    ma200: pd.Series,
    atr20: pd.Series,
    risk_settings: dict[str, Any],
) -> dict[str, str]:
    exits: dict[str, str] = {}
    for symbol in shares.index:
        if shares[symbol] <= 0:
            continue
        close = float(close_prices[symbol])
        entry = float(entry_prices[symbol]) if pd.notna(entry_prices[symbol]) else np.nan
        highest = float(highest_closes[symbol]) if pd.notna(highest_closes[symbol]) else np.nan

        stop_loss_pct = risk_settings["stop_loss_pct"]
        if stop_loss_pct is not None and pd.notna(entry) and close < entry * (1.0 + float(stop_loss_pct)):
            exits[symbol] = "risk_stop_loss"
            continue

        if risk_settings["ma200_exit"] and pd.notna(ma200[symbol]) and close < float(ma200[symbol]):
            exits[symbol] = "risk_ma200_exit"
            continue

        if risk_settings["atr_trailing_stop_enabled"]:
            activation = risk_settings["atr_trailing_stop_activation_profit_pct"]
            multiple = risk_settings["atr_trailing_stop_multiple"]
            if pd.notna(entry) and pd.notna(highest) and pd.notna(atr20[symbol]):
                profit = close / entry - 1.0
                trailing_stop = highest - multiple * float(atr20[symbol])
                if profit >= activation and close < trailing_stop:
                    exits[symbol] = "risk_atr_trailing_stop"
                    continue
    return exits


def _rebalance_at_open(
    date: pd.Timestamp,
    symbols: list[str],
    open_prices: pd.Series,
    target_weights: pd.Series,
    cash: float,
    shares: pd.Series,
    commission_rate: float,
    slippage_rate: float,
    order_rows: list[dict[str, Any]],
    price_type: str = "open",
) -> float:
    equity = _portfolio_value(cash, shares, open_prices)
    min_order_value = max(1.0, equity * 1e-8)
    shares.loc[shares.abs() < 1e-12] = 0.0
    current_values = shares.clip(lower=0.0) * open_prices
    target_values = target_weights.reindex(symbols).fillna(0.0) * equity
    deltas = target_values - current_values

    for symbol in symbols:
        delta = float(deltas[symbol])
        raw_price = float(open_prices[symbol])
        if raw_price <= 0 or np.isnan(raw_price) or delta >= -min_order_value:
            continue
        fill_price = raw_price * (1.0 - slippage_rate)
        sell_value = min(abs(delta), max(float(shares[symbol]), 0.0) * fill_price)
        if sell_value <= min_order_value:
            continue
        sell_shares = sell_value / fill_price
        commission = sell_value * commission_rate
        shares[symbol] = max(0.0, shares[symbol] - sell_shares)
        cash += sell_value - commission
        order_rows.append(
            {
                "symbol": symbol,
                "date": date,
                "action": "SELL",
                "price": fill_price,
                "shares": sell_shares,
                "notional": sell_value,
                "commission": commission,
                "reason": f"rotation_rebalance_{price_type}",
            }
        )

    for symbol in symbols:
        delta = float(deltas[symbol])
        raw_price = float(open_prices[symbol])
        if raw_price <= 0 or np.isnan(raw_price) or delta <= min_order_value:
            continue
        fill_price = raw_price * (1.0 + slippage_rate)
        buy_value = min(delta, cash / (1.0 + commission_rate))
        if buy_value <= min_order_value:
            continue
        buy_shares = buy_value / fill_price
        commission = buy_value * commission_rate
        shares[symbol] += buy_shares
        cash -= buy_value + commission
        order_rows.append(
            {
                "symbol": symbol,
                "date": date,
                "action": "BUY",
                "price": fill_price,
                "shares": buy_shares,
                "notional": buy_value,
                "commission": commission,
                "reason": f"rotation_rebalance_{price_type}",
            }
        )
    return float(cash)


def _portfolio_value(cash: float, shares: pd.Series, prices: pd.Series) -> float:
    return float(cash + (shares * prices).sum())


def _current_weights(shares: pd.Series, prices: pd.Series, equity: float) -> pd.Series:
    if equity <= 0:
        return pd.Series(0.0, index=shares.index)
    return shares * prices / equity


def _metrics(equity_curve: pd.DataFrame, initial_cash: float) -> dict[str, Any]:
    equity = equity_curve["equity"]
    total_return = float(equity.iloc[-1] / initial_cash - 1.0)
    drawdown = equity / equity.cummax() - 1.0
    returns = equity.pct_change().dropna()
    sharpe = None
    if not returns.empty and returns.std(ddof=0) > 0:
        sharpe = float((returns.mean() / returns.std(ddof=0)) * np.sqrt(252))
    return {
        "initial_cash": initial_cash,
        "final_equity": float(equity.iloc[-1]),
        "net_profit": float(equity.iloc[-1] - initial_cash),
        "total_return": total_return,
        "max_drawdown": float(drawdown.min()),
        "sharpe": sharpe,
        "exposure_time": float((equity_curve["open_positions"] > 0).mean()),
    }


def _equal_weight_benchmark(
    close: pd.DataFrame,
    initial_cash: float,
    commission_rate: float,
    slippage_rate: float,
) -> dict[str, Any]:
    first_prices = close.iloc[0]
    valid = first_prices[first_prices > 0].index
    if len(valid) == 0:
        raise RotationBacktestError("Benchmark cannot be calculated because first close prices are invalid")

    normalized = close.loc[:, valid] / first_prices.loc[valid]
    equity = initial_cash * normalized.mean(axis=1)
    drawdown = equity / equity.cummax() - 1.0
    total_return = float(equity.iloc[-1] / initial_cash - 1.0)
    daily = equity.pct_change().dropna()
    sharpe = None
    if not daily.empty and daily.std(ddof=0) > 0:
        sharpe = float((daily.mean() / daily.std(ddof=0)) * np.sqrt(252))
    return {
        "name": "equal_weight_buy_hold",
        "initial_cash": initial_cash,
        "final_equity": float(equity.iloc[-1]),
        "net_profit": float(equity.iloc[-1] - initial_cash),
        "total_return": total_return,
        "max_drawdown": float(drawdown.min()),
        "sharpe": sharpe,
        "total_commission": 0.0,
    }
