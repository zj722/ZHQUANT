from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .compiler import CompiledSignals, CompilerError, StatefulRule, compile_strategy


class BacktestError(ValueError):
    """Raised when a strategy cannot be backtested."""


@dataclass
class Position:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    entry_commission: float
    signal_date: pd.Timestamp
    add_count: int = 0


@dataclass(frozen=True)
class BacktestResult:
    strategy_name: str
    equity_curve: pd.DataFrame
    order_log: pd.DataFrame
    trade_log: pd.DataFrame
    metrics: dict[str, Any]
    compiled: CompiledSignals
    current_actions: pd.DataFrame
    diagnostic_log: pd.DataFrame
    diagnostic_pass_rates: dict[str, float]
    benchmark_metrics: dict[str, Any] | None = None
    score: dict[str, Any] | None = None


class LongOnlyBacktester:
    """Minimal daily long-only backtester.

    Semantics:
    - Signals are evaluated after a daily close.
    - Orders execute at the next daily open.
    - Position sizing uses a fixed fraction of current equity.
    - Only long positions are supported.
    """

    def __init__(
        self,
        strategy: dict[str, Any],
        market_data: dict[str, pd.DataFrame],
        initial_cash: float = 100_000.0,
        events: dict[str, Any] | None = None,
        force_close: bool = True,
    ) -> None:
        if initial_cash <= 0:
            raise BacktestError("initial_cash must be positive")

        self.strategy = strategy
        self.market_data = {symbol.upper(): df.sort_index().copy() for symbol, df in market_data.items()}
        self.initial_cash = float(initial_cash)
        self.events = events
        self.force_close = force_close

        self.compiled = compile_strategy(strategy, self.market_data, events=events)
        self.symbols = list(self.compiled.entry.columns)
        self.index = self.compiled.entry.index

        self.risk = strategy["risk"]
        self.max_position_pct = float(self.risk["max_position_pct"])
        self.max_positions = int(self.risk["max_positions"])
        self.slippage_rate = float(self.risk["slippage_bps"]) / 10_000.0
        self.commission_rate = float(self.risk["commission_bps"]) / 10_000.0
        self.stop_loss_pct = self.risk.get("stop_loss_pct")
        self.take_profit_pct = self.risk.get("take_profit_pct")
        self.max_additions = int(self.risk.get("max_additions", 0))
        self.add_position_pct = float(self.risk.get("add_position_pct", 0.0))
        self.reduce_position_pct = float(self.risk.get("reduce_position_pct", 0.0))
        self.trim_position_pct = float(self.risk.get("trim_position_pct", 0.0))

        self.open_prices = self._price_matrix("open")
        self.close_prices = self._price_matrix("close")

        self.cash = self.initial_cash
        self.positions: dict[str, Position] = {}
        self.order_rows: list[dict[str, Any]] = []
        self.trade_rows: list[dict[str, Any]] = []
        self.equity_rows: list[dict[str, Any]] = []
        self.current_actions = pd.DataFrame()
        self.diagnostic_log = self._diagnostic_log()
        self.diagnostic_pass_rates = self._diagnostic_pass_rates(self.diagnostic_log)

    def run(self) -> BacktestResult:
        pending_entries: list[str] = []
        pending_additions: list[str] = []
        pending_exits: dict[str, str] = {}
        pending_partial_exits: dict[str, tuple[str, float]] = {}

        for idx, date in enumerate(self.index):
            self._execute_exits(date, pending_exits)
            self._execute_partial_exits(date, pending_partial_exits)
            self._execute_additions(date, pending_additions)
            self._execute_entries(date, pending_entries)

            equity = self._portfolio_value(date, price_type="close")
            self.equity_rows.append(
                {
                    "date": date,
                    "cash": self.cash,
                    "equity": equity,
                    "open_positions": len(self.positions),
                }
            )

            if idx == len(self.index) - 1:
                self.current_actions = self._current_actions(date)
                pending_entries = []
                pending_additions = []
                pending_exits = {}
                pending_partial_exits = {}
                continue

            pending_exits = self._collect_exit_orders(date)
            pending_partial_exits = self._collect_partial_exit_orders(date, pending_exits)
            pending_additions = self._collect_add_orders(date, pending_exits)
            pending_entries = self._collect_entry_orders(date)

        if self.force_close and self.positions:
            final_date = self.index[-1]
            for symbol in list(self.positions):
                self._close_position(symbol, final_date, price_type="close", reason="force_close")
            self.equity_rows.append(
                {
                    "date": final_date,
                    "cash": self.cash,
                    "equity": self.cash,
                    "open_positions": 0,
                }
            )

        equity_curve = pd.DataFrame(self.equity_rows).drop_duplicates("date", keep="last").set_index("date")
        order_log = pd.DataFrame(self.order_rows)
        trade_log = pd.DataFrame(self.trade_rows)
        metrics = self._metrics(equity_curve, trade_log)
        benchmark_metrics = self._benchmark_metrics() if len(self.symbols) == 1 else None
        score = score_strategy(metrics, benchmark_metrics)

        return BacktestResult(
            strategy_name=self.strategy["name"],
            equity_curve=equity_curve,
            order_log=order_log,
            trade_log=trade_log,
            metrics=metrics,
            compiled=self.compiled,
            current_actions=self.current_actions,
            diagnostic_log=self.diagnostic_log,
            diagnostic_pass_rates=self.diagnostic_pass_rates,
            benchmark_metrics=benchmark_metrics,
            score=score,
        )

    def _current_actions(self, signal_date: pd.Timestamp) -> pd.DataFrame:
        exit_orders = self._collect_exit_orders(signal_date)
        partial_exit_orders = self._collect_partial_exit_orders(signal_date, exit_orders)
        entry_signals = self._collect_entry_orders(signal_date)
        add_signals = self._collect_add_orders(signal_date, exit_orders)
        available_slots = max(self.max_positions - len(self.positions), 0)
        actionable_entries = set(entry_signals[:available_slots])
        blocked_entries = set(entry_signals[available_slots:])

        rows: list[dict[str, Any]] = []
        for symbol in self.symbols:
            in_position = symbol in self.positions
            entry_signal = bool(self.compiled.entry.loc[signal_date, symbol])
            exit_signal = symbol in exit_orders

            action = "HOLD"
            reason = "in_position" if in_position else "no_signal"
            if exit_signal:
                action = "SELL"
                reason = exit_orders[symbol]
            elif symbol in partial_exit_orders:
                action = "REDUCE" if partial_exit_orders[symbol][1] >= 0.5 else "TRIM"
                reason = partial_exit_orders[symbol][0]
            elif symbol in add_signals:
                action = "ADD"
                reason = "add_entry_signal"
            elif symbol in actionable_entries:
                action = "BUY"
                reason = "entry_signal"
            elif symbol in blocked_entries:
                reason = "max_positions_reached"

            rows.append(
                {
                    "symbol": symbol,
                    "signal_date": signal_date,
                    "action": action,
                    "reason": reason,
                    "in_position": in_position,
                    "entry_signal": entry_signal,
                    "add_entry_signal": symbol in add_signals,
                    "exit_signal": exit_signal,
                    "execution": "next_open",
                }
            )

        return pd.DataFrame(rows)

    def _diagnostic_log(self) -> pd.DataFrame:
        if not self.compiled.diagnostics:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for date in self.index:
            for symbol in self.symbols:
                row: dict[str, Any] = {
                    "symbol": symbol,
                    "signal_date": date,
                }
                for name, frame in self.compiled.diagnostics.items():
                    row[name] = bool(frame.loc[date, symbol])
                rows.append(row)

        return pd.DataFrame(rows)

    def _diagnostic_pass_rates(self, diagnostic_log: pd.DataFrame) -> dict[str, float]:
        if diagnostic_log.empty:
            return {}
        ignored = {"symbol", "signal_date"}
        return {
            column: float(diagnostic_log[column].mean())
            for column in diagnostic_log.columns
            if column not in ignored
        }

    def _collect_exit_orders(self, signal_date: pd.Timestamp) -> dict[str, str]:
        exits: dict[str, str] = {}

        for symbol in list(self.positions):
            if bool(self.compiled.exit.loc[signal_date, symbol]):
                exits[symbol] = "exit_signal"

        for rule in self.compiled.stateful_exit_rules:
            for symbol in list(self.positions):
                if symbol in exits:
                    continue
                if self._eval_stateful_condition(rule.rule, rule.path, symbol, signal_date):
                    exits[symbol] = self._stateful_reason(rule)

        for symbol, position in list(self.positions.items()):
            if symbol in exits:
                continue
            position_return = self._position_return(position, signal_date)
            if self.stop_loss_pct is not None and position_return <= float(self.stop_loss_pct):
                exits[symbol] = "stop_loss"
            elif self.take_profit_pct is not None and position_return >= float(self.take_profit_pct):
                exits[symbol] = "take_profit"

        return exits

    def _collect_entry_orders(self, signal_date: pd.Timestamp) -> list[str]:
        orders: list[str] = []
        for symbol in self.symbols:
            if symbol in self.positions:
                continue
            if bool(self.compiled.entry.loc[signal_date, symbol]):
                orders.append(symbol)
        return orders

    def _collect_add_orders(self, signal_date: pd.Timestamp, pending_exits: dict[str, str]) -> list[str]:
        if self.compiled.add_entry is None or self.max_additions <= 0 or self.add_position_pct <= 0:
            return []

        orders: list[str] = []
        for symbol, position in self.positions.items():
            if symbol in pending_exits:
                continue
            if position.add_count >= self.max_additions:
                continue
            if bool(self.compiled.add_entry.loc[signal_date, symbol]):
                orders.append(symbol)
        return orders

    def _collect_partial_exit_orders(self, signal_date: pd.Timestamp, pending_exits: dict[str, str]) -> dict[str, tuple[str, float]]:
        orders: dict[str, tuple[str, float]] = {}
        for symbol in self.positions:
            if symbol in pending_exits:
                continue
            if (
                self.compiled.reduce_exit is not None
                and self.reduce_position_pct > 0
                and bool(self.compiled.reduce_exit.loc[signal_date, symbol])
            ):
                orders[symbol] = ("reduce_exit", self.reduce_position_pct)
                continue
            if (
                self.compiled.trim_exit is not None
                and self.trim_position_pct > 0
                and bool(self.compiled.trim_exit.loc[signal_date, symbol])
            ):
                orders[symbol] = ("trim_exit", self.trim_position_pct)
        return orders

    def _execute_exits(self, date: pd.Timestamp, exits: dict[str, str]) -> None:
        for symbol, reason in list(exits.items()):
            if symbol in self.positions:
                self._close_position(symbol, date, price_type="open", reason=reason)

    def _execute_partial_exits(self, date: pd.Timestamp, partial_exits: dict[str, tuple[str, float]]) -> None:
        for symbol, (reason, pct) in list(partial_exits.items()):
            if symbol in self.positions:
                self._partial_close_position(symbol, date, pct=pct, reason=reason)

    def _execute_entries(self, date: pd.Timestamp, entries: list[str]) -> None:
        for symbol in entries:
            if symbol in self.positions:
                continue
            if len(self.positions) >= self.max_positions:
                break

            self._buy_position(symbol, date, self.max_position_pct, is_addition=False)

    def _execute_additions(self, date: pd.Timestamp, additions: list[str]) -> None:
        for symbol in additions:
            if symbol not in self.positions:
                continue
            self._buy_position(symbol, date, self.add_position_pct, is_addition=True)

    def _buy_position(self, symbol: str, date: pd.Timestamp, position_pct: float, is_addition: bool) -> None:
        equity = self._portfolio_value(date, price_type="open")
        target_notional = equity * position_pct
        max_affordable = self.cash / (1.0 + self.commission_rate)
        notional = min(target_notional, max_affordable)
        if notional <= 0:
            return

        raw_open = self._price(symbol, date, "open")
        if raw_open <= 0 or np.isnan(raw_open):
            return

        fill_price = raw_open * (1.0 + self.slippage_rate)
        shares = notional / fill_price
        commission = notional * self.commission_rate
        total_cost = notional + commission
        if total_cost > self.cash + 1e-9:
            return

        self.cash -= total_cost

        if is_addition:
            position = self.positions[symbol]
            old_notional = position.shares * position.entry_price
            new_notional = shares * fill_price
            total_shares = position.shares + shares
            position.entry_price = (old_notional + new_notional) / total_shares
            position.shares = total_shares
            position.entry_commission += commission
            position.add_count += 1
            self._record_order(
                symbol=symbol,
                date=date,
                action="ADD",
                price=fill_price,
                shares=shares,
                notional=notional,
                commission=commission,
                reason="add_entry_signal",
            )
            return

        self.positions[symbol] = Position(
            symbol=symbol,
            entry_date=date,
            entry_price=fill_price,
            shares=shares,
            entry_commission=commission,
            signal_date=date,
        )
        self._record_order(
            symbol=symbol,
            date=date,
            action="BUY",
            price=fill_price,
            shares=shares,
            notional=notional,
            commission=commission,
            reason="entry_signal",
        )

    def _close_position(self, symbol: str, date: pd.Timestamp, price_type: str, reason: str) -> None:
        position = self.positions.pop(symbol)
        raw_price = self._price(symbol, date, price_type)
        if raw_price <= 0 or np.isnan(raw_price):
            raise BacktestError(f"Cannot close {symbol} on {date}: invalid {price_type} price")

        exit_price = raw_price * (1.0 - self.slippage_rate)
        gross_proceeds = position.shares * exit_price
        exit_commission = gross_proceeds * self.commission_rate
        net_proceeds = gross_proceeds - exit_commission
        self.cash += net_proceeds
        self._record_order(
            symbol=symbol,
            date=date,
            action="SELL",
            price=exit_price,
            shares=position.shares,
            notional=gross_proceeds,
            commission=exit_commission,
            reason=reason,
        )

        entry_notional = position.shares * position.entry_price
        total_commission = position.entry_commission + exit_commission
        gross_pnl = gross_proceeds - entry_notional
        net_pnl = gross_pnl - total_commission
        net_return = net_pnl / entry_notional if entry_notional else 0.0

        self.trade_rows.append(
            {
                "symbol": symbol,
                "entry_date": position.entry_date,
                "exit_date": date,
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "shares": position.shares,
                "entry_notional": entry_notional,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "net_return": net_return,
                "entry_commission": position.entry_commission,
                "exit_commission": exit_commission,
                "total_commission": total_commission,
                "holding_days": self._holding_days(position, date),
                "add_count": position.add_count,
                "exit_reason": reason,
            }
        )

    def _partial_close_position(self, symbol: str, date: pd.Timestamp, pct: float, reason: str) -> None:
        position = self.positions[symbol]
        pct = max(0.0, min(float(pct), 1.0))
        shares_to_sell = position.shares * pct
        if shares_to_sell <= 0:
            return
        if shares_to_sell >= position.shares * 0.999:
            self._close_position(symbol, date, price_type="open", reason=reason)
            return

        raw_price = self._price(symbol, date, "open")
        if raw_price <= 0 or np.isnan(raw_price):
            raise BacktestError(f"Cannot reduce {symbol} on {date}: invalid open price")

        exit_price = raw_price * (1.0 - self.slippage_rate)
        gross_proceeds = shares_to_sell * exit_price
        exit_commission = gross_proceeds * self.commission_rate
        net_proceeds = gross_proceeds - exit_commission
        self.cash += net_proceeds

        entry_notional = shares_to_sell * position.entry_price
        allocated_entry_commission = position.entry_commission * (shares_to_sell / position.shares)
        position.entry_commission -= allocated_entry_commission
        total_commission = allocated_entry_commission + exit_commission
        gross_pnl = gross_proceeds - entry_notional
        net_pnl = gross_pnl - total_commission
        net_return = net_pnl / entry_notional if entry_notional else 0.0
        position.shares -= shares_to_sell

        action = "REDUCE" if pct >= 0.5 else "TRIM"
        self._record_order(
            symbol=symbol,
            date=date,
            action=action,
            price=exit_price,
            shares=shares_to_sell,
            notional=gross_proceeds,
            commission=exit_commission,
            reason=reason,
        )
        self.trade_rows.append(
            {
                "symbol": symbol,
                "entry_date": position.entry_date,
                "exit_date": date,
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "shares": shares_to_sell,
                "entry_notional": entry_notional,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "net_return": net_return,
                "entry_commission": allocated_entry_commission,
                "exit_commission": exit_commission,
                "total_commission": total_commission,
                "holding_days": self._holding_days(position, date),
                "add_count": position.add_count,
                "exit_reason": reason,
            }
        )

    def _record_order(
        self,
        symbol: str,
        date: pd.Timestamp,
        action: str,
        price: float,
        shares: float,
        notional: float,
        commission: float,
        reason: str,
    ) -> None:
        self.order_rows.append(
            {
                "symbol": symbol,
                "date": date,
                "action": action,
                "price": price,
                "shares": shares,
                "notional": notional,
                "commission": commission,
                "reason": reason,
            }
        )

    def _eval_stateful_condition(self, node: dict[str, Any], path: str, symbol: str, date: pd.Timestamp) -> bool:
        if "all" in node:
            return all(self._eval_stateful_condition(child, f"{path}.all[{idx}]", symbol, date) for idx, child in enumerate(node["all"]))
        if "any" in node:
            return any(self._eval_stateful_condition(child, f"{path}.any[{idx}]", symbol, date) for idx, child in enumerate(node["any"]))
        if "not" in node:
            return not self._eval_stateful_condition(node["not"], f"{path}.not", symbol, date)

        left = self._eval_stateful_operand(node["left"], f"{path}.left", symbol, date)
        right = self._eval_stateful_operand(node["right"], f"{path}.right", symbol, date)
        op = node["op"]
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        raise BacktestError(f"{path}.op: unsupported stateful comparison {op}")

    def _eval_stateful_operand(self, value: Any, path: str, symbol: str, date: pd.Timestamp) -> float:
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            return self._stateful_source_value(symbol, value, date)
        if isinstance(value, dict) and "indicator" in value:
            return self._stateful_indicator_value(value, path, symbol, date)
        if isinstance(value, dict) and "field" in value:
            position = self.positions[symbol]
            field = value["field"]
            if field == "holding_days":
                return float(self._holding_days(position, date))
            if field == "position_return":
                return self._position_return(position, date)
            if field == "current_weight":
                equity = self._portfolio_value(date, price_type="close")
                position_value = position.shares * self._price(symbol, date, "close")
                return position_value / equity if equity else 0.0
            if field == "cash_pct":
                equity = self._portfolio_value(date, price_type="close")
                return self.cash / equity if equity else 0.0
            if field == "entry_price":
                return position.entry_price
            if field == "highest_close_since_entry":
                return self._highest_close_since_entry(position, date)
        if isinstance(value, dict) and "op" in value:
            return self._eval_stateful_math(value, path, symbol, date)
        raise BacktestError(f"{path}: unsupported stateful operand")

    def _stateful_indicator_value(self, spec: dict[str, Any], path: str, symbol: str, date: pd.Timestamp) -> float:
        indicator = spec["indicator"]
        data_symbol = str(spec.get("symbol", symbol)).upper()
        shift = int(spec.get("shift", 0))
        eval_date = self._shifted_date(date, shift)

        if indicator in {"open", "high", "low", "close", "volume"}:
            return self._stateful_source_value(data_symbol, indicator, eval_date)

        if indicator == "rolling_count":
            window = int(spec["window"])
            matching_dates = self.index[self.index <= eval_date]
            if len(matching_dates) < window:
                return float("nan")
            return float(
                sum(
                    self._eval_stateful_condition(spec["condition"], f"{path}.condition", symbol, window_date)
                    for window_date in matching_dates[-window:]
                )
            )

        source_name = spec.get("source", "close")
        source = self._stateful_source_series(data_symbol, source_name).loc[:eval_date]
        window = int(spec["window"])
        if len(source) < window:
            return float("nan")

        if indicator == "sma":
            return float(source.tail(window).mean())
        if indicator == "ema":
            return float(source.ewm(span=window, min_periods=window, adjust=False).mean().iloc[-1])
        if indicator == "return":
            return float(source.pct_change(window).iloc[-1])
        if indicator == "rolling_max":
            return float(source.tail(window).max())
        if indicator == "rolling_min":
            return float(source.tail(window).min())
        if indicator == "zscore":
            sample = source.tail(window)
            std = sample.std(ddof=0)
            return float((sample.iloc[-1] - sample.mean()) / std) if std else float("nan")
        if indicator == "rsi":
            return self._stateful_rsi_value(source, window)
        if indicator == "atr":
            return self._stateful_atr_value(data_symbol, eval_date, window)

        raise BacktestError(f"{path}.indicator: unsupported stateful indicator {indicator}")

    def _eval_stateful_math(self, node: dict[str, Any], path: str, symbol: str, date: pd.Timestamp) -> float:
        op = node["op"]
        if op == "abs":
            return abs(self._eval_stateful_operand(node["value"], f"{path}.value", symbol, date))

        left = self._eval_stateful_operand(node["left"], f"{path}.left", symbol, date)
        right = self._eval_stateful_operand(node["right"], f"{path}.right", symbol, date)
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return left / right
        if op == "min":
            return min(left, right)
        if op == "max":
            return max(left, right)
        raise BacktestError(f"{path}.op: unsupported stateful math op {op}")

    def _stateful_reason(self, rule: StatefulRule) -> str:
        text = str(rule.rule)
        if "holding_days" in text:
            return "holding_days"
        if "position_return" in text:
            return "position_return"
        return "stateful_exit"

    def _holding_days(self, position: Position, date: pd.Timestamp) -> int:
        entry_idx = self.index.get_loc(position.entry_date)
        current_idx = self.index.get_loc(date)
        return int(current_idx - entry_idx)

    def _highest_close_since_entry(self, position: Position, date: pd.Timestamp) -> float:
        close_series = self.close_prices[position.symbol].loc[position.entry_date : date]
        return float(close_series.max())

    def _position_return(self, position: Position, date: pd.Timestamp) -> float:
        close_price = self._price(position.symbol, date, "close")
        return (close_price - position.entry_price) / position.entry_price

    def _shifted_date(self, date: pd.Timestamp, shift: int) -> pd.Timestamp:
        if shift == 0:
            return date
        current_idx = self.index.get_loc(date)
        shifted_idx = current_idx - shift
        if shifted_idx < 0:
            return self.index[0]
        return self.index[shifted_idx]

    def _stateful_source_series(self, symbol: str, source: str) -> pd.Series:
        symbol = symbol.upper()
        if symbol not in self.market_data:
            raise BacktestError(f"Missing market data for referenced symbol: {symbol}")
        return self.market_data[symbol][source].reindex(self.index)

    def _stateful_source_value(self, symbol: str, source: str, date: pd.Timestamp) -> float:
        return float(self._stateful_source_series(symbol, source).loc[date])

    def _stateful_rsi_value(self, source: pd.Series, window: int) -> float:
        delta = source.diff()
        gains = delta.clip(lower=0).rolling(window, min_periods=window).mean()
        losses = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
        rs = gains / losses.replace(0, np.nan)
        return float((100 - (100 / (1 + rs))).iloc[-1])

    def _stateful_atr_value(self, symbol: str, date: pd.Timestamp, window: int) -> float:
        high = self._stateful_source_series(symbol, "high").loc[:date]
        low = self._stateful_source_series(symbol, "low").loc[:date]
        close = self._stateful_source_series(symbol, "close").loc[:date]
        if len(close) < window:
            return float("nan")
        prev_close = close.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return float(true_range.rolling(window, min_periods=window).mean().iloc[-1])

    def _portfolio_value(self, date: pd.Timestamp, price_type: str) -> float:
        value = self.cash
        for symbol, position in self.positions.items():
            price = self._price(symbol, date, price_type)
            value += position.shares * price
        return float(value)

    def _price_matrix(self, field: str) -> pd.DataFrame:
        data = {
            symbol: self.market_data[symbol][field].reindex(self.index)
            for symbol in self.symbols
        }
        return pd.DataFrame(data, index=self.index)

    def _price(self, symbol: str, date: pd.Timestamp, field: str) -> float:
        matrix = self.open_prices if field == "open" else self.close_prices
        return float(matrix.loc[date, symbol])

    def _metrics(self, equity_curve: pd.DataFrame, trade_log: pd.DataFrame) -> dict[str, Any]:
        if equity_curve.empty:
            return {}

        equity = equity_curve["equity"]
        total_return = equity.iloc[-1] / self.initial_cash - 1.0
        daily_returns = equity.pct_change().dropna()
        sharpe = None
        if not daily_returns.empty and daily_returns.std(ddof=0) > 0:
            sharpe = float((daily_returns.mean() / daily_returns.std(ddof=0)) * np.sqrt(252))
        downside = daily_returns[daily_returns < 0]
        sortino = None
        if not downside.empty and downside.std(ddof=0) > 0:
            sortino = float((daily_returns.mean() / downside.std(ddof=0)) * np.sqrt(252))

        drawdown = equity / equity.cummax() - 1.0
        max_drawdown = float(drawdown.min())

        elapsed_days = max((equity.index[-1] - equity.index[0]).days, 1)
        years = elapsed_days / 365.25
        cagr = float((equity.iloc[-1] / self.initial_cash) ** (1 / years) - 1) if years > 0 else None
        calmar = cagr / abs(max_drawdown) if cagr is not None and max_drawdown < 0 else None
        exposure_time = float((equity_curve["open_positions"] > 0).mean()) if "open_positions" in equity_curve else None
        monthly_returns = equity.resample("ME").last().pct_change().dropna()

        metrics: dict[str, Any] = {
            "initial_cash": self.initial_cash,
            "final_equity": float(equity.iloc[-1]),
            "net_profit": float(equity.iloc[-1] - self.initial_cash),
            "total_return": float(total_return),
            "cagr": cagr,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": max_drawdown,
            "exposure_time": exposure_time,
            "monthly_returns": {idx.strftime("%Y-%m"): float(value) for idx, value in monthly_returns.items()},
            "trade_count": int(len(trade_log)),
            "trades_per_year": float(len(trade_log) / years) if years > 0 else None,
        }

        if trade_log.empty:
            metrics.update(
                {
                    "win_rate": None,
                    "avg_return_per_trade": None,
                    "avg_pnl_per_trade": None,
                    "avg_win": None,
                    "avg_loss": None,
                    "gross_profit": 0.0,
                    "gross_loss": 0.0,
                    "profit_factor": None,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "total_commission": 0.0,
                    "avg_holding_days": None,
                    "median_trade_return": None,
                    "max_consecutive_losses": 0,
                    "worst_5_trades": [],
                    "best_trade": None,
                    "worst_trade": None,
                }
            )
            return metrics

        returns = trade_log["net_return"]
        pnl = trade_log["net_pnl"]
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        winning_pnl = pnl[pnl > 0]
        losing_pnl = pnl[pnl < 0]
        gross_profit = float(winning_pnl.sum()) if not winning_pnl.empty else 0.0
        gross_loss = float(abs(losing_pnl.sum())) if not losing_pnl.empty else 0.0
        metrics.update(
            {
                "win_rate": float((returns > 0).mean()),
                "avg_return_per_trade": float(returns.mean()),
                "avg_pnl_per_trade": float(pnl.mean()),
                "avg_win": float(wins.mean()) if not wins.empty else None,
                "avg_loss": float(losses.mean()) if not losses.empty else None,
                "gross_profit": gross_profit,
                "gross_loss": gross_loss,
                "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
                "winning_trades": int((pnl > 0).sum()),
                "losing_trades": int((pnl < 0).sum()),
                "total_commission": float(trade_log["total_commission"].sum()),
                "avg_holding_days": float(trade_log["holding_days"].mean()),
                "median_trade_return": float(returns.median()),
                "max_consecutive_losses": _max_consecutive_losses(pnl),
                "worst_5_trades": [
                    {
                        "symbol": str(row["symbol"]),
                        "exit_date": str(pd.Timestamp(row["exit_date"]).date()),
                        "net_pnl": float(row["net_pnl"]),
                        "net_return": float(row["net_return"]),
                    }
                    for _, row in trade_log.nsmallest(min(5, len(trade_log)), "net_return").iterrows()
                ],
                "best_trade": float(returns.max()),
                "worst_trade": float(returns.min()),
            }
        )
        return metrics

    def _benchmark_metrics(self) -> dict[str, Any]:
        symbol = self.symbols[0]
        open_prices = self.open_prices[symbol].dropna()
        close_prices = self.close_prices[symbol].dropna()
        shared_index = open_prices.index.intersection(close_prices.index)
        if shared_index.empty:
            return {}

        first_date = shared_index[0]
        first_open = float(open_prices.loc[first_date])
        if first_open <= 0:
            return {}

        entry_price = first_open * (1.0 + self.slippage_rate)
        max_notional = self.initial_cash / (1.0 + self.commission_rate)
        shares = max_notional / entry_price
        entry_commission = max_notional * self.commission_rate
        cash = self.initial_cash - max_notional - entry_commission

        close_series = close_prices.reindex(shared_index)
        equity = cash + shares * close_series
        final_exit_price = float(close_series.iloc[-1]) * (1.0 - self.slippage_rate)
        final_proceeds = shares * final_exit_price
        exit_commission = final_proceeds * self.commission_rate
        final_equity_after_exit = cash + final_proceeds - exit_commission
        equity.iloc[-1] = final_equity_after_exit

        total_return = final_equity_after_exit / self.initial_cash - 1.0
        drawdown = equity / equity.cummax() - 1.0
        daily_returns = equity.pct_change().dropna()
        sharpe = None
        if not daily_returns.empty and daily_returns.std(ddof=0) > 0:
            sharpe = float((daily_returns.mean() / daily_returns.std(ddof=0)) * np.sqrt(252))

        return {
            "symbol": symbol,
            "initial_cash": self.initial_cash,
            "final_equity": float(final_equity_after_exit),
            "net_profit": float(final_equity_after_exit - self.initial_cash),
            "total_return": float(total_return),
            "max_drawdown": float(drawdown.min()),
            "sharpe": sharpe,
            "total_commission": float(entry_commission + exit_commission),
        }


def score_strategy(metrics: dict[str, Any], benchmark_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    total_return = _metric(metrics, "total_return")
    max_drawdown = _metric(metrics, "max_drawdown")
    sharpe = _metric(metrics, "sharpe")
    trade_count = int(metrics.get("trade_count") or 0)
    exposure = _metric(metrics, "exposure_time")
    benchmark_return = _metric(benchmark_metrics or {}, "total_return")
    excess_return = total_return - benchmark_return if benchmark_return is not None else None

    return_score = _clamp(total_return / 0.25, -1, 1) * 25 if total_return is not None else 0
    drawdown_score = _clamp((0.30 + max_drawdown) / 0.30, 0, 1) * 20 if max_drawdown is not None else 0
    sharpe_score = _clamp((sharpe or 0) / 2.0, -1, 1) * 20
    trade_count_score = _clamp(trade_count / 20, 0, 1) * 15
    benchmark_score = _clamp((excess_return or 0) / 0.15, -1, 1) * 15 if excess_return is not None else 0
    exposure_bonus = 5 if exposure is not None and 0 < exposure < 0.75 and total_return and total_return > 0 else 0
    score = return_score + drawdown_score + sharpe_score + trade_count_score + benchmark_score + exposure_bonus
    score = float(_clamp(score, 0, 100))

    reasons: list[str] = []
    failures: list[str] = []
    _reason(total_return is not None and total_return > 0, "Return positive", reasons, failures)
    _reason(max_drawdown is not None and max_drawdown >= -0.25, "Max drawdown acceptable", reasons, failures)
    _reason(trade_count >= 3, "Trade count sufficient", reasons, failures)
    if benchmark_return is not None:
        _reason(excess_return is not None and excess_return > 0, "Beats buy and hold", reasons, failures)
    if sharpe is not None:
        _reason(sharpe > 0.5, "Sharpe above threshold", reasons, failures)

    verdict = "PASS" if score >= 60 and not failures else "FAIL"
    return {
        "score": score,
        "verdict": verdict,
        "reasons": reasons,
        "failures": failures,
        "components": {
            "return_score": float(return_score),
            "drawdown_score": float(drawdown_score),
            "sharpe_score": float(sharpe_score),
            "trade_count_score": float(trade_count_score),
            "benchmark_score": float(benchmark_score),
            "exposure_bonus": float(exposure_bonus),
            "complexity_penalty": 0.0,
        },
        "excess_return": excess_return,
    }


def _metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    return float(value)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _reason(condition: bool, message: str, reasons: list[str], failures: list[str]) -> None:
    if condition:
        reasons.append(message)
    else:
        failures.append(message)


def _max_consecutive_losses(pnl: pd.Series) -> int:
    max_streak = 0
    current = 0
    for value in pnl:
        if value < 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def run_backtest(
    strategy: dict[str, Any],
    market_data: dict[str, pd.DataFrame],
    initial_cash: float = 100_000.0,
    events: dict[str, Any] | None = None,
) -> BacktestResult:
    try:
        return LongOnlyBacktester(
            strategy=strategy,
            market_data=market_data,
            initial_cash=initial_cash,
            events=events,
        ).run()
    except CompilerError as exc:
        raise BacktestError(str(exc)) from exc
