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


@dataclass(frozen=True)
class BacktestResult:
    strategy_name: str
    equity_curve: pd.DataFrame
    trade_log: pd.DataFrame
    metrics: dict[str, float | int | None]
    compiled: CompiledSignals


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

        self.open_prices = self._price_matrix("open")
        self.close_prices = self._price_matrix("close")

        self.cash = self.initial_cash
        self.positions: dict[str, Position] = {}
        self.trade_rows: list[dict[str, Any]] = []
        self.equity_rows: list[dict[str, Any]] = []

    def run(self) -> BacktestResult:
        pending_entries: list[str] = []
        pending_exits: dict[str, str] = {}

        for idx, date in enumerate(self.index):
            self._execute_exits(date, pending_exits)
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
                pending_entries = []
                pending_exits = {}
                continue

            pending_exits = self._collect_exit_orders(date)
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
        trade_log = pd.DataFrame(self.trade_rows)
        metrics = self._metrics(equity_curve, trade_log)

        return BacktestResult(
            strategy_name=self.strategy["name"],
            equity_curve=equity_curve,
            trade_log=trade_log,
            metrics=metrics,
            compiled=self.compiled,
        )

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

    def _execute_exits(self, date: pd.Timestamp, exits: dict[str, str]) -> None:
        for symbol, reason in list(exits.items()):
            if symbol in self.positions:
                self._close_position(symbol, date, price_type="open", reason=reason)

    def _execute_entries(self, date: pd.Timestamp, entries: list[str]) -> None:
        for symbol in entries:
            if symbol in self.positions:
                continue
            if len(self.positions) >= self.max_positions:
                break

            equity = self._portfolio_value(date, price_type="open")
            target_notional = equity * self.max_position_pct
            max_affordable = self.cash / (1.0 + self.commission_rate)
            notional = min(target_notional, max_affordable)
            if notional <= 0:
                continue

            raw_open = self._price(symbol, date, "open")
            if raw_open <= 0 or np.isnan(raw_open):
                continue

            fill_price = raw_open * (1.0 + self.slippage_rate)
            shares = notional / fill_price
            commission = notional * self.commission_rate
            total_cost = notional + commission
            if total_cost > self.cash + 1e-9:
                continue

            self.cash -= total_cost
            self.positions[symbol] = Position(
                symbol=symbol,
                entry_date=date,
                entry_price=fill_price,
                shares=shares,
                entry_commission=commission,
                signal_date=date,
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
                "exit_reason": reason,
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
        if isinstance(value, dict) and "op" in value:
            return self._eval_stateful_math(value, path, symbol, date)
        raise BacktestError(f"{path}: unsupported stateful operand")

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

    def _position_return(self, position: Position, date: pd.Timestamp) -> float:
        close_price = self._price(position.symbol, date, "close")
        return (close_price - position.entry_price) / position.entry_price

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

    def _metrics(self, equity_curve: pd.DataFrame, trade_log: pd.DataFrame) -> dict[str, float | int | None]:
        if equity_curve.empty:
            return {}

        equity = equity_curve["equity"]
        total_return = equity.iloc[-1] / self.initial_cash - 1.0
        daily_returns = equity.pct_change().dropna()
        sharpe = None
        if not daily_returns.empty and daily_returns.std(ddof=0) > 0:
            sharpe = float((daily_returns.mean() / daily_returns.std(ddof=0)) * np.sqrt(252))

        drawdown = equity / equity.cummax() - 1.0
        max_drawdown = float(drawdown.min())

        elapsed_days = max((equity.index[-1] - equity.index[0]).days, 1)
        years = elapsed_days / 365.25
        cagr = float((equity.iloc[-1] / self.initial_cash) ** (1 / years) - 1) if years > 0 else None

        metrics: dict[str, float | int | None] = {
            "initial_cash": self.initial_cash,
            "final_equity": float(equity.iloc[-1]),
            "total_return": float(total_return),
            "cagr": cagr,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "trade_count": int(len(trade_log)),
            "trades_per_year": float(len(trade_log) / years) if years > 0 else None,
        }

        if trade_log.empty:
            metrics.update(
                {
                    "win_rate": None,
                    "avg_return_per_trade": None,
                    "avg_win": None,
                    "avg_loss": None,
                    "best_trade": None,
                    "worst_trade": None,
                }
            )
            return metrics

        returns = trade_log["net_return"]
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        metrics.update(
            {
                "win_rate": float((returns > 0).mean()),
                "avg_return_per_trade": float(returns.mean()),
                "avg_win": float(wins.mean()) if not wins.empty else None,
                "avg_loss": float(losses.mean()) if not losses.empty else None,
                "best_trade": float(returns.max()),
                "worst_trade": float(returns.min()),
            }
        )
        return metrics


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

