from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .dsl_validator import BASE_SERIES, StrategyDSLValidator


class CompilerError(ValueError):
    """Raised when a valid DSL cannot be compiled against the supplied data."""


@dataclass(frozen=True)
class StatefulRule:
    path: str
    rule: dict[str, Any]


@dataclass(frozen=True)
class CompiledSignals:
    strategy_name: str
    entry: pd.DataFrame
    exit: pd.DataFrame
    stateful_exit_rules: tuple[StatefulRule, ...]
    data_dependencies: tuple[str, ...]


class StrategyCompiler:
    """Compile strategy DSL into boolean signal matrices.

    Market data input format:
        {
            "AAPL": DataFrame(index=date, columns=["open", "high", "low", "close", "volume"]),
            "QQQ": DataFrame(...),
        }

    Output signal format:
        DataFrame(index=date, columns=universe_symbols, values=bool)
    """

    required_ohlcv = ("open", "high", "low", "close", "volume")

    def __init__(
        self,
        market_data: dict[str, pd.DataFrame],
        events: dict[str, Any] | None = None,
    ) -> None:
        self.market_data = {symbol.upper(): df.copy() for symbol, df in market_data.items()}
        self.events = events or {}
        self.symbols: list[str] = []
        self.index: pd.Index | None = None
        self.stateful_exit_rules: list[StatefulRule] = []
        self.data_dependencies: set[str] = set()

    def compile(self, strategy: dict[str, Any]) -> CompiledSignals:
        StrategyDSLValidator().validate(strategy)

        self.symbols = [symbol.upper() for symbol in strategy["universe"]["symbols"]]
        self._prepare_market_data()
        self.stateful_exit_rules = []
        self.data_dependencies = set()

        entry = self._eval_condition(strategy["entry"], "$.entry", context="entry")
        exit_signal = self._eval_condition(strategy["exit"], "$.exit", context="exit")

        return CompiledSignals(
            strategy_name=strategy["name"],
            entry=entry.fillna(False).astype(bool),
            exit=exit_signal.fillna(False).astype(bool),
            stateful_exit_rules=tuple(self.stateful_exit_rules),
            data_dependencies=tuple(sorted(self.data_dependencies)),
        )

    def _prepare_market_data(self) -> None:
        missing = [symbol for symbol in self.symbols if symbol not in self.market_data]
        if missing:
            raise CompilerError(f"Missing market data for universe symbols: {missing}")

        indexes = []
        for symbol, df in self.market_data.items():
            if not isinstance(df, pd.DataFrame):
                raise CompilerError(f"Market data for {symbol} must be a pandas DataFrame")
            missing_cols = [col for col in self.required_ohlcv if col not in df.columns]
            if missing_cols:
                raise CompilerError(f"Market data for {symbol} is missing columns: {missing_cols}")
            if df.empty:
                raise CompilerError(f"Market data for {symbol} is empty")
            self.market_data[symbol] = df.sort_index()

        for symbol in self.symbols:
            indexes.append(self.market_data[symbol].index)

        shared_index = indexes[0]
        for idx in indexes[1:]:
            shared_index = shared_index.intersection(idx)

        if shared_index.empty:
            raise CompilerError("Universe symbols do not share any dates")

        self.index = shared_index.sort_values()

    def _eval_condition(self, node: dict[str, Any], path: str, context: str) -> pd.DataFrame:
        if "all" in node:
            children = node["all"]
            if context == "exit" and any(self._contains_stateful_field(child) for child in children):
                self.stateful_exit_rules.append(StatefulRule(path=path, rule=node))
                return self._bool_frame(False)
            frames = [self._eval_condition(child, f"{path}.all[{idx}]", context) for idx, child in enumerate(children)]
            result = self._bool_frame(True)
            for frame in frames:
                result = result & frame
            return result

        if "any" in node:
            result = self._bool_frame(False)
            for idx, child in enumerate(node["any"]):
                child_path = f"{path}.any[{idx}]"
                if context == "exit" and self._contains_stateful_field(child):
                    self.stateful_exit_rules.append(StatefulRule(path=child_path, rule=child))
                    continue
                result = result | self._eval_condition(child, child_path, context)
            return result

        if "not" in node:
            if context == "exit" and self._contains_stateful_field(node["not"]):
                self.stateful_exit_rules.append(StatefulRule(path=path, rule=node))
                return self._bool_frame(False)
            return ~self._eval_condition(node["not"], f"{path}.not", context)

        if context == "entry" and self._contains_stateful_field(node):
            raise CompilerError(f"{path}: stateful fields are not supported in entry conditions")

        if context == "exit" and self._contains_stateful_field(node):
            self.stateful_exit_rules.append(StatefulRule(path=path, rule=node))
            return self._bool_frame(False)

        op = node["op"]
        if op in {"within_days_after_event", "days_until_event"}:
            return self._eval_event_condition(node, path)

        left = self._eval_operand(node["left"], f"{path}.left", context)
        right = self._eval_operand(node["right"], f"{path}.right", context)
        result = self._compare(op, left, right)
        return self._as_bool_frame(result, path)

    def _eval_operand(self, value: Any, path: str, context: str) -> float | pd.DataFrame:
        if isinstance(value, int | float):
            return float(value)

        if isinstance(value, str):
            return self._source_matrix(value)

        if "indicator" in value:
            return self._indicator(value, path)

        if "field" in value:
            raise CompilerError(f"{path}: field operands require the backtester state engine")

        return self._math(value, path, context)

    def _indicator(self, spec: dict[str, Any], path: str) -> pd.DataFrame:
        indicator = spec["indicator"]
        symbol = spec.get("symbol")

        if indicator in BASE_SERIES:
            return self._source_matrix(indicator, symbol=symbol)

        source_name = spec.get("source", "close")
        source = self._source_matrix(source_name, symbol=symbol)
        window = int(spec["window"])

        if indicator == "sma":
            return source.rolling(window, min_periods=window).mean()
        if indicator == "ema":
            return source.ewm(span=window, min_periods=window, adjust=False).mean()
        if indicator == "return":
            return source.pct_change(window)
        if indicator == "rolling_max":
            return source.rolling(window, min_periods=window).max()
        if indicator == "rolling_min":
            return source.rolling(window, min_periods=window).min()
        if indicator == "zscore":
            rolling = source.rolling(window, min_periods=window)
            return (source - rolling.mean()) / rolling.std(ddof=0).replace(0, np.nan)
        if indicator == "rsi":
            delta = source.diff()
            gains = delta.clip(lower=0).rolling(window, min_periods=window).mean()
            losses = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
            rs = gains / losses.replace(0, np.nan)
            return 100 - (100 / (1 + rs))
        if indicator == "atr":
            return self._atr_matrix(window, symbol=symbol)

        raise CompilerError(f"{path}.indicator: unsupported indicator {indicator}")

    def _math(self, spec: dict[str, Any], path: str, context: str) -> float | pd.DataFrame:
        op = spec["op"]
        if op == "abs":
            value = self._eval_operand(spec["value"], f"{path}.value", context)
            return abs(value)

        left = self._eval_operand(spec["left"], f"{path}.left", context)
        right = self._eval_operand(spec["right"], f"{path}.right", context)

        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return left / right
        if op == "min":
            return self._binary_ufunc(np.minimum, left, right)
        if op == "max":
            return self._binary_ufunc(np.maximum, left, right)

        raise CompilerError(f"{path}.op: unsupported math op {op}")

    def _compare(self, op: str, left: float | pd.DataFrame, right: float | pd.DataFrame) -> bool | pd.DataFrame:
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
        raise CompilerError(f"Unsupported comparison op {op}")

    def _source_matrix(self, source: str, symbol: str | None = None) -> pd.DataFrame:
        if self.index is None:
            raise CompilerError("Compiler index is not prepared")

        if symbol is not None:
            symbol = symbol.upper()
            series = self._source_series(symbol, source)
            return pd.DataFrame({universe_symbol: series for universe_symbol in self.symbols}, index=self.index)

        data = {
            universe_symbol: self._source_series(universe_symbol, source)
            for universe_symbol in self.symbols
        }
        return pd.DataFrame(data, index=self.index)

    def _source_series(self, symbol: str, source: str) -> pd.Series:
        if self.index is None:
            raise CompilerError("Compiler index is not prepared")
        if symbol not in self.market_data:
            raise CompilerError(f"Missing market data for referenced symbol: {symbol}")
        self.data_dependencies.add(symbol)
        return self.market_data[symbol][source].reindex(self.index)

    def _atr_matrix(self, window: int, symbol: str | None = None) -> pd.DataFrame:
        symbols = [symbol.upper()] if symbol else self.symbols
        columns: dict[str, pd.Series] = {}
        for data_symbol in symbols:
            high = self._source_series(data_symbol, "high")
            low = self._source_series(data_symbol, "low")
            close = self._source_series(data_symbol, "close")
            prev_close = close.shift(1)
            true_range = pd.concat(
                [
                    high - low,
                    (high - prev_close).abs(),
                    (low - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr = true_range.rolling(window, min_periods=window).mean()
            if symbol:
                for universe_symbol in self.symbols:
                    columns[universe_symbol] = atr
            else:
                columns[data_symbol] = atr
        return pd.DataFrame(columns, index=self.index)

    def _eval_event_condition(self, node: dict[str, Any], path: str) -> pd.DataFrame:
        event_name = node["event"]
        event_matrix = self._event_matrix(event_name)
        if node["op"] == "within_days_after_event":
            days = int(node["days"])
            result = self._bool_frame(False)
            for offset in range(days + 1):
                result = result | event_matrix.shift(offset, fill_value=False)
            return result
        if node["op"] == "days_until_event":
            max_days = int(node["max_days"])
            result = self._bool_frame(False)
            for offset in range(max_days + 1):
                result = result | event_matrix.shift(-offset, fill_value=False)
            return result
        raise CompilerError(f"{path}.op: unsupported event op")

    def _event_matrix(self, event_name: str) -> pd.DataFrame:
        if self.index is None:
            raise CompilerError("Compiler index is not prepared")
        if event_name not in self.events:
            raise CompilerError(f"Missing event data for event: {event_name}")

        raw = self.events[event_name]
        if isinstance(raw, pd.DataFrame):
            frame = raw.reindex(index=self.index, columns=self.symbols).fillna(False)
            return frame.astype(bool)

        if isinstance(raw, dict):
            frame = self._bool_frame(False)
            for symbol, dates in raw.items():
                symbol = symbol.upper()
                if symbol not in frame.columns:
                    continue
                event_dates = pd.Index(pd.to_datetime(list(dates)))
                matching = frame.index.intersection(event_dates)
                frame.loc[matching, symbol] = True
            return frame

        raise CompilerError(f"Event data for {event_name} must be a DataFrame or dict")

    def _bool_frame(self, value: bool) -> pd.DataFrame:
        if self.index is None:
            raise CompilerError("Compiler index is not prepared")
        return pd.DataFrame(value, index=self.index, columns=self.symbols)

    def _as_bool_frame(self, value: bool | pd.DataFrame, path: str) -> pd.DataFrame:
        if isinstance(value, bool | np.bool_):
            return self._bool_frame(bool(value))
        if not isinstance(value, pd.DataFrame):
            raise CompilerError(f"{path}: condition did not compile to a boolean matrix")
        return value.reindex(index=self.index, columns=self.symbols).fillna(False).astype(bool)

    def _binary_ufunc(self, func: Any, left: float | pd.DataFrame, right: float | pd.DataFrame) -> float | pd.DataFrame:
        if isinstance(left, pd.DataFrame) or isinstance(right, pd.DataFrame):
            return pd.DataFrame(func(left, right), index=self.index, columns=self.symbols)
        return float(func(left, right))

    def _contains_stateful_field(self, node: Any) -> bool:
        if isinstance(node, dict):
            if "field" in node:
                return True
            return any(self._contains_stateful_field(value) for value in node.values())
        if isinstance(node, list):
            return any(self._contains_stateful_field(item) for item in node)
        return False


def compile_strategy(
    strategy: dict[str, Any],
    market_data: dict[str, pd.DataFrame],
    events: dict[str, Any] | None = None,
) -> CompiledSignals:
    return StrategyCompiler(market_data=market_data, events=events).compile(strategy)

