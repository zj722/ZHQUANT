from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DSLValidationError(ValueError):
    """Raised when a strategy DSL document is invalid."""


@dataclass(frozen=True)
class ValidationErrorItem:
    path: str
    message: str

    def format(self) -> str:
        return f"{self.path}: {self.message}"


BASE_SERIES = {"open", "high", "low", "close", "volume"}
TIMEFRAMES = {"1d"}
UNIVERSE_TYPES = {"static_list"}
LOGICAL_KEYS = {"all", "any", "not"}
COMPARISON_OPS = {">", ">=", "<", "<=", "==", "!="}
MATH_OPS = {"+", "-", "*", "/", "abs", "min", "max"}
EVENT_CONDITION_OPS = {"within_days_after_event", "days_until_event"}
EVENT_NAMES = {"earnings"}
INDICATORS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "sma",
    "ema",
    "rsi",
    "atr",
    "return",
    "rolling_max",
    "rolling_min",
    "rolling_count",
    "rolling_quantile",
    "zscore",
}
FIELDS = {
    "holding_days",
    "position_return",
    "current_weight",
    "cash_pct",
    "entry_price",
    "highest_close_since_entry",
}
EXECUTION_MODES = {"next_open", "next_close", "same_close"}


class StrategyDSLValidator:
    """Validates ZHQUANT strategy DSL JSON.

    The validator is intentionally strict: unknown fields are rejected so that
    later LLM-generated DSL cannot silently drift away from the backtest engine.
    """

    def __init__(self) -> None:
        self.errors: list[ValidationErrorItem] = []

    def validate_file(self, path: str | Path) -> dict[str, Any]:
        file_path = Path(path)
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DSLValidationError(f"{file_path}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc

        self.validate(data)
        return data

    def validate(self, strategy: Any) -> None:
        self.errors = []
        self._strategy(strategy, "$")
        if self.errors:
            details = "\n".join(f"- {err.format()}" for err in self.errors)
            raise DSLValidationError(f"Strategy DSL validation failed:\n{details}")

    def _error(self, path: str, message: str) -> None:
        self.errors.append(ValidationErrorItem(path, message))

    def _require_object(self, value: Any, path: str) -> bool:
        if not isinstance(value, dict):
            self._error(path, "must be an object")
            return False
        return True

    def _strategy(self, value: Any, path: str) -> None:
        if not self._require_object(value, path):
            return

        if "rotation" in value:
            self._rotation_strategy(value, path)
            return

        allowed = {
            "name",
            "description",
            "timeframe",
            "universe",
            "entry",
            "add_entry",
            "exit",
            "reduce_exit",
            "trim_exit",
            "risk",
            "diagnostics",
        }
        self._unknown_keys(value, allowed, path)

        self._required_string(value, "name", path)
        if "description" in value and not isinstance(value["description"], str):
            self._error(f"{path}.description", "must be a string")

        timeframe = value.get("timeframe")
        if timeframe not in TIMEFRAMES:
            self._error(f"{path}.timeframe", f"must be one of {sorted(TIMEFRAMES)}")

        self._universe(value.get("universe"), f"{path}.universe")
        self._condition_tree(value.get("entry"), f"{path}.entry", context="entry")
        if "add_entry" in value:
            self._condition_tree(value["add_entry"], f"{path}.add_entry", context="entry")
        self._condition_tree(value.get("exit"), f"{path}.exit", context="exit")
        if "reduce_exit" in value:
            self._condition_tree(value["reduce_exit"], f"{path}.reduce_exit", context="exit")
        if "trim_exit" in value:
            self._condition_tree(value["trim_exit"], f"{path}.trim_exit", context="exit")
        self._risk(value.get("risk"), f"{path}.risk")
        if "diagnostics" in value:
            self._diagnostics(value["diagnostics"], f"{path}.diagnostics")

    def _rotation_strategy(self, value: dict[str, Any], path: str) -> None:
        allowed = {
            "name",
            "description",
            "timeframe",
            "universe",
            "rotation",
            "risk",
        }
        self._unknown_keys(value, allowed, path)

        self._required_string(value, "name", path)
        if "description" in value and not isinstance(value["description"], str):
            self._error(f"{path}.description", "must be a string")

        timeframe = value.get("timeframe")
        if timeframe not in TIMEFRAMES:
            self._error(f"{path}.timeframe", f"must be one of {sorted(TIMEFRAMES)}")

        self._universe(value.get("universe"), f"{path}.universe")
        self._rotation(value.get("rotation"), f"{path}.rotation")
        self._rotation_risk(value.get("risk"), f"{path}.risk")

    def _rotation(self, value: Any, path: str) -> None:
        if not self._require_object(value, path):
            return
        allowed = {"score", "top_n", "rebalance", "require_positive_score"}
        self._unknown_keys(value, allowed, path)

        score = value.get("score")
        if not self._require_object(score, f"{path}.score"):
            return
        self._unknown_keys(score, {"indicator", "source", "window"}, f"{path}.score")
        if score.get("indicator") != "return":
            self._error(f"{path}.score.indicator", "must be return")
        if score.get("source") != "close":
            self._error(f"{path}.score.source", "must be close")
        self._positive_int(score.get("window"), f"{path}.score.window", min_value=2, max_value=252)

        self._positive_int(value.get("top_n"), f"{path}.top_n", min_value=1, max_value=100)
        if value.get("rebalance") not in {"weekly", "monthly"}:
            self._error(f"{path}.rebalance", "must be weekly or monthly")
        if "require_positive_score" in value and not isinstance(value["require_positive_score"], bool):
            self._error(f"{path}.require_positive_score", "must be boolean")

    def _rotation_risk(self, value: Any, path: str) -> None:
        if not self._require_object(value, path):
            return
        allowed = {"execution", "slippage_bps", "commission_bps", "stop_loss_pct", "ma200_exit", "atr_trailing_stop"}
        self._unknown_keys(value, allowed, path)

        if value.get("execution") != "next_open":
            self._error(f"{path}.execution", "must be next_open")
        self._number_range(
            value.get("slippage_bps"),
            f"{path}.slippage_bps",
            min_value=0,
            max_value=1000,
            include_min=True,
        )
        self._number_range(
            value.get("commission_bps"),
            f"{path}.commission_bps",
            min_value=0,
            max_value=1000,
            include_min=True,
        )
        if "stop_loss_pct" in value:
            self._number_range(
                value["stop_loss_pct"],
                f"{path}.stop_loss_pct",
                min_value=-1,
                max_value=0,
                include_max=False,
            )
        if "ma200_exit" in value and not isinstance(value["ma200_exit"], bool):
            self._error(f"{path}.ma200_exit", "must be boolean")
        if "atr_trailing_stop" in value:
            self._atr_trailing_stop(value["atr_trailing_stop"], f"{path}.atr_trailing_stop")

    def _atr_trailing_stop(self, value: Any, path: str) -> None:
        if not self._require_object(value, path):
            return
        self._unknown_keys(value, {"enabled", "multiple", "activation_profit_pct"}, path)
        if "enabled" in value and not isinstance(value["enabled"], bool):
            self._error(f"{path}.enabled", "must be boolean")
        if "multiple" in value:
            self._number_range(value["multiple"], f"{path}.multiple", min_value=0, max_value=20)
        if "activation_profit_pct" in value:
            self._number_range(
                value["activation_profit_pct"],
                f"{path}.activation_profit_pct",
                min_value=0,
                max_value=10,
                include_min=True,
            )

    def _universe(self, value: Any, path: str) -> None:
        if not self._require_object(value, path):
            return

        allowed = {"type", "symbols"}
        self._unknown_keys(value, allowed, path)

        if value.get("type") not in UNIVERSE_TYPES:
            self._error(f"{path}.type", f"must be one of {sorted(UNIVERSE_TYPES)}")

        symbols = value.get("symbols")
        if not isinstance(symbols, list) or not symbols:
            self._error(f"{path}.symbols", "must be a non-empty list of ticker strings")
            return

        seen: set[str] = set()
        for idx, symbol in enumerate(symbols):
            item_path = f"{path}.symbols[{idx}]"
            if not isinstance(symbol, str) or not symbol.strip():
                self._error(item_path, "must be a non-empty string")
                continue
            normalized = symbol.strip().upper()
            if normalized in seen:
                self._error(item_path, f"duplicate symbol {normalized}")
            seen.add(normalized)

    def _condition_tree(self, value: Any, path: str, context: str) -> None:
        if not self._require_object(value, path):
            return

        logical_keys = [key for key in LOGICAL_KEYS if key in value]
        if logical_keys:
            if len(logical_keys) != 1 or len(value) != 1:
                self._error(path, "logical node must contain exactly one of all, any, not")
                return
            key = logical_keys[0]
            child = value[key]
            if key in {"all", "any"}:
                if not isinstance(child, list) or not child:
                    self._error(f"{path}.{key}", "must be a non-empty list")
                    return
                for idx, item in enumerate(child):
                    self._condition_tree(item, f"{path}.{key}[{idx}]", context=context)
                return
            self._condition_tree(child, f"{path}.not", context=context)
            return

        op = value.get("op")
        if op in COMPARISON_OPS:
            self._unknown_keys(value, {"op", "left", "right"}, path)
            if "left" not in value:
                self._error(f"{path}.left", "is required")
            else:
                self._operand(value["left"], f"{path}.left", context=context)
            if "right" not in value:
                self._error(f"{path}.right", "is required")
            else:
                self._operand(value["right"], f"{path}.right", context=context)
            return

        if op in EVENT_CONDITION_OPS:
            self._event_condition(value, path)
            return

        self._error(f"{path}.op", f"must be a comparison op, event op, or logical node")

    def _operand(self, value: Any, path: str, context: str) -> None:
        if isinstance(value, int | float):
            return
        if isinstance(value, str):
            if value not in BASE_SERIES:
                self._error(path, f"string operands must be one of {sorted(BASE_SERIES)}")
            return
        if not self._require_object(value, path):
            return

        if "indicator" in value:
            self._indicator(value, path)
            return

        if "field" in value:
            self._field(value, path, context=context)
            return

        op = value.get("op")
        if op in MATH_OPS:
            self._math_expression(value, path, context=context)
            return

        self._error(path, "operand must be a number, base series, indicator, field, or math expression")

    def _indicator(self, value: dict[str, Any], path: str) -> None:
        indicator = value.get("indicator")
        if indicator not in INDICATORS:
            self._error(f"{path}.indicator", f"must be one of {sorted(INDICATORS)}")
            return

        if indicator in BASE_SERIES:
            self._unknown_keys(value, {"indicator", "symbol", "shift"}, path)
            self._optional_symbol(value, path)
            self._optional_shift(value, path)
            return

        if indicator == "rolling_count":
            self._unknown_keys(value, {"indicator", "condition", "window", "shift"}, path)
            if "condition" not in value:
                self._error(f"{path}.condition", "is required")
            else:
                self._condition_tree(value["condition"], f"{path}.condition", context="entry")
            self._positive_int(value.get("window"), f"{path}.window", min_value=1, max_value=252)
            self._optional_shift(value, path)
            return

        if indicator == "rolling_quantile":
            self._unknown_keys(value, {"indicator", "source", "window", "return_window", "quantile", "symbol", "shift"}, path)
            source = value.get("source")
            if source not in BASE_SERIES:
                self._error(f"{path}.source", f"must be one of {sorted(BASE_SERIES)}")
            self._optional_symbol(value, path)
            self._positive_int(value.get("window"), f"{path}.window", min_value=1, max_value=252)
            self._positive_int(value.get("return_window"), f"{path}.return_window", min_value=1, max_value=252)
            self._number_range(value.get("quantile"), f"{path}.quantile", min_value=0, max_value=1, include_min=True)
            self._optional_shift(value, path)
            return

        allowed = {"indicator", "source", "window", "symbol", "shift"}
        self._unknown_keys(value, allowed, path)
        self._optional_symbol(value, path)
        self._optional_shift(value, path)

        if indicator == "atr":
            if "source" in value:
                self._error(f"{path}.source", "is not supported for atr")
        else:
            source = value.get("source")
            if source not in BASE_SERIES:
                self._error(f"{path}.source", f"must be one of {sorted(BASE_SERIES)}")

        self._positive_int(value.get("window"), f"{path}.window", min_value=1, max_value=252)

    def _field(self, value: dict[str, Any], path: str, context: str) -> None:
        self._unknown_keys(value, {"field"}, path)
        field = value.get("field")
        if field not in FIELDS:
            self._error(f"{path}.field", f"must be one of {sorted(FIELDS)}")
            return
        if context == "entry" and field in {"holding_days", "position_return", "current_weight", "entry_price", "highest_close_since_entry"}:
            self._error(f"{path}.field", f"{field} is only valid after a position exists")

    def _math_expression(self, value: dict[str, Any], path: str, context: str) -> None:
        op = value.get("op")
        if op == "abs":
            self._unknown_keys(value, {"op", "value"}, path)
            if "value" not in value:
                self._error(f"{path}.value", "is required")
            else:
                self._operand(value["value"], f"{path}.value", context=context)
            return

        self._unknown_keys(value, {"op", "left", "right"}, path)
        if "left" not in value:
            self._error(f"{path}.left", "is required")
        else:
            self._operand(value["left"], f"{path}.left", context=context)
        if "right" not in value:
            self._error(f"{path}.right", "is required")
        else:
            self._operand(value["right"], f"{path}.right", context=context)

    def _event_condition(self, value: dict[str, Any], path: str) -> None:
        op = value.get("op")
        if op == "within_days_after_event":
            self._unknown_keys(value, {"op", "event", "days"}, path)
            if value.get("event") not in EVENT_NAMES:
                self._error(f"{path}.event", f"must be one of {sorted(EVENT_NAMES)}")
            self._positive_int(value.get("days"), f"{path}.days", min_value=1, max_value=60)
            return

        if op == "days_until_event":
            self._unknown_keys(value, {"op", "event", "max_days"}, path)
            if value.get("event") not in EVENT_NAMES:
                self._error(f"{path}.event", f"must be one of {sorted(EVENT_NAMES)}")
            self._positive_int(value.get("max_days"), f"{path}.max_days", min_value=1, max_value=60)

    def _risk(self, value: Any, path: str) -> None:
        if not self._require_object(value, path):
            return

        allowed = {
            "max_position_pct",
            "max_positions",
            "execution",
            "slippage_bps",
            "commission_bps",
            "stop_loss_pct",
            "take_profit_pct",
            "max_additions",
            "add_position_pct",
            "reduce_position_pct",
            "trim_position_pct",
        }
        self._unknown_keys(value, allowed, path)

        self._number_range(value.get("max_position_pct"), f"{path}.max_position_pct", min_value=0, max_value=1)
        self._positive_int(value.get("max_positions"), f"{path}.max_positions", min_value=1, max_value=100)

        if value.get("execution") not in EXECUTION_MODES:
            self._error(f"{path}.execution", f"must be one of {sorted(EXECUTION_MODES)}")

        self._number_range(
            value.get("slippage_bps"),
            f"{path}.slippage_bps",
            min_value=0,
            max_value=1000,
            include_min=True,
        )
        self._number_range(
            value.get("commission_bps"),
            f"{path}.commission_bps",
            min_value=0,
            max_value=1000,
            include_min=True,
        )
        if "stop_loss_pct" in value:
            self._number_range(
                value["stop_loss_pct"],
                f"{path}.stop_loss_pct",
                min_value=-1,
                max_value=0,
                include_max=False,
            )
        if "take_profit_pct" in value:
            self._number_range(value["take_profit_pct"], f"{path}.take_profit_pct", min_value=0, max_value=10)
        if "max_additions" in value:
            self._positive_int(value["max_additions"], f"{path}.max_additions", min_value=0, max_value=10)
        if "add_position_pct" in value:
            self._number_range(value["add_position_pct"], f"{path}.add_position_pct", min_value=0, max_value=1)
        if "reduce_position_pct" in value:
            self._number_range(value["reduce_position_pct"], f"{path}.reduce_position_pct", min_value=0, max_value=1)
        if "trim_position_pct" in value:
            self._number_range(value["trim_position_pct"], f"{path}.trim_position_pct", min_value=0, max_value=1)

    def _diagnostics(self, value: Any, path: str) -> None:
        if not self._require_object(value, path):
            return
        if not value:
            self._error(path, "must contain at least one named diagnostic condition")
            return
        for name, condition in value.items():
            if not isinstance(name, str) or not name.strip():
                self._error(path, "diagnostic names must be non-empty strings")
                continue
            self._condition_tree(condition, f"{path}.{name}", context="entry")

    def _required_string(self, value: dict[str, Any], key: str, path: str) -> None:
        if key not in value:
            self._error(f"{path}.{key}", "is required")
            return
        if not isinstance(value[key], str) or not value[key].strip():
            self._error(f"{path}.{key}", "must be a non-empty string")

    def _positive_int(self, value: Any, path: str, min_value: int, max_value: int) -> None:
        if not isinstance(value, int) or isinstance(value, bool):
            self._error(path, "must be an integer")
            return
        if value < min_value or value > max_value:
            self._error(path, f"must be between {min_value} and {max_value}")

    def _number_range(
        self,
        value: Any,
        path: str,
        min_value: float,
        max_value: float,
        include_min: bool = False,
        include_max: bool = True,
    ) -> None:
        if not isinstance(value, int | float) or isinstance(value, bool):
            self._error(path, "must be a number")
            return
        min_ok = value >= min_value if include_min else value > min_value
        max_ok = value <= max_value if include_max else value < max_value
        if not min_ok or not max_ok:
            op = ">=" if include_min else ">"
            max_op = "<=" if include_max else "<"
            self._error(path, f"must be {op} {min_value} and {max_op} {max_value}")

    def _optional_symbol(self, value: dict[str, Any], path: str) -> None:
        if "symbol" not in value:
            return
        symbol = value["symbol"]
        if not isinstance(symbol, str) or not symbol.strip():
            self._error(f"{path}.symbol", "must be a non-empty string")

    def _optional_shift(self, value: dict[str, Any], path: str) -> None:
        if "shift" not in value:
            return
        self._positive_int(value["shift"], f"{path}.shift", min_value=0, max_value=252)

    def _unknown_keys(self, value: dict[str, Any], allowed: set[str], path: str) -> None:
        for key in sorted(set(value) - allowed):
            self._error(f"{path}.{key}", "is not allowed")
