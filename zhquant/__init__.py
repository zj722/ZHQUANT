"""ZHQUANT strategy DSL tooling."""

__version__ = "0.1.0"

from .backtest import BacktestResult, LongOnlyBacktester, run_backtest
from .compiler import CompiledSignals, StrategyCompiler, compile_strategy
from .data_loader import download_yfinance_ohlcv
from .rotation_backtest import RotationBacktestResult, run_rotation_backtest
from .strategy_batch import StrategyBatchRun, run_strategy_directory

__all__ = [
    "BacktestResult",
    "CompiledSignals",
    "LongOnlyBacktester",
    "RotationBacktestResult",
    "StrategyCompiler",
    "StrategyBatchRun",
    "compile_strategy",
    "download_yfinance_ohlcv",
    "run_backtest",
    "run_rotation_backtest",
    "run_strategy_directory",
]
