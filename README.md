# ZHQUANT

Minimal strategy DSL validation pipeline.

This repo starts with one simple contract:

1. Put strategy DSL files in `strategies/*.json`.
2. Run the validator.
3. Invalid DSL fails with explicit error messages.

## Validate A Strategy

```powershell
python -m zhquant.validate strategies/basic_momentum.json
```

Validate every JSON strategy:

```powershell
python -m zhquant.validate strategies
```

## Current DSL Scope

The first version supports:

- Static symbol universe
- Daily timeframe
- Entry and exit condition trees
- Whitelisted indicators and operators
- Basic execution and risk settings

The validator intentionally does not execute arbitrary Python. Strategy JSON is treated as data, not code.

## Compile DSL To Signals

The compiler turns a validated strategy into boolean signal matrices:

```python
from zhquant.compiler import compile_strategy

compiled = compile_strategy(strategy_json, market_data)

entry_signal = compiled.entry
exit_signal = compiled.exit
stateful_exit_rules = compiled.stateful_exit_rules
```

`market_data` is a dictionary keyed by ticker. Each value is a pandas DataFrame indexed by date with these columns:

```text
open, high, low, close, volume
```

The compiler handles stateless conditions immediately. Rules that require live position state, such as `holding_days` or `position_return`, are returned in `stateful_exit_rules` for the future backtester.

## Run A Backtest

The first backtester is intentionally narrow:

- Long-only
- Daily bars
- Signals generated at close
- `next_open` execution
- Fixed position sizing through `max_position_pct`
- Portfolio cap through `max_positions`
- `entry` buys and `exit` sells
- Stateful exits for `holding_days` and `position_return`
- Optional `risk.stop_loss_pct` and `risk.take_profit_pct`
- Slippage and commission in basis points

```python
from zhquant.backtest import run_backtest

result = run_backtest(strategy_json, market_data, initial_cash=100_000)

print(result.trade_log)
print(result.metrics)
```

Or run it directly from the command line with yfinance data:

```powershell
python -m zhquant.backtest_cli strategies/basic_momentum.json AAPL --period 1mo
```

For a simpler single-stock demo strategy:

```powershell
python -m zhquant.backtest_cli strategies/simple_sma_pullback.json AAPL --period 1mo
```

With explicit dates:

```powershell
python -m zhquant.backtest_cli strategies/basic_momentum.json NVDA --start 2024-01-01 --end 2024-06-01 --initial-cash 100000
```

Batch mode:

```powershell
python -m zhquant.backtest_cli strategies/simple_sma_pullback.json --tickers AAPL,MSFT,NVDA,MU,AMD --start 2024-01-01 --end 2024-06-01
```

The CLI overrides the strategy universe with the ticker you pass in, while still downloading referenced benchmark symbols such as `QQQ`.

The result contains:

```text
equity_curve
trade_log
metrics
compiled
```

## Current Report Fields

Single ticker reports include:

- Strategy return, net P/L, Sharpe, Sortino, Calmar, max drawdown
- Buy-and-hold benchmark return and max drawdown
- Excess return versus buy and hold
- Exposure time
- Gross profit, gross loss, total commission
- Trade count, win rate, profit factor, average holding days
- Median trade return, best/worst trade, max consecutive losses
- PASS/FAIL verdict and score

Batch reports summarize those fields across tickers.

## Run A Strategy Directory

LLM-generated strategies should be stored as one JSON file per strategy. You can run a whole directory:

```powershell
python -m zhquant.strategy_batch_cli strategies --tickers AAPL,MSFT,NVDA,MU,AMD --start 2024-01-01 --end 2024-06-01
```

The command prints a strategy leaderboard and saves:

```text
results/runs/<timestamp>/summary.csv
results/runs/<timestamp>/details.json
```

`summary.csv` is for ranking strategies. `details.json` keeps ticker-level metrics and score reasons for feeding back into the next LLM generation round.
