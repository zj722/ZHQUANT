# ZHQUANT

Minimal strategy DSL validation, signal compilation, backtesting, diagnostics, and charting toolkit.

Strategy files live in `strategies/*.json`. The DSL is data-only JSON; it does not execute arbitrary Python.

## Install

```powershell
pip install -e .
```

Current project dependencies:

```text
pandas
yfinance
matplotlib
```

## Validate Strategies

Validate one strategy:

```powershell
python -m zhquant.validate strategies/volume_trend_pullback_breakout.json
```

Validate every JSON strategy in a directory:

```powershell
python -m zhquant.validate strategies
```

## Run A Single-Ticker Backtest

Run the current v4 trend pullback strategy on AMD:

```powershell
python -m zhquant.backtest_cli strategies/volume_trend_pullback_breakout.json AMD --start 2024-01-01 --end 2026-06-30 --initial-cash 100000
```

Run with a yfinance period instead of explicit dates:

```powershell
python -m zhquant.backtest_cli strategies/volume_trend_pullback_breakout.json AMD --period 2y --initial-cash 100000
```

The single-ticker report includes:

```text
Current Action
Summary
Benchmark
Strategy Score
Money Made / Lost
Trades
Trade Log
```

`Current Action` tells you what the strategy says at the latest signal date:

```text
BUY
REDUCE
TRIM
SELL
HOLD
```

Signals are evaluated after the daily close and executed at the next daily open.

## Run With Daily Diagnostics

Print daily condition diagnostics and pass rates:

```powershell
python -m zhquant.backtest_cli strategies/volume_trend_pullback_breakout.json AMD --start 2024-01-01 --end 2026-06-30 --initial-cash 100000 --show-diagnostics
```

Save diagnostics to CSV:

```powershell
python -m zhquant.backtest_cli strategies/volume_trend_pullback_breakout.json AMD --start 2024-01-01 --end 2026-06-30 --initial-cash 100000 --diagnostics-csv results/amd_v4_diagnostics.csv
```

Use both:

```powershell
python -m zhquant.backtest_cli strategies/volume_trend_pullback_breakout.json AMD --start 2024-01-01 --end 2026-06-30 --initial-cash 100000 --show-diagnostics --diagnostics-csv results/amd_v4_diagnostics.csv
```

The v4 strategy currently reports these diagnostics:

```text
trend_ok
healthy_pullback
buy_trigger
do_not_buy
short_term_extended
overextended
reversal_signal
final_buy_signal
```

`final_buy_signal` is generated from the actual `entry` condition, so it matches real buy eligibility.

## Generate A K-Line Chart

Write a candlestick chart with BUY / ADD / REDUCE / TRIM / SELL labels:

```powershell
python -m zhquant.backtest_cli strategies/volume_trend_pullback_breakout.json AMD --start 2024-01-01 --end 2026-06-30 --initial-cash 100000 --plot results/amd_v4_chart.png
```

Run report, diagnostics, CSV, and chart together:

```powershell
python -m zhquant.backtest_cli strategies/volume_trend_pullback_breakout.json AMD --start 2024-01-01 --end 2026-06-30 --initial-cash 100000 --show-diagnostics --diagnostics-csv results/amd_v4_diagnostics.csv --plot results/amd_v4_chart.png
```

The chart uses the backtest `order_log`, so it marks every:

```text
BUY
ADD
REDUCE
TRIM
SELL
```

The chart also plots the main indicators used by the current strategy:

```text
Price panel: EMA20, MA50, MA200
Volume panel: Volume MA5, Volume MA20
RSI panel: RSI14 with 35 / 50 / 68 / 72 reference levels
```

## Run A Batch Backtest

Run the same strategy across several tickers:

```powershell
python -m zhquant.backtest_cli strategies/volume_trend_pullback_breakout.json --tickers AMD,MU,NVDA,AVGO,TSM --start 2024-01-01 --end 2026-06-30 --initial-cash 100000
```

Save one chart per ticker:

```powershell
python -m zhquant.backtest_cli strategies/volume_trend_pullback_breakout.json --tickers AMD,MU,NVDA,AVGO,TSM --start 2024-01-01 --end 2026-06-30 --initial-cash 100000 --plot results/charts
```

Save batch diagnostics:

```powershell
python -m zhquant.backtest_cli strategies/volume_trend_pullback_breakout.json --tickers AMD,MU,NVDA,AVGO,TSM --start 2024-01-01 --end 2026-06-30 --initial-cash 100000 --diagnostics-csv results/batch_v4_diagnostics.csv
```

If `--plot` points to a directory in batch mode, files are written as:

```text
results/charts/AMD.png
results/charts/MU.png
...
```

If `--plot` points to a file in batch mode, ticker suffixes are added:

```text
results/chart_AMD.png
results/chart_MU.png
...
```

## Run A Strategy Directory

Run every strategy JSON in a directory:

```powershell
python -m zhquant.strategy_batch_cli strategies --tickers AMD,MU,NVDA,AVGO,TSM --start 2024-01-01 --end 2026-06-30
```

The command prints a strategy leaderboard and saves:

```text
results/runs/<timestamp>/summary.csv
results/runs/<timestamp>/details.json
```

## Python API

Compile a strategy to signal matrices:

```python
from zhquant.compiler import compile_strategy

compiled = compile_strategy(strategy_json, market_data)

entry_signal = compiled.entry
exit_signal = compiled.exit
diagnostics = compiled.diagnostics
```

Run a backtest:

```python
from zhquant.backtest import run_backtest

result = run_backtest(strategy_json, market_data, initial_cash=100_000)

print(result.current_actions)
print(result.order_log)
print(result.trade_log)
print(result.diagnostic_pass_rates)
print(result.metrics)
```

Write a chart from Python:

```python
from zhquant.plotting import plot_candlestick_with_orders

plot_candlestick_with_orders(
    market_data=market_data["AMD"],
    order_log=result.order_log,
    ticker="AMD",
    output_path="results/amd_chart.png",
)
```

## Current DSL Scope

Supported features include:

- Static symbol universe
- Daily timeframe
- `entry`, optional `add_entry`, `exit`, `reduce_exit`, and `trim_exit` condition trees
- Named `diagnostics` condition trees
- Long-only backtesting
- Next-open execution
- BUY / ADD / REDUCE / TRIM / SELL order logging
- Fixed position sizing with optional add-on buys and partial exits
- Risk settings for max positions, add count, slippage, commission, stop loss, take profit
- Stateful exits using fields such as `holding_days`, `position_return`, `entry_price`, `highest_close_since_entry`
- Indicators including `sma`, `ema`, `rsi`, `atr`, `return`, `rolling_max`, `rolling_min`, `rolling_count`, `rolling_quantile`, `zscore`
- Indicator `shift` for prior-day comparisons

## Notes

For strategies using long lookbacks such as MA200, use enough historical data for warmup. A short date range may show no trades simply because indicators are not mature.

Single-ticker benchmark returns are full buy-and-hold for that ticker. If a strategy uses partial position sizing, benchmark comparison is not position-size neutral.

## Run A Semiconductor Rotation Backtest

Run the current semiconductor / memory momentum rotation strategy:

```powershell
python -m zhquant.rotation_backtest_cli strategies/semiconductor_momentum_rotation.json --start 2024-01-01 --end 2026-06-30 --initial-cash 100000
```

Save equity, weights, and rebalance orders:

```powershell
python -m zhquant.rotation_backtest_cli strategies/semiconductor_momentum_rotation.json --start 2024-01-01 --end 2026-06-30 --initial-cash 100000 --equity-csv results/rotation_equity.csv --weights-csv results/rotation_weights.csv --orders-csv results/rotation_orders.csv
```

Check the latest action using the most recent yfinance trading data:

```powershell
python -m zhquant.rotation_backtest_cli strategies/semiconductor_momentum_rotation.json --start 2024-01-01 --initial-cash 100000 --live-action
```

`--live-action` keeps the final simulated positions open and prints:

```text
Current Action
Target Weights
Weight Changes
```

Use this mode for daily operational checks. A monthly rotation signal is generated after the first trading day of a new month closes, then executed at the next open.

Check a zero-position new-entry plan:

```powershell
python -m zhquant.rotation_backtest_cli strategies/semiconductor_momentum_rotation.json --start 2025-01-01 --initial-cash 100000 --new-entry
```

Use `--new-entry` when you are starting from cash today. It treats the rotation output as a candidate list, then applies a stricter new-entry gate before suggesting any initial position.

You can print both existing-portfolio maintenance and new-cash entry analysis:

```powershell
python -m zhquant.rotation_backtest_cli strategies/semiconductor_momentum_rotation.json --start 2025-01-01 --initial-cash 100000 --live-action --new-entry
```

Interpretation:

```text
Current Action / Weight Changes:
Use this only if you have already been following the strategy and need to rebalance existing positions.

New Entry Plan:
Use this if you are starting from zero position today.
```

The rotation strategy is a portfolio-level strategy, not a single-ticker entry/exit strategy. It ranks the configured stock pool by 126-trading-day return, holds the top 3 names with positive momentum, and rebalances monthly at the next open.

Its benchmark is static equal-weight buy-and-hold of the same universe, so the comparison answers:

```text
Did dynamic rotation beat simply holding the semiconductor / memory basket?
```

Current rotation strategy file:

```text
strategies/semiconductor_momentum_rotation.json
```

Current universe:

```text
AMD, MU, SNDK, NVDA, AVGO, TSM, MRVL, IREN, NBIS, INTC, QCOM, ASML, LITE, ETN, ANET, ZS
```

Current rotation settings:

```text
Score: 126-day close-to-close return
Selection: top 3 positive-momentum stocks
Rebalance: monthly
Execution: next open
Slippage: 5 bps
Commission: 1 bp
Catastrophic stop loss: -30% from entry
MA200 exit: enabled
ATR trailing stop: enabled after +25% profit, using highest close - 4.5 * ATR20
```

## Semiconductor Rotation Strategy Details

The rotation strategy has two separate decision layers.

Layer 1 selects what the system wants to own:

```text
Rotation Score = 126-trading-day close-to-close return
Eligible symbol = Rotation Score > 0
Selection = top 3 eligible symbols
Target weight = equal weight across selected symbols
Rebalance schedule = monthly
Execution = next open after the rotation signal date
```

This layer answers:

```text
Which stocks are the current leaders in the configured semiconductor / AI infrastructure basket?
```

Layer 2 decides whether fresh cash should enter now:

```text
New Entry Gate
```

This layer exists because a portfolio that started months ago may already have strong cost basis, while a new account starting today can easily chase a short-term high. For a zero-position account, do not blindly buy `Target Weights`. Use `New Entry Plan`.

New-entry candidates are the current rotation selections. Each candidate is classified as:

```text
BUY_NOW
WAIT_PULLBACK
TOO_EXTENDED
INVALID_TREND
INSUFFICIENT_DATA
```

Trend gate:

```text
Trend_OK =
Close > MA50
AND Close > MA200
AND 126d return > 0
```

Do-not-chase filter:

```text
TOO_EXTENDED if:
RSI14 > 72
OR Close > 1.12 * EMA20
OR Close > 1.25 * MA50
OR Close > MA50 + 4 * ATR20
OR 5d return > trailing 252-day 90th percentile of 5d returns
```

Healthy pullback entry:

```text
BUY_NOW if:
Trend_OK
AND not TOO_EXTENDED
AND Close > MA50
AND RSI14 between 35 and 68
AND Close <= 1.08 * EMA20
AND Pullback_From_20d_High between -2% and -12%
AND Volume_5d_avg <= Volume_20d_avg
```

EMA20 reclaim entry:

```text
BUY_NOW if:
Trend_OK
AND not TOO_EXTENDED
AND Low <= EMA20
AND Close >= EMA20
AND Close > Previous_Day_High
AND not a high-volume down day
```

High-volume down day check:

```text
Close < Previous_Close
AND Volume > 1.2 * Volume20
```

Initial sizing for a new account:

```text
If status = BUY_NOW:
Suggested initial weight = 50% of target weight

If target weight is 33.33%:
Initial buy is 16.67%

If status is not BUY_NOW:
Suggested initial weight = 0%
```

This is intentionally conservative. The rotation layer tells you what is strong; the new-entry layer prevents a new account from opening full target size after a large short-term run.

Operational workflow:

```text
1. Run --new-entry.
2. Buy only candidates marked BUY_NOW.
3. Use suggested_initial_weight for the first tranche.
4. Keep WAIT_PULLBACK and TOO_EXTENDED names on watch.
5. Re-run after pullbacks, EMA20 reclaims, or the next monthly rotation signal.
6. After you have live positions, use --live-action for ongoing rebalance checks.
```

For the current universe, `--start 2025-01-01` is usually enough because SNDK data starts in 2025 and the strategy needs 126 trading days of warmup. If you remove short-history symbols, you can use an earlier start date for longer backtests.

Layer 3 manages open-position risk every day:

```text
Daily Risk Exit
```

Monthly rotation decides the preferred holdings, but risk exits are checked after every daily close. If a risk exit triggers, the position is sold at the next open. The strategy does not have to wait until the next monthly rebalance.

Current daily risk rules:

```text
Catastrophic_Stop:
Sell if Close < Entry_Price * 0.70

MA200_Exit:
Sell if Close < MA200

ATR_Trailing_Stop:
Activate only after unrealized profit >= 25%
Trailing_Stop = Highest_Close_Since_Entry - 4.5 * ATR20
Sell if Close < Trailing_Stop
```

Risk order reasons in the order log:

```text
risk_stop_loss
risk_ma200_exit
risk_atr_trailing_stop
```

The fixed stop is intentionally wide. Semiconductor, memory, AI infrastructure, and neocloud-related stocks can have normal pullbacks larger than 10%-15%. A tight fixed stop can turn a trend strategy into a whipsaw strategy. The current default uses:

```text
-30% catastrophic stop for true damage control
4.5 ATR trailing stop for profit protection after a strong move
MA200 exit for major trend failure
```

Operational interpretation:

```text
Rotation target says what the strategy wants to own.
New Entry Plan says whether fresh cash should enter today.
Daily Risk Exit says when an existing position is wrong or profit protection has triggered.
```
