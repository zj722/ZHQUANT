from __future__ import annotations

from pathlib import Path

import pandas as pd


class PlotError(ValueError):
    """Raised when a backtest chart cannot be written."""


def plot_candlestick_with_orders(
    market_data: pd.DataFrame,
    order_log: pd.DataFrame,
    ticker: str,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise PlotError("matplotlib is required to write backtest charts") from exc

    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required - set(market_data.columns))
    if missing:
        raise PlotError(f"Market data is missing columns: {missing}")

    data = market_data.sort_index().copy()
    if data.empty:
        raise PlotError("Cannot plot empty market data")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    dates = pd.to_datetime(data.index)
    x_values = mdates.date2num(dates.to_pydatetime())
    candle_width = 0.6

    fig, (price_ax, volume_ax, rsi_ax) = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1, 1]},
    )
    fig.suptitle(title or f"{ticker.upper()} Backtest Orders", fontsize=14)

    for x_value, (_, row) in zip(x_values, data.iterrows(), strict=False):
        open_price = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        color = "#1f9d55" if close >= open_price else "#d64545"

        price_ax.vlines(x_value, low, high, color=color, linewidth=1)
        body_low = min(open_price, close)
        body_height = abs(close - open_price)
        if body_height == 0:
            body_height = max(close * 0.001, 0.01)
        price_ax.add_patch(
            Rectangle(
                (x_value - candle_width / 2, body_low),
                candle_width,
                body_height,
                facecolor=color,
                edgecolor=color,
                alpha=0.85,
            )
        )

    indicators = _strategy_indicators(data)
    _plot_price_indicators(price_ax, x_values, indicators)

    volume_colors = ["#1f9d55" if close >= open_ else "#d64545" for open_, close in zip(data["open"], data["close"], strict=False)]
    volume_ax.bar(x_values, data["volume"].astype(float), width=candle_width, color=volume_colors, alpha=0.35)
    _plot_volume_indicators(volume_ax, x_values, indicators)
    _plot_rsi(rsi_ax, x_values, indicators)

    if not order_log.empty:
        orders = order_log.copy()
        orders = orders[orders["symbol"].astype(str).str.upper() == ticker.upper()]
        if not orders.empty:
            orders["date"] = pd.to_datetime(orders["date"])
            _plot_order_markers(price_ax, mdates, orders)

    price_ax.set_ylabel("Price")
    volume_ax.set_ylabel("Volume")
    rsi_ax.set_ylabel("RSI14")
    price_ax.grid(True, color="#e5e7eb", linewidth=0.8)
    volume_ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    rsi_ax.grid(True, color="#e5e7eb", linewidth=0.8)
    price_ax.margins(x=0.01)
    price_ax.legend(loc="upper left", fontsize=8)
    volume_ax.legend(loc="upper left", fontsize=8)
    rsi_ax.legend(loc="upper left", fontsize=8)
    rsi_ax.set_ylim(0, 100)
    rsi_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output


def _strategy_indicators(data: pd.DataFrame) -> pd.DataFrame:
    close = data["close"].astype(float)
    volume = data["volume"].astype(float)
    indicators = pd.DataFrame(index=data.index)
    indicators["ema20"] = close.ewm(span=20, min_periods=20, adjust=False).mean()
    indicators["ma50"] = close.rolling(50, min_periods=50).mean()
    indicators["ma200"] = close.rolling(200, min_periods=200).mean()
    indicators["volume_ma5"] = volume.rolling(5, min_periods=5).mean()
    indicators["volume_ma20"] = volume.rolling(20, min_periods=20).mean()
    indicators["rsi14"] = _rsi(close, 14)
    return indicators


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    losses = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gains / losses.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def _plot_price_indicators(price_ax: object, x_values: object, indicators: pd.DataFrame) -> None:
    price_ax.plot(x_values, indicators["ema20"], color="#2563eb", linewidth=1.2, label="EMA20")
    price_ax.plot(x_values, indicators["ma50"], color="#f59e0b", linewidth=1.2, label="MA50")
    price_ax.plot(x_values, indicators["ma200"], color="#7c3aed", linewidth=1.2, label="MA200")


def _plot_volume_indicators(volume_ax: object, x_values: object, indicators: pd.DataFrame) -> None:
    volume_ax.plot(x_values, indicators["volume_ma5"], color="#2563eb", linewidth=1.0, label="Vol MA5")
    volume_ax.plot(x_values, indicators["volume_ma20"], color="#111827", linewidth=1.0, label="Vol MA20")


def _plot_rsi(rsi_ax: object, x_values: object, indicators: pd.DataFrame) -> None:
    rsi_ax.plot(x_values, indicators["rsi14"], color="#7c3aed", linewidth=1.2, label="RSI14")
    for level, color, label in [
        (35, "#9ca3af", "RSI 35"),
        (50, "#d1d5db", "RSI 50"),
        (68, "#9ca3af", "RSI 68"),
        (72, "#ef4444", "RSI 72"),
    ]:
        rsi_ax.axhline(level, color=color, linewidth=0.8, linestyle="--", label=label)


def _plot_order_markers(price_ax: object, mdates: object, orders: pd.DataFrame) -> None:
    styles = {
        "BUY": {"marker": "^", "color": "#047857", "offset": -0.025},
        "ADD": {"marker": "^", "color": "#2563eb", "offset": -0.045},
        "SELL": {"marker": "v", "color": "#dc2626", "offset": 0.025},
        "REDUCE": {"marker": "v", "color": "#f97316", "offset": 0.045},
        "TRIM": {"marker": "v", "color": "#a855f7", "offset": 0.065},
    }

    for _, order in orders.iterrows():
        action = str(order["action"]).upper()
        style = styles.get(action)
        if style is None:
            continue
        x_value = mdates.date2num(pd.Timestamp(order["date"]).to_pydatetime())
        price = float(order["price"])
        y_value = price * (1.0 + float(style["offset"]))
        price_ax.scatter(
            [x_value],
            [y_value],
            marker=style["marker"],
            s=85,
            color=style["color"],
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        price_ax.annotate(
            action,
            xy=(x_value, y_value),
            xytext=(0, 10 if action == "SELL" else -14),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=8,
            color=style["color"],
            weight="bold",
            zorder=6,
        )
