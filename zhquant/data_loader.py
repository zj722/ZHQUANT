from __future__ import annotations

from typing import Iterable

import pandas as pd


class DataLoadError(ValueError):
    """Raised when market data cannot be loaded."""


REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def download_yfinance_ohlcv(
    symbols: Iterable[str],
    period: str | None = "1mo",
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Download OHLCV data from yfinance.

    Returns a dictionary keyed by upper-case ticker. Each DataFrame is indexed
    by date and contains lower-case open/high/low/close/volume columns.
    """

    normalized_symbols = sorted({symbol.strip().upper() for symbol in symbols if symbol and symbol.strip()})
    if not normalized_symbols:
        raise DataLoadError("At least one symbol is required")

    try:
        import yfinance as yf
    except ImportError as exc:
        raise DataLoadError("yfinance is not installed. Run: pip install yfinance") from exc

    kwargs: dict[str, object] = {
        "tickers": normalized_symbols,
        "interval": interval,
        "group_by": "ticker",
        "auto_adjust": False,
        "progress": False,
        "threads": True,
    }
    if start or end:
        kwargs["start"] = start
        kwargs["end"] = end
    else:
        kwargs["period"] = period or "1mo"

    raw = yf.download(**kwargs)
    if raw.empty:
        raise DataLoadError(f"No data returned by yfinance for {', '.join(normalized_symbols)}")

    market_data: dict[str, pd.DataFrame] = {}
    for symbol in normalized_symbols:
        frame = _extract_symbol_frame(raw, symbol, len(normalized_symbols) == 1)
        frame = _normalize_ohlcv(frame)
        if frame.empty:
            raise DataLoadError(f"No usable OHLCV rows for {symbol}")
        market_data[symbol] = frame

    return market_data


def _extract_symbol_frame(raw: pd.DataFrame, symbol: str, single_symbol: bool) -> pd.DataFrame:
    if not isinstance(raw.columns, pd.MultiIndex):
        if not single_symbol:
            raise DataLoadError(f"Unexpected yfinance format while reading {symbol}")
        return raw.copy()

    level0 = raw.columns.get_level_values(0)
    level1 = raw.columns.get_level_values(1)
    if symbol in level0:
        return raw[symbol].copy()
    if symbol in level1:
        return raw.xs(symbol, axis=1, level=1).copy()
    raise DataLoadError(f"Missing yfinance data for {symbol}")


def _normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {str(col): str(col).strip().lower().replace(" ", "_") for col in frame.columns}
    frame = frame.rename(columns=renamed)
    frame = frame.rename(columns={"adj_close": "adj_close"})

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataLoadError(f"Downloaded data is missing columns: {missing}")

    result = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    result.index = pd.to_datetime(result.index).tz_localize(None)
    result = result.sort_index()
    result = result.apply(pd.to_numeric, errors="coerce")
    result = result.dropna(subset=list(REQUIRED_COLUMNS))
    return result

