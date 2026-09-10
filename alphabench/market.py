"""Market data: fetch, cache, point-in-time slicing, indicators.

Design contract
---------------
`MarketData` holds five wide frames (date x symbol): open, high, low, close, volume.
Everything downstream receives a `MarketData` produced by `.asof(date)`, which
contains ONLY bars with index <= date. That is the single look-ahead guard: if
you only ever build payloads and screens from an `.asof()` slice, you cannot
peek. Fills use `md.open.loc[next_day]` explicitly in the engine, never a slice.

Prices are split- and dividend-adjusted (yfinance auto_adjust=True) so that
long-run equity curves are not distorted by corporate actions. The prompt shows
adjusted candles; that is a known, documented simplification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FIELDS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class MarketData:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame

    # ---- construction -----------------------------------------------------
    @classmethod
    def from_long(cls, df: pd.DataFrame) -> "MarketData":
        """df columns: date, symbol, open, high, low, close, volume."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        wide = {f: df.pivot(index="date", columns="symbol", values=f).sort_index() for f in FIELDS}
        return cls(**wide)

    def to_long(self) -> pd.DataFrame:
        parts = []
        for f in FIELDS:
            s = getattr(self, f).stack(future_stack=True).rename(f)
            parts.append(s)
        out = pd.concat(parts, axis=1).reset_index()
        out.columns = ["date", "symbol", *FIELDS]
        return out.dropna(subset=["close"])

    # ---- properties -------------------------------------------------------
    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    @property
    def last_date(self) -> pd.Timestamp:
        return self.close.index[-1]

    # ---- point-in-time ------------------------------------------------------
    def asof(self, date) -> "MarketData":
        """Return only bars with index <= date. The look-ahead guard."""
        date = pd.Timestamp(date)
        return MarketData(**{f: getattr(self, f).loc[:date] for f in FIELDS})

    def between(self, start, end) -> "MarketData":
        return MarketData(**{f: getattr(self, f).loc[pd.Timestamp(start): pd.Timestamp(end)] for f in FIELDS})

    def symbol_bars(self, symbol: str) -> pd.DataFrame:
        """OHLCV frame for one symbol (daily)."""
        return pd.DataFrame({f: getattr(self, f)[symbol] for f in FIELDS}).dropna(subset=["close"])

    # ---- indicators ---------------------------------------------------------
    def weekly(self, symbol: str) -> pd.DataFrame:
        """Resample daily bars to weekly (W-FRI). The last row may be a partial
        week — that is correct point-in-time behaviour, not a bug."""
        d = self.symbol_bars(symbol)
        d = d.assign(last_bar=d.index)
        w = d.resample("W-FRI").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "last_bar": "last"}
        ).dropna(subset=["close"])
        # label each week by its last actual trading day so no candle is ever dated in the future
        return w.set_index("last_bar").rename_axis("date")

    def summary_table(self, symbols: list[str] | None = None) -> pd.DataFrame:
        """One row per symbol with the fields shown in the universe table.
        Uses only the data in this (already sliced) object."""
        c = self.close if symbols is None else self.close[symbols]
        v = self.volume if symbols is None else self.volume[symbols]
        ret = lambda n: (c.iloc[-1] / c.iloc[-1 - n] - 1.0) if len(c) > n else pd.Series(np.nan, index=c.columns)
        logret = np.log(c).diff()
        out = pd.DataFrame({
            "price": c.iloc[-1],
            "ret_1d": ret(1),
            "ret_10d": ret(10),
            "ret_30d": ret(30),
            "vol_30d_ann": logret.tail(30).std() * np.sqrt(252),
            "rsi_1d": rsi(c, 14).iloc[-1],
            "pct_from_30d_high": c.iloc[-1] / c.tail(30).max() - 1.0,
            "adv_20d_usd": (c * v).tail(20).mean(),
        })
        out.index.name = "symbol"
        return out


# ---- indicators -------------------------------------------------------------
def rsi(close: pd.DataFrame | pd.Series, n: int = 14) -> pd.DataFrame | pd.Series:
    """Wilder RSI computed on past data only (EWM with alpha=1/n)."""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    ru = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rd = down.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = ru / rd.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.where(rd != 0.0, 100.0)


# ---- fetch / cache ------------------------------------------------------------
def fetch_yfinance(symbols: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Download adjusted daily OHLCV from Yahoo Finance. Returns a long frame."""
    import yfinance as yf  # imported lazily so the engine works offline
    logging.getLogger('yfinance').setLevel(logging.CRITICAL)

    raw = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False, group_by="column", threads=True)
    if raw.empty:
        raise RuntimeError("yfinance returned no data — check network / tickers")
    frames = []
    for f, F in zip(FIELDS, ("Open", "High", "Low", "Close", "Volume")):
        s = raw[F] if isinstance(raw.columns, pd.MultiIndex) else raw[[F]].rename(columns={F: symbols[0]})
        frames.append(s.stack(future_stack=True).rename(f))
    out = pd.concat(frames, axis=1).reset_index()
    out.columns = ["date", "symbol", *FIELDS]
    out = out.dropna(subset=["close"]).sort_values(["date", "symbol"]).reset_index(drop=True)
    missing = set(symbols) - set(out["symbol"].unique())
    if missing:
        log.warning("no data for: %s", sorted(missing))
    return out


def load_prices(symbols: list[str], start: str, end: str | None, cache_path: Path, refresh: bool = False) -> MarketData:
    """Load from parquet cache if present, else fetch and cache."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        df = pd.read_parquet(cache_path)
        log.info("loaded %d rows from %s", len(df), cache_path)
    else:
        df = fetch_yfinance(symbols, start, end)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        log.info("fetched %d rows and cached to %s", len(df), cache_path)
    return MarketData.from_long(df)


# ---- synthetic data for offline development -----------------------------------
def synthetic_prices(symbols: list[str], start: str, end: str, seed: int = 7, sectors: dict[str, str] | None = None) -> MarketData:
    """Geometric-Brownian daily bars with a market factor and a sector factor.
    Only for testing the plumbing; never for reporting results."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)  # business days ~ trading days (holidays ignored)
    n, k = len(dates), len(symbols)
    sectors = sectors or {}
    sec_names = sorted(set(sectors.get(s, "none") for s in symbols))
    mkt = rng.normal(0.0004, 0.010, n)
    sec = {name: rng.normal(0.0, 0.006, n) for name in sec_names}
    close = np.empty((n, k))
    for j, s in enumerate(symbols):
        beta = rng.uniform(0.6, 1.4)
        idio = rng.normal(0.0, rng.uniform(0.008, 0.02), n)
        r = beta * mkt + 0.7 * sec[sectors.get(s, "none")] + idio
        close[:, j] = 100.0 * rng.uniform(0.3, 5.0) * np.exp(np.cumsum(r))
    close = pd.DataFrame(close, index=dates, columns=symbols)
    gap = rng.normal(0, 0.004, (n, k))
    open_ = close.shift(1).fillna(close.iloc[0]) * np.exp(gap)
    hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.006, (n, k))))
    lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.006, (n, k))))
    vol = pd.DataFrame(rng.lognormal(15, 0.5, (n, k)), index=dates, columns=symbols)
    return MarketData(open=open_, high=hi, low=lo, close=close, volume=vol)
