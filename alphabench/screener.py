"""Deterministic daily screener (TradeRank rule, equities-only).

score = 20-day average daily dollar volume * (1 + |10-day return|)

The top-N by score are shown to the agent alongside any held positions. The
screener must be called on an `.asof()` slice; it uses only trailing windows.
"""

from __future__ import annotations

import pandas as pd

from .market import MarketData


def screen(md: MarketData, tickers: list[str], top_n: int = 5) -> pd.DataFrame:
    c = md.close[tickers]
    v = md.volume[tickers]
    adv = (c * v).tail(20).mean()
    mom = c.iloc[-1] / c.iloc[-11] - 1.0 if len(c) > 10 else pd.Series(0.0, index=c.columns)
    score = adv * (1.0 + mom.abs())
    out = pd.DataFrame({"adv_20d_usd": adv, "mom_10d": mom, "score": score}).dropna()
    out = out.sort_values("score", ascending=False)
    out["rank"] = range(1, len(out) + 1)
    out.index.name = "symbol"
    return out.head(top_n)
