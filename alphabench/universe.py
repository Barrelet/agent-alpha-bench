"""Tradeable universe and benchmark definition.

Universe rule (v1): the 50 largest US-listed, US-domiciled companies by market
capitalisation as of 2024-12-31 (one share class per company). The list below is
an approximate snapshot compiled from memory and should be verified once against a
point-in-time source (e.g. a Dec-2024 S&P 500 constituent file with market caps)
before the headline run.

Survivorship note: a list fixed at 2024-12-31 and used for a replay starting
2024-01-01 has look-ahead in *membership* (we know these names were big at the end
of 2024). This is documented, not hidden; a point-in-time constituent list is the
planned upgrade.
"""

from __future__ import annotations

UNIVERSE_ASOF = "2024-12-31"
BENCHMARK = "SPY"  # never tradeable by LLM agents; buy-and-hold control only

# ticker -> sector (GICS-style, coarse)
UNIVERSE: dict[str, str] = {
    # Technology
    "AAPL": "Technology", "NVDA": "Technology", "MSFT": "Technology",
    "AVGO": "Technology", "ORCL": "Technology", "CRM": "Technology",
    "CSCO": "Technology", "ACN": "Technology", "NOW": "Technology",
    "AMD": "Technology", "ADBE": "Technology", "IBM": "Technology",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary",
    # Communication Services
    "GOOGL": "Communication Services", "META": "Communication Services",
    "NFLX": "Communication Services", "TMUS": "Communication Services",
    "DIS": "Communication Services",
    # Financials
    "BRK-B": "Financials", "JPM": "Financials", "V": "Financials",
    "MA": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "AXP": "Financials", "MS": "Financials",
    # Consumer Staples
    "WMT": "Consumer Staples", "COST": "Consumer Staples", "PG": "Consumer Staples",
    "KO": "Consumer Staples", "PEP": "Consumer Staples", "PM": "Consumer Staples",
    # Health Care
    "LLY": "Health Care", "UNH": "Health Care", "JNJ": "Health Care",
    "ABBV": "Health Care", "MRK": "Health Care", "ABT": "Health Care",
    "TMO": "Health Care",
    # Energy / Industrials / Materials
    "XOM": "Energy", "CVX": "Energy",
    "GE": "Industrials", "CAT": "Industrials", "RTX": "Industrials",
    "LIN": "Materials",
}

TICKERS: list[str] = list(UNIVERSE.keys())
ALL_SYMBOLS: list[str] = TICKERS + [BENCHMARK]

assert len(TICKERS) == 50, f"expected 50 tickers, got {len(TICKERS)}"
assert len(TICKERS) == len(set(TICKERS)), "duplicate tickers"


def sector_of(symbol: str) -> str:
    return UNIVERSE.get(symbol, "Benchmark" if symbol == BENCHMARK else "Unknown")
