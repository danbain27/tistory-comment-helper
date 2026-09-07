"""Universe construction: every USDT perpetual Bybit currently lists, filtered down to
what is actually researchable (liquidity, listing age, available history).

Survivorship warning: Bybit's instrument list contains only LIVE contracts. Coins that
were delisted (usually after collapsing) are absent, so any statistic pooled over this
universe is biased upward. The scan reports this; do not read universe-wide numbers as
what a live bot would have earned.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from . import data as dataio

EXCLUDE_PATTERNS = (r"-\d{2}[A-Z]{3}\d{2}$",)   # dated futures, just in case


def fetch_universe(min_turnover: float = 5_000_000, min_listed_days: int = 365,
                   top_n: int | None = None, exclude: tuple[str, ...] = (),
                   category: str = "linear") -> pd.DataFrame:
    """Live USDT perps, ranked by 24h turnover."""
    inst = dataio.fetch_bybit_instruments(category=category)
    tick = dataio.fetch_bybit_tickers(category=category)
    if inst.empty:
        return inst
    u = inst.merge(tick, on="symbol", how="left") if not tick.empty else inst
    u["turnover24h"] = pd.to_numeric(u.get("turnover24h"), errors="coerce").fillna(0.0)
    for pat in EXCLUDE_PATTERNS + tuple(exclude):
        u = u[~u["symbol"].str.contains(pat, regex=True, na=False)]
    u = u[(u["turnover24h"] >= min_turnover) & (u["listed_days"].fillna(0) >= min_listed_days)]
    u = u.sort_values("turnover24h", ascending=False).reset_index(drop=True)
    if top_n:
        u = u.head(top_n).copy()
    u["rank_turnover"] = np.arange(1, len(u) + 1)
    return u


def local_universe(data_dir: str, interval: str = "15") -> list[str]:
    """Symbols already downloaded into data/."""
    if not os.path.isdir(data_dir):
        return []
    suf = f"_{interval}m.csv"
    return sorted(f[:-len(suf)] for f in os.listdir(data_dir) if f.endswith(suf))


def synthetic_universe(n: int = 12) -> pd.DataFrame:
    """Offline placeholder so the pipeline can be exercised without exchange access."""
    return pd.DataFrame({"symbol": [f"SYN{i:02d}USDT" for i in range(1, n + 1)],
                         "turnover24h": np.linspace(5e8, 5e6, n),
                         "listed_days": 1000.0,
                         "rank_turnover": np.arange(1, n + 1)})


def resolve(a, data_dir: str) -> tuple[list[str], pd.DataFrame, str]:
    """Turn CLI options into a symbol list.

    --symbols A B C   explicit
    --universe all    every live USDT perp passing the liquidity/age filters
    --universe local  whatever is already in data/
    --universe topN   e.g. 'top50'
    """
    if a.symbols:
        return list(a.symbols), pd.DataFrame({"symbol": a.symbols}), "explicit"

    spec = (a.universe or "local").lower()
    if spec == "local":
        syms = local_universe(data_dir, a.interval)
        if syms:
            return syms, pd.DataFrame({"symbol": syms}), "local"
    top_n = None
    m = re.fullmatch(r"top(\d+)", spec)
    if m:
        top_n = int(m.group(1))
    try:
        u = fetch_universe(min_turnover=a.min_turnover, min_listed_days=a.min_listed_days,
                           top_n=top_n)
        if len(u):
            return u["symbol"].tolist(), u, "bybit"
    except Exception as exc:
        print(f"[universe] Bybit instrument list unavailable: {exc}")
    syms = local_universe(data_dir, a.interval)
    if syms:
        return syms, pd.DataFrame({"symbol": syms}), "local"
    if getattr(a, "allow_synthetic", False):
        u = synthetic_universe(getattr(a, "synthetic_symbols", 12))
        return u["symbol"].tolist(), u, "synthetic"
    raise RuntimeError("no universe: run fetch_data.py first, or pass --symbols")
