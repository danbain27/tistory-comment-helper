"""Universe construction: every USDT perpetual Bybit currently lists, filtered down to
what is actually researchable (instrument class, liquidity, listing age, history).

Two filters, and they are NOT the same thing:

  1. Instrument class. Bybit's `linear` category is no longer crypto-only - it carries
     tokenised equities ("stock"), leveraged/index ETFs ("ETF") and commodities
     ("commodity", e.g. XAU/XAG gold-silver, CL/BZ crude). Those trade on exchange
     hours, gap over weekends and mean-revert on a different clock. Pooling them into a
     crypto study contaminates the universe-wide statistics, so they are excluded by
     default. "" and "innovation" are crypto and stay.

  2. Liquidity. `turnover24h` is a SNAPSHOT of today. Selecting a 3-year backtest
     universe on today's turnover is look-ahead: a coin whose volume exploded last week
     gets its whole illiquid history admitted as if it had been tradeable. The
     point-in-time gate (`pit_liquidity_mask`) re-checks liquidity bar by bar from the
     candles themselves and keeps only the stretch a bot could actually have traded.

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

# Bybit `symbolType` values that are not crypto.
NON_CRYPTO_SYMBOL_TYPES = ("stock", "ETF", "commodity")

TYPE_CACHE = "instrument_types.csv"

# Point-in-time liquidity gate defaults (15m bars).
BARS_PER_DAY = 96
PIT_WINDOW_DAYS = 30
PIT_MIN_WINDOW_DAYS = 10


# --------------------------------------------------------------------------
# instrument class
# --------------------------------------------------------------------------
def _norm_types(types) -> set[str]:
    return {str(t).strip().lower() for t in types}


def drop_non_crypto(u: pd.DataFrame,
                    exclude_types: tuple[str, ...] = NON_CRYPTO_SYMBOL_TYPES
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a universe frame into (crypto, non-crypto). Missing symbolType -> crypto."""
    if u.empty or "symbolType" not in u.columns:
        return u, u.iloc[0:0]
    bad = _norm_types(exclude_types)
    is_bad = u["symbolType"].fillna("").astype(str).str.strip().str.lower().isin(bad)
    return u[~is_bad].reset_index(drop=True), u[is_bad].reset_index(drop=True)


def save_type_map(u: pd.DataFrame, data_dir: str) -> str | None:
    """Cache symbol -> symbolType so an offline/local run can still exclude equities."""
    if u.empty or "symbolType" not in u.columns:
        return None
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, TYPE_CACHE)
    u[["symbol", "symbolType"]].to_csv(path, index=False)
    return path


def load_type_map(data_dir: str) -> dict[str, str]:
    path = os.path.join(data_dir, TYPE_CACHE)
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path).fillna("")
    return dict(zip(df["symbol"], df["symbolType"].astype(str)))


def filter_symbol_types(symbols: list[str], data_dir: str,
                        exclude_types: tuple[str, ...] = NON_CRYPTO_SYMBOL_TYPES
                        ) -> tuple[list[str], list[str]]:
    """Apply the instrument-class filter to a bare symbol list using the cached map.

    Returns (kept, dropped). With no cache nothing is dropped - and the caller says so
    out loud, because silently admitting 200 tokenised stocks is the failure mode here.
    """
    tmap = load_type_map(data_dir)
    if not tmap:
        return list(symbols), []
    bad = _norm_types(exclude_types)
    kept, dropped = [], []
    for s in symbols:
        (dropped if str(tmap.get(s, "")).strip().lower() in bad else kept).append(s)
    return kept, dropped


# --------------------------------------------------------------------------
# point-in-time liquidity
# --------------------------------------------------------------------------
def pit_liquidity_mask(df: pd.DataFrame, min_daily_turnover: float,
                       bars_per_day: int = BARS_PER_DAY,
                       window_days: int = PIT_WINDOW_DAYS,
                       min_window_days: int = PIT_MIN_WINDOW_DAYS) -> pd.Series:
    """True where the symbol was liquid AT THAT BAR.

    Turnover is approximated per bar as close*volume (quote units) - the cached CSVs
    keep base volume only. The test is the median of trailing-24h turnover over a
    30-day window, so a single volume spike cannot open the gate. Past bars only.
    """
    if min_daily_turnover <= 0 or df.empty:
        return pd.Series(True, index=df.index)
    turnover = df["close"].astype(float) * df["volume"].astype(float)
    daily = turnover.rolling(bars_per_day, min_periods=bars_per_day).sum()
    med = daily.rolling(bars_per_day * window_days,
                        min_periods=bars_per_day * min_window_days).median()
    return (med >= min_daily_turnover).fillna(False)


def first_liquid_index(mask) -> int:
    """Position of the first liquid bar, or -1 if the symbol never qualifies."""
    idx = np.flatnonzero(np.asarray(getattr(mask, "values", mask), dtype=bool))
    return int(idx[0]) if len(idx) else -1


def longest_liquid_run(mask) -> tuple[int, int]:
    """Longest contiguous True run as positional [start, end). All-False -> (0, 0).

    Diagnostic only - see apply_pit_liquidity for why the cut is leading-edge.
    """
    v = np.asarray(getattr(mask, "values", mask), dtype=bool)
    best = best_end = cur = 0
    for i, x in enumerate(v):
        cur = cur + 1 if x else 0
        if cur > best:
            best, best_end = cur, i + 1
    return best_end - best, best_end


def apply_pit_liquidity(d: pd.DataFrame, min_daily_turnover: float,
                        **kw) -> tuple[pd.DataFrame, int]:
    """Trim the LEADING illiquid history. Returns (df, bars_dropped).

    Only the front of the series is cut, never the tail, and the frame is kept
    contiguous so indicators are never computed across a stitched-together price
    series. Two reasons for leading-edge only:

      * It kills the actual bias - today's turnover ranking back-dating itself onto
        years when the coin traded a few hundred thousand a day.
      * The cut point is then fixed by early bars alone, so how much data the test
        window contains cannot move the train/validation boundary. Cutting to the
        longest liquid RUN would make the split depend on test-window volume, which
        is the leak the scan is built to avoid.

    A symbol that dries up later keeps its illiquid tail; `pit_liquid_frac` in the
    scan summary reports that, and min_bars removes the hopeless cases.
    """
    if min_daily_turnover <= 0 or d.empty:
        return d, 0
    i = first_liquid_index(pit_liquidity_mask(d, min_daily_turnover, **kw))
    if i < 0:
        return d.iloc[0:0], len(d)
    return d.iloc[i:].reset_index(drop=True), i


def liquid_fraction(d: pd.DataFrame, min_daily_turnover: float,
                    upto_frac: float = 0.75, **kw) -> float:
    """Share of bars that clear the liquidity floor, measured over the first
    `upto_frac` of the frame (train+validation) so the number never reads the test
    window."""
    if min_daily_turnover <= 0 or d.empty:
        return 1.0
    cut = max(1, int(len(d) * upto_frac))
    return float(pit_liquidity_mask(d.iloc[:cut], min_daily_turnover, **kw).mean())


# --------------------------------------------------------------------------
# universe assembly
# --------------------------------------------------------------------------
def fetch_universe(min_turnover: float = 10_000_000, min_listed_days: int = 365,
                   top_n: int | None = None, exclude: tuple[str, ...] = (),
                   category: str = "linear",
                   exclude_types: tuple[str, ...] = NON_CRYPTO_SYMBOL_TYPES,
                   data_dir: str | None = None, verbose: bool = True) -> pd.DataFrame:
    """Live USDT crypto perps, ranked by 24h turnover."""
    inst = dataio.fetch_bybit_instruments(category=category)
    tick = dataio.fetch_bybit_tickers(category=category)
    if inst.empty:
        return inst
    u = inst.merge(tick, on="symbol", how="left") if not tick.empty else inst
    u["turnover24h"] = pd.to_numeric(u.get("turnover24h"), errors="coerce").fillna(0.0)
    for pat in EXCLUDE_PATTERNS + tuple(exclude):
        u = u[~u["symbol"].str.contains(pat, regex=True, na=False)]
    if data_dir:
        save_type_map(u, data_dir)
    u, non_crypto = drop_non_crypto(u, exclude_types)
    if verbose and len(non_crypto):
        by = non_crypto.groupby(non_crypto["symbolType"].str.lower()).size().to_dict()
        top = non_crypto.nlargest(5, "turnover24h")["symbol"].tolist()
        print(f"[universe] excluded {len(non_crypto)} non-crypto instruments {by}; "
              f"largest by turnover: {', '.join(top)}")
    u = u[(u["turnover24h"] >= min_turnover) & (u["listed_days"].fillna(0) >= min_listed_days)]
    u = u.sort_values("turnover24h", ascending=False).reset_index(drop=True)
    if top_n:
        u = u.head(top_n).copy()
    u["rank_turnover"] = np.arange(1, len(u) + 1)
    return u


def local_universe(data_dir: str, interval: str = "15",
                   exclude_types: tuple[str, ...] = NON_CRYPTO_SYMBOL_TYPES,
                   verbose: bool = True) -> list[str]:
    """Symbols already downloaded into data/, minus the non-crypto instruments."""
    if not os.path.isdir(data_dir):
        return []
    suf = f"_{interval}m.csv"
    syms = sorted(f[:-len(suf)] for f in os.listdir(data_dir) if f.endswith(suf))
    kept, dropped = filter_symbol_types(syms, data_dir, exclude_types)
    if verbose:
        if dropped:
            more = " ..." if len(dropped) > 6 else ""
            print(f"[universe] local: dropped {len(dropped)} non-crypto "
                  f"({', '.join(dropped[:6])}{more})")
        elif exclude_types and not load_type_map(data_dir):
            print(f"[universe] WARNING: no {TYPE_CACHE} in {data_dir} - instrument class "
                  f"cannot be checked, so tokenised stocks/ETFs may be scanned as coins. "
                  f"Run fetch_data.py once to write it.")
    return kept


def synthetic_universe(n: int = 12) -> pd.DataFrame:
    """Offline placeholder so the pipeline can be exercised without exchange access."""
    return pd.DataFrame({"symbol": [f"SYN{i:02d}USDT" for i in range(1, n + 1)],
                         "turnover24h": np.linspace(5e8, 5e6, n),
                         "listed_days": 1000.0,
                         "symbolType": "",
                         "rank_turnover": np.arange(1, n + 1)})


def resolve(a, data_dir: str) -> tuple[list[str], pd.DataFrame, str]:
    """Turn CLI options into a symbol list.

    --symbols A B C   explicit (instrument-class filter NOT applied - you named them)
    --universe all    every live USDT crypto perp passing the liquidity/age filters
    --universe local  whatever is already in data/, minus non-crypto instruments
    --universe topN   e.g. 'top50'
    """
    ex_types = tuple(getattr(a, "exclude_symbol_types", NON_CRYPTO_SYMBOL_TYPES) or ())
    if a.symbols:
        return list(a.symbols), pd.DataFrame({"symbol": a.symbols}), "explicit"

    spec = (a.universe or "local").lower()
    if spec == "local":
        syms = local_universe(data_dir, a.interval, ex_types)
        if syms:
            return syms, pd.DataFrame({"symbol": syms}), "local"
    top_n = None
    m = re.fullmatch(r"top(\d+)", spec)
    if m:
        top_n = int(m.group(1))
    try:
        u = fetch_universe(min_turnover=a.min_turnover, min_listed_days=a.min_listed_days,
                           top_n=top_n, exclude_types=ex_types, data_dir=data_dir)
        if len(u):
            return u["symbol"].tolist(), u, "bybit"
    except Exception as exc:
        print(f"[universe] Bybit instrument list unavailable: {exc}")
    syms = local_universe(data_dir, a.interval, ex_types)
    if syms:
        return syms, pd.DataFrame({"symbol": syms}), "local"
    if getattr(a, "allow_synthetic", False):
        u = synthetic_universe(getattr(a, "synthetic_symbols", 12))
        return u["symbol"].tolist(), u, "synthetic"
    raise RuntimeError("no universe: run fetch_data.py first, or pass --symbols")
