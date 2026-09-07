"""Data layer: Bybit fetch, CSV load, integrity checks, synthetic fallback."""
from __future__ import annotations

import os
import time
import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

def symbol_seed(symbol: str) -> int:
    """Stable seed so a synthetic symbol looks the same in every phase of a run."""
    return zlib.crc32(symbol.encode()) % 100_000


BYBIT_BASE = "https://api.bybit.com"
COLS = ["timestamp", "open", "high", "low", "close", "volume"]
INTERVAL_MS = {"1": 60_000, "5": 300_000, "15": 900_000, "60": 3_600_000, "240": 14_400_000}


# --------------------------------------------------------------------------
# Bybit REST (works wherever api.bybit.com is reachable; blocked in some CI)
# --------------------------------------------------------------------------
def fetch_bybit_klines(symbol: str, interval: str = "15", start_ms: int | None = None,
                       end_ms: int | None = None, category: str = "linear",
                       sleep: float = 0.12) -> pd.DataFrame:
    import requests  # imported lazily so offline use needs no network dep

    end_ms = end_ms or int(time.time() * 1000)
    start_ms = start_ms or end_ms - 365 * 24 * 3600 * 1000
    out, cursor = [], end_ms
    while cursor > start_ms:
        r = requests.get(f"{BYBIT_BASE}/v5/market/kline", timeout=30, params={
            "category": category, "symbol": symbol, "interval": interval,
            "end": cursor, "limit": 1000})
        r.raise_for_status()
        rows = r.json().get("result", {}).get("list", [])
        if not rows:
            break
        out.extend(rows)
        oldest = min(int(x[0]) for x in rows)
        if oldest <= start_ms or oldest >= cursor:
            break
        cursor = oldest - 1
        time.sleep(sleep)

    df = pd.DataFrame(out, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    df = df[COLS].astype(float)
    df["timestamp"] = df["timestamp"].astype("int64")
    df = df[df["timestamp"] >= start_ms]
    return _finalize(df)


def fetch_bybit_funding(symbol: str, start_ms: int, end_ms: int,
                        category: str = "linear", sleep: float = 0.12) -> pd.DataFrame:
    """Historical funding rates (8h). Returns columns [timestamp, funding_rate]."""
    import requests

    out, cursor = [], end_ms
    while cursor > start_ms:
        r = requests.get(f"{BYBIT_BASE}/v5/market/funding/history", timeout=30, params={
            "category": category, "symbol": symbol, "endTime": cursor, "limit": 200})
        r.raise_for_status()
        rows = r.json().get("result", {}).get("list", [])
        if not rows:
            break
        out.extend(rows)
        oldest = min(int(x["fundingRateTimestamp"]) for x in rows)
        if oldest <= start_ms or oldest >= cursor:
            break
        cursor = oldest - 1
        time.sleep(sleep)

    if not out:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    df = pd.DataFrame(out)
    df = pd.DataFrame({"timestamp": df["fundingRateTimestamp"].astype("int64"),
                       "funding_rate": df["fundingRate"].astype(float)})
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


# --------------------------------------------------------------------------
# Load / validate
# --------------------------------------------------------------------------
def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def load_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "timestamp" not in df.columns:
        for cand in ("open_time", "time", "date", "datetime"):
            if cand in df.columns:
                ts = pd.to_datetime(df[cand], utc=True, errors="coerce")
                df["timestamp"] = (ts.astype("int64") // 1_000_000)
                break
    if df["timestamp"].max() < 1e12:  # seconds -> ms
        df["timestamp"] = df["timestamp"].astype("int64") * 1000
    df = df[COLS].astype({c: float for c in COLS[1:]})
    df["timestamp"] = df["timestamp"].astype("int64")
    return _finalize(df)


@dataclass
class DataQuality:
    rows: int
    start: str
    end: str
    duplicates: int
    nan_rows: int
    gaps: int
    missing_bars: int
    ohlc_violations: int
    zero_volume: int
    extreme_returns: int

    def to_text(self) -> str:
        return (f"rows={self.rows} range=[{self.start} .. {self.end}] dup={self.duplicates} "
                f"nan={self.nan_rows} gaps={self.gaps} missing_bars={self.missing_bars} "
                f"ohlc_violations={self.ohlc_violations} zero_vol={self.zero_volume} "
                f"extreme_ret(>20%)={self.extreme_returns}")


def check_quality(df: pd.DataFrame, interval_ms: int = 900_000) -> DataQuality:
    dup = int(df["timestamp"].duplicated().sum())
    nan_rows = int(df[COLS].isna().any(axis=1).sum())
    d = df["timestamp"].diff().dropna()
    gaps = int((d != interval_ms).sum())
    missing = int(((d - interval_ms) / interval_ms).clip(lower=0).sum())
    viol = int(((df["high"] < df[["open", "close"]].max(axis=1)) |
                (df["low"] > df[["open", "close"]].min(axis=1)) |
                (df["high"] < df["low"])).sum())
    zero_vol = int((df["volume"] <= 0).sum())
    ret = df["close"].pct_change()
    extreme = int((ret.abs() > 0.20).sum())
    return DataQuality(len(df), str(df["datetime"].iloc[0]), str(df["datetime"].iloc[-1]),
                       dup, nan_rows, gaps, missing, viol, zero_vol, extreme)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop NaN/invalid bars. Gaps are left as-is (no forward fill -> no fake bars)."""
    df = df.dropna(subset=COLS).copy()
    ok = (df["high"] >= df["low"]) & (df["high"] >= df[["open", "close"]].max(axis=1)) & \
         (df["low"] <= df[["open", "close"]].min(axis=1)) & (df["close"] > 0)
    return df[ok].reset_index(drop=True)


# --------------------------------------------------------------------------
# Synthetic data (pipeline validation only - NOT for strategy selection)
# --------------------------------------------------------------------------
def make_synthetic_ohlcv(n: int = 70_000, seed: int = 0, start: str = "2022-01-01",
                         interval_ms: int = 900_000, p0: float = 20_000.0) -> pd.DataFrame:
    """Regime-switching GBM with vol clustering + volume, for smoke-testing the engine."""
    rng = np.random.default_rng(seed)
    regimes = np.array([+0.00020, +0.00006, 0.0, -0.00006, -0.00020])  # per-bar drift
    reg_vol = np.array([1.3, 0.85, 0.7, 0.95, 1.5])
    state, drift, volm = 2, np.empty(n), np.empty(n)
    for i in range(n):
        if rng.random() < 1 / 900:  # ~ regime change every 900 bars
            state = int(rng.integers(0, 5))
        drift[i], volm[i] = regimes[state], reg_vol[state]
    sig = np.empty(n)
    s = 0.004
    for i in range(n):  # GARCH-ish vol clustering
        s = np.sqrt(0.000004 + 0.90 * s ** 2 + 0.08 * (rng.normal(0, s)) ** 2)
        sig[i] = np.clip(s, 0.0008, 0.02) * volm[i]
    ret = drift + sig * rng.standard_normal(n)
    close = p0 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[p0], close[:-1]])
    hl = np.abs(rng.normal(0, 1, n)) * sig * close
    high = np.maximum(open_, close) + hl * 0.7
    low = np.minimum(open_, close) - hl * 0.7
    vol = np.exp(rng.normal(0, 0.5, n)) * (1 + 40 * np.abs(ret) / sig.mean()) * 1000
    ts0 = int(pd.Timestamp(start, tz="UTC").value // 1_000_000)
    df = pd.DataFrame({"timestamp": ts0 + np.arange(n, dtype="int64") * interval_ms,
                       "open": open_, "high": high, "low": low, "close": close, "volume": vol})
    return _finalize(df)


def ensure_dataset(symbol: str, data_dir: str, interval: str = "15",
                   allow_synthetic: bool = True, days: int = 1095,
                   seed: int = 0) -> tuple[pd.DataFrame, str]:
    """Return (df, source). Uses cached CSV, else Bybit, else synthetic."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"{symbol}_{interval}m.csv")
    if os.path.exists(path):
        return load_ohlcv(path), "csv"
    try:
        df, how = update_dataset(symbol, data_dir, interval, days)
        if len(df) > 1000:
            return df, how
    except Exception as exc:  # offline / blocked egress
        print(f"[data] bybit fetch failed for {symbol}: {exc}")
    if not allow_synthetic:
        raise RuntimeError(f"no data for {symbol}")
    df = make_synthetic_ohlcv(n=days * 96, seed=seed)
    return df, "synthetic"

# --------------------------------------------------------------------------
# Universe discovery
# --------------------------------------------------------------------------
def fetch_bybit_instruments(category: str = "linear", quote: str = "USDT") -> pd.DataFrame:
    """Every instrument Bybit currently lists. Note: this is a SURVIVORSHIP-BIASED
    universe - delisted contracts are gone, so a scan over it flatters any strategy."""
    import requests

    rows, cursor = [], None
    while True:
        params = {"category": category, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        r = requests.get(f"{BYBIT_BASE}/v5/market/instruments-info", params=params, timeout=30)
        r.raise_for_status()
        res = r.json().get("result", {})
        rows.extend(res.get("list", []))
        cursor = res.get("nextPageCursor")
        if not cursor:
            break
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    keep = ["symbol", "contractType", "status", "baseCoin", "quoteCoin", "launchTime"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["launchTime"] = pd.to_numeric(df.get("launchTime"), errors="coerce")
    df = df[(df["quoteCoin"] == quote) & (df["status"] == "Trading")]
    if "contractType" in df:
        df = df[df["contractType"].str.contains("Perpetual", na=False)]
    df["listed_days"] = (time.time() * 1000 - df["launchTime"]) / 86_400_000
    return df.reset_index(drop=True)


def fetch_bybit_tickers(category: str = "linear") -> pd.DataFrame:
    """24h turnover / volume per symbol - the liquidity filter for the universe."""
    import requests

    r = requests.get(f"{BYBIT_BASE}/v5/market/tickers", params={"category": category}, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json().get("result", {}).get("list", []))
    if df.empty:
        return df
    out = pd.DataFrame({"symbol": df["symbol"]})
    for c in ("turnover24h", "volume24h", "lastPrice", "fundingRate", "openInterestValue"):
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")
    return out


def update_dataset(symbol: str, data_dir: str, interval: str = "15", days: int = 1095,
                   category: str = "linear") -> tuple[pd.DataFrame, str]:
    """Download or incrementally extend data/<symbol>_<interval>m.csv."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"{symbol}_{interval}m.csv")
    end = int(time.time() * 1000)
    start = end - days * 86_400_000
    old = None
    if os.path.exists(path):
        old = load_ohlcv(path)
        if len(old):
            start = max(start, int(old["timestamp"].max()) - INTERVAL_MS[interval])
            if start >= end - INTERVAL_MS[interval]:
                return old, "cached"
    new = fetch_bybit_klines(symbol, interval, start, end, category=category)
    df = new if old is None else _finalize(pd.concat([old[COLS], new[COLS]], ignore_index=True))
    df[COLS].to_csv(path, index=False)
    return df, "updated" if old is not None else "downloaded"

