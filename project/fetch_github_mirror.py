"""Fallback data source when exchange APIs are unreachable (e.g. sandboxed network).

Downloads Binance SPOT 1m klines from the public GitHub mirrors
`parshant231234-eng/<SYMBOL>-Historical-Data` (monthly CSVs sourced from
data.binance.vision) and resamples them to 15m, writing data/<SYMBOL>_15m.csv in the
same format as fetch_data.py.

Caveats (also printed in the report): spot, not perps -> no funding; Binance, not Bybit.

    python fetch_github_mirror.py --months 36 --workers 16
"""
from __future__ import annotations

import argparse
import io
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

OWNER = "parshant231234-eng"
RAW = "https://raw.githubusercontent.com/{o}/{s}-Historical-Data/HEAD/1m/{y}/{m:02d}/{s}_1m_{y}-{m:02d}.csv"


def list_symbols() -> list[str]:
    out = subprocess.run(
        ["gh", "api", "-X", "GET", "search/repositories", "-f",
         f"q=user:{OWNER} Historical-Data", "-f", "per_page=100", "--paginate",
         "--jq", ".items[].name"], capture_output=True, text=True, check=True).stdout
    return sorted({n.replace("-Historical-Data", "") for n in out.split() if n.endswith("USDT-Historical-Data")})


def months_back(n: int, end: pd.Timestamp) -> list[tuple[int, int]]:
    p = pd.Period(end, "M") - 1  # last complete month
    return [((p - i).year, (p - i).month) for i in range(n)][::-1]


def fetch_month(sym: str, y: int, m: int, sess: requests.Session) -> pd.DataFrame | None:
    url = RAW.format(o=OWNER, s=sym, y=y, m=m)
    for _ in range(3):
        try:
            r = sess.get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            d = pd.read_csv(io.BytesIO(r.content),
                            usecols=["timestamp", "open", "high", "low", "close", "volume"])
            return d
        except Exception:
            continue
    return None


def to_15m(d: pd.DataFrame) -> pd.DataFrame:
    d = d.drop_duplicates("timestamp").sort_values("timestamp")
    d["timestamp"] = d["timestamp"].astype("int64")
    if d["timestamp"].max() > 1e14:  # binance switched spot files to microseconds in 2025
        d.loc[d["timestamp"] > 1e14, "timestamp"] //= 1000
    d.index = pd.to_datetime(d["timestamp"], unit="ms", utc=True)
    o = d.resample("15min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    o.insert(0, "timestamp", ((o.index - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)).astype("int64"))
    return o.reset_index(drop=True)


def fetch_symbol(sym: str, months, data_dir: str, force: bool) -> str:
    path = os.path.join(data_dir, f"{sym}_15m.csv")
    if os.path.exists(path) and not force:
        return f"{sym}: exists"
    sess = requests.Session()
    parts = [p for p in (fetch_month(sym, y, m, sess) for y, m in months) if p is not None and len(p)]
    if not parts:
        return f"{sym}: no data"
    out = to_15m(pd.concat(parts, ignore_index=True))
    out.to_csv(path, index=False)
    return f"{sym}: {len(out)} bars {pd.to_datetime(out.timestamp.iloc[0], unit='ms').date()}..{pd.to_datetime(out.timestamp.iloc[-1], unit='ms').date()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=36)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.data_dir, exist_ok=True)
    syms = a.symbols or list_symbols()
    months = months_back(a.months, pd.Timestamp.now("UTC"))
    print(f"{len(syms)} symbols x {len(months)} months ({months[0]}..{months[-1]})", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for msg in ex.map(lambda s: fetch_symbol(s, months, a.data_dir, a.force), syms):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
