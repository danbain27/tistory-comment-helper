#!/usr/bin/env python3
"""Download Bybit USDT-perp klines (+ funding history) into ./data.

    python fetch_data.py --symbols BTCUSDT ETHUSDT --days 1095 --interval 15
"""
import argparse
import os
import time

from src import data as dataio

p = argparse.ArgumentParser()
p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT",
                                                "BNBUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT"])
p.add_argument("--interval", default="15")
p.add_argument("--days", type=int, default=1095)
p.add_argument("--out", default="data")
p.add_argument("--funding", action="store_true", help="also download funding-rate history")
a = p.parse_args()

os.makedirs(a.out, exist_ok=True)
end = int(time.time() * 1000)
start = end - a.days * 86_400_000
for s in a.symbols:
    df = dataio.fetch_bybit_klines(s, a.interval, start, end)
    q = dataio.check_quality(df, dataio.INTERVAL_MS[a.interval])
    path = f"{a.out}/{s}_{a.interval}m.csv"
    df[dataio.COLS].to_csv(path, index=False)
    print(f"{s}: {len(df)} bars -> {path}\n   {q.to_text()}")
    if a.funding:
        f = dataio.fetch_bybit_funding(s, start, end)
        f.to_csv(f"{a.out}/{s}_funding.csv", index=False)
        print(f"   funding: {len(f)} records")
