#!/usr/bin/env python3
"""Download Bybit USDT-perp klines (+ funding history) into ./data.

    # 상장된 USDT 무기한 전 종목 (유동성/상장기간 필터 통과분)
    python fetch_data.py --universe all --days 1095 --workers 4

    # 거래대금 상위 100개만
    python fetch_data.py --universe top100 --days 1095

    # 특정 종목
    python fetch_data.py --symbols BTCUSDT ETHUSDT --funding

이미 받아둔 CSV가 있으면 마지막 봉 이후만 이어서 받습니다 (--force 로 무시).
"""
import argparse
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src import data as dataio
from src import universe as uni

p = argparse.ArgumentParser()
p.add_argument("--symbols", nargs="*", default=None)
p.add_argument("--universe", default="all", help="'all', 'topN' (e.g. top100)")
p.add_argument("--min-turnover", type=float, default=1e7)
p.add_argument("--min-listed-days", type=int, default=365)
p.add_argument("--exclude-symbol-types", nargs="*", default=list(uni.NON_CRYPTO_SYMBOL_TYPES),
               help="Bybit symbolType values to drop (tokenised stocks/ETFs/commodities). "
                    "Pass with no values to keep everything.")
p.add_argument("--interval", default="15")
p.add_argument("--days", type=int, default=1095)
p.add_argument("--out", default="data")
p.add_argument("--workers", type=int, default=4)
p.add_argument("--funding", action="store_true", help="also download funding-rate history")
p.add_argument("--force", action="store_true", help="re-download instead of extending")
a = p.parse_args()

os.makedirs(a.out, exist_ok=True)

if a.symbols:
    symbols = a.symbols
    # still cache symbol -> symbolType, so a later `--universe local` run can tell a coin
    # from a tokenised stock without hitting the API again
    try:
        uni.save_type_map(dataio.fetch_bybit_instruments(), a.out)
    except Exception as exc:
        print(f"[universe] could not cache instrument types: {exc}")
else:
    m = re.fullmatch(r"top(\d+)", a.universe.lower())
    u = uni.fetch_universe(min_turnover=a.min_turnover, min_listed_days=a.min_listed_days,
                           top_n=int(m.group(1)) if m else None,
                           exclude_types=tuple(a.exclude_symbol_types),
                           data_dir=a.out)
    symbols = u["symbol"].tolist()
    u.to_csv(f"{a.out}/universe.csv", index=False)
    print(f"universe: {len(symbols)} symbols "
          f"(turnover >= {a.min_turnover:,.0f}, listed >= {a.min_listed_days}d, "
          f"excluding {a.exclude_symbol_types or 'nothing'})"
          f" -> {a.out}/universe.csv")

bars = a.days * (1440 // int(a.interval))
print(f"{len(symbols)} symbols x ~{bars:,} bars -> ~{len(symbols) * bars // 1000:,} requests, "
      f"{a.workers} workers. Ctrl-C to abort.")
t0 = time.time()


def one(sym: str) -> str:
    try:
        if a.force:
            path = f"{a.out}/{sym}_{a.interval}m.csv"
            if os.path.exists(path):
                os.remove(path)
        df, how = dataio.update_dataset(sym, a.out, a.interval, a.days)
        q = dataio.check_quality(df, dataio.INTERVAL_MS[a.interval])
        msg = f"{sym:<16} {how:<10} {len(df):>7} bars  {q.start[:10]}..{q.end[:10]}  " \
              f"missing={q.missing_bars}"
        if a.funding:
            end = int(time.time() * 1000)
            f = dataio.fetch_bybit_funding(sym, end - a.days * 86_400_000, end)
            f.to_csv(f"{a.out}/{sym}_funding.csv", index=False)
            msg += f"  funding={len(f)}"
        return msg
    except Exception as exc:
        return f"{sym:<16} FAILED  {type(exc).__name__}: {exc}"


done = 0
with ThreadPoolExecutor(max_workers=a.workers) as ex:
    futs = {ex.submit(one, s): s for s in symbols}
    for f in as_completed(futs):
        done += 1
        print(f"[{done}/{len(symbols)}] {f.result()}", flush=True)

print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {a.out}/")
