"""Integrity tests: no look-ahead, PCA train-only fitting, cost accounting.

    python -m tests.test_engine
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src import data as dataio
from src import features as F
from src.backtest import ExecConfig, run_backtest
from src.grid import GridParams
from src.pca_model import PCAModel
from src.risk import RiskParams
from src.signal import EntryParams, build_signal

EP = EntryParams(rsi_max=40, bbp_max=0.25, vol_min=None, trend="none")
GP = GridParams(step_atr=1.0, max_entries=3)
RP = RiskParams(tp_value=0.01, sl_value=0.05)
FAILS = []


def check(name: str, ok: bool, detail: str = ""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


def prep(n=12_000, seed=11):
    d = F.build_features(dataio.make_synthetic_ohlcv(n, seed=seed))
    tr, _, _ = F.time_split(d)
    m = PCAModel().fit(d, tr)
    return m.attach(d), m


def test_entry_is_next_bar_open():
    d, m = prep()
    cfg = ExecConfig(slippage_bp=2.0)
    sig = build_signal(d, EP, m.quantiles_)
    r = run_backtest(d, sig, GP, RP, cfg)
    t = r.trades.iloc[0]
    ts = d["timestamp"].to_numpy("int64")
    i = int(np.flatnonzero(ts == t["entry_time"].value // 1_000_000)[0])
    sig_i = int(np.flatnonzero(ts == t["timestamp"].value // 1_000_000)[0])
    expect = d["open"].iloc[i] * (1 + cfg.slippage_bp / 10_000)
    check("entry fills at next bar open + slippage", abs(t["entry_price"] - expect) < 1e-6)
    check("signal bar precedes entry bar by entry_delay_bars", i - sig_i == cfg.entry_delay_bars)
    check("signal bar actually carried a signal", bool(sig[sig_i]))


def test_future_mutation_does_not_change_past():
    """The strongest look-ahead test: scramble the tail of the data and verify that every
    trade that closed before the cut is bit-identical."""
    d, m = prep()
    cut = int(len(d) * 0.6)
    cfg = ExecConfig()
    sig = build_signal(d, EP, m.quantiles_)
    base = run_backtest(d, sig, GP, RP, cfg).trades

    d2 = d.copy()
    rng = np.random.default_rng(0)
    for c in ("open", "high", "low", "close", "volume"):
        v = d2[c].to_numpy().copy()
        v[cut:] = v[cut:] * rng.uniform(0.5, 1.5, len(v) - cut)
        d2[c] = v
    d2 = F.build_features(d2[["timestamp", "datetime", "open", "high", "low", "close", "volume"]])
    d2 = m.attach(d2)                       # same (train-fitted) PCA
    sig2 = build_signal(d2, EP, m.quantiles_)
    mut = run_backtest(d2, sig2, GP, RP, cfg).trades

    cutoff = d["datetime"].iloc[cut - 1]
    a = base[base["exit_time"] < cutoff].reset_index(drop=True)
    b = mut[mut["exit_time"] < cutoff].reset_index(drop=True)
    cols = ["entry_price", "average_entry", "exit_price", "profit", "exit_reason"]
    same = len(a) == len(b) and all(
        np.allclose(a[c].to_numpy(float), b[c].to_numpy(float)) if c != "exit_reason"
        else (a[c] == b[c]).all() for c in cols)
    check("future bars cannot change already-closed trades", same,
          f"({len(a)} trades compared)")
    check("signals before the cut are unchanged", bool((sig[:cut] == sig2[:cut]).all()))


def test_pca_is_train_only():
    d, _ = prep()
    tr, va, te = F.time_split(d)
    m1 = PCAModel().fit(d, tr)
    d_mut = d.copy()
    rng = np.random.default_rng(1)
    idx = np.flatnonzero(te)
    for c in F.FEATURES:
        v = d_mut[c].to_numpy().copy()
        v[idx] = v[idx] + rng.normal(0, 5, len(idx))
        d_mut[c] = v
    m2 = PCAModel().fit(d_mut, tr)
    check("PCA loadings do not depend on test-slice data",
          np.allclose(m1.loadings().to_numpy(), m2.loadings().to_numpy()))
    check("PCA train quantiles do not depend on test-slice data",
          np.allclose(m1.quantiles_.to_numpy(), m2.quantiles_.to_numpy()))


def test_costs_are_charged():
    d, m = prep()
    sig = build_signal(d, EP, m.quantiles_)
    free = ExecConfig(fee_taker=0.0, fee_maker=0.0, slippage_bp=0.0, use_funding=False)
    real = ExecConfig(fee_taker=0.00055, fee_maker=0.0002, slippage_bp=2.0, use_funding=True)
    a = run_backtest(d, sig, GP, RP, free)
    b = run_backtest(d, sig, GP, RP, real)
    check("fees/slippage/funding reduce the result",
          b.equity.iloc[-1] < a.equity.iloc[-1],
          f"({a.equity.iloc[-1]:.0f} -> {b.equity.iloc[-1]:.0f})")
    check("fees are logged per trade", float(b.trades["fees"].sum()) > 0)
    check("funding is logged per trade", float(b.trades["funding"].abs().sum()) > 0)


def test_position_and_grid_limits():
    d, m = prep()
    sig = build_signal(d, EP, m.quantiles_)
    gp = GridParams(step_atr=0.3, max_entries=4)
    r = run_backtest(d, sig, gp, RiskParams(sl_mode="none", tp_value=0.02), ExecConfig())
    t = r.trades
    check("grid entries never exceed max_entries", int(t["grid_entry_count"].max()) <= gp.max_entries)
    check("notional never exceeds equity * leverage",
          bool((t["notional"] <= t["equity_at_entry"] * 2.0 * 1.02 + 1e-6).all()))
    check("only one position at a time",
          bool((pd.to_datetime(t["entry_time"]).to_numpy()[1:] >
                pd.to_datetime(t["exit_time"]).to_numpy()[:-1]).all()))


def test_indicators_are_causal():
    d = F.build_features(dataio.make_synthetic_ohlcv(3000, seed=13))
    k = 2000
    trunc = F.build_features(dataio.make_synthetic_ohlcv(3000, seed=13).iloc[:k].copy())
    cols = ["rsi", "atr", "ema_fast", "ema_slow", "adx", "BB_percent_b"]
    ok = all(np.allclose(d[c].to_numpy()[200:k], trunc[c].to_numpy()[200:], equal_nan=True)
             for c in cols)
    check("indicators computed on a truncated series match the full series", ok)


if __name__ == "__main__":
    for fn in (test_entry_is_next_bar_open, test_future_mutation_does_not_change_past,
               test_pca_is_train_only, test_costs_are_charged, test_position_and_grid_limits,
               test_indicators_are_causal):
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)
