"""Entry-signal construction. A signal at bar t uses only information known at the
close of bar t; execution happens at the open of bar t+1 (see backtest.py)."""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace

import numpy as np
import pandas as pd

TREND_MODES = ("none", "close>ema20", "close>ema50", "ema20>ema50",
               "ema20>ema50 & close>ema20", "close>ema50 & ema20>ema50")


@dataclass
class EntryParams:
    rsi_max: float | None = 30.0
    bbp_max: float | None = 0.10
    vol_min: float | None = 1.2
    trend: str = "ema20>ema50"
    adx_min: float | None = None
    adx_max: float | None = None
    # PCA filters are expressed as TRAIN-distribution quantiles (0..1), so they stay
    # meaningful after a walk-forward refit where raw PC scale drifts.
    pc1_min_q: float | None = None
    pc1_max_q: float | None = None
    pc2_min_q: float | None = None
    pc2_max_q: float | None = None
    pc3_min_q: float | None = None
    pc3_max_q: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _trend_mask(d: pd.DataFrame, mode: str) -> np.ndarray:
    c, f, s = d["close"], d["ema_fast"], d["ema_slow"]
    if mode == "none":
        return np.ones(len(d), bool)
    if mode == "close>ema20":
        return (c > f).to_numpy()
    if mode == "close>ema50":
        return (c > s).to_numpy()
    if mode == "ema20>ema50":
        return (f > s).to_numpy()
    if mode == "ema20>ema50 & close>ema20":
        return ((f > s) & (c > f)).to_numpy()
    if mode == "close>ema50 & ema20>ema50":
        return ((c > s) & (f > s)).to_numpy()
    raise ValueError(f"unknown trend mode {mode}")


def build_signal(d: pd.DataFrame, p: EntryParams, pca_quantiles: pd.DataFrame | None = None) -> np.ndarray:
    m = d["feat_ok"].to_numpy() & d["atr"].notna().to_numpy() & (d["atr"].to_numpy() > 0)
    if p.rsi_max is not None:
        m &= (d["rsi"] <= p.rsi_max).to_numpy()
    if p.bbp_max is not None:
        m &= (d["BB_percent_b"] <= p.bbp_max).to_numpy()
    if p.vol_min is not None:
        m &= (d["Volume_ratio"] >= p.vol_min).to_numpy()
    m &= _trend_mask(d, p.trend)
    if p.adx_min is not None:
        m &= (d["ADX"] >= p.adx_min).to_numpy()
    if p.adx_max is not None:
        m &= (d["ADX"] <= p.adx_max).to_numpy()

    for pc in ("PC1", "PC2", "PC3"):
        lo_q = getattr(p, f"{pc.lower()}_min_q")
        hi_q = getattr(p, f"{pc.lower()}_max_q")
        if lo_q is None and hi_q is None:
            continue
        if pc not in d.columns or pca_quantiles is None:
            raise ValueError(f"{pc} filter requested but PC columns/quantiles are missing")
        v = d[pc].to_numpy()
        if lo_q is not None:
            m &= np.nan_to_num(v, nan=-np.inf) >= _qval(pca_quantiles, pc, lo_q)
        if hi_q is not None:
            m &= np.nan_to_num(v, nan=np.inf) <= _qval(pca_quantiles, pc, hi_q)
    return m


def _qval(qdf: pd.DataFrame, pc: str, q: float) -> float:
    if q in qdf.index:
        return float(qdf.loc[q, pc])
    idx = np.asarray(qdf.index, float)
    return float(np.interp(q, idx, qdf[pc].to_numpy(float)))


# Incremental filter stacks used for the "does each filter help?" study (Step 10).
def filter_stack(base: EntryParams) -> dict[str, EntryParams]:
    A = replace(base, bbp_max=None, vol_min=None, trend="none", adx_min=None,
                pc1_min_q=None, pc1_max_q=None, pc2_min_q=None, pc2_max_q=None,
                pc3_min_q=None, pc3_max_q=None)
    B = replace(A, bbp_max=base.bbp_max)
    C = replace(B, vol_min=base.vol_min)
    D = replace(C, trend=base.trend)
    E = replace(D, pc1_min_q=base.pc1_min_q, pc1_max_q=base.pc1_max_q,
                pc2_min_q=base.pc2_min_q, pc2_max_q=base.pc2_max_q,
                pc3_min_q=base.pc3_min_q, pc3_max_q=base.pc3_max_q)
    F = replace(E, adx_min=base.adx_min, adx_max=base.adx_max)
    return {"A_RSI": A, "B_+BB": B, "C_+Volume": C, "D_+EMA": D, "E_+PCA": E, "F_+ADX": F}
