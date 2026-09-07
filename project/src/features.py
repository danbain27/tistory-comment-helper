"""Feature construction + forward-return labels (labels are for ANALYSIS ONLY)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import add_indicators

FEATURES = ["RSI_norm", "BB_percent_b", "BB_width", "Volume_ratio", "ATR_percent",
            "EMA20_distance", "EMA50_distance", "Return_1", "Return_4", "Return_16", "ADX"]
FWD_HORIZONS = (1, 4, 8, 16)


def build_features(df: pd.DataFrame, **ind_kw) -> pd.DataFrame:
    d = add_indicators(df, **ind_kw)
    c = d["close"]
    bb_range = (d["bb_upper"] - d["bb_lower"]).replace(0, np.nan)
    d["RSI_norm"] = (d["rsi"] - 50.0) / 50.0
    d["BB_percent_b"] = (c - d["bb_lower"]) / bb_range
    d["BB_width"] = (d["bb_upper"] - d["bb_lower"]) / d["bb_mid"]
    d["Volume_ratio"] = d["volume"] / d["vol_ma"].replace(0, np.nan)
    d["ATR_percent"] = d["atr"] / c
    d["EMA20_distance"] = c / d["ema_fast"] - 1.0
    d["EMA50_distance"] = c / d["ema_slow"] - 1.0
    d["Return_1"] = c.pct_change(1)
    d["Return_4"] = c.pct_change(4)
    d["Return_16"] = c.pct_change(16)
    d["ADX"] = d["adx"]
    d["feat_ok"] = d[FEATURES].notna().all(axis=1) & np.isfinite(d[FEATURES]).all(axis=1)
    return d


def add_forward_returns(d: pd.DataFrame, horizons=FWD_HORIZONS) -> pd.DataFrame:
    """Forward stats measured from the NEXT bar's open (the earliest tradable price)."""
    d = d.copy()
    entry = d["open"].shift(-1)
    for n in horizons:
        d[f"fwd_ret_{n}"] = d["close"].shift(-n) / entry - 1.0
        d[f"fwd_mfe_{n}"] = d["high"].shift(-1).rolling(n, min_periods=n).max().shift(-(n - 1)) / entry - 1.0
        d[f"fwd_mae_{n}"] = d["low"].shift(-1).rolling(n, min_periods=n).min().shift(-(n - 1)) / entry - 1.0
    return d


def time_split(d: pd.DataFrame, train=0.5, val=0.25):
    """Chronological split -> (train_idx, val_idx, test_idx) as boolean masks."""
    n = len(d)
    i1, i2 = int(n * train), int(n * (train + val))
    m = np.zeros(n, dtype=int)
    m[i1:i2], m[i2:] = 1, 2
    return m == 0, m == 1, m == 2
