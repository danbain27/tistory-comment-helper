"""Technical indicators (Wilder smoothing where standard). All causal: value at bar t
uses only data up to and including bar t."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def _wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = _wilder(d.clip(lower=0), n)
    loss = _wilder((-d).clip(lower=0), n)
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    zero_loss = (loss == 0) & gain.notna()
    out = out.mask(zero_loss & (gain > 0), 100.0).mask(zero_loss & (gain <= 0), 50.0)
    return out.where(gain.notna())


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return mid - k * sd, mid, mid + k * sd


def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    pc = c.shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    return _wilder(true_range(h, l, c), n)


def adx(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    up, dn = h.diff(), -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=h.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=h.index)
    tr_n = _wilder(true_range(h, l, c), n)
    pdi = 100 * _wilder(plus_dm, n) / tr_n.replace(0, np.nan)
    mdi = 100 * _wilder(minus_dm, n) / tr_n.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return _wilder(dx.fillna(0), n)


def add_indicators(df: pd.DataFrame, rsi_n=14, bb_n=20, bb_k=2.0, vol_n=20,
                   atr_n=14, ema_fast=20, ema_slow=50, adx_n=14) -> pd.DataFrame:
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
    out = df.copy()
    out["rsi"] = rsi(c, rsi_n)
    lo, mid, up = bollinger(c, bb_n, bb_k)
    out["bb_lower"], out["bb_mid"], out["bb_upper"] = lo, mid, up
    out["vol_ma"] = sma(v, vol_n)
    out["atr"] = atr(h, l, c, atr_n)
    out["ema_fast"] = ema(c, ema_fast)
    out["ema_slow"] = ema(c, ema_slow)
    out["adx"] = adx(h, l, c, adx_n)
    return out
