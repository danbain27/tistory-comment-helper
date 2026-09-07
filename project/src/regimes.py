"""Market-regime labelling (trailing windows only) and per-regime attribution."""
from __future__ import annotations

import numpy as np
import pandas as pd

TREND_LABELS = ["strong_down", "weak_down", "sideways", "weak_up", "strong_up"]


def label_regimes(d: pd.DataFrame, trend_bars: int = 960, vol_lookback: int = 5000,
                  strong: float = 0.10, weak: float = 0.02) -> pd.DataFrame:
    """trend_bars=960 -> 10 days on 15m. All inputs are trailing: no future leakage."""
    out = d.copy()
    r = out["close"] / out["close"].shift(trend_bars) - 1.0
    out["regime_trend"] = pd.cut(r, [-np.inf, -strong, -weak, weak, strong, np.inf],
                                 labels=TREND_LABELS).astype(object)
    atrp = out["ATR_percent"]
    rank = atrp.rolling(vol_lookback, min_periods=500).rank(pct=True)
    out["regime_vol"] = np.where(rank.isna(), None, np.where(rank > 0.7, "high_vol",
                                 np.where(rank < 0.3, "low_vol", "mid_vol")))
    return out


def regime_table(d: pd.DataFrame, trades: pd.DataFrame, col: str = "regime_trend") -> pd.DataFrame:
    """Per-regime trade performance (trades attributed by their signal bar)."""
    if trades is None or len(trades) == 0:
        return pd.DataFrame()
    lut = pd.Series(d[col].to_numpy(), index=pd.to_datetime(d["datetime"]))
    reg = pd.to_datetime(trades["timestamp"]).map(lut)
    t = trades.assign(_regime=reg.to_numpy())
    rows = []
    for name, g in t.groupby("_regime", dropna=False):
        p = g["profit"]
        wins, losses = p[p > 0], p[p <= 0]
        bars = int(((d[col] == name).sum())) if name is not None else 0
        rows.append({"regime": name, "bars_in_regime": bars, "trades": len(g),
                     "trades_per_1k_bars": len(g) / max(bars, 1) * 1000,
                     "win_rate": float((p > 0).mean()), "avg_trade": float(p.mean()),
                     "total_profit": float(p.sum()),
                     "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.sum() else np.inf,
                     "avg_return": float(g["trade_return"].mean()),
                     "worst_trade": float(p.min()),
                     "liquidations": int((g["exit_reason"] == "liquidation").sum())})
    return pd.DataFrame(rows).sort_values("trades", ascending=False)


def pca_state_table(d: pd.DataFrame, pcs=("PC1", "PC2", "PC3"), col: str = "regime_trend") -> pd.DataFrame:
    """Does PCA actually separate market states? Mean PC value per regime."""
    sub = d[d["feat_ok"] & d[col].notna()]
    have = [p for p in pcs if p in sub.columns]
    if not have:
        return pd.DataFrame()
    g = sub.groupby(col, observed=True)[have].agg(["mean", "std", "count"])
    return g.round(3)
