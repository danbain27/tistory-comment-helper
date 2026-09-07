"""Step 6-7: conditional forward-return research (what states favour a LONG?)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FWD_HORIZONS

RSI_BINS = [-np.inf, 20, 25, 30, 35, 40, 45, 50, 60, 70, np.inf]
BB_BINS = [-np.inf, 0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0, np.inf]
VOL_BINS = [0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, np.inf]
ATR_BINS = [0, 0.003, 0.005, 0.0075, 0.010, 0.015, 0.02, np.inf]
ADX_BINS = [0, 15, 20, 25, 30, 40, np.inf]


def fwd_stats(r: pd.Series, mfe: pd.Series | None = None, mae: pd.Series | None = None) -> dict:
    r = r.dropna()
    if len(r) == 0:
        return {"n": 0}
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    out = {"n": int(len(r)), "mean": r.mean(), "median": r.median(),
           "win_rate": float((r > 0).mean()), "std": r.std(),
           "profit_factor": float(pos / neg) if neg > 0 else np.inf,
           "expectancy": r.mean(), "t_stat": float(r.mean() / (r.std() / np.sqrt(len(r)))) if r.std() > 0 else 0.0}
    if mfe is not None:
        out["avg_max_up"] = float(mfe.reindex(r.index).mean())
    if mae is not None:
        out["avg_max_down"] = float(mae.reindex(r.index).mean())
    return out


def bucket_analysis(d: pd.DataFrame, col: str, bins, horizons=FWD_HORIZONS,
                    mask=None, label: str | None = None) -> pd.DataFrame:
    sub = d[mask] if mask is not None else d
    sub = sub[sub["feat_ok"]]
    cats = pd.cut(sub[col], bins)
    rows = []
    for cat, idx in sub.groupby(cats, observed=True).groups.items():
        g = sub.loc[idx]
        row = {"feature": label or col, "bucket": str(cat), "n": len(g)}
        for h in horizons:
            st = fwd_stats(g[f"fwd_ret_{h}"], g.get(f"fwd_mfe_{h}"), g.get(f"fwd_mae_{h}"))
            if st.get("n", 0) == 0:
                continue
            row[f"h{h}_mean"] = st["mean"]
            row[f"h{h}_median"] = st["median"]
            row[f"h{h}_win"] = st["win_rate"]
            row[f"h{h}_pf"] = st["profit_factor"]
            row[f"h{h}_std"] = st["std"]
            row[f"h{h}_t"] = st["t_stat"]
            row[f"h{h}_maxup"] = st.get("avg_max_up", np.nan)
            row[f"h{h}_maxdn"] = st.get("avg_max_down", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def quantile_analysis(d: pd.DataFrame, col: str, qs: int = 5, horizons=FWD_HORIZONS,
                      mask=None) -> pd.DataFrame:
    sub = d[mask] if mask is not None else d
    sub = sub[sub["feat_ok"] & sub[col].notna()]
    edges = np.unique(np.quantile(sub[col], np.linspace(0, 1, qs + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return bucket_analysis(d, col, edges, horizons, mask, label=f"{col} (quantile)")


def full_feature_scan(d: pd.DataFrame, mask=None, pcs=("PC1", "PC2", "PC3")) -> pd.DataFrame:
    parts = [
        bucket_analysis(d, "rsi", RSI_BINS, mask=mask, label="RSI"),
        bucket_analysis(d, "BB_percent_b", BB_BINS, mask=mask, label="BB %B"),
        bucket_analysis(d, "Volume_ratio", VOL_BINS, mask=mask, label="Volume ratio"),
        bucket_analysis(d, "ATR_percent", ATR_BINS, mask=mask, label="ATR %"),
        bucket_analysis(d, "ADX", ADX_BINS, mask=mask, label="ADX"),
    ]
    parts += [quantile_analysis(d, pc, 5, mask=mask) for pc in pcs if pc in d.columns]
    return pd.concat(parts, ignore_index=True)


def conditional_edge(d: pd.DataFrame, cond: pd.Series, mask=None, horizon: int = 4) -> dict:
    """Edge of an arbitrary boolean condition vs the unconditional baseline."""
    sub = d[mask] if mask is not None else d
    sub = sub[sub["feat_ok"]]
    c = cond.reindex(sub.index).fillna(False)
    st = fwd_stats(sub.loc[c, f"fwd_ret_{horizon}"])
    base = fwd_stats(sub[f"fwd_ret_{horizon}"])
    if st.get("n", 0) == 0:
        return {"n": 0}
    return {"n": st["n"], "coverage": st["n"] / max(len(sub), 1), "mean": st["mean"],
            "baseline_mean": base["mean"], "edge": st["mean"] - base["mean"],
            "win_rate": st["win_rate"], "baseline_win": base["win_rate"],
            "profit_factor": st["profit_factor"], "t_stat": st["t_stat"]}
