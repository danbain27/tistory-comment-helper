"""Chart output (matplotlib Agg)."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import drawdown, monthly_returns


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def equity_curve(curves: dict[str, pd.Series], path: str, title="Equity curve"):
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, eq in curves.items():
        if eq is not None and len(eq):
            ax.plot(eq.index, eq.to_numpy(), label=name, lw=1.2)
    ax.set_title(title); ax.set_ylabel("equity"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    return _save(fig, path)


def drawdown_curve(eq: pd.Series, path: str, title="Drawdown"):
    dd = drawdown(eq)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.fill_between(dd.index, dd.to_numpy() * 100, 0, color="crimson", alpha=0.5)
    ax.set_title(f"{title} (max {dd.min():.2%})"); ax.set_ylabel("%"); ax.grid(alpha=0.3)
    return _save(fig, path)


def monthly_returns_chart(eq: pd.Series, path: str, title="Monthly returns"):
    m = monthly_returns(eq)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    if len(m):
        ax.bar([str(i)[:7] for i in m.index], m.to_numpy() * 100,
               color=np.where(m.to_numpy() >= 0, "seagreen", "crimson"))
        ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.set_title(title); ax.set_ylabel("%"); ax.grid(alpha=0.3, axis="y")
    return _save(fig, path)


def trade_distribution(trades: pd.DataFrame, path: str, title="Trade return distribution"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    if trades is not None and len(trades):
        axes[0].hist(trades["trade_return"] * 100, bins=50, color="steelblue")
        axes[0].set_title("trade return %")
        axes[1].hist(trades["holding_hours"], bins=50, color="darkorange")
        axes[1].set_title("holding time (h)")
        vc = trades["exit_reason"].value_counts()
        axes[2].bar(vc.index, vc.to_numpy(), color="slateblue")
        axes[2].set_title("exit reason"); axes[2].tick_params(axis="x", rotation=30, labelsize=8)
    for a in axes:
        a.grid(alpha=0.3)
    fig.suptitle(title)
    return _save(fig, path)


def pca_analysis(d: pd.DataFrame, path: str, horizon: int = 4, pcs=("PC1", "PC2", "PC3"),
                 sample: int = 30_000):
    have = [p for p in pcs if p in d.columns]
    sub = d[d["feat_ok"] & d[f"fwd_ret_{horizon}"].notna()]
    if len(sub) > sample:
        sub = sub.sample(sample, random_state=0)
    fig, axes = plt.subplots(2, max(len(have), 2), figsize=(4.6 * max(len(have), 2), 7.5))
    for j, pc in enumerate(have):
        axes[0, j].scatter(sub[pc], sub[f"fwd_ret_{horizon}"] * 100, s=2, alpha=0.15)
        b = pd.qcut(sub[pc], 10, duplicates="drop")
        mu = sub.groupby(b, observed=True)[f"fwd_ret_{horizon}"].mean() * 100
        axes[0, j].set_title(f"{pc} vs fwd {horizon}-bar return")
        axes[0, j].set_xlabel(pc); axes[0, j].grid(alpha=0.3)
        axes[1, j].bar(range(len(mu)), mu.to_numpy(),
                       color=np.where(mu.to_numpy() >= 0, "seagreen", "crimson"))
        axes[1, j].set_title(f"{pc} decile -> mean fwd return %"); axes[1, j].grid(alpha=0.3, axis="y")
    if len(have) >= 2:
        pass
    fig.suptitle("PCA vs forward returns")
    return _save(fig, path)


def pca_scatter(d: pd.DataFrame, path: str, regime_col="regime_trend", sample=20_000):
    sub = d[d["feat_ok"] & d.get(regime_col, pd.Series(index=d.index, dtype=object)).notna()]
    if not len(sub) or "PC1" not in sub.columns:
        return None
    if len(sub) > sample:
        sub = sub.sample(sample, random_state=0)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, g in sub.groupby(regime_col, observed=True):
        ax.scatter(g["PC1"], g["PC2"], s=3, alpha=0.3, label=str(name))
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, markerscale=3); ax.set_title("PC1/PC2 by market regime")
    return _save(fig, path)


def sensitivity_chart(sens: pd.DataFrame, path: str, params: list[str]):
    ps = [p for p in params if p in set(sens["param"])]
    if not ps:
        return None
    fig, axes = plt.subplots(1, len(ps), figsize=(4.2 * len(ps), 3.6), squeeze=False)
    for i, p in enumerate(ps):
        s = sens[sens["param"] == p]
        axes[0, i].bar(s["value"].astype(str), s["score_median"].to_numpy(), color="steelblue")
        axes[0, i].set_title(p, fontsize=9); axes[0, i].tick_params(axis="x", rotation=45, labelsize=7)
        axes[0, i].grid(alpha=0.3, axis="y")
    fig.suptitle("Parameter sensitivity (median robust score on validation)")
    return _save(fig, path)
