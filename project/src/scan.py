"""Universe-wide streaming scan.

One symbol is loaded, screened and thrown away before the next one starts, so the scan
runs over hundreds of coins in bounded memory. Only TRAIN and VALIDATION data are used
here - the scan both pools the statistical research and picks the deep-research basket,
so letting it see the test slice would leak.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import data as dataio
from .analysis import ADX_BINS, ATR_BINS, BB_BINS, RSI_BINS, VOL_BINS
from .backtest import ExecConfig
from .features import add_forward_returns, build_features, time_split
from .grid import GridParams
from .metrics import robust_score
from .optimization import evaluate, mask_to_slice
from .pca_model import PCAModel
from .risk import RiskParams
from .signal import EntryParams

BASELINE_ENTRY = EntryParams(rsi_max=35, bbp_max=0.15, vol_min=1.0, trend="ema20>ema50")
BASELINE_GRID = GridParams(step_atr=1.0, max_entries=3, weights="equal")
BASELINE_RISK = RiskParams(tp_mode="pct", tp_value=0.007, sl_mode="pct", sl_value=0.04,
                           max_hold_bars=480)
POOL_SPECS = [("RSI", "rsi", RSI_BINS), ("BB %B", "BB_percent_b", BB_BINS),
              ("Volume ratio", "Volume_ratio", VOL_BINS), ("ATR %", "ATR_percent", ATR_BINS),
              ("ADX", "ADX", ADX_BINS)]
PC_QUANTILE_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SCAN_MIN_TRADES = 10
HORIZONS = (4, 16)


@dataclass
class ScanConfig:
    data_dir: str = "data"
    interval: str = "15"
    days: int = 1095
    min_bars: int = 20_000          # ~7 months of 15m bars
    n_components: int = 3
    train_frac: float = 0.5
    val_frac: float = 0.25
    allow_synthetic: bool = False
    exec_cfg: ExecConfig = None
    leverage: float = 2.0


def _pool_rows(d: pd.DataFrame, mask, symbol: str) -> list[dict]:
    """Raw aggregates (sums, not means) so the universe pool can be combined exactly."""
    sub = d[mask & d["feat_ok"]]
    out = []
    specs = list(POOL_SPECS)
    for pc in ("PC1", "PC2", "PC3"):
        if pc in sub.columns and sub[pc].notna().any():
            edges = np.unique(np.quantile(sub[pc].dropna(), PC_QUANTILE_EDGES))
            if len(edges) > 2:
                edges[0], edges[-1] = -np.inf, np.inf
                specs.append((f"{pc} (own quintile)", pc, list(edges)))
    for label, col, bins in specs:
        cats = pd.cut(sub[col], bins)
        for cat, g in sub.groupby(cats, observed=True):
            row = {"symbol": symbol, "feature": label,
                   "bucket": (f"q{int(cats.cat.categories.get_loc(cat)) + 1}"
                              if label.endswith("(own quintile)") else str(cat)),
                   "n": len(g)}
            for h in HORIZONS:
                r = g[f"fwd_ret_{h}"].dropna()
                row[f"h{h}_n"] = int(len(r))
                row[f"h{h}_sum"] = float(r.sum())
                row[f"h{h}_sumsq"] = float((r ** 2).sum())
                row[f"h{h}_wins"] = int((r > 0).sum())
                row[f"h{h}_gain"] = float(r[r > 0].sum())
                row[f"h{h}_loss"] = float(-r[r < 0].sum())
            out.append(row)
    return out


def scan_symbol(symbol: str, cfg: ScanConfig, seed: int = 0,
                df: pd.DataFrame | None = None) -> dict:
    """Screen one symbol. Returns {'summary':dict, 'pool':[rows], 'loadings':[rows]}.
    `df` bypasses disk loading (used by the tests)."""
    ex = cfg.exec_cfg or ExecConfig()
    try:
        path = os.path.join(cfg.data_dir, f"{symbol}_{cfg.interval}m.csv")
        if df is not None:
            src = "given"
        elif os.path.exists(path):
            df, src = dataio.load_ohlcv(path), "csv"
        elif cfg.allow_synthetic:
            df, src = dataio.make_synthetic_ohlcv(n=cfg.days * 96, seed=seed), "synthetic"
        else:
            return {"summary": {"symbol": symbol, "status": "no_data"}}
        q = dataio.check_quality(df, dataio.INTERVAL_MS[cfg.interval])
        df = dataio.clean(df)
        if len(df) < cfg.min_bars:
            return {"summary": {"symbol": symbol, "status": "too_short", "bars": len(df),
                                "source": src}}

        d = add_forward_returns(build_features(df))
        tr, va, te = time_split(d, cfg.train_frac, cfg.val_frac)
        model = PCAModel(cfg.n_components).fit(d, tr)
        d = model.attach(d)

        rp = replace(BASELINE_RISK, leverage=cfg.leverage)
        s_tr, _ = evaluate(d, mask_to_slice(tr), BASELINE_ENTRY, BASELINE_GRID, rp, ex,
                           model.quantiles_, "train", symbol=symbol)
        s_va, _ = evaluate(d, mask_to_slice(va), BASELINE_ENTRY, BASELINE_GRID, rp, ex,
                           model.quantiles_, "validation", symbol=symbol)

        summary = {
            "symbol": symbol, "status": "ok", "source": src, "bars": len(d),
            "start": str(d["datetime"].iloc[0])[:10], "end": str(d["datetime"].iloc[-1])[:10],
            "missing_bars": q.missing_bars, "extreme_returns": q.extreme_returns,
            "atr_pct_median": float(d["ATR_percent"].median()),
            "pc1_var": float(model.pca.explained_variance_ratio_[0]),
            "pc_var_total": float(model.pca.explained_variance_ratio_.sum()),
            "train_trades": s_tr["trades"], "train_roi": s_tr["roi"],
            "train_pf": s_tr["profit_factor"], "train_mdd": s_tr["max_drawdown"],
            "val_trades": s_va["trades"], "val_roi": s_va["roi"], "val_pf": s_va["profit_factor"],
            "val_mdd": s_va["max_drawdown"], "val_win": s_va["win_rate"],
            "val_sharpe": s_va["sharpe"],
            # screening uses a lower trade floor than final selection: a symbol is being
            # ranked here, not adopted
            "val_score": robust_score(s_va, min_trades=SCAN_MIN_TRADES),
            "train_score": robust_score(s_tr, min_trades=SCAN_MIN_TRADES),
            "liquidations": s_tr["liquidations"] + s_va["liquidations"],
        }
        summary["screen_score"] = _screen_score(summary)
        loadings = [{"symbol": symbol, "feature": f, **model.loadings().loc[f].to_dict()}
                    for f in model.features]
        return {"summary": summary, "pool": _pool_rows(d, tr, symbol), "loadings": loadings}
    except Exception as exc:                                    # keep the scan going
        return {"summary": {"symbol": symbol, "status": f"error: {type(exc).__name__}: {exc}"}}


def _screen_score(s: dict) -> float:
    """Basket ranking. Train and validation must BOTH hold up; the test slice is untouched.
    Deliberately conservative: consistency beats a single strong window."""
    if s["liquidations"] > 0 or min(s["val_trades"], s["train_trades"]) < SCAN_MIN_TRADES:
        return -np.inf
    tr, va = s["train_score"], s["val_score"]
    if not np.isfinite(tr) or not np.isfinite(va):
        return -np.inf
    both_positive = 1.0 if (s["train_roi"] > 0 and s["val_roi"] > 0) else 0.0
    consistency = 1.0 - min(abs(tr - va) / (abs(tr) + abs(va) + 1e-9), 1.0)
    sample = min(1.0, (s["train_trades"] + s["val_trades"]) / 120.0)
    return float((min(tr, va) * 0.6 + (tr + va) / 2 * 0.4) * (0.4 + 0.6 * consistency)
                 * (0.3 + 0.7 * sample) + both_positive * 0.2)


def _worker(args):
    return scan_symbol(*args)


def run_scan(symbols: list[str], cfg: ScanConfig, jobs: int = 1, verbose: bool = True):
    """Returns (summary_df, pooled_df, loadings_df)."""
    tasks = [(s, cfg, dataio.symbol_seed(s)) for s in symbols]
    results = []
    if jobs > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(jobs) as pool:
            for i, r in enumerate(pool.imap_unordered(_worker, tasks, chunksize=1), 1):
                results.append(r)
                if verbose and i % 10 == 0:
                    print(f"  [scan] {i}/{len(tasks)}")
    else:
        for i, t in enumerate(tasks, 1):
            results.append(_worker(t))
            if verbose and i % 10 == 0:
                print(f"  [scan] {i}/{len(tasks)}")

    summary = pd.DataFrame([r["summary"] for r in results])
    pool = pd.DataFrame([row for r in results for row in r.get("pool", [])])
    load = pd.DataFrame([row for r in results for row in r.get("loadings", [])])
    if "screen_score" in summary:
        summary = summary.sort_values("screen_score", ascending=False, na_position="last")
    return summary.reset_index(drop=True), pool, load


def pool_feature_stats(pool: pd.DataFrame) -> pd.DataFrame:
    """Combine per-symbol raw aggregates into universe-wide conditional statistics."""
    if pool is None or pool.empty:
        return pd.DataFrame()
    g = pool.groupby(["feature", "bucket"], observed=True)
    rows = []
    for (feat, bucket), sub in g:
        row = {"feature": feat, "bucket": bucket, "symbols": sub["symbol"].nunique(),
               "n": int(sub["n"].sum())}
        for h in HORIZONS:
            n = int(sub[f"h{h}_n"].sum())
            if n == 0:
                continue
            ssum, ssq = sub[f"h{h}_sum"].sum(), sub[f"h{h}_sumsq"].sum()
            mean = ssum / n
            var = max(ssq / n - mean ** 2, 0.0)
            gain, loss = sub[f"h{h}_gain"].sum(), sub[f"h{h}_loss"].sum()
            row[f"h{h}_n"] = n
            row[f"h{h}_mean"] = mean
            row[f"h{h}_win"] = sub[f"h{h}_wins"].sum() / n
            row[f"h{h}_pf"] = float(gain / loss) if loss > 0 else np.inf
            row[f"h{h}_t"] = float(mean / (np.sqrt(var / n))) if var > 0 else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["feature", "bucket"]).reset_index(drop=True)


def pool_loadings(load: pd.DataFrame) -> pd.DataFrame:
    """Are the principal components the SAME thing across coins? (stability check)"""
    if load is None or load.empty:
        return pd.DataFrame()
    pcs = [c for c in load.columns if c.startswith("PC")]
    g = load.groupby("feature")[pcs]
    out = g.mean().round(3).add_suffix("_mean").join(g.std().round(3).add_suffix("_std"))
    return out.reset_index()


def select_basket(summary: pd.DataFrame, k: int = 8, min_bars: int = 20_000,
                  spread: bool = True) -> list[str]:
    """Top-k by screen_score. With spread=True the basket is not allowed to be all
    top-of-book majors: it takes the best from each screen_score quartile band."""
    ok = summary[(summary.get("status") == "ok") & (summary["bars"] >= min_bars)
                 & np.isfinite(summary["screen_score"])]
    if ok.empty:
        return summary.head(k)["symbol"].tolist()
    ok = ok.sort_values("screen_score", ascending=False)
    if not spread or len(ok) <= k:
        return ok.head(k)["symbol"].tolist()
    head = ok.head(max(k - 2, 1))["symbol"].tolist()
    rest = ok.iloc[max(k - 2, 1):]
    extra = rest.iloc[[len(rest) // 3, 2 * len(rest) // 3]]["symbol"].tolist()  # mid-pack sanity
    return (head + extra)[:k]


# --------------------------------------------------------------------------
# Universe-wide validation of ONE fixed configuration (test slice only)
# --------------------------------------------------------------------------
def validate_symbol(symbol: str, cfg: ScanConfig, ep, gp, rp, seed: int = 0,
                    split: str = "test") -> dict:
    """Apply an already-chosen config to a symbol that had no say in choosing it.
    PCA is still fit on that symbol's own train slice - exactly what a live bot does."""
    ex = cfg.exec_cfg or ExecConfig()
    try:
        path = os.path.join(cfg.data_dir, f"{symbol}_{cfg.interval}m.csv")
        if os.path.exists(path):
            df = dataio.load_ohlcv(path)
        elif cfg.allow_synthetic:
            df = dataio.make_synthetic_ohlcv(n=cfg.days * 96, seed=seed)
        else:
            return {"symbol": symbol, "status": "no_data"}
        df = dataio.clean(df)
        if len(df) < cfg.min_bars:
            return {"symbol": symbol, "status": "too_short", "bars": len(df)}
        d = build_features(df)
        tr, va, te = time_split(d, cfg.train_frac, cfg.val_frac)
        model = PCAModel(cfg.n_components).fit(d, tr)
        d = model.attach(d)
        sl = {"train": tr, "validation": va, "test": te}[split]
        s, _ = evaluate(d, mask_to_slice(sl), ep, gp, rp, ex, model.quantiles_, split, symbol=symbol)
        return {"symbol": symbol, "status": "ok", "split": split, "trades": s["trades"],
                "roi": s["roi"], "cagr": s["cagr"], "max_drawdown": s["max_drawdown"],
                "sharpe": s["sharpe"], "profit_factor": s["profit_factor"],
                "win_rate": s["win_rate"], "avg_trade": s["avg_trade"],
                "avg_holding_hours": s["avg_holding_hours"], "liquidations": s["liquidations"],
                "signals": s["signals"]}
    except Exception as exc:
        return {"symbol": symbol, "status": f"error: {type(exc).__name__}: {exc}"}


def _val_worker(args):
    return validate_symbol(*args)


def validate_universe(symbols: list[str], cfg: ScanConfig, ep, gp, rp, jobs: int = 1,
                      split: str = "test", verbose: bool = True) -> pd.DataFrame:
    tasks = [(s, cfg, ep, gp, rp, dataio.symbol_seed(s), split) for s in symbols]
    rows = []
    if jobs > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(jobs) as pool:
            for i, r in enumerate(pool.imap_unordered(_val_worker, tasks, chunksize=1), 1):
                rows.append(r)
                if verbose and i % 25 == 0:
                    print(f"  [validate] {i}/{len(tasks)}")
    else:
        for i, t in enumerate(tasks, 1):
            rows.append(_val_worker(t))
            if verbose and i % 25 == 0:
                print(f"  [validate] {i}/{len(tasks)}")
    return pd.DataFrame(rows)


def universe_verdict_stats(val: pd.DataFrame) -> dict:
    ok = val[val["status"] == "ok"] if "status" in val else val
    if ok.empty:
        return {}
    return {"symbols_tested": int(len(ok)), "total_trades": int(ok["trades"].sum()),
            "roi_median": float(ok["roi"].median()), "roi_mean": float(ok["roi"].mean()),
            "positive_fraction": float((ok["roi"] > 0).mean()),
            "mdd_median": float(ok["max_drawdown"].median()),
            "mdd_worst": float(ok["max_drawdown"].min()),
            "pf_median": float(ok["profit_factor"].replace(np.inf, 5).median()),
            "win_rate_median": float(ok["win_rate"].median()),
            "symbols_with_liquidation": int((ok["liquidations"] > 0).sum())}
