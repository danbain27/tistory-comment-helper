"""Parameter search on the VALIDATION slice only, with stability-aware selection."""
from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pandas as pd

from .backtest import ExecConfig, run_backtest
from .grid import GridParams
from .metrics import robust_score, summarize
from .risk import RiskParams
from .signal import EntryParams, build_signal


def mask_to_slice(mask: np.ndarray) -> tuple[int, int]:
    idx = np.flatnonzero(mask)
    return int(idx[0]), int(idx[-1]) + 1


def evaluate(d: pd.DataFrame, sl: tuple[int, int], ep: EntryParams, gp: GridParams,
             rp: RiskParams, cfg: ExecConfig, quantiles: pd.DataFrame | None,
             label: str = "", funding: pd.DataFrame | None = None,
             symbol: str = "", keep=False):
    sub = d.iloc[sl[0]:sl[1]].reset_index(drop=True)
    sig = build_signal(sub, ep, quantiles)
    res = run_backtest(sub, sig, gp, rp, cfg, funding=funding, symbol=symbol)
    s = summarize(res.equity, res.trades, label, initial=cfg.initial_capital)
    s["signals"] = int(sig.sum())
    s["ruin"] = res.ruin
    s["score"] = robust_score(s)
    return (s, res) if keep else (s, None)


def evaluate_multi(datasets: dict, sl_map: dict, ep, gp, rp, cfg, q_map, label="") -> dict:
    """Average performance across symbols; the weakest symbol is reported too."""
    rows = []
    for sym, d in datasets.items():
        s, _ = evaluate(d, sl_map[sym], ep, gp, rp, cfg, q_map.get(sym), label, symbol=sym)
        s["symbol"] = sym
        rows.append(s)
    df = pd.DataFrame(rows)
    agg = {"strategy": label, "symbols": len(rows),
           "trades": int(df["trades"].sum()),
           "roi_mean": float(df["roi"].mean()), "roi_median": float(df["roi"].median()),
           "roi_min": float(df["roi"].min()),
           "mdd_mean": float(df["max_drawdown"].mean()), "mdd_worst": float(df["max_drawdown"].min()),
           "pf_median": float(df["profit_factor"].replace(np.inf, 5).median()),
           "win_rate_mean": float(df["win_rate"].mean()),
           "sharpe_mean": float(df["sharpe"].mean()),
           "score_mean": float(df["score"].replace(-np.inf, -5).mean()),
           "score_min": float(df["score"].replace(-np.inf, -5).min()),
           "positive_symbols": int((df["roi"] > 0).sum()),
           "liquidations": int(df["liquidations"].sum())}
    return agg, df


# ---------------------------------------------------------------- search spaces
def entry_candidates(rsi=(25, 30, 35, 40), bbp=(0.05, 0.10, 0.15, 0.20),
                     vol=(None, 1.0, 1.2, 1.5), trend=("none", "ema20>ema50", "close>ema50"),
                     adx=(None, 20, 25), pca=(None,)) -> list[EntryParams]:
    out = []
    for r, b, v, t, a in itertools.product(rsi, bbp, vol, trend, adx):
        for pc in pca:
            ep = EntryParams(rsi_max=r, bbp_max=b, vol_min=v, trend=t, adx_min=a)
            if pc:
                ep = replace(ep, **pc)
            out.append(ep)
    return out


def grid_candidates(steps=(0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
                    counts=(1, 2, 3, 4, 5, 6),
                    weights=("equal", "mild", "aggressive")) -> list[GridParams]:
    out = []
    for s, c, w in itertools.product(steps, counts, weights):
        if c == 1 and (w != "equal" or s != steps[0]):
            continue  # no DCA -> step/weights are meaningless, keep one representative
        out.append(GridParams(step_atr=s, max_entries=c, weights=w))
    return out


def exit_candidates(tp_pct=(0.003, 0.005, 0.007, 0.010, 0.015, 0.020),
                    tp_atr=(0.5, 1.0, 1.5, 2.0),
                    sl_pct=(0.02, 0.03, 0.04, 0.05, 0.06, 0.08),
                    sl_atr=(1.0, 1.5, 2.0, 2.5, 3.0),
                    include_no_sl=True, max_hold=(96, 480, 960)) -> list[RiskParams]:
    out = []
    for mh in max_hold:
        for tpm, tpv in [("pct", v) for v in tp_pct] + [("atr", v) for v in tp_atr]:
            sls = [("pct", v) for v in sl_pct] + [("atr", v) for v in sl_atr]
            if include_no_sl:
                sls.append(("none", 0.0))
            for slm, slv in sls:
                out.append(RiskParams(tp_mode=tpm, tp_value=tpv, sl_mode=slm,
                                      sl_value=slv, max_hold_bars=mh))
    return out


# ---------------------------------------------------------------- search driver
def search(d: pd.DataFrame, sl, combos: list[tuple], cfg: ExecConfig,
           quantiles=None, symbol="", progress_every: int = 0) -> pd.DataFrame:
    rows = []
    for i, (ep, gp, rp, tag) in enumerate(combos):
        s, _ = evaluate(d, sl, ep, gp, rp, cfg, quantiles, tag, symbol=symbol)
        s.update({f"e_{k}": v for k, v in ep.to_dict().items()})
        s.update({f"g_{k}": v for k, v in gp.to_dict().items()})
        s.update({f"r_{k}": v for k, v in rp.to_dict().items()})
        rows.append(s)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  [search] {i+1}/{len(combos)}")
    return pd.DataFrame(rows)


def sensitivity_table(df: pd.DataFrame, params: list[str], score: str = "score") -> pd.DataFrame:
    """Marginal stability: how the score behaves across each parameter's values."""
    rows = []
    for p in params:
        if p not in df.columns:
            continue
        g = df.groupby(df[p].astype(str), dropna=False)[score]
        for val, sub in g:
            v = sub.replace(-np.inf, np.nan).dropna()
            rows.append({"param": p, "value": val, "n": len(sub),
                         "score_mean": float(v.mean()) if len(v) else np.nan,
                         "score_median": float(v.median()) if len(v) else np.nan,
                         "score_p25": float(v.quantile(0.25)) if len(v) else np.nan,
                         "frac_positive_roi": float((df.loc[sub.index, "roi"] > 0).mean())})
    return pd.DataFrame(rows)


def plateau_pick(df: pd.DataFrame, numeric_params: list[str], score: str = "score",
                 k: int = 1, min_trades: int = 30) -> pd.Series:
    """Pick the combo sitting on a stable PLATEAU rather than the single best spike:
    each candidate is re-scored as the mean score of its parameter-space neighbours."""
    d = pd.DataFrame()
    for floor in (min_trades, max(min_trades // 2, 5), 5, 1):   # graduated relaxation
        d = df[(df["trades"] >= floor) & np.isfinite(df[score])].copy()
        if not d.empty:
            break
    if d.empty:
        # nothing tradable: prefer the candidate that at least produced trades
        return df.sort_values(["trades", score], ascending=False).iloc[0]
    ranks = {}
    for p in numeric_params:
        if p in d.columns and pd.api.types.is_numeric_dtype(d[p]):
            vals = np.sort(d[p].dropna().unique())
            ranks[p] = d[p].map({v: i for i, v in enumerate(vals)})
    if not ranks:
        return d.sort_values(score, ascending=False).iloc[0]
    R = pd.DataFrame(ranks).fillna(0).to_numpy(float)
    S = d[score].to_numpy(float)
    nbr = np.empty(len(d))
    for i in range(len(d)):
        close = (np.abs(R - R[i]) <= k).all(axis=1)
        nbr[i] = S[close].mean()
    d["neighbour_score"] = nbr
    return d.sort_values(["neighbour_score", score], ascending=False).iloc[0]


def params_from_row(row: pd.Series) -> tuple[EntryParams, GridParams, RiskParams]:
    def sub(prefix, cls):
        f = {k[len(prefix):]: row[k] for k in row.index if k.startswith(prefix)}
        f = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in f.items()}
        for k, v in list(f.items()):
            if isinstance(v, np.bool_):
                f[k] = bool(v)
            elif isinstance(v, np.integer):
                f[k] = int(v)
            elif isinstance(v, np.floating):
                f[k] = float(v)
        return cls(**f)
    ep = sub("e_", EntryParams)
    gp = sub("g_", GridParams)
    gp.max_entries = int(gp.max_entries)
    rp = sub("r_", RiskParams)
    rp.max_hold_bars = int(rp.max_hold_bars)
    rp.cooldown_bars = int(rp.cooldown_bars)
    return ep, gp, rp
