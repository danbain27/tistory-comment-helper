"""Multi-strategy universe lab: grid search + IS/OOS split + walk-forward.

Complements run_research.py (which studies ONE strategy in depth). Here several simple,
well-known strategy families are evaluated on the WHOLE local universe at once as an
equal-weight portfolio, so a parameter set is only "good" if it works across coins.

    python -m src.strategy_lab --data-dir data --out results/lab

Design / anti-overfitting
  * Panel (bars x symbols). Signal at bar t close -> position held over bar t+1.
  * Costs per unit of turnover: taker fee + slippage (default 5.5bp + 2bp per side),
    plus an 8h funding drag on perp longs (and credit on shorts) - 0.01%/8h default.
  * Point-in-time liquidity gate: a symbol only trades while its trailing 30d median daily
    quote volume >= --min-daily-turnover (computed from past data only).
  * In-sample = first --is-frac of the timeline, out-of-sample = the rest. Parameters are
    RANKED ON IS ONLY; OOS is measured, never used for choice.
  * Selection uses a plateau score (mean of the combo and its grid neighbours), not the
    single best cell.
  * Walk-forward: rolling train/test windows, parameters re-chosen in every fold, OOS
    stitched together.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import time

import numpy as np
import pandas as pd

BARS_PER_DAY = 96
ANN = 365


# ------------------------------------------------------------------ data panel
def load_panel(data_dir: str, interval: str = "15", min_bars: int = 20_000):
    files = sorted(glob.glob(os.path.join(data_dir, f"*_{interval}m.csv")))
    cols = {}
    for f in files:
        sym = os.path.basename(f).split("_")[0]
        d = pd.read_csv(f)
        if len(d) < min_bars:
            continue
        d.index = pd.to_datetime(d["timestamp"], unit="ms", utc=True)
        d = d[~d.index.duplicated()]
        cols[sym] = d[["open", "high", "low", "close", "volume"]].astype("float32")
    if not cols:
        raise SystemExit(f"no data in {data_dir}")
    idx = sorted(set().union(*[c.index for c in cols.values()]))
    idx = pd.DatetimeIndex(idx)
    P = {k: pd.DataFrame({s: c[k].reindex(idx) for s, c in cols.items()}) for k in
         ("open", "high", "low", "close", "volume")}
    P["close"] = P["close"].ffill(limit=8)
    return P


def liquidity_gate(P, min_daily_turnover: float) -> pd.DataFrame:
    qv = (P["close"] * P["volume"]).fillna(0.0)
    daily = qv.rolling(BARS_PER_DAY, min_periods=BARS_PER_DAY).sum()
    med = daily.rolling(30 * BARS_PER_DAY, min_periods=7 * BARS_PER_DAY).median()
    return (med >= min_daily_turnover) & P["close"].notna()


# ------------------------------------------------------------------ indicators (panel)
def ema(x, n):
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(P, n):
    c = P["close"]
    pc = c.shift()
    tr = np.maximum(P["high"] - P["low"], np.maximum((P["high"] - pc).abs(), (P["low"] - pc).abs()))
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def state(entry, exit_):
    """Entry/exit boolean panels -> 0/1 position panel via forward fill (entry wins ties)."""
    s = pd.DataFrame(np.nan, index=entry.index, columns=entry.columns)
    s = s.mask(exit_, 0.0).mask(entry, 1.0)
    return s.ffill().fillna(0.0)


# ------------------------------------------------------------------ strategies
# Each returns a position panel in [-1, 1] decided at bar close.
def s_ema_cross(P, C, fast, slow, side):
    c = P["close"]
    f, s = ema(c, fast), ema(c, slow)
    pos = (f > s).astype("float32")
    if side == "ls":
        pos = pos - (f < s).astype("float32")
    return pos.where(s.notna(), 0.0)


def s_donchian(P, C, n_in, n_out, side):
    c, h, l = P["close"], P["high"], P["low"]
    hh = h.rolling(n_in).max().shift()
    ll_out = l.rolling(n_out).min().shift()
    pos = state(c > hh, c < ll_out)
    if side == "ls":
        ll = l.rolling(n_in).min().shift()
        hh_out = h.rolling(n_out).max().shift()
        short = state(c < ll, c > hh_out)
        pos = pos - short
    return pos.clip(-1, 1)


def s_tsmom(P, C, lookback_d, band, side):
    c = P["close"]
    r = c / c.shift(lookback_d * BARS_PER_DAY) - 1
    pos = state(r > band, r < 0)
    if side == "ls":
        pos = pos - state(r < -band, r > 0)
    return pos.clip(-1, 1)


def s_rsi_mr(P, C, n, lo, hi, trend):
    c = P["close"]
    x = rsi(c, n)
    entry = x < lo
    if trend:
        entry &= c > ema(c, 200 * 4)  # only buy dips in an up-trend (~200h EMA)
    return state(entry, x > hi)


def s_bb_mr(P, C, n, k, trend):
    c = P["close"]
    m = c.rolling(n).mean()
    sd = c.rolling(n).std()
    entry = c < m - k * sd
    if trend:
        entry &= c > ema(c, 200 * 4)
    stop = c < m - (k + 2.0) * sd  # crash guard
    return state(entry, (c > m) | stop)


def s_atr_trend(P, C, n_ema, mult, side):
    """EMA trend with ATR band entry (Keltner breakout) and exit back through the EMA."""
    c = P["close"]
    e = ema(c, n_ema)
    a = atr(P, 14 * 4)
    pos = state(c > e + mult * a, c < e)
    if side == "ls":
        pos = pos - state(c < e - mult * a, c > e)
    return pos.clip(-1, 1)


def s_xsmom(P, C, lookback_d, top_frac, side):
    """Cross-sectional momentum: once a day rank liquid coins by past return,
    long the top fraction (and short the bottom for side='ls')."""
    c = P["close"]
    day_end = c.index.hour.isin([23]) & (c.index.minute == 45)
    r = (c / c.shift(lookback_d * BARS_PER_DAY) - 1).where(C["gate"])
    rk = r[day_end].rank(axis=1, pct=True)
    pos = (rk >= 1 - top_frac).astype("float32")
    if side == "ls":
        pos = pos - (rk <= top_frac).astype("float32")
    pos = pos.where(rk.notna())
    return pos.reindex(c.index).ffill().fillna(0.0)


STRATEGIES = {
    "ema_cross": (s_ema_cross, {"fast": [24, 48, 96], "slow": [192, 384, 768], "side": ["long", "ls"]}),
    "donchian": (s_donchian, {"n_in": [96, 192, 480, 960], "n_out": [48, 96, 288], "side": ["long", "ls"]}),
    "tsmom": (s_tsmom, {"lookback_d": [3, 7, 14, 30], "band": [0.0, 0.05, 0.10], "side": ["long", "ls"]}),
    "atr_trend": (s_atr_trend, {"n_ema": [96, 192, 384], "mult": [1.0, 2.0, 3.0], "side": ["long", "ls"]}),
    "rsi_mr": (s_rsi_mr, {"n": [7, 14, 28], "lo": [15, 20, 25, 30], "hi": [50, 60, 70], "trend": [False, True]}),
    "bb_mr": (s_bb_mr, {"n": [20, 48, 96], "k": [2.0, 2.5, 3.0], "trend": [False, True]}),
    "xsmom": (s_xsmom, {"lookback_d": [3, 7, 14, 30], "top_frac": [0.1, 0.2, 0.3], "side": ["long", "ls"]}),
}


# ------------------------------------------------------------------ evaluation
class Costs:
    def __init__(self, fee=0.00055, slip_bp=2.0, funding_8h=0.0001):
        self.side = fee + slip_bp / 1e4
        self.fund_bar = funding_8h / 32.0  # 32 x 15m bars per 8h


def bar_pnl(pos: pd.DataFrame, ret: pd.DataFrame, gate: pd.DataFrame, costs: Costs):
    """Per-symbol net return panel. pos decided at t is held over bar t+1."""
    p = pos.where(gate, 0.0).fillna(0.0).astype("float32")
    held = p.shift().fillna(0.0)
    turn = (p - p.shift().fillna(0.0)).abs().shift().fillna(0.0)
    net = held * ret - turn * costs.side - held * costs.fund_bar
    return net, held


def portfolio(net: pd.DataFrame, gate: pd.DataFrame) -> pd.Series:
    """Equal capital split across currently tradeable symbols."""
    n = gate.shift().fillna(False).sum(axis=1).replace(0, np.nan)
    return (net.sum(axis=1) / n).fillna(0.0)


def trades(held_col: np.ndarray, net_col: np.ndarray):
    """Trade-level returns of one symbol (a trade = a run of same-sign position)."""
    sgn = np.sign(held_col)
    chg = np.flatnonzero(np.diff(np.r_[0, sgn]) != 0)
    out = []
    for i, st in enumerate(chg):
        if sgn[st] == 0:
            continue
        en = chg[i + 1] if i + 1 < len(chg) else len(sgn)
        # exit cost lands on the bar after the run ends
        seg = net_col[st:min(en + 1, len(net_col))]
        out.append(np.prod(1 + seg) - 1)
    return out


def metrics(r: pd.Series, net=None, held=None) -> dict:
    r = r.fillna(0.0)
    eq = (1 + r).cumprod()
    days = max((r.index[-1] - r.index[0]).total_seconds() / 86400, 1)
    tot = float(eq.iloc[-1] - 1)
    cagr = float(eq.iloc[-1] ** (ANN / days) - 1) if eq.iloc[-1] > 0 else -1.0
    mdd = float((eq / eq.cummax() - 1).min())
    dr = r.resample("1D").apply(lambda x: (1 + x).prod() - 1)
    sh = float(dr.mean() / dr.std() * np.sqrt(ANN)) if dr.std() > 0 else 0.0
    out = {"ret": tot, "cagr": cagr, "mdd": mdd, "sharpe": sh,
           "calmar": cagr / abs(mdd) if mdd < 0 else 0.0}
    if net is not None:
        tr = []
        sym_ret = []
        H, N = held.values, net.values
        for j in range(H.shape[1]):
            t = trades(H[:, j], N[:, j])
            tr += t
            if np.abs(H[:, j]).sum() > 0:
                sym_ret.append(np.prod(1 + N[:, j]) - 1)
        tr = np.asarray(tr)
        out.update({"trades": int(len(tr)),
                    "win_rate": float((tr > 0).mean()) if len(tr) else np.nan,
                    "avg_trade": float(tr.mean()) if len(tr) else np.nan,
                    "pf": float(tr[tr > 0].sum() / -tr[tr < 0].sum()) if (tr < 0).any() else np.nan,
                    "sym_pos_frac": float(np.mean(np.asarray(sym_ret) > 0)) if sym_ret else np.nan,
                    "exposure": float(np.abs(H).sum() / max(1, (H.shape[0] * H.shape[1])))})
    return out


def combos(grid: dict):
    keys = list(grid)
    for vals in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, vals))


def plateau(df: pd.DataFrame, grid: dict, score: str) -> pd.Series:
    """Mean score over the combo and every combo that differs by one grid step in one param."""
    keys = list(grid)
    pos = {k: {v: i for i, v in enumerate(grid[k])} for k in keys}
    lut = {tuple(r[k] for k in keys): r[score] for _, r in df.iterrows()}
    out = []
    for _, r in df.iterrows():
        vals = [r[k] for k in keys]
        sc = [r[score]]
        for i, k in enumerate(keys):
            for d in (-1, 1):
                j = pos[k][vals[i]] + d
                if 0 <= j < len(grid[k]):
                    nb = list(vals)
                    nb[i] = grid[k][j]
                    if tuple(nb) in lut:
                        sc.append(lut[tuple(nb)])
        out.append(float(np.nanmean(sc)))
    return pd.Series(out, index=df.index)


def is_score(m: dict) -> float:
    """Selection score on IS: Sharpe, but disqualify thin / ruinous results."""
    if m.get("trades", 0) < 100 or m["mdd"] < -0.6:
        return -9.0
    return m["sharpe"]


# ------------------------------------------------------------------ main
def run(a):
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    P = load_panel(a.data_dir, min_bars=a.min_bars)
    if a.symbols:
        P = {k: v[[s for s in a.symbols if s in v]] for k, v in P.items()}
    gate = liquidity_gate(P, a.min_daily_turnover)
    C = {"gate": gate}
    ret = P["close"].pct_change(fill_method=None).fillna(0.0).clip(-0.5, 1.0)
    idx = ret.index
    cut = idx[int(len(idx) * a.is_frac)]
    isel, osel = idx < cut, idx >= cut
    costs = Costs(a.fee, a.slippage_bp, a.funding)
    print(f"panel {len(idx)} bars x {ret.shape[1]} symbols  {idx[0].date()}..{idx[-1].date()}  "
          f"IS<{cut.date()}  liquid sym-bars {gate.values.mean():.1%}", flush=True)

    # benchmark: equal-weight buy & hold of liquid coins
    bh_net, bh_held = bar_pnl(pd.DataFrame(1.0, index=idx, columns=ret.columns), ret, gate, costs)
    bench = {}
    for name, sel in (("is", isel), ("oos", osel)):
        bench[name] = metrics(portfolio(bh_net[sel], gate[sel]))

    folds = make_folds(idx, a.wf_train_m, a.wf_test_m)
    rows, store = [], {}
    for sname, (fn, grid) in STRATEGIES.items():
        if a.strategies and sname not in a.strategies:
            continue
        for prm in combos(grid):
            pos = fn(P, C, **prm)
            net, held = bar_pnl(pos, ret, gate, costs)
            pr = portfolio(net, gate)
            rec = {"strategy": sname, **{f"p_{k}": v for k, v in prm.items()}}
            for name, sel in (("is", isel), ("oos", osel)):
                m = metrics(pr[sel], net[sel], held[sel])
                rec.update({f"{name}_{k}": v for k, v in m.items()})
            rec["is_score"] = is_score({k[3:]: v for k, v in rec.items() if k.startswith("is_")})
            fstats = []
            for tr0, tr1, te1 in folds:
                trs, tes = (idx >= tr0) & (idx < tr1), (idx >= tr1) & (idx < te1)
                ntr = int((held[trs].diff().abs() > 0).values.sum())
                fstats.append({"train_entries": ntr, "test": metrics(pr[tes], net[tes], held[tes])})
            store.setdefault(sname, []).append((prm, pr.astype("float64"), fstats))
            rows.append(rec)
            print(f"  {sname:9s} {prm}  IS sh={rec['is_sharpe']:.2f} ret={rec['is_ret']:+.1%}  "
                  f"OOS sh={rec['oos_sharpe']:.2f} ret={rec['oos_ret']:+.1%} mdd={rec['oos_mdd']:.1%} "
                  f"n={rec['oos_trades']}  [{time.time() - t0:.0f}s]", flush=True)
            del pos, net, held
    res = pd.DataFrame(rows)

    # plateau score per strategy (IS only) and the chosen combo
    picks = []
    for sname, g in res.groupby("strategy", sort=False):
        grid = STRATEGIES[sname][1]
        gg = g.rename(columns={f"p_{k}": k for k in grid})
        res.loc[g.index, "is_plateau"] = plateau(gg, grid, "is_score").values
        picks.append(res.loc[g.index].sort_values("is_plateau", ascending=False).iloc[0])
    res.to_csv(f"{a.out}/grid_results.csv", index=False)
    pd.DataFrame(picks).to_csv(f"{a.out}/is_selected.csv", index=False)

    # IS-vs-OOS rank correlation per strategy (overfitting diagnostic)
    diag = []
    for sname, g in res.groupby("strategy", sort=False):
        diag.append({"strategy": sname, "combos": len(g),
                     "is_oos_spearman": g["is_sharpe"].corr(g["oos_sharpe"], method="spearman"),
                     "oos_sharpe_median": g["oos_sharpe"].median(),
                     "oos_pos_frac": (g["oos_ret"] > 0).mean()})
    pd.DataFrame(diag).to_csv(f"{a.out}/overfit_diag.csv", index=False)

    # walk-forward: re-pick params on each train window (plateau of train Sharpe)
    wf_rows, curves = [], {}
    for sname, items in store.items():
        grid = STRATEGIES[sname][1]
        parts = []
        for k, (tr0, tr1, te1) in enumerate(folds):
            trs, tes = (idx >= tr0) & (idx < tr1), (idx >= tr1) & (idx < te1)
            sc = []
            for prm, pr, fs in items:
                m = metrics(pr[trs])
                ok = fs[k]["train_entries"] >= 100 and m["mdd"] > -0.6
                sc.append(m["sharpe"] if ok else -9.0)
            df = pd.DataFrame([{**prm, "s": s_} for (prm, _, _), s_ in zip(items, sc)])
            df["pl"] = plateau(df, grid, "s").values
            j = int(df["pl"].idxmax())
            prm, pr, fs = items[j]
            parts.append(pr[tes])
            wf_rows.append({"strategy": sname, "fold": k, "train": f"{tr0.date()}..{tr1.date()}",
                            "test": f"{tr1.date()}..{te1.date()}", "params": json.dumps(prm),
                            "train_sharpe": sc[j],
                            **{f"test_{x}": v for x, v in fs[k]["test"].items()}})
        curves[sname] = pd.concat(parts)
        f = pd.DataFrame([r for r in wf_rows if r["strategy"] == sname])
        curves[sname].to_frame("ret").to_csv(f"{a.out}/wf_curve_{sname}.csv")
        print(f"  WF {sname}: {metrics(curves[sname])}  trades={int(f.test_trades.sum())}", flush=True)
    pd.DataFrame(wf_rows).to_csv(f"{a.out}/walkforward_folds.csv", index=False)
    wf_sum = []
    for sname, r in curves.items():
        f = pd.DataFrame([x for x in wf_rows if x["strategy"] == sname])
        wf_sum.append({"strategy": sname, **metrics(r), "trades": int(f.test_trades.sum()),
                       "win_rate": float((f.test_win_rate * f.test_trades).sum() / max(1, f.test_trades.sum())),
                       "folds_pos": int((f.test_ret > 0).sum()), "folds": len(f)})
    pd.DataFrame(wf_sum).to_csv(f"{a.out}/walkforward_summary.csv", index=False)
    bh = portfolio(bh_net, gate)
    wf0 = folds[0][1] if folds else idx[0]
    bh.to_frame("ret").to_csv(f"{a.out}/bench_curve.csv")
    meta = {"bars": len(idx), "symbols": int(ret.shape[1]), "start": str(idx[0]), "end": str(idx[-1]),
            "is_cut": str(cut), "bench": bench, "bench_wf": metrics(bh[bh.index >= wf0]),
            "folds": [[str(x.date()) for x in f] for f in folds],
            "costs": {"fee": a.fee, "slippage_bp": a.slippage_bp, "funding_8h": a.funding},
            "min_daily_turnover": a.min_daily_turnover, "runtime_s": round(time.time() - t0)}
    json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=2, default=str)
    print(json.dumps(meta, indent=2, default=str))
    return res


def make_folds(idx, train_m, test_m):
    folds = []
    t = idx[0] + pd.DateOffset(months=train_m)
    while t + pd.DateOffset(months=test_m) <= idx[-1] + pd.Timedelta(days=10):
        folds.append((t - pd.DateOffset(months=train_m), t, t + pd.DateOffset(months=test_m)))
        t += pd.DateOffset(months=test_m)
    return folds


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="results/lab")
    p.add_argument("--symbols", nargs="*")
    p.add_argument("--strategies", nargs="*")
    p.add_argument("--min-bars", type=int, default=20_000)
    p.add_argument("--min-daily-turnover", type=float, default=5e6)
    p.add_argument("--is-frac", type=float, default=0.6)
    p.add_argument("--fee", type=float, default=0.00055)
    p.add_argument("--slippage-bp", type=float, default=2.0)
    p.add_argument("--funding", type=float, default=0.0001)
    p.add_argument("--wf-train-m", type=int, default=12)
    p.add_argument("--wf-test-m", type=int, default=3)
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
