"""Monte Carlo robustness: trade resampling + execution-cost stress tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import ExecConfig
from .optimization import evaluate


def _path_stats(returns: np.ndarray, initial: float) -> tuple[float, float]:
    eq = initial * np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(np.concatenate([[initial], eq]))
    dd = (np.concatenate([[initial], eq]) / peak - 1.0).min()
    return float(eq[-1]), float(dd)


def simulate(trades: pd.DataFrame, initial: float = 10_000.0, n_sims: int = 2000,
             mode: str = "bootstrap", extra_cost: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """mode: 'shuffle' (same trades, random order) or 'bootstrap' (resample with
    replacement). extra_cost is subtracted from every trade return (fee/slippage stress)."""
    if trades is None or len(trades) == 0:
        return pd.DataFrame()
    r = trades["trade_return"].to_numpy(float) - extra_cost
    rng = np.random.default_rng(seed)
    finals, mdds = np.empty(n_sims), np.empty(n_sims)
    for i in range(n_sims):
        s = rng.permutation(r) if mode == "shuffle" else rng.choice(r, size=len(r), replace=True)
        finals[i], mdds[i] = _path_stats(s, initial)
    return pd.DataFrame({"final_equity": finals, "roi": finals / initial - 1.0, "mdd": mdds})


def summarize_mc(sim: pd.DataFrame, initial: float = 10_000.0, ruin_level: float = 0.5) -> dict:
    if sim is None or len(sim) == 0:
        return {}
    return {"sims": len(sim),
            "roi_mean": float(sim["roi"].mean()), "roi_median": float(sim["roi"].median()),
            "roi_p05": float(sim["roi"].quantile(0.05)), "roi_p95": float(sim["roi"].quantile(0.95)),
            "prob_profit": float((sim["roi"] > 0).mean()),
            "mdd_median": float(sim["mdd"].median()),
            "mdd_p95_worst": float(sim["mdd"].quantile(0.05)),
            "mdd_worst": float(sim["mdd"].min()),
            "prob_dd_gt_20pct": float((sim["mdd"] < -0.20).mean()),
            "prob_dd_gt_40pct": float((sim["mdd"] < -0.40).mean()),
            "prob_ruin_50pct_loss": float((sim["final_equity"] < initial * ruin_level).mean())}


def max_consecutive_loss_distribution(trades: pd.DataFrame, n_sims: int = 2000, seed: int = 0) -> dict:
    if trades is None or len(trades) == 0:
        return {}
    r = (trades["trade_return"].to_numpy(float) <= 0).astype(int)
    rng = np.random.default_rng(seed)
    out = np.empty(n_sims, int)
    for i in range(n_sims):
        s = rng.permutation(r)
        best = cur = 0
        for x in s:
            cur = cur + 1 if x else 0
            best = max(best, cur)
        out[i] = best
    return {"consec_loss_median": int(np.median(out)), "consec_loss_p95": int(np.quantile(out, 0.95)),
            "consec_loss_max": int(out.max())}


def cost_stress(d: pd.DataFrame, sl, ep, gp, rp, cfg: ExecConfig, quantiles,
                fee_mults=(1.0, 1.5, 2.0), slip_bps=(2.0, 5.0, 10.0),
                delays=(1, 2, 3), symbol="") -> pd.DataFrame:
    """Re-runs the engine under harsher execution assumptions (this is the honest test:
    resampling trade returns cannot capture a changed fill path)."""
    from dataclasses import replace
    rows = []
    for fm in fee_mults:
        for sb in slip_bps:
            for dl in delays:
                c = replace(cfg, fee_taker=cfg.fee_taker * fm, fee_maker=cfg.fee_maker * fm,
                            slippage_bp=sb, entry_delay_bars=dl)
                s, _ = evaluate(d, sl, ep, gp, rp, c, quantiles,
                                f"fee x{fm} slip {sb}bp delay {dl}", symbol=symbol)
                rows.append({"fee_mult": fm, "slippage_bp": sb, "entry_delay_bars": dl,
                             "trades": s["trades"], "roi": s["roi"],
                             "max_drawdown": s["max_drawdown"],
                             "profit_factor": s["profit_factor"], "win_rate": s["win_rate"],
                             "sharpe": s["sharpe"]})
    return pd.DataFrame(rows)
