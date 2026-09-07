"""Walk-forward validation: PCA and parameters are re-fit on past data only, and every
reported number comes from the untouched test window that follows them."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import ExecConfig
from .metrics import summarize
from .optimization import evaluate, plateau_pick, search
from .pca_model import PCAModel


@dataclass
class WFConfig:
    train_bars: int = 26_000      # ~9 months of 15m bars
    val_bars: int = 8_700         # ~3 months
    test_bars: int = 8_700        # ~3 months
    anchored: bool = True         # expanding train window (False -> rolling)
    n_components: int = 3


def make_folds(n: int, w: WFConfig) -> list[dict]:
    folds, start = [], 0
    while True:
        tr_end = start + w.train_bars if not w.anchored else w.train_bars + len(folds) * w.test_bars
        va_end, te_end = tr_end + w.val_bars, tr_end + w.val_bars + w.test_bars
        if te_end > n:
            break
        folds.append({"fold": len(folds) + 1,
                      "train": (0 if w.anchored else start, tr_end),
                      "val": (tr_end, va_end), "test": (va_end, te_end)})
        start += w.test_bars
    return folds


def run_walk_forward(d: pd.DataFrame, combos, cfg: ExecConfig, w: WFConfig,
                     symbol: str = "", numeric_params=None, min_trades: int = 15,
                     verbose: bool = True):
    """Returns (fold_summary_df, oos_trades_df, oos_equity_series, chosen_params)."""
    numeric_params = numeric_params or ["e_rsi_max", "e_bbp_max", "g_step_atr",
                                        "g_max_entries", "r_tp_value", "r_sl_value"]
    folds = make_folds(len(d), w)
    rows, all_trades, eq_parts, chosen = [], [], [], []
    equity_level = cfg.initial_capital

    for f in folds:
        model = PCAModel(w.n_components).fit(d, _mask(len(d), f["train"]))
        dd = model.attach(d)
        val = search(dd, f["val"], combos, cfg, model.quantiles_, symbol=symbol)
        pick = plateau_pick(val, numeric_params, min_trades=min_trades)
        from .optimization import params_from_row
        ep, gp, rp = params_from_row(pick)
        s_test, res = evaluate(dd, f["test"], ep, gp, rp, cfg, model.quantiles_,
                               f"fold{f['fold']}_test", symbol=symbol, keep=True)
        s_val = {k: pick[k] for k in ("roi", "max_drawdown", "profit_factor", "trades", "score")}

        # stitch OOS equity by compounding fold returns
        curve = res.equity / cfg.initial_capital * equity_level
        equity_level = float(curve.iloc[-1])
        eq_parts.append(curve)
        t = res.trades.copy()
        if len(t):
            t["fold"] = f["fold"]
            all_trades.append(t)

        rows.append({"fold": f["fold"], "symbol": symbol,
                     "train_end": str(d["datetime"].iloc[f["train"][1] - 1]),
                     "test_start": str(d["datetime"].iloc[f["test"][0]]),
                     "test_end": str(d["datetime"].iloc[f["test"][1] - 1]),
                     "val_roi": s_val["roi"], "val_pf": s_val["profit_factor"],
                     "val_trades": s_val["trades"],
                     "test_roi": s_test["roi"], "test_mdd": s_test["max_drawdown"],
                     "test_pf": s_test["profit_factor"], "test_win": s_test["win_rate"],
                     "test_trades": s_test["trades"], "test_sharpe": s_test["sharpe"],
                     "liquidations": s_test["liquidations"],
                     "params": _pstr(ep, gp, rp)})
        chosen.append({"fold": f["fold"], **{f"e_{k}": v for k, v in ep.to_dict().items()},
                       **{f"g_{k}": v for k, v in gp.to_dict().items()},
                       **{f"r_{k}": v for k, v in rp.to_dict().items()}})
        if verbose:
            print(f"  fold {f['fold']}: val ROI {s_val['roi']:+.2%} -> test ROI "
                  f"{s_test['roi']:+.2%} (MDD {s_test['max_drawdown']:.2%}, "
                  f"{s_test['trades']} trades) | {_pstr(ep, gp, rp)}")

    fold_df = pd.DataFrame(rows)
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    equity = pd.concat(eq_parts) if eq_parts else pd.Series(dtype=float)
    equity = equity[~equity.index.duplicated(keep="last")]
    return fold_df, trades, equity, pd.DataFrame(chosen)


def oos_summary(fold_df: pd.DataFrame, trades: pd.DataFrame, equity: pd.Series,
                cfg: ExecConfig, label="walk_forward_OOS") -> dict:
    s = summarize(equity, trades, label, initial=cfg.initial_capital)
    if len(fold_df):
        s.update({"folds": len(fold_df),
                  "folds_profitable": int((fold_df["test_roi"] > 0).sum()),
                  "fold_roi_mean": float(fold_df["test_roi"].mean()),
                  "fold_roi_std": float(fold_df["test_roi"].std()),
                  "worst_fold_roi": float(fold_df["test_roi"].min()),
                  "val_test_corr": float(fold_df["val_roi"].corr(fold_df["test_roi"]))
                  if len(fold_df) > 2 else np.nan})
    return s


def _mask(n: int, sl: tuple[int, int]) -> np.ndarray:
    m = np.zeros(n, bool)
    m[sl[0]:sl[1]] = True
    return m


def _pstr(ep, gp, rp) -> str:
    return (f"RSI<={ep.rsi_max} BB<={ep.bbp_max} Vol>={ep.vol_min} {ep.trend} "
            f"PC1q>={ep.pc1_min_q} PC2q<={ep.pc2_max_q} | grid {gp.step_atr}ATR x{gp.max_entries} "
            f"{gp.weights} | TP {rp.tp_mode} {rp.tp_value} SL {rp.sl_mode} {rp.sl_value}")
