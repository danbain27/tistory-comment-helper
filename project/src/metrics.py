"""Performance statistics for an equity curve + trade log."""
from __future__ import annotations

import numpy as np
import pandas as pd

BARS_PER_YEAR = 35_040  # 15m


def drawdown(eq: pd.Series) -> pd.Series:
    return eq / eq.cummax() - 1.0


def max_consecutive_losses(profits: pd.Series) -> int:
    best = cur = 0
    for p in profits:
        cur = cur + 1 if p <= 0 else 0
        best = max(best, cur)
    return best


def monthly_returns(eq: pd.Series) -> pd.Series:
    m = eq.resample("ME").last()
    first = eq.iloc[0]
    return (m / pd.concat([pd.Series([first], index=[m.index[0]]), m.shift(1).iloc[1:]]) - 1).dropna() \
        if len(m) else pd.Series(dtype=float)


def summarize(eq: pd.Series, trades: pd.DataFrame, label: str = "",
              bars_per_year: int = BARS_PER_YEAR, initial: float | None = None) -> dict:
    eq = eq.dropna()
    if len(eq) < 2:
        return {"strategy": label, "trades": 0}
    init = float(initial if initial is not None else eq.iloc[0])
    final = float(eq.iloc[-1])
    years = len(eq) / bars_per_year
    ret = eq.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    dd = drawdown(eq)
    downside = ret[ret < 0]
    sharpe = float(ret.mean() / ret.std() * np.sqrt(bars_per_year)) if ret.std() > 0 else 0.0
    sortino = float(ret.mean() / downside.std() * np.sqrt(bars_per_year)) if len(downside) and downside.std() > 0 else np.nan
    mdd = float(dd.min())

    out = {"strategy": label, "period_start": str(eq.index[0]), "period_end": str(eq.index[-1]),
           "years": round(years, 3), "initial_capital": init, "final_capital": final,
           "net_profit": final - init, "roi": final / init - 1.0,
           "cagr": (final / init) ** (1 / years) - 1 if years > 0 and final > 0 else -1.0,
           "max_drawdown": mdd, "sharpe": sharpe, "sortino": sortino,
           "calmar": (final / init) ** (1 / years) - 1 if years > 0 and final > 0 and mdd < 0 else np.nan}
    if out["calmar"] is not np.nan and mdd < 0:
        out["calmar"] = out["cagr"] / abs(mdd)

    if trades is None or len(trades) == 0:
        out.update({"trades": 0, "win_rate": np.nan, "profit_factor": np.nan,
                    "avg_trade": np.nan, "max_consecutive_losses": 0,
                    "avg_holding_hours": np.nan, "avg_grid_entries": np.nan,
                    "tp_rate": np.nan, "sl_rate": np.nan, "liquidations": 0})
        return out

    p = trades["profit"]
    wins, losses = p[p > 0], p[p <= 0]
    out.update({
        "trades": int(len(trades)),
        "win_rate": float((p > 0).mean()),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else np.inf,
        "avg_trade": float(p.mean()),
        "avg_win": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss": float(losses.mean()) if len(losses) else np.nan,
        "max_consecutive_losses": max_consecutive_losses(p),
        "avg_holding_hours": float(trades["holding_hours"].mean()),
        "max_holding_hours": float(trades["holding_hours"].max()),
        "avg_grid_entries": float(trades["grid_entry_count"].mean()),
        "tp_rate": float((trades["exit_reason"] == "tp").mean()),
        "sl_rate": float((trades["exit_reason"] == "sl").mean()),
        "liquidations": int((trades["exit_reason"] == "liquidation").sum()),
        "worst_trade": float(p.min()),
        "best_trade": float(p.max()),
        "total_fees": float(trades["fees"].sum()) if "fees" in trades else np.nan,
        "total_funding": float(trades["funding"].sum()) if "funding" in trades else np.nan,
        "exposure": float(trades["holding_bars"].sum() / max(len(eq), 1)),
    })
    return out


def robust_score(s: dict, min_trades: int = 30) -> float:
    """Selection score: rewards risk-adjusted return, penalises thin samples, hard-fails
    on ruin/liquidation. Deliberately NOT raw ROI."""
    if s.get("trades", 0) < min_trades:
        return -np.inf
    if s.get("liquidations", 0) > 0:
        return -np.inf
    mdd = abs(s.get("max_drawdown", 0.0)) or 1e-6
    pf = s.get("profit_factor", 0.0)
    pf = 3.0 if not np.isfinite(pf) else min(pf, 3.0)
    sharpe = s.get("sharpe", 0.0) or 0.0
    cagr = s.get("cagr", 0.0)
    if cagr <= 0:
        return -abs(cagr) - mdd
    sample = min(1.0, s["trades"] / 100.0)
    return float((cagr / mdd) * 0.5 + sharpe * 0.3 + (pf - 1.0) * 0.2) * (0.5 + 0.5 * sample)


def to_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)
