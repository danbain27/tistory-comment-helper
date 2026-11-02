"""Render strategy_lab outputs as markdown tables (pasted into docs/ report).

    python -m src.lab_report --lab results/lab > /tmp/tables.md
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

PCT = ("ret", "cagr", "mdd", "win_rate", "sym_pos_frac", "exposure", "avg_trade")


def fmt(df: pd.DataFrame) -> str:
    d = df.copy()
    for c in d.columns:
        if any(c.endswith(p) for p in PCT):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:+.1%}" if "ret" in c or "cagr" in c or "mdd" in c or "avg" in c else f"{v:.0%}")
        elif d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    return d.to_markdown(index=False)


def params(row) -> str:
    return ", ".join(f"{k[2:]}={row[k]}" for k in row.index if k.startswith("p_") and pd.notna(row[k]))


def main(lab: str):
    g = pd.read_csv(f"{lab}/grid_results.csv")
    meta = json.load(open(f"{lab}/meta.json"))
    g["params"] = g.apply(params, axis=1)
    cols = ["strategy", "params", "is_ret", "is_sharpe", "is_mdd", "oos_ret", "oos_mdd",
            "oos_sharpe", "oos_win_rate", "oos_trades", "oos_pf", "oos_sym_pos_frac"]
    print("## benchmark (equal-weight buy&hold)\n")
    print(fmt(pd.DataFrame([{"segment": k, **v} for k, v in meta["bench"].items()]
                           + [{"segment": "wf", **meta["bench_wf"]}])))
    print("\n## IS-selected (plateau) per strategy\n")
    print(fmt(pd.read_csv(f"{lab}/is_selected.csv").assign(
        params=lambda d: d.apply(params, axis=1))[cols]))
    print("\n## top 15 by IS plateau score (all strategies)\n")
    print(fmt(g.sort_values("is_plateau", ascending=False)[cols].head(15)))
    print("\n## top 15 by OOS Sharpe (hindsight - NOT a selection rule)\n")
    print(fmt(g.sort_values("oos_sharpe", ascending=False)[cols].head(15)))
    print("\n## overfitting diagnostic\n")
    print(fmt(pd.read_csv(f"{lab}/overfit_diag.csv")))
    print("\n## walk-forward (OOS stitched)\n")
    print(fmt(pd.read_csv(f"{lab}/walkforward_summary.csv")))
    wf = pd.read_csv(f"{lab}/walkforward_folds.csv")
    print("\n## walk-forward folds\n")
    print(fmt(wf[["strategy", "fold", "test", "params", "train_sharpe", "test_ret", "test_mdd",
                  "test_sharpe", "test_trades", "test_win_rate"]]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", default="results/lab")
    main(ap.parse_args().lab)
