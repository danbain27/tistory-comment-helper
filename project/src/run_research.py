"""End-to-end research pipeline (Steps 1-16).

    python -m src.run_research --symbols BTCUSDT ETHUSDT --quick

Every number that is used to CHOOSE something comes from train/validation.
The test slice and the walk-forward test windows are only ever measured.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import yaml

from . import charts, data as dataio, universe as uni
from .analysis import conditional_edge, full_feature_scan
from .backtest import ExecConfig
from .features import add_forward_returns, build_features, time_split
from .grid import GridParams
from .monte_carlo import (cost_stress, max_consecutive_loss_distribution, simulate,
                          summarize_mc)
from .optimization import (entry_candidates, evaluate, evaluate_multi, exit_candidates,
                           grid_candidates, mask_to_slice, params_from_row, plateau_pick,
                           search, sensitivity_table)
from .pca_model import PCAModel
from .regimes import label_regimes, pca_state_table, regime_table
from .risk import RiskParams
from .scan import (ScanConfig, pool_feature_stats, pool_loadings, run_scan,
                   select_basket, universe_verdict_stats, validate_universe)
from .signal import filter_stack
from .walk_forward import WFConfig, oos_summary, run_walk_forward

pd.set_option("display.width", 200)


# ----------------------------------------------------------------------- utils
def hr(t: str):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def save(df: pd.DataFrame, path: str, note: str = ""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  -> {path} ({len(df)} rows) {note}")


def main(a):
    os.makedirs(a.results, exist_ok=True)
    os.makedirs(a.charts, exist_ok=True)
    cfg = ExecConfig(initial_capital=a.capital, fee_taker=a.fee_taker, fee_maker=a.fee_maker,
                     slippage_bp=a.slippage_bp, funding_rate_8h=a.funding,
                     use_funding=not a.no_funding, entry_delay_bars=1)
    report: list[str] = []

    # ------------------------------------------------------------------ Step 0
    hr("STEP 0  universe")
    all_symbols, ulist, usrc = uni.resolve(a, a.data_dir)
    print(f"  universe source: {usrc}  ({len(all_symbols)} symbols)")
    if usrc == "bybit":
        save(ulist, f"{a.results}/universe.csv")
        print("  주의: Bybit 상장 목록에는 상장폐지된 코인이 없습니다 (생존 편향). "
              "유니버스 전체 통계는 실제보다 좋게 나옵니다.")
        report.append("> **생존 편향 주의**: 현재 상장 중인 코인만 스캔합니다. "
                      "상장폐지된(대개 폭락한) 코인이 빠져 있으므로 유니버스 통계는 낙관적입니다.")
    print(f"  제외 instrument class: {', '.join(a.exclude_symbol_types) or '(없음)'}")
    print(f"  point-in-time 유동성 하한: {a.min_daily_turnover:,.0f} USDT/day "
          f"({'적용' if a.min_daily_turnover > 0 else '해제'})")
    if a.min_daily_turnover > 0:
        report.append(f"> **유동성 게이트**: 24h 거래대금은 오늘 기준 스냅샷이라 그대로 3년 "
                      f"백테스트에 쓰면 룩어헤드입니다. 심볼별로 30일 이동중앙값 일거래대금이 "
                      f"{a.min_daily_turnover:,.0f} USDT 를 처음 넘는 시점 이전 구간을 버리고, "
                      f"train+validation 구간에서 이 하한을 넘는 봉 비율이 "
                      f"{a.min_liquid_frac:.0%} 미만인 심볼은 제외합니다. "
                      f"컷 위치는 앞구간만 보고 정해지므로 test 구간 거래량이 "
                      f"train/validation 경계를 움직이지 못합니다.")
    scfg = ScanConfig(data_dir=a.data_dir, interval=a.interval, days=a.days,
                      min_bars=a.min_bars, min_daily_turnover=a.min_daily_turnover,
                      min_liquid_frac=a.min_liquid_frac,
                      n_components=a.n_components,
                      train_frac=a.train_frac, val_frac=a.val_frac,
                      allow_synthetic=a.allow_synthetic, exec_cfg=cfg, leverage=a.leverage)

    scan_df = pd.DataFrame()
    if len(all_symbols) > a.basket_size and not a.skip_scan:
        hr(f"STEP 0b  streaming scan of {len(all_symbols)} symbols "
           f"(TRAIN+VALIDATION only, jobs={a.jobs})")
        scan_df, pool, load = run_scan(all_symbols, scfg, jobs=a.jobs)
        save(scan_df, f"{a.results}/universe_scan.csv")
        ok = scan_df[scan_df.status == "ok"] if "status" in scan_df else scan_df
        bad = scan_df[scan_df.status != "ok"] if "status" in scan_df else pd.DataFrame()
        print(f"  scanned ok: {len(ok)}   skipped: {len(bad)} "
              f"({dict(bad['status'].str.slice(0, 12).value_counts()) if len(bad) else {}})")
        if len(ok):
            cols = ["symbol", "bars", "train_trades", "train_roi", "val_trades", "val_roi",
                    "val_pf", "val_mdd", "screen_score"]
            print("\n  상위 15 (screen_score = train/val 일관성 가중):")
            print(ok[cols].head(15).round(4).to_string(index=False))
            print("\n  하위 5:")
            print(ok[cols].tail(5).round(4).to_string(index=False))
            report.append(f"### 유니버스 스캔 ({len(ok)}개 심볼, train+val)\n```\n"
                          + ok[cols].head(15).round(4).to_string(index=False) + "\n```")
        pooled = pool_feature_stats(pool)
        save(pooled, f"{a.results}/pooled_feature_analysis.csv")
        if len(pooled):
            print("\n  유니버스 통합 조건별 forward return (표본이 심볼 수만큼 커짐):")
            pc = pooled[["feature", "bucket", "symbols", "n", "h4_mean", "h4_win", "h4_pf", "h4_t"]]
            print(pc.round(4).to_string(index=False))
            report.append("### 유니버스 통합 Feature -> 4봉 forward return\n```\n"
                          + pc.round(4).to_string(index=False) + "\n```")
        stab = pool_loadings(load)
        save(stab, f"{a.results}/pca_loading_stability.csv")
        if len(stab):
            print("\n  PCA loading 안정성 (심볼 간 평균 +- 표준편차):")
            print(stab.round(3).to_string(index=False))
            report.append("### PCA loading 안정성 (심볼 간)\n```\n"
                          + stab.round(3).to_string(index=False) + "\n```")
        symbols = select_basket(scan_df, a.basket_size, a.min_bars)
        print(f"\n  심층 연구 바스켓 ({len(symbols)}): {', '.join(symbols)}")
        report.append(f"### 심층 연구 바스켓\n{', '.join(symbols)}")
    else:
        symbols = all_symbols[:a.basket_size] if len(all_symbols) > a.basket_size else all_symbols
        print(f"  scan 생략, 바스켓 = {', '.join(symbols)}")

    # ---------------------------------------------------------------- Step 1-3
    hr("STEP 1-3  load / quality check / features (basket)")
    D, sources, quality = {}, {}, []
    for sym in symbols:
        df, src = dataio.ensure_dataset(sym, a.data_dir, a.interval, allow_synthetic=a.allow_synthetic,
                                        days=a.days, seed=dataio.symbol_seed(sym))
        q = dataio.check_quality(df)
        print(f"  {sym:10s} [{src}] {q.to_text()}")
        quality.append({"symbol": sym, "source": src, **asdict(q)})
        df = dataio.clean(df)
        d = add_forward_returns(build_features(df))
        d = label_regimes(d)
        D[sym], sources[sym] = d, src
    save(pd.DataFrame(quality), f"{a.results}/data_quality.csv")
    if set(sources.values()) == {"synthetic"}:
        report.append("**경고: 실데이터를 받지 못해 합성 데이터로 실행되었습니다. "
                      "아래 수치는 파이프라인 검증용이며 전략 근거로 쓸 수 없습니다.**")

    splits = {s: time_split(D[s], a.train_frac, a.val_frac) for s in symbols}
    sl_train = {s: mask_to_slice(splits[s][0]) for s in symbols}
    sl_val = {s: mask_to_slice(splits[s][1]) for s in symbols}
    sl_test = {s: mask_to_slice(splits[s][2]) for s in symbols}
    primary = symbols[0]
    print(f"  split ({primary}): train {sl_train[primary]}  val {sl_val[primary]}  test {sl_test[primary]}")

    # ---------------------------------------------------------------- Step 4-5
    hr("STEP 4-5  PCA fit on TRAIN only + loadings")
    models, pca_rows = {}, []
    for s in symbols:
        m = PCAModel(a.n_components).fit(D[s], splits[s][0])
        D[s] = m.attach(D[s])
        models[s] = m
        for _, r in m.interpret().iterrows():
            pca_rows.append({"symbol": s, **r.to_dict()})
    print(models[primary].report())
    save(pd.DataFrame(pca_rows), f"{a.results}/pca_components.csv")
    save(models[primary].loadings().round(4).reset_index().rename(columns={"index": "feature"}),
         f"{a.results}/pca_loadings_{primary}.csv")
    report.append("### PCA 해석 (" + primary + ")\n```\n" + models[primary].report() + "\n```")

    # ------------------------------------------------------------------ Step 6
    hr("STEP 6  feature / PCA -> forward return (TRAIN only)")
    scans = []
    for s in symbols:
        sc = full_feature_scan(D[s], mask=splits[s][0])
        sc.insert(0, "symbol", s)
        scans.append(sc)
    scan = pd.concat(scans, ignore_index=True)
    save(scan, f"{a.results}/feature_analysis.csv")
    show = scan[scan.symbol == primary][["feature", "bucket", "n", "h4_mean", "h4_win", "h4_pf", "h4_t"]]
    print(show.round(4).to_string(index=False))
    report.append(f"### Feature -> 4봉 forward return ({primary}, train)\n```\n"
                  + show.round(4).to_string(index=False) + "\n```")

    # ------------------------------------------------------------------ Step 7
    hr("STEP 7  entry-condition screening (forward-return edge, TRAIN)")
    cond_rows = []
    for s in symbols:
        d = D[s]
        tests = {}
        for x in (20, 25, 30, 35, 40):
            tests[f"RSI<{x}"] = d["rsi"] < x
        for x in (0.0, 0.05, 0.10, 0.15, 0.20):
            tests[f"BB%B<{x}"] = d["BB_percent_b"] < x
        for x in (1.0, 1.1, 1.2, 1.3, 1.5, 2.0):
            tests[f"Vol>{x}"] = d["Volume_ratio"] > x
        tests["close>EMA20"] = d["close"] > d["ema_fast"]
        tests["close>EMA50"] = d["close"] > d["ema_slow"]
        tests["EMA20>EMA50"] = d["ema_fast"] > d["ema_slow"]
        for x in (15, 20, 25, 30):
            tests[f"ADX>{x}"] = d["ADX"] > x
        q = models[s].quantiles_
        for pc in ("PC1", "PC2", "PC3"):
            for qq in (0.25, 0.5, 0.75):
                tests[f"{pc}>q{int(qq*100)}"] = d[pc] > q.loc[qq, pc]
                tests[f"{pc}<q{int(qq*100)}"] = d[pc] < q.loc[qq, pc]
        for name, c in tests.items():
            for h in (4, 16):
                e = conditional_edge(d, c, mask=splits[s][0], horizon=h)
                if e.get("n", 0) > 50:
                    cond_rows.append({"symbol": s, "condition": name, "horizon": h, **e})
    cond = pd.DataFrame(cond_rows)
    save(cond, f"{a.results}/entry_conditions.csv")
    agg = (cond[cond.horizon == 4].groupby("condition")
           .agg(edge=("edge", "mean"), win=("win_rate", "mean"), pf=("profit_factor", "mean"),
                t=("t_stat", "mean"), cov=("coverage", "mean"), n=("n", "sum"))
           .sort_values("edge", ascending=False))
    print(agg.round(4).head(20).to_string())
    report.append("### 조건별 4봉 forward-return edge (전 심볼 평균, train)\n```\n"
                  + agg.round(4).head(20).to_string() + "\n```")

    # ---------------------------------------------------------- Step 7b/8 search
    hr("STEP 7b-8  entry parameter search on VALIDATION (primary symbol)")
    base_grid = GridParams(step_atr=1.0, max_entries=3, weights="equal")
    base_risk = RiskParams(leverage=a.leverage, tp_mode="pct", tp_value=0.007,
                           sl_mode="pct", sl_value=0.04, max_hold_bars=480)
    ecs = entry_candidates(
        rsi=(25, 30, 35, 40) if not a.quick else (30, 35),
        bbp=(0.05, 0.10, 0.15, 0.20) if not a.quick else (0.10, 0.20),
        vol=(None, 1.0, 1.2, 1.5) if not a.quick else (None, 1.2),
        trend=("none", "ema20>ema50", "close>ema50", "ema20>ema50 & close>ema20")
        if not a.quick else ("none", "ema20>ema50"),
        adx=(None, 20, 25) if not a.quick else (None,))
    combos = [(ep, base_grid, base_risk, f"entry_{i}") for i, ep in enumerate(ecs)]
    r_entry = search(D[primary], sl_val[primary], combos, cfg, models[primary].quantiles_,
                     symbol=primary, progress_every=100)
    r_entry["stage"] = "entry"
    top_entry = r_entry.sort_values("score", ascending=False).head(10)
    print(top_entry[["strategy", "e_rsi_max", "e_bbp_max", "e_vol_min", "e_trend", "e_adx_min",
                     "trades", "roi", "max_drawdown", "profit_factor", "score"]].round(4).to_string(index=False))
    best_entry_row = plateau_pick(r_entry, ["e_rsi_max", "e_bbp_max"], min_trades=a.min_trades)
    ep0, _, _ = params_from_row(best_entry_row)
    print(f"  plateau pick: {ep0}")

    # ------------------------------------------------------------------ Step 8b
    hr("STEP 8b  does a PCA filter help? (VALIDATION)")
    pca_variants = {"no_PCA": {}}
    for q1 in (0.25, 0.5):
        pca_variants[f"PC1>q{int(q1*100)}"] = {"pc1_min_q": q1}
    for q1 in (0.5, 0.75):
        pca_variants[f"PC1<q{int(q1*100)}"] = {"pc1_max_q": q1}
    for q2 in (0.25, 0.5, 0.75):
        pca_variants[f"PC2<q{int(q2*100)}"] = {"pc2_max_q": q2}
        pca_variants[f"PC2>q{int(q2*100)}"] = {"pc2_min_q": q2}
    for q3 in (0.25, 0.75):
        pca_variants[f"PC3<q{int(q3*100)}"] = {"pc3_max_q": q3}
        pca_variants[f"PC3>q{int(q3*100)}"] = {"pc3_min_q": q3}
    pca_variants["PC1>q25 & PC2<q75"] = {"pc1_min_q": 0.25, "pc2_max_q": 0.75}
    pca_variants["PC1>q50 & PC2<q50"] = {"pc1_min_q": 0.5, "pc2_max_q": 0.5}
    pv_combos = [(replace(ep0, **kw), base_grid, base_risk, name) for name, kw in pca_variants.items()]
    r_pca = search(D[primary], sl_val[primary], pv_combos, cfg, models[primary].quantiles_, symbol=primary)
    r_pca["stage"] = "pca_filter"
    print(r_pca[["strategy", "trades", "roi", "max_drawdown", "profit_factor", "win_rate", "score"]]
          .round(4).to_string(index=False))
    base_score = float(r_pca.loc[r_pca.strategy == "no_PCA", "score"].iloc[0])
    best_pca = r_pca.sort_values("score", ascending=False).iloc[0]
    pca_helps = bool(best_pca["strategy"] != "no_PCA" and best_pca["score"] > base_score + a.pca_margin)
    ep1 = params_from_row(best_pca)[0] if pca_helps else ep0
    print(f"  PCA filter {'IS' if pca_helps else 'is NOT'} kept "
          f"(best {best_pca['strategy']} score {best_pca['score']:.3f} vs no_PCA {base_score:.3f})")
    report.append("### PCA 필터 효과 (validation)\n```\n"
                  + r_pca[["strategy", "trades", "roi", "max_drawdown", "profit_factor", "score"]]
                  .round(4).to_string(index=False)
                  + f"\n-> PCA 채택: {pca_helps}\n```")

    # ------------------------------------------------------------------- Step 9
    hr("STEP 9  ATR grid parameters (VALIDATION)")
    gcs = grid_candidates(steps=(0.5, 0.75, 1.0, 1.25, 1.5) if a.quick else (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
                          counts=(1, 2, 3, 4, 5) if a.quick else (1, 2, 3, 4, 5, 6),
                          weights=("equal", "mild") if a.quick else ("equal", "mild", "aggressive"))
    r_grid = search(D[primary], sl_val[primary], [(ep1, g, base_risk, f"grid_{i}") for i, g in enumerate(gcs)],
                    cfg, models[primary].quantiles_, symbol=primary, progress_every=100)
    r_grid["stage"] = "grid"
    print(r_grid.sort_values("score", ascending=False)
          [["g_step_atr", "g_max_entries", "g_weights", "trades", "roi", "max_drawdown",
            "profit_factor", "avg_grid_entries", "score"]].round(4).head(12).to_string(index=False))
    gp1 = params_from_row(plateau_pick(r_grid, ["g_step_atr", "g_max_entries"], min_trades=a.min_trades))[1]
    print(f"  plateau pick: {gp1}")

    # ------------------------------------------------------------------ Step 10
    hr("STEP 10  TP / SL (VALIDATION)")
    rcs = exit_candidates(
        tp_pct=(0.003, 0.005, 0.007, 0.010, 0.015, 0.020),
        tp_atr=(0.5, 1.0, 1.5, 2.0) if not a.quick else (1.0,),
        sl_pct=(0.02, 0.03, 0.04, 0.05, 0.06, 0.08),
        sl_atr=(1.0, 1.5, 2.0, 2.5, 3.0) if not a.quick else (2.0,),
        include_no_sl=True, max_hold=(96, 480, 960) if not a.quick else (480,))
    rcs = [replace(r, leverage=a.leverage) for r in rcs]
    r_exit = search(D[primary], sl_val[primary], [(ep1, gp1, r, f"exit_{i}") for i, r in enumerate(rcs)],
                    cfg, models[primary].quantiles_, symbol=primary, progress_every=200)
    r_exit["stage"] = "exit"
    print(r_exit.sort_values("score", ascending=False)
          [["r_tp_mode", "r_tp_value", "r_sl_mode", "r_sl_value", "r_max_hold_bars", "trades",
            "win_rate", "roi", "max_drawdown", "profit_factor", "liquidations", "score"]]
          .round(4).head(12).to_string(index=False))
    no_sl = r_exit[r_exit.r_sl_mode == "none"]
    if len(no_sl):
        print("\n  손절 없음(no-SL) 전략 리스크:")
        print(no_sl[["r_tp_mode", "r_tp_value", "trades", "roi", "max_drawdown", "max_holding_hours",
                     "liquidations", "worst_trade"]].round(3).head(8).to_string(index=False))
        report.append("### 손절 없음(no-SL) 리스크 (validation)\n```\n"
                      + no_sl[["r_tp_value", "roi", "max_drawdown", "max_holding_hours",
                               "liquidations", "worst_trade"]].round(3).head(8).to_string(index=False) + "\n```")
    rp1 = params_from_row(plateau_pick(r_exit, ["r_tp_value", "r_sl_value"], min_trades=a.min_trades))[2]
    rp1 = replace(rp1, leverage=a.leverage)
    print(f"  plateau pick: {rp1}")

    param_test = pd.concat([r_entry, r_pca, r_grid, r_exit], ignore_index=True)
    save(param_test, f"{a.results}/parameter_test.csv")

    # ------------------------------------------------------- Step 10b filter stack
    hr("STEP 10b  incremental filter study A->F (VALIDATION, all symbols)")
    stack_rows = []
    for name, ep in filter_stack(ep1).items():
        aggr, per = evaluate_multi(D, sl_val, ep, gp1, rp1, cfg, {s: models[s].quantiles_ for s in symbols}, name)
        stack_rows.append(aggr)
    stack = pd.DataFrame(stack_rows)
    print(stack.round(4).to_string(index=False))
    save(stack, f"{a.results}/filter_stack.csv")
    report.append("### 필터 단계별 효과 (validation, 전 심볼)\n```\n" + stack.round(4).to_string(index=False) + "\n```")

    # ------------------------------------------------------- Step 11-12 + sens.
    hr("STEP 11-12  chosen strategy on TRAIN / VALIDATION / TEST")
    final_rows = []
    for s in symbols:
        for name, sl in (("train", sl_train[s]), ("validation", sl_val[s]), ("test(OOS)", sl_test[s])):
            st, res = evaluate(D[s], sl, ep1, gp1, rp1, cfg, models[s].quantiles_, name, symbol=s, keep=True)
            st["symbol"], st["split"] = s, name
            final_rows.append(st)
            if s == primary and name == "test(OOS)":
                primary_test = res
    final = pd.DataFrame(final_rows)
    cols = ["symbol", "split", "trades", "win_rate", "roi", "cagr", "max_drawdown", "sharpe",
            "profit_factor", "avg_trade", "max_consecutive_losses", "avg_holding_hours", "liquidations"]
    print(final[cols].round(4).to_string(index=False))
    save(final, f"{a.results}/backtest_results.csv")
    report.append("### 최종 전략 구간별 성과\n```\n" + final[cols].round(4).to_string(index=False) + "\n```")

    sens = sensitivity_table(param_test, ["e_rsi_max", "e_bbp_max", "e_vol_min", "e_trend",
                                          "g_step_atr", "g_max_entries", "g_weights",
                                          "r_tp_value", "r_sl_value", "r_sl_mode"])
    save(sens, f"{a.results}/parameter_sensitivity.csv")
    print("\n  parameter sensitivity (median score by value):")
    print(sens.round(3).to_string(index=False))
    report.append("### 파라미터 민감도\n```\n" + sens.round(3).to_string(index=False) + "\n```")

    # ------------------------------------------------------------------ Step 13
    hr("STEP 13  walk-forward (PCA + parameters re-fit per fold)")
    wf_combos = _wf_combo_set(ep1, gp1, rp1, a)
    n_primary = len(D[primary])
    w = WFConfig(train_bars=int(n_primary * 0.4), val_bars=int(n_primary * 0.15),
                 test_bars=int(n_primary * 0.15), anchored=True, n_components=a.n_components)
    wf_all, wf_trades_all, wf_summaries = [], [], []
    wf_equity = {}
    for s in symbols:
        print(f"  [{s}]")
        fdf, tdf, eq, chosen = run_walk_forward(D[s], wf_combos, cfg, w, symbol=s,
                                                min_trades=max(5, a.min_trades // 3))
        if len(fdf) == 0:
            print("    (not enough bars for a fold)")
            continue
        wf_all.append(fdf)
        if len(tdf):
            tdf["symbol"] = s
            wf_trades_all.append(tdf)
        wf_equity[s] = eq
        os_ = oos_summary(fdf, tdf, eq, cfg, f"{s}_WF_OOS")
        os_["symbol"] = s
        wf_summaries.append(os_)
    wf = pd.concat(wf_all, ignore_index=True) if wf_all else pd.DataFrame()
    wf_trades = pd.concat(wf_trades_all, ignore_index=True) if wf_trades_all else pd.DataFrame()
    save(wf, f"{a.results}/walkforward.csv")
    wfs = pd.DataFrame(wf_summaries)
    if len(wfs):
        c = ["symbol", "folds", "folds_profitable", "trades", "roi", "max_drawdown", "sharpe",
             "profit_factor", "win_rate", "worst_fold_roi", "val_test_corr", "liquidations"]
        c = [x for x in c if x in wfs.columns]
        print(wfs[c].round(4).to_string(index=False))
        save(wfs, f"{a.results}/walkforward_summary.csv")
        report.append("### Walk-forward OOS 요약\n```\n" + wfs[c].round(4).to_string(index=False) + "\n```")

    # ----------------------------------------------------------------- Step 13b
    hr(f"STEP 13b  universe-wide validation of the chosen config "
       f"({len(all_symbols)} symbols, test slice only)")
    uval = validate_universe(all_symbols, scfg, ep1, gp1, rp1, jobs=a.jobs, split="test")
    save(uval, f"{a.results}/universe_validation.csv")
    ustats = universe_verdict_stats(uval)
    uok = uval[uval.status == "ok"] if "status" in uval else uval
    if ustats:
        print("  " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in ustats.items()))
        if len(uok):
            best = uok.sort_values("roi", ascending=False).head(10)
            worst = uok.sort_values("roi").head(10)
            c = ["symbol", "trades", "roi", "max_drawdown", "profit_factor", "win_rate", "liquidations"]
            print("\n  최고 10:"); print(best[c].round(4).to_string(index=False))
            print("\n  최악 10:"); print(worst[c].round(4).to_string(index=False))
            report.append("### 유니버스 전체 OOS 검증 (선택에 관여하지 않은 심볼 포함)\n```\n"
                          + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                      for k, v in ustats.items())
                          + "\n\n최악 10:\n" + worst[c].round(4).to_string(index=False) + "\n```")

    # ------------------------------------------------------------------ Step 14
    hr("STEP 14  Monte Carlo + cost stress (OOS trades)")
    mc_source = wf_trades if len(wf_trades) else pd.concat(
        [primary_test.trades], ignore_index=True) if primary_test.n_trades else pd.DataFrame()
    mc_rows = []
    for mode in ("shuffle", "bootstrap"):
        for extra in (0.0, 0.0005, 0.001):
            sim = simulate(mc_source, cfg.initial_capital, a.mc_sims, mode=mode, extra_cost=extra)
            st = summarize_mc(sim, cfg.initial_capital)
            if st:
                mc_rows.append({"mode": mode, "extra_cost_per_trade": extra, **st})
    mc = pd.DataFrame(mc_rows)
    if len(mc):
        print(mc.round(4).to_string(index=False))
        print(" ", max_consecutive_loss_distribution(mc_source, a.mc_sims))
        report.append("### Monte Carlo (OOS 거래 재표본)\n```\n" + mc.round(4).to_string(index=False) + "\n```")
    save(mc, f"{a.results}/monte_carlo.csv")

    stress = cost_stress(D[primary], sl_test[primary], ep1, gp1, rp1, cfg, models[primary].quantiles_,
                         fee_mults=(1.0, 2.0), slip_bps=(2.0, 5.0, 10.0), delays=(1, 2, 3),
                         symbol=primary)
    print("\n  cost / latency stress (test slice, primary):")
    print(stress.round(4).to_string(index=False))
    save(stress, f"{a.results}/cost_stress.csv")
    report.append("### 비용/지연 스트레스 (test)\n```\n" + stress.round(4).to_string(index=False) + "\n```")

    # ------------------------------------------------------------ Step 14b regime
    hr("STEP 14b  market-regime attribution")
    reg_rows = []
    for s in symbols:
        t = wf_trades[wf_trades.symbol == s] if len(wf_trades) and "symbol" in wf_trades else pd.DataFrame()
        if not len(t):
            _, res = evaluate(D[s], sl_test[s], ep1, gp1, rp1, cfg, models[s].quantiles_, "t", symbol=s, keep=True)
            t = res.trades
        rt = regime_table(D[s], t)
        if len(rt):
            rt.insert(0, "symbol", s)
            reg_rows.append(rt)
    reg = pd.concat(reg_rows, ignore_index=True) if reg_rows else pd.DataFrame()
    if len(reg):
        allreg = (reg.groupby("regime").agg(trades=("trades", "sum"), win=("win_rate", "mean"),
                                            avg_ret=("avg_return", "mean"),
                                            pf=("profit_factor", "median")).round(4))
        print(allreg.to_string())
        report.append("### 시장 국면별 성과 (OOS)\n```\n" + allreg.to_string() + "\n```")
    save(reg, f"{a.results}/regime_analysis.csv")
    print("\n  PCA vs regime separation (primary):")
    print(pca_state_table(D[primary]).to_string())

    # ---------------------------------------------------------------- trades csv
    tr_out = wf_trades if len(wf_trades) else (primary_test.trades if primary_test.n_trades else pd.DataFrame())
    save(tr_out, f"{a.results}/trades.csv", "(OOS trades)")

    # -------------------------------------------------------------- Step 15 charts
    hr("STEP 15  charts")
    curves = {k: v for k, v in wf_equity.items() if v is not None and len(v)}
    if not curves and primary_test.n_trades:
        curves = {f"{primary} test": primary_test.equity}
    if curves:
        charts.equity_curve(curves, f"{a.charts}/equity.png", "Out-of-sample equity (walk-forward)")
        main_eq = curves.get(primary, list(curves.values())[0])
        charts.drawdown_curve(main_eq, f"{a.charts}/drawdown.png")
        charts.monthly_returns_chart(main_eq, f"{a.charts}/monthly_returns.png")
    if len(tr_out):
        charts.trade_distribution(tr_out, f"{a.charts}/trade_distribution.png")
    charts.pca_analysis(D[primary], f"{a.charts}/pca_analysis.png")
    charts.pca_scatter(D[primary], f"{a.charts}/pca_scatter.png")
    charts.sensitivity_chart(sens, f"{a.charts}/parameter_sensitivity.png",
                             ["e_rsi_max", "e_bbp_max", "g_step_atr", "r_tp_value", "r_sl_value"])
    print(f"  -> {a.charts}/*.png")

    # ------------------------------------------------------------ Step 16 verdict
    hr("STEP 16  final selection + verdict")
    oos = final[final.split == "test(OOS)"]
    wf_ok = wfs if len(wfs) else pd.DataFrame()
    checks = {
        "OOS 수익 (test split, 심볼 중앙값 > 0)": float(oos["roi"].median()) > 0,
        "OOS MDD <= 25%": float(abs(oos["max_drawdown"]).max()) <= 0.25,
        "OOS Profit Factor >= 1.2 (중앙값)": float(oos["profit_factor"].replace(np.inf, 5).median()) >= 1.2,
        "거래 표본 >= 100 (전체 OOS)": int(oos["trades"].sum()) >= 100,
        "심볼 재현성 (과반 심볼 수익)": int((oos["roi"] > 0).sum()) > len(symbols) / 2,
        "Walk-forward 과반 fold 수익": (float(wf_ok["folds_profitable"].sum()) / max(float(wf_ok["folds"].sum()), 1) > 0.5)
        if len(wf_ok) else False,
        "청산 0건": int(final["liquidations"].sum()) == 0,
        "비용 2배에서도 수익": bool((stress[stress.fee_mult == 2.0]["roi"] > 0).mean() > 0.5) if len(stress) else False,
        "MC 50% 손실 확률 < 5%": (float(mc["prob_ruin_50pct_loss"].max()) < 0.05) if len(mc) else False,
        "유니버스 과반 심볼 OOS 수익": (ustats.get("positive_fraction", 0.0) > 0.5) if ustats else False,
        "유니버스 청산 심볼 0개": (ustats.get("symbols_with_liquidation", 1) == 0) if ustats else False,
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    passed = sum(checks.values())
    verdict = ("실전 적용 가능 (소액부터)" if passed >= len(checks) - 1 else
               "추가 연구 필요" if passed >= len(checks) * 0.55 else "폐기")
    if set(sources.values()) == {"synthetic"}:
        verdict += "  [합성 데이터 실행 - 판정 무효]"
    print(f"\n  통과 {passed}/{len(checks)}  ->  결론: {verdict}")

    conf = {
        "meta": {"generated_by": "src/run_research.py", "symbols": symbols,
                 "universe_source": usrc, "universe_size": len(all_symbols),
                 "universe_stats_oos": ustats,
                 "interval_minutes": int(a.interval), "data_source": sources,
                 "primary_symbol": primary, "verdict": verdict,
                 "checks_passed": f"{passed}/{len(checks)}"},
        "execution": cfg.to_dict(),
        "entry": ep1.to_dict(),
        "grid": gp1.to_dict(),
        "risk": rp1.to_dict(),
        "pca": {"n_components": a.n_components, "fit_policy": "train slice only; refit on a "
                "rolling window in live trading (see README)",
                "features": models[primary].features,
                "explained_variance_ratio": [float(x) for x in models[primary].pca.explained_variance_ratio_],
                "train_quantiles": json.loads(models[primary].quantiles_.round(4).to_json()),
                "loadings": json.loads(models[primary].loadings().round(4).to_json())},
        "performance": {
            "validation_primary": _pick(final, primary, "validation"),
            "test_oos_primary": _pick(final, primary, "test(OOS)"),
            "test_oos_median_roi": float(oos["roi"].median()),
            "walk_forward": (wfs[["symbol", "roi", "max_drawdown", "profit_factor", "trades"]]
                             .round(4).to_dict("records") if len(wfs) else []),
            "monte_carlo": (mc.round(4).to_dict("records")[:3] if len(mc) else []),
        },
        "checks": {k: bool(v) for k, v in checks.items()},
    }
    with open(f"{a.out_config}", "w") as f:
        yaml.safe_dump(_yamlable(conf), f, allow_unicode=True, sort_keys=False)
    print(f"  -> {a.out_config}")

    report.insert(0, f"# 연구 리포트\n\n- 심볼: {', '.join(symbols)}\n- 타임프레임: {a.interval}m\n"
                     f"- 데이터: {sources}\n- 결론: **{verdict}** (체크 {passed}/{len(checks)})\n")
    report.append("### 최종 파라미터\n```yaml\n"
                  + yaml.safe_dump(_yamlable({"entry": ep1.to_dict(), "grid": gp1.to_dict(),
                                    "risk": rp1.to_dict()}), allow_unicode=True, sort_keys=False) + "```")
    report.append("### 체크리스트\n" + "\n".join(f"- [{'x' if v else ' '}] {k}" for k, v in checks.items()))

    summary = _final_summary(symbols, a, ep1, gp1, rp1, models[primary], final, wfs, mc, stress,
                             pca_helps, verdict, primary, all_symbols, ustats)
    print(summary)
    report.append("### 최종 전략 요약\n```\n" + summary + "\n```")
    with open(f"{a.results}/REPORT.md", "w") as f:
        f.write("\n\n".join(report))
    print(f"  -> {a.results}/REPORT.md")
    return conf


def _final_summary(symbols, a, ep, gp, rp, model, final, wfs, mc, stress, pca_helps,
                   verdict, primary, all_symbols=None, ustats=None) -> str:
    """The human-readable conclusion (Step 24)."""
    oos = final[final.split == "test(OOS)"]
    val = final[final.split == "validation"]
    L = []
    L.append("최종 전략")
    L.append("")
    L.append(f"유니버스:    {len(all_symbols or symbols)}개 심볼 스캔")
    L.append(f"연구 바스켓: {', '.join(symbols)}")
    L.append(f"Timeframe:   {a.interval}m,  LONG only,  leverage {rp.leverage}x")
    L.append(f"PCA:         {a.n_components} components "
             f"(explained var {model.pca.explained_variance_ratio_.sum():.1%}), "
             f"{'필터 채택' if pca_helps else '필터 미채택 (성능 개선 없음)'}")
    if pca_helps:
        for pc in ("PC1", "PC2", "PC3"):
            lo, hi = getattr(ep, f"{pc.lower()}_min_q"), getattr(ep, f"{pc.lower()}_max_q")
            if lo is not None:
                L.append(f"LONG filter: {pc} > train q{lo:.2f} (= {model.q(pc, lo):+.3f})")
            if hi is not None:
                L.append(f"LONG filter: {pc} < train q{hi:.2f} (= {model.q(pc, hi):+.3f})")
    L.append(f"Entry:       RSI <= {ep.rsi_max},  BB %B <= {ep.bbp_max},  "
             f"Volume ratio >= {ep.vol_min}")
    L.append(f"Trend:       {ep.trend}" + (f",  ADX >= {ep.adx_min}" if ep.adx_min else ""))
    offs = ", ".join(f"-{o:.2f}ATR" for o in gp.offsets()) or "(no DCA)"
    L.append(f"Grid:        step {gp.step_atr} ATR -> {offs}")
    L.append(f"Max grid:    {gp.max_entries} entries, size weights '{gp.weights}', "
             f"max notional = equity x {rp.leverage * rp.position_pct:.1f}")
    tp_txt = f"+{rp.tp_value * 100:.3g}%" if rp.tp_mode == "pct" else f"+ {rp.tp_value:g} x ATR"
    sl_txt = ("없음" if rp.sl_mode == "none"
              else f"-{rp.sl_value * 100:.3g}%" if rp.sl_mode == "pct"
              else f"-{rp.sl_value:g} x ATR")
    L.append(f"TP:          average entry {tp_txt}")
    L.append(f"SL:          {sl_txt},  max hold {rp.max_hold_bars} bars "
             f"({rp.max_hold_bars * 0.25:.0f}h)")
    L.append("")
    L.append(f"Validation:  ROI {val['roi'].median():+.2%} (중앙값)  MDD {val['max_drawdown'].min():.2%}  "
             f"WinRate {val['win_rate'].mean():.1%}  PF {val['profit_factor'].replace(np.inf, 5).median():.2f}")
    L.append(f"Out-of-Sample: ROI {oos['roi'].median():+.2%} (중앙값)  MDD {oos['max_drawdown'].min():.2%}  "
             f"거래 {int(oos['trades'].sum())}건  수익 심볼 {int((oos['roi'] > 0).sum())}/{len(oos)}")
    if len(wfs):
        L.append(f"Walk-forward: fold {int(wfs['folds_profitable'].sum())}/{int(wfs['folds'].sum())} 수익, "
                 f"OOS ROI 중앙값 {wfs['roi'].median():+.2%}, MDD {wfs['max_drawdown'].min():.2%}, "
                 f"val-test 상관 {wfs['val_test_corr'].mean():.2f}")
    if ustats:
        L.append(f"유니버스 OOS: {ustats['symbols_tested']}개 심볼, 수익 심볼 비율 "
                 f"{ustats['positive_fraction']:.1%}, ROI 중앙값 {ustats['roi_median']:+.2%}, "
                 f"MDD 중앙값 {ustats['mdd_median']:.2%}, 거래 {ustats['total_trades']}건")
    if len(mc):
        m0 = mc.iloc[0]
        L.append(f"Monte Carlo: 수익 확률 {m0['prob_profit']:.1%}, MDD 중앙값 {m0['mdd_median']:.2%}, "
                 f"자본 -50% 확률 {m0['prob_ruin_50pct_loss']:.1%}")
    if len(stress):
        s2 = stress[stress.fee_mult == 2.0]
        L.append(f"비용 2배 스트레스: 수익 케이스 {(s2['roi'] > 0).mean():.0%}, ROI 중앙값 {s2['roi'].median():+.2%}")
    L.append("")
    L.append(f"결론: {verdict}")
    return "\n".join(L)


def _yamlable(o):
    """numpy scalars / NaN -> plain python so yaml.safe_dump can write the config."""
    if isinstance(o, dict):
        return {str(k): _yamlable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_yamlable(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    return o


def _pick(final: pd.DataFrame, symbol: str, split: str) -> dict:
    row = final[(final.symbol == symbol) & (final.split == split)]
    if not len(row):
        return {}
    keep = ["trades", "win_rate", "roi", "cagr", "max_drawdown", "sharpe", "profit_factor",
            "avg_trade", "max_consecutive_losses", "avg_holding_hours", "liquidations"]
    return {k: (float(row.iloc[0][k]) if not isinstance(row.iloc[0][k], str) else row.iloc[0][k])
            for k in keep if k in row.columns}


def _wf_combo_set(ep, gp, rp, a) -> list[tuple]:
    """Small, deliberately coarse candidate set re-optimised inside every fold."""
    out = []
    for rsi in (25, 30, 35):
        for bbp in (0.05, 0.10, 0.20):
            e = replace(ep, rsi_max=rsi, bbp_max=bbp)
            for step in (0.75, 1.25):
                for cnt in (2, 4):
                    g = replace(gp, step_atr=step, max_entries=cnt)
                    for tp in (0.005, 0.010):
                        r = replace(rp, tp_value=tp, tp_mode="pct")
                        out.append((e, g, r, f"wf_{rsi}_{bbp}_{step}_{cnt}_{tp}"))
    return out


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="*", default=None,
                   help="explicit symbol list; overrides --universe")
    p.add_argument("--universe", default="local",
                   help="'all' (every live USDT perp passing filters), 'topN' (e.g. top100), "
                        "'local' (whatever is in data/)")
    p.add_argument("--min-turnover", type=float, default=1e7,
                   help="24h turnover floor for --universe all/topN")
    p.add_argument("--min-listed-days", type=int, default=365)
    p.add_argument("--min-daily-turnover", type=float, default=1e7,
                   help="point-in-time liquidity floor (quote/day, 30d rolling median). "
                        "History before the symbol first clears it is dropped. 0 disables.")
    p.add_argument("--min-liquid-frac", type=float, default=0.5,
                   help="drop symbols that clear the liquidity floor on less than this "
                        "share of their train+validation bars")
    p.add_argument("--exclude-symbol-types", nargs="*",
                   default=list(uni.NON_CRYPTO_SYMBOL_TYPES),
                   help="Bybit symbolType values to drop (tokenised stocks/ETFs/commodities). "
                        "Pass with no values to keep everything.")
    p.add_argument("--min-bars", type=int, default=20_000,
                   help="skip symbols with less history than this (15m bars, after the "
                        "liquidity cut)")
    p.add_argument("--basket-size", type=int, default=8,
                   help="how many symbols go into the deep parameter research")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--skip-scan", action="store_true")
    p.add_argument("--synthetic-symbols", type=int, default=12)
    p.add_argument("--interval", default="15")
    p.add_argument("--days", type=int, default=1095)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--results", default="results")
    p.add_argument("--charts", default="charts")
    p.add_argument("--out-config", default="strategy_config.yaml")
    p.add_argument("--capital", type=float, default=10_000.0)
    p.add_argument("--leverage", type=float, default=2.0)
    p.add_argument("--fee-taker", type=float, default=0.00055)
    p.add_argument("--fee-maker", type=float, default=0.0002)
    p.add_argument("--slippage-bp", type=float, default=2.0)
    p.add_argument("--funding", type=float, default=0.0001)
    p.add_argument("--no-funding", action="store_true")
    p.add_argument("--n-components", type=int, default=3)
    p.add_argument("--train-frac", type=float, default=0.5)
    p.add_argument("--val-frac", type=float, default=0.25)
    p.add_argument("--min-trades", type=int, default=30)
    p.add_argument("--pca-margin", type=float, default=0.05,
                   help="PCA filter is kept only if it beats no-PCA by this score margin")
    p.add_argument("--mc-sims", type=int, default=2000)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--allow-synthetic", action="store_true",
                   help="fall back to synthetic bars when Bybit is unreachable (pipeline test only)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse())
