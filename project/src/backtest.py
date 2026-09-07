"""Event-driven LONG-only backtester.

Timing contract (no look-ahead):
  * a signal is evaluated on the CLOSE of bar t using only bars <= t;
  * the first entry executes at the OPEN of bar t + entry_delay_bars (default 1);
  * grid add-ons are resting limit buys, TP is a resting limit sell, SL is a stop;
  * within one bar the order is funding -> liquidation -> SL -> TP(pre-bar average)
    -> grid fills -> max-hold. That ordering is deliberately pessimistic.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd

from .grid import GridParams
from .risk import RiskParams, planned_notional, tp_price, sl_price, liquidation_price

FUNDING_MS = 8 * 3600 * 1000
LOG_COLS = ["rsi", "BB_percent_b", "Volume_ratio", "atr", "ema_fast", "ema_slow", "ADX",
            "PC1", "PC2", "PC3"]


@dataclass
class ExecConfig:
    initial_capital: float = 10_000.0
    fee_taker: float = 0.00055        # Bybit USDT perp taker
    fee_maker: float = 0.0002         # Bybit USDT perp maker
    grid_uses_maker: bool = False     # False = charge taker everywhere (conservative)
    slippage_bp: float = 2.0          # applied to market fills (entry, SL, max-hold, EOD)
    funding_rate_8h: float = 0.0001   # fallback when no funding history is supplied
    use_funding: bool = True
    entry_delay_bars: int = 1
    bars_per_year: int = 35_040       # 15m bars
    allow_reentry_same_bar: bool = False
    exit_from_next_bar: bool = True   # no TP/SL/grid on the entry bar (pessimistic)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    config: dict = field(default_factory=dict)
    ruin: bool = False
    symbol: str = ""

    @property
    def n_trades(self) -> int:
        return len(self.trades)


def run_backtest(d: pd.DataFrame, signal: np.ndarray, grid: GridParams, risk: RiskParams,
                 cfg: ExecConfig, funding: pd.DataFrame | None = None,
                 symbol: str = "") -> BacktestResult:
    n = len(d)
    o = d["open"].to_numpy(float); h = d["high"].to_numpy(float)
    lo = d["low"].to_numpy(float); c = d["close"].to_numpy(float)
    atr = d["atr"].to_numpy(float); ts = d["timestamp"].to_numpy("int64")
    logv = {k: (d[k].to_numpy(float) if k in d.columns else np.full(n, np.nan)) for k in LOG_COLS}

    fr = np.full(n, cfg.funding_rate_8h if cfg.use_funding else 0.0)
    if funding is not None and len(funding):
        fr = pd.Series(ts).map(dict(zip(funding["timestamp"], funding["funding_rate"]))) \
               .astype(float).to_numpy()
        fr = np.where(np.isnan(fr), cfg.funding_rate_8h, fr)
    is_funding_bar = (ts % FUNDING_MS == 0) & cfg.use_funding

    slip = cfg.slippage_bp / 10_000.0
    offsets = grid.offsets()
    weights = grid.size_weights()
    grid_fee = cfg.fee_maker if cfg.grid_uses_maker else cfg.fee_taker

    cash = cfg.initial_capital
    equity = np.empty(n); equity[:] = np.nan
    in_pos = False
    ruin = False
    last_exit_bar = -10 ** 9
    trades: list[dict] = []

    # live position state
    qty = avg = notional = margin = atr_e = tp = equity_at_entry = 0.0
    sl: float | None = None
    filled = 0
    fills: list[float] = []
    entry_bar = sig_bar = 0
    entry_fee_paid = funding_paid = 0.0
    first_price = 0.0
    planned = 0.0

    def close_position(i: int, price: float, reason: str):
        nonlocal cash, in_pos, qty, avg, notional, margin, filled, fills, last_exit_bar
        gross = qty * (price - avg)
        exit_fee = qty * price * cfg.fee_taker
        cash += gross - exit_fee
        profit = gross - exit_fee - entry_fee_paid - funding_paid
        rec = {"symbol": symbol,
               "timestamp": pd.to_datetime(ts[sig_bar], unit="ms", utc=True),
               "entry_time": pd.to_datetime(ts[entry_bar], unit="ms", utc=True),
               "exit_time": pd.to_datetime(ts[i], unit="ms", utc=True)}
        rec.update({"PCA1": logv["PC1"][sig_bar], "PCA2": logv["PC2"][sig_bar],
                    "PCA3": logv["PC3"][sig_bar], "RSI": logv["rsi"][sig_bar],
                    "BB_percent_b": logv["BB_percent_b"][sig_bar],
                    "Volume_ratio": logv["Volume_ratio"][sig_bar],
                    "ATR": logv["atr"][sig_bar], "EMA20": logv["ema_fast"][sig_bar],
                    "EMA50": logv["ema_slow"][sig_bar], "ADX": logv["ADX"][sig_bar]})
        rec.update({"entry_price": first_price, "entry_size": planned * weights[0] / first_price,
                    "grid_entry_count": filled, "grid_entry_prices": "|".join(f"{p:.6f}" for p in fills),
                    "average_entry": avg, "TP_price": tp, "SL_price": (sl if sl is not None else np.nan),
                    "exit_price": price, "profit": profit,
                    "profit_pct": profit / (avg * qty) if qty > 0 else 0.0,
                    "equity_at_entry": equity_at_entry,
                    "trade_return": profit / max(equity_at_entry, 1e-9),
                    "return_on_equity": profit / max(cfg.initial_capital, 1e-9),
                    "notional": avg * qty, "fees": entry_fee_paid + exit_fee,
                    "funding": funding_paid,
                    "holding_bars": i - entry_bar,
                    "holding_hours": (i - entry_bar) * 0.25,
                    "exit_reason": reason})
        trades.append(rec)
        in_pos = False
        qty = avg = notional = margin = 0.0
        filled = 0
        fills = []
        last_exit_bar = i

    delay = cfg.entry_delay_bars
    for i in range(n):
        # ---------------- entry ----------------
        if not in_pos and not ruin:
            s = i - delay
            gap = 0 if cfg.allow_reentry_same_bar else risk.cooldown_bars
            if s >= 0 and signal[s] and (i - last_exit_bar) > gap and cash > 0 \
                    and atr[s] > 0 and np.isfinite(o[i]) and o[i] > 0:
                sig_bar, entry_bar = s, i
                equity_at_entry = cash
                planned = planned_notional(cash, risk)
                first_price = o[i] * (1 + slip)
                notional = planned * weights[0]
                qty = notional / first_price
                avg = first_price
                entry_fee_paid = notional * cfg.fee_taker
                funding_paid = 0.0
                cash -= entry_fee_paid
                margin = notional / risk.leverage
                atr_e = atr[s]
                filled, fills = 1, [first_price]
                tp = tp_price(avg, atr_e, risk)
                sl = sl_price(avg, atr_e, risk)
                in_pos = True

        # ---------------- manage open position ----------------
        if in_pos and not (cfg.exit_from_next_bar and i == entry_bar):
            if is_funding_bar[i]:
                f = qty * c[i] * fr[i]
                cash -= f
                funding_paid += f

            liq = liquidation_price(avg, qty, margin, risk)
            if lo[i] <= liq and liq > 0:
                close_position(i, liq, "liquidation")
            elif sl is not None and lo[i] <= sl:
                close_position(i, sl * (1 - slip), "sl")
            elif h[i] >= tp:
                close_position(i, tp, "tp")
            else:
                # grid add-ons (resting limit buys below the first entry)
                while filled < grid.max_entries:
                    lvl = first_price - offsets[filled - 1] * atr_e
                    if lo[i] > lvl or lvl <= 0:
                        break
                    add_n = planned * weights[filled]
                    add_q = add_n / lvl
                    fee = add_n * grid_fee
                    cash -= fee
                    entry_fee_paid += fee
                    avg = (avg * qty + lvl * add_q) / (qty + add_q)
                    qty += add_q
                    notional += add_n
                    margin = notional / risk.leverage
                    fills.append(lvl)
                    filled += 1
                    tp = tp_price(avg, atr_e, risk)
                    sl = sl_price(avg, atr_e, risk)
                if in_pos and (i - entry_bar) >= risk.max_hold_bars:
                    close_position(i, c[i] * (1 - slip), "max_hold")

        eq = cash + (qty * (c[i] - avg) if in_pos else 0.0)
        equity[i] = eq
        if eq <= 0 and not ruin:
            if in_pos:
                close_position(i, c[i] * (1 - slip), "ruin")
            ruin = True
            equity[i] = max(cash, 0.0)

    if in_pos:
        close_position(n - 1, c[n - 1] * (1 - slip), "end_of_data")
        equity[n - 1] = cash

    tdf = pd.DataFrame(trades)
    eq = pd.Series(equity, index=d["datetime"].to_numpy(), name="equity")
    return BacktestResult(tdf, eq, {"grid": grid.to_dict(), "risk": risk.to_dict(),
                                    "exec": cfg.to_dict()}, ruin, symbol)
