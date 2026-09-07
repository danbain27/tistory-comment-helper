"""Position sizing, exposure caps and (approximate) liquidation modelling."""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class RiskParams:
    leverage: float = 2.0
    position_pct: float = 1.0        # fraction of (equity * leverage) committed when ALL
                                     # grid legs fill -> hard cap on total notional
    tp_mode: str = "pct"             # 'pct' | 'atr'
    tp_value: float = 0.007          # 0.7% above average entry, or X * ATR
    sl_mode: str = "pct"             # 'pct' | 'atr' | 'none'
    sl_value: float = 0.04           # 4% below average entry, or X * ATR
    max_hold_bars: int = 480         # 480 * 15m = 5 days
    cooldown_bars: int = 1
    maint_margin_rate: float = 0.005  # Bybit-ish maintenance margin for liquidation calc

    def to_dict(self) -> dict:
        return asdict(self)


def planned_notional(equity: float, r: RiskParams) -> float:
    """Total notional if every grid leg fills. Caps leverage at r.leverage."""
    return max(equity, 0.0) * r.leverage * r.position_pct


def tp_price(avg: float, atr_at_entry: float, r: RiskParams) -> float:
    return avg * (1 + r.tp_value) if r.tp_mode == "pct" else avg + r.tp_value * atr_at_entry


def sl_price(avg: float, atr_at_entry: float, r: RiskParams) -> float | None:
    if r.sl_mode == "none":
        return None
    return avg * (1 - r.sl_value) if r.sl_mode == "pct" else avg - r.sl_value * atr_at_entry


def liquidation_price(avg: float, qty: float, margin: float, r: RiskParams) -> float:
    """Isolated-margin LONG liquidation approximation:
    loss = qty * (avg - P); liquidated when loss >= margin - maint_margin * notional."""
    if qty <= 0:
        return 0.0
    buffer = margin - r.maint_margin_rate * qty * avg
    return max(avg - buffer / qty, 0.0)
