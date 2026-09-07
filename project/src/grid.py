"""ATR grid (DCA) geometry: where the add-on entries sit and how big each one is."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

WEIGHT_SCHEMES = {
    "equal": [1, 1, 1, 1, 1, 1],
    "mild": [1, 1, 1.25, 1.5, 1.75, 2.0],
    "aggressive": [1, 1.25, 1.5, 2.0, 2.5, 3.0],
    "front": [1.5, 1.25, 1, 1, 1, 1],
}


@dataclass
class GridParams:
    step_atr: float = 0.7          # spacing between add-ons, in ATR at first entry
    max_entries: int = 4           # 1 = no DCA
    weights: str = "equal"
    geometric: bool = False        # True -> spacing grows (1x, 2x, 3x ... step)

    def to_dict(self) -> dict:
        return asdict(self)

    def offsets(self) -> np.ndarray:
        """ATR distances below the FIRST entry price for add-on k = 1..max_entries-1."""
        k = np.arange(1, self.max_entries)
        return (k * (k + 1) / 2.0 * self.step_atr) if self.geometric else k * self.step_atr

    def size_weights(self) -> np.ndarray:
        w = np.asarray(WEIGHT_SCHEMES[self.weights][: self.max_entries], float)
        if len(w) < self.max_entries:
            w = np.concatenate([w, np.full(self.max_entries - len(w), w[-1])])
        return w / w.sum()
