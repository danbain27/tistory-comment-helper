"""PCA market-state model. Scaler and PCA are fit on the TRAIN slice only and then
applied (transform) to validation/test. Never fit on the full sample."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .features import FEATURES

THEMES = {
    "trend/momentum": ["EMA20_distance", "EMA50_distance", "Return_16", "Return_4"],
    "volatility": ["ATR_percent", "BB_width"],
    "trend strength": ["ADX"],
    "volume/participation": ["Volume_ratio"],
    "short-term over-extension": ["RSI_norm", "BB_percent_b", "Return_1"],
}


class PCAModel:
    """Fit-on-train PCA with deterministic component signs and train-based quantiles."""

    def __init__(self, n_components: int = 3, features: list[str] | None = None):
        self.n = n_components
        self.features = list(features or FEATURES)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, svd_solver="full")
        self.signs_: np.ndarray | None = None
        self.quantiles_: pd.DataFrame | None = None
        self.fit_range_: tuple[str, str] | None = None

    # ---------------- fit / transform ----------------
    def fit(self, df: pd.DataFrame, mask=None) -> "PCAModel":
        sub = df[mask] if mask is not None else df
        sub = sub[sub["feat_ok"]]
        X = sub[self.features].to_numpy(float)
        self.scaler.fit(X)
        self.pca.fit(self.scaler.transform(X))
        # Deterministic sign: the largest-|loading| feature of each PC gets a + sign.
        comp = self.pca.components_
        self.signs_ = np.sign(comp[np.arange(self.n), np.abs(comp).argmax(axis=1)])
        self.signs_[self.signs_ == 0] = 1.0
        pcs = self._raw_transform(sub)
        self.quantiles_ = pcs.quantile([0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95])
        self.fit_range_ = (str(sub["datetime"].iloc[0]), str(sub["datetime"].iloc[-1]))
        return self

    def _raw_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features].to_numpy(float)
        Z = self.pca.transform(self.scaler.transform(np.nan_to_num(X)))
        Z = Z * self.signs_
        return pd.DataFrame(Z, index=df.index, columns=self.pc_names)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._raw_transform(df)
        out[~df["feat_ok"].to_numpy()] = np.nan
        return out

    def attach(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d[self.pc_names] = self.transform(df)
        return d

    # ---------------- reporting ----------------
    @property
    def pc_names(self) -> list[str]:
        return [f"PC{i+1}" for i in range(self.n)]

    def loadings(self) -> pd.DataFrame:
        return pd.DataFrame((self.pca.components_ * self.signs_[:, None]).T,
                            index=self.features, columns=self.pc_names)

    def q(self, pc: str, quantile: float) -> float:
        """Threshold value at a TRAIN-distribution quantile (keeps filters comparable
        across walk-forward refits, where raw PC scale drifts)."""
        return float(self.quantiles_.loc[quantile, pc]) if quantile in self.quantiles_.index \
            else float(np.quantile(self.quantiles_[pc], quantile))

    def interpret(self, top: int = 4) -> pd.DataFrame:
        """Human-readable meaning of each PC (top loadings + heuristic label)."""
        L, evr, rows = self.loadings(), self.pca.explained_variance_ratio_, []
        for i, pc in enumerate(self.pc_names):
            s = L[pc].reindex(L[pc].abs().sort_values(ascending=False).index)[:top]
            desc = "  ".join(f"{k} {v:+.2f}" for k, v in s.items())
            score = {k: float(L[pc].reindex(v).abs().sum()) for k, v in THEMES.items()}
            ranked = sorted(score, key=score.get, reverse=True)
            label = ranked[0] if score[ranked[0]] > 1.3 * score[ranked[1]] \
                else f"{ranked[0]} + {ranked[1]}"
            rows.append({"PC": pc, "explained_var": round(float(evr[i]), 4),
                         "top_loadings": desc, "reading": label})
        return pd.DataFrame(rows)

    def report(self) -> str:
        L = self.loadings().round(3)
        evr = self.pca.explained_variance_ratio_
        lines = [f"PCA fit window: {self.fit_range_[0]} .. {self.fit_range_[1]}",
                 "explained variance ratio: " + ", ".join(f"PC{i+1}={v:.3f}" for i, v in enumerate(evr)) +
                 f"  (cumulative {evr.sum():.3f})", "", "Loadings:", L.to_string(), ""]
        for _, r in self.interpret().iterrows():
            lines.append(f"{r['PC']} ({r['explained_var']:.1%}): {r['top_loadings']}  -> {r['reading']}")
        return "\n".join(lines)
