"""Pre-submission denial-risk model: features, training, cost-based threshold, drivers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None

CATEGORICAL = ["payer", "service_line", "provider_specialty"]
NUMERIC = [
    "needs_prior_auth",
    "prior_auth_obtained",
    "days_to_submit",
    "billed_amount",
    "dx_px_mismatch",
    "missing_modifier",
    "duplicate_flag",
    "in_network",
    "missing_fields",
    "electronic_submission",
    "patient_age",
]


def build_features(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    X = df[NUMERIC].copy()
    X["auth_gap"] = ((df["needs_prior_auth"] == 1) & (df["prior_auth_obtained"] == 0)).astype(int)
    X["log_billed"] = np.log1p(df["billed_amount"])
    X = pd.concat([X, pd.get_dummies(df[CATEGORICAL], dtype=int)], axis=1)
    if columns is not None:  # align to training columns (unseen categories -> 0)
        X = X.reindex(columns=columns, fill_value=0)
    return X


@dataclass
class CostModel:
    """Per-claim business assumptions (ESTIMATES, change them for a real setting)."""

    review_cost: float = 6.0          # staff minutes x rate to review one flagged claim
    rework_cost: float = 45.0         # cost to work a denial after the fact (appeal/resubmit)
    fix_success: float = 0.6          # share of real denials a pre-submission review prevents


def choose_threshold(y: np.ndarray, p: np.ndarray, cost: CostModel) -> tuple[float, float]:
    """Pick the probability cut-off that maximises expected net saving per claim."""
    best_t, best_s = 0.5, -np.inf
    for t in np.linspace(0.02, 0.9, 89):
        flag = p >= t
        saving = (flag & (y == 1)).sum() * cost.fix_success * cost.rework_cost - flag.sum() * cost.review_cost
        if saving > best_s:
            best_t, best_s = float(t), float(saving)
    return best_t, best_s / len(y)


class DenialModel:
    def __init__(self, seed: int = 7):
        self.seed = seed
        self.columns: list[str] = []
        self.booster = None
        self.threshold = 0.5

    def fit(self, df: pd.DataFrame, cost: CostModel | None = None) -> dict:
        cost = cost or CostModel()
        X = build_features(df)
        self.columns = list(X.columns)
        y = df["denied"].values
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=self.seed, stratify=y)
        params = dict(n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.9,
                      colsample_bytree=0.9, eval_metric="aucpr", random_state=self.seed)
        self.booster = xgb.XGBClassifier(**params)
        self.booster.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        p_val = self.booster.predict_proba(X_val)[:, 1]
        self.threshold, saving = choose_threshold(y_val, p_val, cost)
        return {"val_auc": roc_auc_score(y_val, p_val), "val_ap": average_precision_score(y_val, p_val),
                "threshold": self.threshold, "val_saving_per_claim": saving}

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.booster.predict_proba(build_features(df, self.columns))[:, 1]

    def drivers(self, df: pd.DataFrame, top_k: int = 3) -> list[list[tuple[str, float]]]:
        """Per-claim top contributing features (XGBoost's built-in SHAP values)."""
        X = build_features(df, self.columns)
        contrib = self.booster.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
        out = []
        for row in contrib:
            idx = np.argsort(-row)[:top_k]
            out.append([(self.columns[i], float(row[i])) for i in idx if row[i] > 0])
        return out


def evaluate(model: DenialModel, df: pd.DataFrame, cost: CostModel | None = None) -> dict:
    cost = cost or CostModel()
    y = df["denied"].values
    p = model.predict_proba(df)
    flag = p >= model.threshold
    tp = int((flag & (y == 1)).sum())
    saving = tp * cost.fix_success * cost.rework_cost - flag.sum() * cost.review_cost
    return {
        "n": int(len(y)),
        "denial_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, p)),
        "avg_precision": float(average_precision_score(y, p)),
        "flag_rate": float(flag.mean()),
        "precision_at_threshold": float(tp / max(flag.sum(), 1)),
        "recall_at_threshold": float(tp / max(y.sum(), 1)),
        "est_net_saving_per_1000_claims": float(1000 * saving / len(y)),
    }
