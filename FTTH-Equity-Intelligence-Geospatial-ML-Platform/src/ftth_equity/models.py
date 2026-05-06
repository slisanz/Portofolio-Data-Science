from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, train_test_split

from . import config


@dataclass
class TrainResult:
    model: object
    cv_auc: float
    cv_ap: float
    holdout_auc: float
    holdout_ap: float
    holdout_brier: float
    feature_names: list[str]


def _make_estimator():
    # lightgbm if available, else sklearn HGB as fallback
    try:
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=50,
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.05,
            random_state=config.RANDOM_SEED,
        )


def train_lag_model(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    target_col: str = "is_lagging",
    group_col: str = "code_insee",
) -> TrainResult:
    feature_cols = list(feature_cols)
    X = df[feature_cols].fillna(-1).to_numpy()
    y = df[target_col].astype(int).to_numpy()
    groups = df[group_col].fillna("UNK").to_numpy()

    X_tr, X_te, y_tr, y_te, g_tr, _ = train_test_split(
        X, y, groups, test_size=0.2, random_state=config.RANDOM_SEED, stratify=y
    )

    est = _make_estimator()
    cv = GroupKFold(n_splits=5)
    cv_aucs, cv_aps = [], []
    for tr, va in cv.split(X_tr, y_tr, g_tr):
        est.fit(X_tr[tr], y_tr[tr])
        p = est.predict_proba(X_tr[va])[:, 1]
        cv_aucs.append(roc_auc_score(y_tr[va], p))
        cv_aps.append(average_precision_score(y_tr[va], p))

    final = CalibratedClassifierCV(_make_estimator(), method="isotonic", cv=3)
    final.fit(X_tr, y_tr)
    p_te = final.predict_proba(X_te)[:, 1]

    return TrainResult(
        model=final,
        cv_auc=float(np.mean(cv_aucs)),
        cv_ap=float(np.mean(cv_aps)),
        holdout_auc=float(roc_auc_score(y_te, p_te)),
        holdout_ap=float(average_precision_score(y_te, p_te)),
        holdout_brier=float(brier_score_loss(y_te, p_te)),
        feature_names=feature_cols,
    )


def save(result: TrainResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": result.model,
            "feature_names": result.feature_names,
            "metrics": {
                "cv_auc": result.cv_auc,
                "cv_ap": result.cv_ap,
                "holdout_auc": result.holdout_auc,
                "holdout_ap": result.holdout_ap,
                "holdout_brier": result.holdout_brier,
            },
        },
        path,
    )
    return path


def load(path: Path) -> dict:
    return joblib.load(path)
