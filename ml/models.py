from __future__ import annotations
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


def get_model(name: str, random_state: int = 42):
    if name == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, class_weight="balanced",
                                       max_iter=2000, random_state=random_state)),
        ])
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced",
            n_jobs=-1, random_state=random_state,
        )
    if name == "xgb":
        if not XGB_AVAILABLE:
            raise ImportError("pip install xgboost")
        return XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            n_jobs=-1, random_state=random_state, verbosity=0,
        )
    if name == "mlp":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4,
                                  max_iter=500, random_state=random_state)),
        ])
    raise ValueError(f"Unknown model: {name}")


def available_models():
    return ["logreg", "rf", "mlp"] + (["xgb"] if XGB_AVAILABLE else [])