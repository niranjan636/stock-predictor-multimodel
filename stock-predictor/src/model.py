from typing import Tuple, Dict, List
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    _HAS_SM = True
except Exception:
    _HAS_SM = False

SUPPORTED_MODELS = ["RandomForest", "XGBoost", "LinearRegression", "ARIMA_close_only"]

def train_model(model_name: str, train: pd.DataFrame) -> Tuple[object, List[str], str]:
    model_name = (model_name or "RandomForest").strip()
    if model_name == "ARIMA_close_only":
        if not _HAS_SM:
            raise ImportError("statsmodels not installed; cannot use ARIMA_close_only.")
        return None, [], "close_only"

    features = [c for c in train.columns if c != "Target"]
    X_train = train[features].select_dtypes(include=[np.number])
    y_train = train["Target"]

    if model_name == "RandomForest":
        model = RandomForestRegressor(n_estimators=400, max_depth=8, random_state=42, n_jobs=-1)
    elif model_name == "XGBoost":
        if not _HAS_XGB:
            raise ImportError("xgboost not installed; cannot use XGBoost.")
        model = XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, tree_method="hist")
    elif model_name == "LinearRegression":
        model = LinearRegression()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(X_train, y_train)
    return model, list(X_train.columns), "feature_based"

def evaluate(model, cols: List[str], df: pd.DataFrame, model_kind: str) -> Dict[str, float]:
    if model_kind == "close_only":
        return {"MAE": float("nan"), "MAPE": float("nan")}
    X_test = df[cols]
    y_true = df["Target"]
    y_pred = model.predict(X_test)
    return {"MAE": float(mean_absolute_error(y_true, y_pred)), "MAPE": float(mean_absolute_percentage_error(y_true, y_pred))}

def forecast_next_days(model, used_cols, raw, days=5, model_kind="multi"):
    tmp = raw.copy()

    # Flatten MultiIndex if needed
    if isinstance(tmp.columns, pd.MultiIndex):
        tmp.columns = ['_'.join([str(c) for c in col if c]) for col in tmp.columns]

    # Standardize casing
    tmp.columns = [str(c).title() for c in tmp.columns]

    # Create features (no model_kind arg here)
    tmp = make_features(tmp)

    # If close_only model, keep only Close column features
    if model_kind == "close_only":
        tmp = tmp[[col for col in tmp.columns if "Close" in col]]

    # Match used_cols with available columns
    available_cols = [col for col in used_cols if col in tmp.columns]
    if not available_cols:
        raise ValueError(
            f"No matching columns found.\nused_cols: {used_cols}\ncurrent_cols: {list(tmp.columns)}"
        )

    X_last = tmp[available_cols].iloc[-1:]
    preds = []

    for _ in range(days):
        y_pred = model.predict(X_last)[0]
        preds.append(y_pred)
        X_last = X_last.shift(-1, axis=0)
        X_last.iloc[-1] = y_pred

    future_dates = pd.date_range(start=tmp.index[-1] + pd.Timedelta(days=1), periods=days)
    return pd.DataFrame({"Date": future_dates, "Prediction": preds})
def feature_importance(model, feature_names: List[str]):
    try:
        if hasattr(model, "feature_importances_"):
            return dict(zip(feature_names, model.feature_importances_.tolist()))
        if hasattr(model, "coef_"):
            coefs = model.coef_.ravel().tolist() if hasattr(model.coef_, "ravel") else list(model.coef_)
            return dict(zip(feature_names, [float(x) for x in coefs]))
        if hasattr(model, "get_booster"):
            booster = model.get_booster(); fmap = booster.get_fscore()
            return {feature_names[int(k[1:])]: float(v) for k, v in fmap.items() if k.startswith("f")}
    except Exception:
        return None
    return None
