"""
Predictive Analytics Engine
============================
Supports multiple forecasting models:
  - Linear Regression
  - Polynomial Regression
  - Random Forest Regressor
  - Moving Average (Time-Series)
  - Exponential Smoothing (Time-Series)
"""

import numpy as np
import pandas as pd
import json
import warnings
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")


# ─── Data Generation ──────────────────────────────────────────────────────────

DATASETS = {
    "sales": {
        "label": "Monthly Sales Revenue",
        "unit": "$",
        "description": "E-commerce sales trend with seasonal spikes and growth.",
    },
    "temperature": {
        "label": "Average Temperature (°C)",
        "unit": "°C",
        "description": "Annual temperature readings showing climate trend.",
    },
    "stock": {
        "label": "Stock Price (USD)",
        "unit": "$",
        "description": "Tech stock price movement with volatility.",
    },
    "energy": {
        "label": "Energy Consumption (MWh)",
        "unit": "MWh",
        "description": "Monthly energy usage with seasonal patterns.",
    },
    "users": {
        "label": "Monthly Active Users",
        "unit": "K",
        "description": "SaaS platform growth with adoption curve.",
    },
}


def generate_dataset(name: str, n_points: int = 60, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic time-series dataset."""
    np.random.seed(seed)
    t = np.arange(n_points)
    months = pd.date_range("2019-01", periods=n_points, freq="ME")

    if name == "sales":
        trend = 50000 + 1200 * t
        seasonal = 8000 * np.sin(2 * np.pi * t / 12) + 4000 * np.sin(2 * np.pi * t / 6)
        noise = np.random.normal(0, 3500, n_points)
        values = trend + seasonal + noise

    elif name == "temperature":
        trend = 14.5 + 0.03 * t
        seasonal = 12 * np.sin(2 * np.pi * (t - 3) / 12)
        noise = np.random.normal(0, 1.2, n_points)
        values = trend + seasonal + noise

    elif name == "stock":
        returns = np.random.normal(0.008, 0.04, n_points)
        values = 100 * np.cumprod(1 + returns)
        values += np.random.normal(0, 1.5, n_points)

    elif name == "energy":
        trend = 2200 - 0.5 * t
        seasonal = 600 * np.cos(2 * np.pi * t / 12)
        noise = np.random.normal(0, 80, n_points)
        values = trend + seasonal + noise

    elif name == "users":
        values = 5000 * (1 - np.exp(-t / 20)) + 500 * t / n_points * np.random.normal(1, 0.1, n_points)
        values += np.random.normal(0, 100, n_points)

    else:
        raise ValueError(f"Unknown dataset: {name}")

    return pd.DataFrame({"date": months, "value": np.clip(values, 0, None), "time_index": t})


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, lag_features: int = 3) -> pd.DataFrame:
    """
    Clean and enrich a time-series DataFrame.
    Steps:
      1. Drop exact duplicate rows
      2. Impute missing values with column median
      3. Add lag features (t-1, t-2, t-3)
      4. Add rolling mean (window=3)
    """
    df = df.copy()

    # 1. Deduplication
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)

    # 2. Missing-value imputation
    imputer = SimpleImputer(strategy="median")
    df["value"] = imputer.fit_transform(df[["value"]])

    # 3. Lag features
    for lag in range(1, lag_features + 1):
        df[f"lag_{lag}"] = df["value"].shift(lag)

    # 4. Rolling mean
    df["rolling_mean_3"] = df["value"].rolling(3).mean()

    df = df.dropna().reset_index(drop=True)

    preprocessing_log = {
        "duplicates_removed": removed,
        "missing_imputed": True,
        "lag_features": lag_features,
        "final_rows": len(df),
    }
    return df, preprocessing_log


# ─── Model Training ───────────────────────────────────────────────────────────

MODELS = {
    "linear": {
        "label": "Linear Regression",
        "model": LinearRegression(),
        "type": "ml",
    },
    "polynomial": {
        "label": "Polynomial Regression (deg 3)",
        "model": Pipeline([
            ("poly", PolynomialFeatures(degree=3, include_bias=False)),
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0)),
        ]),
        "type": "ml",
    },
    "random_forest": {
        "label": "Random Forest",
        "model": RandomForestRegressor(n_estimators=100, random_state=42),
        "type": "ml",
    },
    "gradient_boost": {
        "label": "Gradient Boosting",
        "model": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "type": "ml",
    },
    "moving_average": {
        "label": "Moving Average (window=6)",
        "model": None,
        "type": "timeseries",
        "window": 6,
    },
    "exp_smoothing": {
        "label": "Exponential Smoothing (α=0.3)",
        "model": None,
        "type": "timeseries",
        "alpha": 0.3,
    },
}


def _moving_average_forecast(series: np.ndarray, window: int, forecast_steps: int):
    """Rolling window forecast."""
    predictions = []
    history = list(series)
    for _ in range(forecast_steps):
        pred = np.mean(history[-window:])
        predictions.append(pred)
        history.append(pred)
    return np.array(predictions)


def _exp_smoothing_forecast(series: np.ndarray, alpha: float, forecast_steps: int):
    """Simple exponential smoothing."""
    level = series[0]
    for val in series[1:]:
        level = alpha * val + (1 - alpha) * level
    return np.full(forecast_steps, level)


def train_and_forecast(
    df: pd.DataFrame,
    model_key: str,
    forecast_steps: int = 12,
    test_size: float = 0.2,
) -> dict:
    """
    Train the chosen model and return metrics + forecasts.
    Returns a JSON-serialisable dict.
    """
    cfg = MODELS[model_key]
    values = df["value"].values
    t = df["time_index"].values.reshape(-1, 1)

    result = {
        "model_key": model_key,
        "model_label": cfg["label"],
        "forecast_steps": forecast_steps,
    }

    # ── Time-series models ──
    if cfg["type"] == "timeseries":
        split = int(len(values) * (1 - test_size))
        train_vals, test_vals = values[:split], values[split:]

        if model_key == "moving_average":
            test_preds = _moving_average_forecast(train_vals, cfg["window"], len(test_vals))
            future_preds = _moving_average_forecast(values, cfg["window"], forecast_steps)
        else:
            test_preds = np.array([
                _exp_smoothing_forecast(train_vals[:i+1], cfg["alpha"], 1)[0]
                for i in range(len(test_vals))
            ])
            future_preds = _exp_smoothing_forecast(values, cfg["alpha"], forecast_steps)

        mae = mean_absolute_error(test_vals, test_preds)
        rmse = np.sqrt(mean_squared_error(test_vals, test_preds))
        r2 = r2_score(test_vals, test_preds)

        result.update({
            "train_actual": values[:split].tolist(),
            "test_actual": test_vals.tolist(),
            "test_predicted": test_preds.tolist(),
            "future_forecast": future_preds.tolist(),
            "train_indices": t[:split].flatten().tolist(),
            "test_indices": t[split:].flatten().tolist(),
            "cv_r2_scores": None,
            "cv_r2_mean": None,
            "cv_r2_std": None,
        })

    # ── ML models ──
    else:
        feature_cols = ["time_index"] + [c for c in df.columns if c.startswith("lag_") or c == "rolling_mean_3"]
        X = df[feature_cols].values
        y = values

        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(cfg["model"], X, y, cv=tscv, scoring="r2")

        split = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        cfg["model"].fit(X_train, y_train)
        test_preds = cfg["model"].predict(X_test)

        # Build future feature matrix
        last_t = df["time_index"].max()
        future_rows = []
        lag_buffer = list(y[-3:])
        for step in range(1, forecast_steps + 1):
            future_t = last_t + step
            lags = lag_buffer[-3:][::-1]
            while len(lags) < 3:
                lags.append(lags[-1])
            rm3 = np.mean(lag_buffer[-3:])
            row = [future_t] + lags + [rm3]
            future_rows.append(row)
            pred = cfg["model"].predict(np.array([row]))[0]
            lag_buffer.append(pred)

        future_preds = cfg["model"].predict(np.array(future_rows))

        mae = mean_absolute_error(y_test, test_preds)
        rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        r2 = r2_score(y_test, test_preds)

        result.update({
            "train_actual": y_train.tolist(),
            "test_actual": y_test.tolist(),
            "test_predicted": test_preds.tolist(),
            "future_forecast": future_preds.tolist(),
            "train_indices": t[:split].flatten().tolist(),
            "test_indices": t[split:].flatten().tolist(),
            "cv_r2_scores": cv_scores.tolist(),
            "cv_r2_mean": float(cv_scores.mean()),
            "cv_r2_std": float(cv_scores.std()),
        })

    result["metrics"] = {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
        "mape": round(float(np.mean(np.abs((test_vals if cfg["type"] == "timeseries" else y_test) / (result["test_actual"] or [1]) - 1)) * 100), 2)
        if any(v != 0 for v in result["test_actual"]) else None,
    }

    return result


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    dataset = sys.argv[1] if len(sys.argv) > 1 else "sales"
    model = sys.argv[2] if len(sys.argv) > 2 else "random_forest"

    print(f"\n{'='*55}")
    print(f"  Predictive Analytics — {DATASETS[dataset]['label']}")
    print(f"  Model: {MODELS[model]['label']}")
    print(f"{'='*55}\n")

    raw_df = generate_dataset(dataset)
    df, log = preprocess(raw_df)
    print("Preprocessing log:", json.dumps(log, indent=2))

    result = train_and_forecast(df, model)

    print(f"\nMetrics:")
    for k, v in result["metrics"].items():
        print(f"  {k.upper():6s}: {v}")

    print(f"\nNext 12-period forecast:")
    for i, val in enumerate(result["future_forecast"], 1):
        print(f"  Period +{i:2d}: {val:,.2f}")
