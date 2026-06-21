# 📊 Predictive Analytics — Forecasting Engine

> A complete, production-ready predictive analytics project.  
> Train 6 models, evaluate accuracy, and visualise 12-period forecasts — all in Python.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)
![Models](https://img.shields.io/badge/Models-6-purple)

---

## 🚀 Features

| Feature | Details |
|---|---|
| **6 Forecasting Models** | Linear, Polynomial, Random Forest, Gradient Boosting, Moving Average, Exponential Smoothing |
| **5 Built-in Datasets** | Sales, Temperature, Stock Price, Energy, Monthly Active Users |
| **Full Preprocessing** | Dedup → impute → lag features → rolling mean |
| **Robust Evaluation** | MAE, RMSE, R², MAPE + TimeSeriesSplit cross-validation |
| **Publication Charts** | Dark-theme matplotlib — forecast, residuals, model comparison |
| **Zero Config** | Run with a single command, no YAML, no config files |

---

## 📁 Project Structure

```
predictive-analytics/
│
├── src/
│   ├── predictive_model.py     ← Core engine: data generation, preprocessing, training
│   └── generate_charts.py      ← Generates all charts as PNG files
│
├── static/
│   ├── dataset_overview.png    ← All 5 datasets visualised
│   ├── forecast_chart.png      ← Train / test / forecast chart
│   └── model_comparison.png    ← Side-by-side model metrics
│
├── data/                       ← Drop your own CSVs here
├── index.html                  ← Project website (host on GitHub Pages)
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

```bash
# 1. Clone
git clone https://github.com/yourname/predictive-analytics
cd predictive-analytics

# 2. Install
pip install -r requirements.txt

# 3. Run (uses sales dataset + gradient_boost by default)
python3 src/predictive_model.py

# 4. Choose dataset and model
python3 src/predictive_model.py temperature random_forest

# 5. Regenerate all charts
python3 src/generate_charts.py
```

**Available datasets:** `sales`, `temperature`, `stock`, `energy`, `users`  
**Available models:** `linear`, `polynomial`, `random_forest`, `gradient_boost`, `moving_average`, `exp_smoothing`

---

## 🐍 Use as a Module

```python
from src.predictive_model import generate_dataset, preprocess, train_and_forecast

# 1. Load
df = generate_dataset("sales")           # or pd.read_csv("data/your_data.csv")

# 2. Preprocess
df, log = preprocess(df)
print(log)
# {'duplicates_removed': 0, 'missing_imputed': True, 'lag_features': 3, 'final_rows': 57}

# 3. Train + Forecast
result = train_and_forecast(df, "gradient_boost", forecast_steps=12)

# 4. Metrics
print(result["metrics"])
# {'mae': 3241.2, 'rmse': 4108.7, 'r2': 0.941, 'mape': 3.8}

# 5. Future values
print(result["future_forecast"])
# [98234.1, 97881.4, 99012.6, 100450.2, ...]
```

---

## 🤖 Models

### Machine Learning Models
These use lag features + rolling mean as input features, trained with `TimeSeriesSplit` cross-validation:

| Model | Key Strength |
|---|---|
| **Linear Regression** | Fastest, most interpretable baseline |
| **Polynomial Regression (deg 3)** | Captures curved trends; Ridge regularisation prevents overfitting |
| **Random Forest (n=100)** | Robust ensemble; handles outliers and interactions |
| **Gradient Boosting (n=100)** | Highest accuracy on structured data; sequential error correction |

### Time-Series Models
No training step — pure statistical methods:

| Model | Key Strength |
|---|---|
| **Moving Average (window=6)** | Simple, interpretable; good for stable series |
| **Exponential Smoothing (α=0.3)** | Gives higher weight to recent observations |

---

## 📐 Evaluation Metrics

| Metric | Formula | Meaning |
|---|---|---|
| **MAE** | mean(|actual − predicted|) | Average absolute error in original units |
| **RMSE** | √mean((actual − predicted)²) | Penalises large errors more than MAE |
| **R²** | 1 − SS_res / SS_tot | Fraction of variance explained (1.0 = perfect) |
| **MAPE** | mean(|actual − predicted| / actual) × 100 | Error as a percentage of actual value |

---

## 📊 Sample Output

```
=======================================================
  Predictive Analytics — Monthly Sales Revenue
  Model: Gradient Boosting
=======================================================

Preprocessing log:
  duplicates_removed: 0
  missing_imputed:    True
  lag_features:       3
  final_rows:         57

Metrics:
  MAE   : 3,241.20
  RMSE  : 4,108.70
  R2    : 0.941
  MAPE  : 3.8%

Next 12-period forecast:
  Period + 1: 98,234.10
  Period + 2: 97,881.40
  ...
```

---

## 🌐 GitHub Pages

This project includes a ready-to-deploy `index.html` project site.

1. Push to GitHub
2. Go to **Settings → Pages → Source → main branch / root**
3. Your site goes live at `https://yourname.github.io/predictive-analytics`

To link it in your portfolio, add this to your `<a href>`:
```
https://yourname.github.io/predictive-analytics
```

---

## 📦 Requirements

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
```

Install: `pip install -r requirements.txt`

---

## 📚 What You'll Learn

- ✅ **Data Preprocessing** — dedup, imputation, feature engineering
- ✅ **Time-Series Splitting** — why random split leaks data; how `TimeSeriesSplit` fixes it
- ✅ **Lag Features** — encoding temporal dependencies as ML input features
- ✅ **Model Selection** — comparing 6 approaches on the same task
- ✅ **Evaluation** — MAE vs RMSE vs R² — when to use each
- ✅ **Multi-Step Forecasting** — iteratively feeding predictions back as inputs
- ✅ **Visualisation** — publication-ready matplotlib charts

---

## 📄 License

MIT — free for personal and commercial use.

---

*Built with Python · scikit-learn · matplotlib · pandas*
