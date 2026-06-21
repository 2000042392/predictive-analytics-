"""
Generate publication-quality charts for the project README.
"""
import sys
sys.path.insert(0, "src")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings("ignore")

from predictive_model import (
    generate_dataset, preprocess, train_and_forecast, DATASETS, MODELS
)

# ── Style ──────────────────────────────────────────────────────────────────────
DARK  = "#0D1117"
PANEL = "#161B22"
BLUE  = "#58A6FF"
GREEN = "#3FB950"
ORANGE= "#F78166"
PURPLE= "#BC8CFF"
GRAY  = "#8B949E"
WHITE = "#E6EDF3"
GOLD  = "#E3B341"

plt.rcParams.update({
    "figure.facecolor": DARK,
    "axes.facecolor":   PANEL,
    "axes.edgecolor":   "#30363D",
    "axes.labelcolor":  WHITE,
    "xtick.color":      GRAY,
    "ytick.color":      GRAY,
    "text.color":       WHITE,
    "grid.color":       "#21262D",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.5,
    "font.family":      "DejaVu Sans",
    "legend.facecolor": PANEL,
    "legend.edgecolor": "#30363D",
})


def make_forecast_chart(dataset="sales", model_key="gradient_boost"):
    raw = generate_dataset(dataset)
    df, _ = preprocess(raw)
    res  = train_and_forecast(df, model_key, forecast_steps=12)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor=DARK)
    fig.suptitle(
        f"Predictive Analytics · {DATASETS[dataset]['label']}  ×  {MODELS[model_key]['label']}",
        color=WHITE, fontsize=13, fontweight="bold", y=1.02
    )

    # ── Left: full history + forecast ──
    ax = axes[0]
    train_x = res["train_indices"]
    test_x  = res["test_indices"]
    n = len(train_x) + len(test_x)
    future_x = list(range(n, n + res["forecast_steps"]))

    ax.plot(train_x, res["train_actual"],   color=BLUE,   lw=1.8, label="Train")
    ax.plot(test_x,  res["test_actual"],    color=GREEN,  lw=1.8, label="Test (actual)")
    ax.plot(test_x,  res["test_predicted"], color=ORANGE, lw=1.8, linestyle="--", label="Test (predicted)")
    ax.plot(future_x, res["future_forecast"], color=GOLD, lw=2.2, linestyle="-.", label="Forecast")
    ax.fill_between(future_x,
        np.array(res["future_forecast"]) * 0.92,
        np.array(res["future_forecast"]) * 1.08,
        color=GOLD, alpha=0.15, label="±8% CI"
    )
    ax.axvline(x=test_x[0], color=GRAY, linestyle=":", lw=1, alpha=0.7)
    ax.text(test_x[0]+0.3, ax.get_ylim()[0], "train/test split", color=GRAY, fontsize=7, va="bottom")
    ax.set_title("Historical + Forecast", color=WHITE, fontsize=11)
    ax.set_xlabel("Time Period", fontsize=9)
    ax.set_ylabel(f"{DATASETS[dataset]['label']}", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.4)

    # ── Right: residuals ──
    ax2 = axes[1]
    residuals = np.array(res["test_actual"]) - np.array(res["test_predicted"])
    ax2.bar(range(len(residuals)), residuals,
            color=[GREEN if r > 0 else ORANGE for r in residuals], alpha=0.8, width=0.7)
    ax2.axhline(0, color=WHITE, lw=1, alpha=0.5)
    ax2.set_title("Residuals (Test Set)", color=WHITE, fontsize=11)
    ax2.set_xlabel("Test Sample", fontsize=9)
    ax2.set_ylabel("Actual − Predicted", fontsize=9)

    # metrics box
    m = res["metrics"]
    box_txt = f"MAE  {m['mae']:,.1f}\nRMSE {m['rmse']:,.1f}\nR²   {m['r2']:.3f}"
    ax2.text(0.97, 0.97, box_txt, transform=ax2.transAxes,
             ha="right", va="top", fontsize=8.5, color=WHITE,
             bbox=dict(facecolor="#21262D", edgecolor="#30363D", boxstyle="round,pad=0.5"))
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    path = "static/forecast_chart.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"Saved {path}")
    return path


def make_model_comparison():
    raw = generate_dataset("sales")
    df, _ = preprocess(raw)

    model_keys = ["linear", "polynomial", "random_forest", "gradient_boost", "moving_average", "exp_smoothing"]
    labels, r2s, maes, rmses = [], [], [], []

    for key in model_keys:
        res = train_and_forecast(df, key)
        labels.append(MODELS[key]["label"].replace(" Regression","").replace(" (","\n("))
        r2s.append(max(res["metrics"]["r2"], -1))
        maes.append(res["metrics"]["mae"])
        rmses.append(res["metrics"]["rmse"])

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), facecolor=DARK)
    fig.suptitle("Model Comparison — Monthly Sales", color=WHITE, fontsize=13, fontweight="bold")

    colors = [BLUE, PURPLE, GREEN, GOLD, ORANGE, "#F0883E"]

    for ax, vals, title, good_high in zip(
        axes,
        [r2s, maes, rmses],
        ["R² Score (higher = better)", "MAE (lower = better)", "RMSE (lower = better)"],
        [True, False, False]
    ):
        bars = ax.barh(labels, vals, color=colors, alpha=0.85, height=0.6)
        ax.set_title(title, color=WHITE, fontsize=10)
        ax.grid(True, axis="x", alpha=0.4)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + abs(bar.get_width()) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:,.2f}", va="center", ha="left", fontsize=8, color=WHITE)
        ax.tick_params(axis="y", labelsize=7.5)

    plt.tight_layout()
    path = "static/model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"Saved {path}")
    return path


def make_dataset_overview():
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), facecolor=DARK)
    fig.suptitle("Dataset Overview — All Built-in Time Series", color=WHITE, fontsize=13, fontweight="bold")
    colors = [BLUE, GREEN, GOLD, PURPLE, ORANGE]

    datasets = list(DATASETS.keys())
    for idx, (name, ax, color) in enumerate(zip(datasets, axes.flat[:5], colors)):
        raw = generate_dataset(name)
        ax.plot(raw["value"].values, color=color, lw=1.5)
        ax.fill_between(range(len(raw)), raw["value"].values, alpha=0.1, color=color)
        ax.set_title(DATASETS[name]["label"], color=WHITE, fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Month", fontsize=8, color=GRAY)
        ax.set_ylabel(DATASETS[name]["unit"], fontsize=8, color=GRAY)
        ax.grid(True, alpha=0.35)
        ax.tick_params(labelsize=7)

    # 6th cell: preprocessing stats
    ax6 = axes.flat[5]
    ax6.axis("off")
    steps = [
        "① Duplicate Row Removal",
        "② Median Value Imputation",
        "③ Lag Features  (t-1, t-2, t-3)",
        "④ Rolling Mean  (window=3)",
        "⑤ Time-Series Train/Test Split",
        "⑥ Cross-Validation (TimeSeriesSplit)",
    ]
    ax6.text(0.5, 0.92, "Preprocessing Pipeline", ha="center", va="top",
             fontsize=11, fontweight="bold", color=WHITE, transform=ax6.transAxes)
    for i, step in enumerate(steps):
        ax6.text(0.1, 0.76 - i * 0.12, step, ha="left", va="top",
                 fontsize=9, color=GREEN if i % 2 == 0 else BLUE,
                 transform=ax6.transAxes)

    plt.tight_layout()
    path = "static/dataset_overview.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"Saved {path}")
    return path


if __name__ == "__main__":
    make_dataset_overview()
    make_forecast_chart()
    make_model_comparison()
    print("\n✅ All charts generated in static/")
