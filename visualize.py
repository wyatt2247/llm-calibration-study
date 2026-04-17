# this is the script that creates the actual charts and bars from the oranized data from anlayze script

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from pathlib import Path

# Creates graphs and charts from the analyse.py script -> takes the output of that and feeds it into visualize.py
# ================== Config =====================
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

summary = pd.read_csv(RESULTS_DIR / "summary.csv")
acc_by_subject = pd.read_csv(RESULTS_DIR / "accuracy_by_subject.csv")

csv_files = sorted(RESULTS_DIR.glob("results_*.csv"))
df = pd.read_csv(csv_files[-1])

# ================== ACCURACY VS CONFIDENCE =====================
def plot_accuracy_vs_confidence():
    fig, ax = plt.subplots(figsize = (8,5))
    models = summary["model"]
    x = np.arange(len(models))
    w = 0.35

    ax.bar(x - w/2, summary["accuracy"] * 100, w, label = "Accuracy", color= "#378ADD")
    ax.bar(x + w/2, summary["mean_confidence"], w, label = "Confidence", color="#D85A30")

    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha= "right")
    ax.set_ylim(0,100)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "accuracy_vs_confidence.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved accurracy_vs_confidence.png")

#=================== CALIBRATION GAP ======================
def plot_calibration_gap():
    fig, ax = plt.subplots(figsize=(8,5))
    models = summary["model"]
    gaps = summary["calibration_gap"]

    bars = ax.bar(models, gaps, color="#E24B4A")
    for bar, v in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"+{v:.1f}%", ha="center", fontsize=9)
    
    ax.set_ylabel("Calibration Gap (%)")
    ax.set_ylim(0, max(gaps) + 10)
    ax.spines["top"].set_visible(False)
    plt.xticks(rotation=15, ha="right")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "calibration_gap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved calibration_gap.png")


#=================== CALIBRATION CURVE =====================
def plot_calibration_curve():
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot([0,1], [0,1], linestyle="--", color="grey", label = "Perfect Calibration")
    colors = ["#378ADD", "#D4537E", "#1D9E75", "#534AB7", "#D85A30"]
    markers= ["o", "s", "^", "D", "v"]

    for i, model in enumerate(df["model"].unique()):
        model_df = df[df["model"] == model]
        if len(model_df) < 5:
            continue

        y_true = model_df["is_correct"].astype(int)
        y_prob = model_df["confidence"] / 100

        try:
            prop_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=5, strategy ="uniform")
            ax.plot(prob_pred, prop_true, marker=markers[i], color=colors[i],
                label=model, markersize=6)
        except:
            continue

    ax.set_xlabel("Confidence")
    ax.set_ylabel("actual accuracy")
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.legend(loc="upper left", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "calibration_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved calibration_curve.png")


# ================== HEATMAP =====================

def plot_heatmap():
    pivot = acc_by_subject.pivot(index="model", columns="subject", values="is_correct")
    pivot = pivot * 100

    fig, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn",
                vmin=20, vmax=100, center= 50, cbar_kws={"label": "Accuracy (%):"})

    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved heatmap.png")


# ================== CONSISTENCY =====================
def plot_consistency():
    fig, ax = plt.subplots(figsize=(8,5))
    models = summary["model"]
    cons = summary["consistency"]

    bars = ax.bar(models, cons, color="#534AB7", width=0.6)
    for bar, v in zip(bars, cons):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f"{v:.3f}", ha="center", fontsize=9)
    
    ax.set_ylabel("Consistency Score")
    ax.set_ylim(0,1.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=15, ha="right")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "consistency.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved consistency.png")


#=================== CONFIDENCE BOX PLOT =====================
def plot_confidence_boxplot():
    fig, ax = plt.subplots(figsize=(8,5))

    models = df["model"].unique()
    data = [df[df["model"]== m]["confidence"].values for m in models]
    colors_list = ["#378ADD", "#D4537E", "#1D9E75", "#534AB7", "#D85A30"]

    bp = ax.boxplot(data, tick_labels= models, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors_list):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    
    ax.set_ylabel("Confidence Score")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=15, ha="right")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confidence_boxplot.png", dpi=300, bbox_inches = "tight")
    plt.close()
    print("Saved confidence_boxplot.png")

#=================== ECE AND BRIER SCORE =====================

def plot_ece_brier():
    if "ece" not in summary.columns or "brier_score" not in summary.columns:
        print("SKIPPING ECE/BRIER chart - run updated analysis.py first")
        return
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,5))
    models = summary["model"]
    
    # Creates the ECE Chart 
    bars1 = ax1.bar(models, summary["ece"], color= "#1D9E75", width=0.6)
    for bar, v in zip(bars1, summary["ece"]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", fontsize=9)
    ax1.set_ylabel("ECE Score")
    ax1.set_ylim(0, max(summary["ece"]) + 0.1)
    ax1.set_title("Expected Calibration Error (lower = better)", fontsize=11, fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    plt.sca(ax1)
    plt.xticks(rotation=15, ha="right")


    # Creates the  Brier Chart
    bars2 = ax2.bar(models, summary["brier_score"], color= "#BA7517", width=0.6)
    for bar, v in zip(bars2, summary["brier_score"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", fontsize=9)
    
    ax2.set_ylabel("Brier Score")
    ax2.set_ylim(0, max(summary["brier_score"]) + 0.1)
    ax2.set_title("Brier Score (lower = better)", fontsize=11, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    plt.sca(ax2)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ece_brier.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved ece_brier.png")

#Runs all 7 parts of the script to create each of the 7 figures needed for the report
# ================== RUN ALL =====================

if __name__ == "__main__":
    plot_accuracy_vs_confidence()
    plot_calibration_gap()
    plot_calibration_curve()
    plot_heatmap()
    plot_consistency()
    plot_confidence_boxplot()
    plot_ece_brier()
    print(f"\nAll figures saved to {FIGURES_DIR}/")
