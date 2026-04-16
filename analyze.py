from math import nan
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import binomtest
import statsmodels.stats.contingency_tables as smct

RESULTS_DIR = Path('results')

cvs_files = sorted(RESULTS_DIR.glob("results_*.csv"))
if not cvs_files:
    print ("No results CSV found!")
    exit()

latest = cvs_files[-1]
print(f"Loading: {latest}")
df = pd.read_csv(latest)


# ================== MAJORITY VOTE ACCURACY + CI ===========================
def majority_vote(group):
    counts = group["model_answer"].value_counts()
    return counts.index[0] if len(counts) > 0 else None

votes = df.groupby(["model", "subject", "question"]).apply(majority_vote).reset_index()
votes.columns = ["model", "subject", "question", "voted_answer"]

correct = df.drop_duplicates(subset =["model", "subject", "question"])[["model", "subject", "question", "correct_answer"]]
votes = votes.merge(correct, on=["model", "subject", "question"])
votes["is_correct"] = votes["voted_answer"] == votes["correct_answer"]

# ================== ACCURACY WITH 95% CI ===========================
def accuracy_with_ci(group):
    n = len(group)
    successes = group["is_correct"].sum()
    acc = successes / n
    ci = binomtest(successes, n, alternative="two-sided").proportion_ci(confidence_level=0.95)
    return pd.Series({"accuracy": acc, "ci_low": ci.low, "ci_high": ci.high, "n": n})

accuracy = votes.groupby(["model"]).apply(accuracy_with_ci).reset_index()

# =================== CALIBRATION & OTHER METRICS =======================
mean_confidence = df.groupby("model")["confidence"].mean().reset_index()
mean_confidence.columns = ["model", "mean_confidence"]


summary = accuracy.merge(mean_confidence, on="model")
summary["calibration_gap"] = summary["mean_confidence"] - (summary["accuracy"] * 100)

# consistency
def consistency(group): 
    return 1.0 if group["model_answer"].nunique() == 1 else 0.0

cons = df.groupby(["model", "subject", "question"]).apply(consistency).reset_index(name="consistency")
mean_con = cons.groupby("model")["consistency"].mean().reset_index()
summary = summary.merge(mean_con, on = "model")

def compute_ece(model_df, n_bins=10): 
    confidences = model_df["confidence"].values / 100.0
    corrects = model_df["is_correct"].values.astype(float)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(confidences)
    for i in range(n_bins):
        low = bin_boundaries[i]
        high = bin_boundaries[i+1]
        mask = (confidences >= low) & (confidences <= high) if i == n_bins - 1 else (confidences >= low) & (confidences < high)
        count = mask.sum()
        if count == 0:
            continue
        bin_acc = corrects[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (count / total) * abs(bin_acc - bin_conf)
    return round(ece, 4) 

ece_results = [{"model": model, "ece": compute_ece(df[df["model"] == model])} for model in df["model"].unique()]
summary = summary.merge(pd.DataFrame(ece_results), on="model")

# brier Score
brier = df.groupby("model").apply(lambda g: ((g["confidence"] / 100 - g["is_correct"].astype(int))**2).mean()).reset_index()
brier.columns = ["model", "brier_score"]
summary = summary.merge(brier, on="model")


#Overconfidence Rate
def overconfidence_rate(group):
    wrong = group[~group["is_correct"]]
    return (wrong["confidence"] >= 80).mean() if len(wrong) > 0 else 0.0

overconf = df.groupby("model").apply(overconfidence_rate).reset_index(name="overconfidence_rate")
summary = summary.merge(overconf, on="model")


# ======================== SIGNIFICANCE (MCNEMAR vs GROK) =====================
grok_votes = votes[votes["model"] == "Grok-4.1-Fast"].rename(columns={"is_correct":"grok_correct"})

summary["p_value_vs_grok"] = np.nan
for model in summary["model"]:
    if model == "Grok-4.1-Fast":
        continue
    model_votes = votes[votes["model"] == model].rename(columns={"is_correct":"model_correct"})
    merged = grok_votes.merge(model_votes, on=["subject", "question"])
    table = pd.crosstab(merged["grok_correct"], merged["model_correct"])
    if table.shape == (2, 2):
        try:
            result = smct.mcnemar(table.values, exact=True)
            summary.loc[summary["model"] == model, "p_value_vs_grok"] = round(result.pvalue, 4)
        except:
            pass

# ======================== OUTPUT =====================

print("\n=== Final Summary ===")
print(summary.to_string(index=False))

acc_by_subject = votes.groupby(["model", "subject"])["is_correct"].mean().reset_index()
acc_by_subject.to_csv(RESULTS_DIR / "accuracy_by_subject.csv")
summary.to_csv(RESULTS_DIR / "summary.csv")


print(f"\nSaved summary.csv and accuracy_by_subject.csv")