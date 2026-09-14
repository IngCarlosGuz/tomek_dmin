import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    accuracy_score,
)
from statsmodels.stats.multitest import multipletests

def evaluation_metrics(y_true, y_pred, rare_scores, rare_class):
    y_true_bin = (y_true == rare_class).astype(int)
    y_pred_bin = (y_pred == rare_class).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    ba_rare = 0.5 * (sensitivity + specificity)

    try:
        roc_auc = roc_auc_score(y_true_bin, rare_scores)
    except ValueError:
        roc_auc = np.nan

    try:
        pr_auc = average_precision_score(y_true_bin, rare_scores)
    except ValueError:
        pr_auc = np.nan

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "npv": float(npv),
        "balanced_accuracy_rare": float(ba_rare),
        "roc_auc_rare": float(roc_auc),
        "pr_auc_rare": float(pr_auc),
        "accuracy_multiclass": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy_multiclass": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
    }

def safe_wilcoxon_greater(post, pre):
    delta = np.asarray(post) - np.asarray(pre)
    if np.allclose(delta, 0):
        return np.nan, 1.0
    try:
        stat, p = wilcoxon(
            post,
            pre,
            alternative="greater",
            zero_method="wilcox",
        )
        return stat, p
    except ValueError:
        return np.nan, 1.0

def holm_table(rows, alpha=0.05):
    import pandas as pd
    df = pd.DataFrame(rows)
    reject, p_holm, _, _ = multipletests(
        df["p_raw"],
        alpha=alpha,
        method="holm",
    )
    df["p_holm"] = p_holm
    df["reject_H0_alpha_0.05"] = reject
    return df

def bootstrap_delta_ci(deltas, n_boot=20000, seed=42):
    deltas = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(deltas, size=len(deltas), replace=True)
        boot[b] = np.mean(sample)
    return {
        "mean_delta": float(np.mean(deltas)),
        "IC95_low": float(np.quantile(boot, 0.025)),
        "IC95_high": float(np.quantile(boot, 0.975)),
    }
