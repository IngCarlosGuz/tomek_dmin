import numpy as np
import pandas as pd

from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

class MedianMADScaler:
    def __init__(self, consistency_constant=1.4826):
        self.c = consistency_constant

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.center_ = np.median(X, axis=0)
        mad = np.median(np.abs(X - self.center_), axis=0)
        scale = self.c * mad

        q75 = np.percentile(X, 75, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        iqr_sigma = (q75 - q25) / 1.349

        scale = np.where(scale > 1e-12, scale, iqr_sigma)
        scale = np.where(scale > 1e-12, scale, 1.0)

        self.scale_ = scale
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.center_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

def apply_tomek_majority(X, y):
    sampler = TomekLinks(sampling_strategy="majority")
    X_new, y_new = sampler.fit_resample(X, y)

    kept = np.asarray(sampler.sample_indices_)
    removed_mask = np.ones(len(y), dtype=bool)
    removed_mask[kept] = False
    removed_idx = np.where(removed_mask)[0]
    return X_new, y_new, removed_idx

def random_matched_undersample(X, y, n_remove, random_state):
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    majority_class = classes[np.argmax(counts)]
    idx_majority = np.where(y == majority_class)[0]

    n_remove = min(int(n_remove), max(0, len(idx_majority) - 1))
    if n_remove <= 0:
        return X.copy(), y.copy(), np.array([], dtype=int)

    removed_idx = rng.choice(idx_majority, size=n_remove, replace=False)
    keep = np.ones(len(y), dtype=bool)
    keep[removed_idx] = False
    return X[keep], y[keep], removed_idx

def apply_smote_tomek(X, y, random_state):
    counts = pd.Series(y).value_counts()
    min_count = int(counts.min())
    k_smote = max(1, min(5, min_count - 1))

    smote = SMOTE(
        sampling_strategy="not majority",
        k_neighbors=k_smote,
        random_state=random_state,
    )

    sampler = SMOTETomek(
        smote=smote,
        tomek=TomekLinks(sampling_strategy="all"),
        random_state=random_state,
    )
    return sampler.fit_resample(X, y)
