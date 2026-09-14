import numpy as np
from sklearn.datasets import make_classification

def generate_dataset(cfg):
    X, y = make_classification(
        n_samples=cfg.n_samples,
        n_features=cfg.n_features,
        n_informative=cfg.n_informative,
        n_redundant=cfg.n_redundant,
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=1,
        weights=list(cfg.class_weights),
        class_sep=cfg.class_sep,
        flip_y=0.0,
        shuffle=False,
        random_state=cfg.seed,
    )

    rng = np.random.default_rng(cfg.seed)
    n_outlier_rows = max(1, int(round(cfg.n_samples * cfg.outlier_row_rate)))
    outlier_rows = rng.choice(cfg.n_samples, size=n_outlier_rows, replace=False)
    n_outlier_features = max(1, int(round(cfg.n_features * cfg.outlier_feature_rate)))
    feature_sd = X.std(axis=0, ddof=1)

    for row in outlier_rows:
        feat_idx = rng.choice(cfg.n_features, size=n_outlier_features, replace=False)
        perturb = (
            rng.standard_t(df=2, size=n_outlier_features)
            * cfg.outlier_strength
            * feature_sd[feat_idx]
        )
        X[row, feat_idx] += perturb

    artifact_mask = np.zeros(cfg.n_samples, dtype=bool)
    artifact_mask[outlier_rows] = True
    return X, y, artifact_mask
