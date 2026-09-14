import numpy as np
from src.config import ExperimentConfig
from src.data import generate_dataset
from src.preprocessing import MedianMADScaler, apply_tomek_majority
from src.classifier import DMinClassifier

def test_dataset_shape_and_rare_class():
    cfg = ExperimentConfig(n_samples=500, class_weights=(0.98, 0.01, 0.01))
    X, y, artifact_mask = generate_dataset(cfg)
    assert X.shape == (500, cfg.n_features)
    assert len(y) == 500
    assert artifact_mask.shape == (500,)
    assert set(np.unique(y)) == {0, 1, 2}

def test_scaler_finite():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 5))
    Z = MedianMADScaler().fit_transform(X)
    assert np.isfinite(Z).all()

def test_tomek_preserves_rare_when_majority_only():
    cfg = ExperimentConfig(n_samples=600, class_weights=(0.97, 0.02, 0.01))
    X, y, _ = generate_dataset(cfg)
    Z = MedianMADScaler().fit_transform(X)
    _, y2, _ = apply_tomek_majority(Z, y)
    assert np.sum(y2 == cfg.rare_class) == np.sum(y == cfg.rare_class)

def test_dmin_predicts_known_labels_shape():
    cfg = ExperimentConfig(n_samples=500, class_weights=(0.98, 0.01, 0.01))
    X, y, _ = generate_dataset(cfg)
    Z = MedianMADScaler().fit_transform(X)
    clf = DMinClassifier(metric="euclidean", rare_class=cfg.rare_class).fit(Z, y)
    pred = clf.predict(Z[:25])
    score = clf.rare_score(Z[:25])
    assert pred.shape == (25,)
    assert score.shape == (25,)
    assert np.isfinite(score).all()
