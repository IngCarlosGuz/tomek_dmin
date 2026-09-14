from dataclasses import dataclass, asdict
from typing import Tuple

@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    n_samples: int = 2000
    n_features: int = 10
    n_informative: int = 6
    n_redundant: int = 2
    class_weights: Tuple[float, float, float] = (0.982, 0.010, 0.008)
    rare_class: int = 2
    class_sep: float = 0.65
    outlier_row_rate: float = 0.02
    outlier_feature_rate: float = 0.20
    outlier_strength: float = 6.0
    min_rare_per_test_fold: int = 5
    max_folds: int = 5
    n_repeats: int = 10
    k_kdn: int = 5
    primary_distance: str = "euclidean"
    metrics: Tuple[str, ...] = (
        "euclidean",
        "cityblock",
        "minkowski_p3",
        "chebyshev",
        "canberra",
        "mahalanobis",
    )
    methods: Tuple[str, ...] = (
        "baseline",
        "tomek",
        "random_matched",
        "smote_tomek",
    )

    def to_dict(self):
        out = asdict(self)
        out["class_weights"] = list(self.class_weights)
        out["metrics"] = list(self.metrics)
        out["methods"] = list(self.methods)
        return out
