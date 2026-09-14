import numpy as np
from .geometry import class_centroids, distance_matrix, inverse_covariance_ledoit

class DMinClassifier:
    def __init__(self, metric="euclidean", rare_class=2):
        self.metric = metric
        self.rare_class = rare_class

    def fit(self, X, y):
        self.classes_, self.centroids_ = class_centroids(X, y)
        self.VI_ = None
        if self.metric == "mahalanobis":
            self.VI_ = inverse_covariance_ledoit(X)
        return self

    def _distances(self, X):
        return distance_matrix(
            X,
            self.centroids_,
            metric=self.metric,
            VI=self.VI_,
        )

    def predict(self, X):
        D = self._distances(X)
        return self.classes_[np.argmin(D, axis=1)]

    def rare_score(self, X):
        D = self._distances(X)
        rare_idx = np.where(self.classes_ == self.rare_class)[0][0]
        rare_d = D[:, rare_idx]
        other_cols = np.arange(D.shape[1]) != rare_idx
        nearest_other = D[:, other_cols].min(axis=1)
        return nearest_other - rare_d
