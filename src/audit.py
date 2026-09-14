import numpy as np
from sklearn.neighbors import NearestNeighbors
from .geometry import class_centroids, distance_matrix

def point_audit_metrics(X, y, point_indices, artifact_mask_local=None, k=5):
    if len(point_indices) == 0:
        return []

    classes, centroids = class_centroids(X, y)

    # kDN for all points (vectorized neighbors)
    k_eff = min(k, len(y) - 1)
    nn_all = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean").fit(X)
    _, idx_all = nn_all.kneighbors(X)
    neigh = idx_all[:, 1:]
    kdn_all = np.mean(y[neigh] != y[:, None], axis=1)

    # distance to own/rival centroid (matrix-wise cdist)
    D_cent = distance_matrix(X, centroids, metric="euclidean")
    class_to_col = {int(c): i for i, c in enumerate(classes)}

    rows = []
    for idx in point_indices:
        own_col = class_to_col[int(y[idx])]
        own_d = float(D_cent[idx, own_col])
        rival_d = float(np.min(np.delete(D_cent[idx], own_col)))
        margin = rival_d - own_d

        same_idx = np.where(y == y[idx])[0]
        same_idx = same_idx[same_idx != idx]
        if len(same_idx):
            d_same = distance_matrix(
                X[idx:idx+1],
                X[same_idx],
                metric="euclidean",
            ).ravel()
            kk = min(k, len(d_same))
            mean_same_knn = float(np.partition(d_same, kk - 1)[:kk].mean())
        else:
            mean_same_knn = np.nan

        density = (
            1.0 / (mean_same_knn + 1e-12)
            if np.isfinite(mean_same_knn)
            else np.nan
        )

        rows.append({
            "local_index": int(idx),
            "class": int(y[idx]),
            "kdn": float(kdn_all[idx]),
            "distance_own_centroid": own_d,
            "distance_nearest_rival_centroid": rival_d,
            "centroid_margin": margin,
            "mean_same_class_knn_distance": mean_same_knn,
            "local_density_proxy": density,
            "is_injected_artifact": (
                bool(artifact_mask_local[idx])
                if artifact_mask_local is not None
                else False
            ),
        })
    return rows
