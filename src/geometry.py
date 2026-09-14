import numpy as np
from scipy.spatial.distance import cdist
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors

def inverse_covariance_ledoit(X):
    lw = LedoitWolf().fit(X)
    return np.linalg.inv(lw.covariance_)

def distance_matrix(XA, XB, metric, VI=None):
    if metric == "euclidean":
        return cdist(XA, XB, metric="euclidean")
    if metric == "cityblock":
        return cdist(XA, XB, metric="cityblock")
    if metric == "minkowski_p3":
        return cdist(XA, XB, metric="minkowski", p=3)
    if metric == "chebyshev":
        return cdist(XA, XB, metric="chebyshev")
    if metric == "canberra":
        return cdist(XA, XB, metric="canberra")
    if metric == "mahalanobis":
        if VI is None:
            raise ValueError("Mahalanobis requiere VI.")
        return cdist(XA, XB, metric="mahalanobis", VI=VI)
    raise ValueError(f"Métrica no soportada: {metric}")

def class_centroids(X, y):
    classes = np.unique(y)
    centroids = np.vstack([X[y == c].mean(axis=0) for c in classes])
    return classes, centroids

def geometry_metrics(X, y, rare_class, metric="euclidean", VI=None):
    classes, centroids = class_centroids(X, y)

    intra = {}
    for i, c in enumerate(classes):
        d = distance_matrix(X[y == c], centroids[i:i+1], metric, VI).ravel()
        intra[int(c)] = float(np.mean(d))

    D_intra_bal = float(np.mean(list(intra.values())))

    centroid_dist = distance_matrix(centroids, centroids, metric, VI)
    upper = centroid_dist[np.triu_indices_from(centroid_dist, k=1)]
    D_inter_macro = float(np.mean(upper))

    rare_idx = np.where(classes == rare_class)[0][0]
    rare_to_all = centroid_dist[rare_idx]
    D_inter_rare = float(np.mean(np.delete(rare_to_all, rare_idx)))
    D_intra_rare = float(intra[int(rare_class)])

    return {
        "D_intra_bal": D_intra_bal,
        "D_intra_rare": D_intra_rare,
        "D_inter_macro": D_inter_macro,
        "D_inter_rare": D_inter_rare,
        "G_macro": D_inter_macro / D_intra_bal if D_intra_bal > 0 else np.nan,
        "G_rare": D_inter_rare / D_intra_rare if D_intra_rare > 0 else np.nan,
        **{f"D_intra_class_{c}": v for c, v in intra.items()},
    }

def kdn_metrics(X, y, k, rare_class):
    n = len(y)
    k_eff = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    neigh_idx = indices[:, 1:]
    neigh_labels = y[neigh_idx]
    kdn = np.mean(neigh_labels != y[:, None], axis=1)

    classes = np.unique(y)
    class_means = {int(c): float(np.mean(kdn[y == c])) for c in classes}

    return {
        "kdn_micro": float(np.mean(kdn)),
        "kdn_macro": float(np.mean(list(class_means.values()))),
        "kdn_rare": class_means[int(rare_class)],
        **{f"kdn_class_{c}": v for c, v in class_means.items()},
    }, kdn
