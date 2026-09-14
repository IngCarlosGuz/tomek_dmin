from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def save_class_distribution(y, out_dir):
    counts = pd.Series(y).value_counts().sort_index()
    fig = plt.figure(figsize=(7, 5))
    counts.plot(kind="bar")
    plt.xlabel("Clase")
    plt.ylabel("Número de observaciones")
    plt.title("Distribución multiclase del dataset sintético")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    path = Path(out_dir) / "fig_01_distribucion_clases.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path

def save_2d_view(X, y, rare_class, out_dir):
    markers = {0: "o", 1: "s", 2: "^"}
    fig = plt.figure(figsize=(8, 6))
    for c in np.unique(y):
        plt.scatter(
            X[y == c, 0],
            X[y == c, 1],
            s=18 if c != rare_class else 55,
            alpha=0.25 if c == 0 else 0.85,
            marker=markers.get(int(c), "o"),
            label=f"Clase {c}",
        )
    plt.xlabel("Característica original 1 — escala Mediana/MAD")
    plt.ylabel("Característica original 2 — escala Mediana/MAD")
    plt.title("Vista 2D del espacio original — sin PCA")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    path = Path(out_dir) / "fig_02_vista_2D_sin_PCA.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path

def save_delta_plot(pairs, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (name, pair_df) in zip(axes.ravel(), pairs.items()):
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.plot(pair_df["repeat"], pair_df["delta"], marker="o")
        ax.set_title(f"Delta {name}")
        ax.set_xlabel("Repetición")
        ax.set_ylabel("Post - Pre")
        ax.grid(alpha=0.2)
    plt.tight_layout()
    path = Path(out_dir) / "fig_04_deltas_hipotesis.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path

def save_kdn_plot(kdn_repeat_df, out_dir):
    plot_df = kdn_repeat_df[
        kdn_repeat_df["method"].isin(["baseline", "tomek"])
    ]
    fig = plt.figure(figsize=(8, 5))
    for method, grp in plot_df.groupby("method"):
        plt.plot(grp["repeat"], grp["kdn_macro"], marker="o", label=method)
    plt.xlabel("Repetición")
    plt.ylabel("kDN macro")
    plt.title("Ambigüedad local pre/post Tomek")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    path = Path(out_dir) / "fig_05_kdn_macro.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path
