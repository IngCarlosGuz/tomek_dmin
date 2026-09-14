from pathlib import Path
import json
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from .data import generate_dataset
from .preprocessing import (
    MedianMADScaler,
    apply_tomek_majority,
    random_matched_undersample,
    apply_smote_tomek,
)
from .geometry import (
    inverse_covariance_ledoit,
    geometry_metrics,
    kdn_metrics,
)
from .classifier import DMinClassifier
from .evaluation import (
    evaluation_metrics,
    safe_wilcoxon_greater,
    holm_table,
    bootstrap_delta_ci,
)
from .audit import point_audit_metrics
from .plots import (
    save_class_distribution,
    save_2d_view,
    save_delta_plot,
    save_kdn_plot,
)

def determine_folds(y, rare_class, min_rare_per_test_fold, max_folds):
    n_rare = int(np.sum(y == rare_class))
    K = min(max_folds, n_rare // min_rare_per_test_fold)
    return max(2, K)

def _paired_geo(geo_repeat_df, variable, metric):
    pre = geo_repeat_df[
        (geo_repeat_df["method"] == "baseline")
        & (geo_repeat_df["metric"] == metric)
    ][["repeat", variable]].rename(columns={variable: "pre"})
    post = geo_repeat_df[
        (geo_repeat_df["method"] == "tomek")
        & (geo_repeat_df["metric"] == metric)
    ][["repeat", variable]].rename(columns={variable: "post"})
    out = pre.merge(post, on="repeat")
    out["delta"] = out["post"] - out["pre"]
    return out

def _paired_class(class_repeat_df, variable, metric):
    pre = class_repeat_df[
        (class_repeat_df["method"] == "baseline")
        & (class_repeat_df["metric"] == metric)
    ][["repeat", variable]].rename(columns={variable: "pre"})
    post = class_repeat_df[
        (class_repeat_df["method"] == "tomek")
        & (class_repeat_df["metric"] == metric)
    ][["repeat", variable]].rename(columns={variable: "post"})
    out = pre.merge(post, on="repeat")
    out["delta"] = out["post"] - out["pre"]
    return out

def _paired_kdn(kdn_repeat_df, variable):
    pre = kdn_repeat_df[kdn_repeat_df["method"] == "baseline"][
        ["repeat", variable]
    ].rename(columns={variable: "pre"})
    post = kdn_repeat_df[kdn_repeat_df["method"] == "tomek"][
        ["repeat", variable]
    ].rename(columns={variable: "post"})
    out = pre.merge(post, on="repeat")
    out["delta"] = out["post"] - out["pre"]
    return out

def run_experiment(cfg, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, artifact_mask = generate_dataset(cfg)
    n_rare = int(np.sum(y == cfg.rare_class))
    K = determine_folds(
        y,
        cfg.rare_class,
        cfg.min_rare_per_test_fold,
        cfg.max_folds,
    )

    config_dict = cfg.to_dict()
    config_dict["folds"] = K
    with open(output_dir / "00_configuracion_experimento.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)

    counts = pd.Series(y).value_counts().sort_index()
    dataset_summary = pd.DataFrame({
        "clase": counts.index,
        "n": counts.values,
        "proporcion": (counts / len(y)).values,
    })
    dataset_summary.to_csv(output_dir / "00_resumen_dataset.csv", index=False)

    cv = RepeatedStratifiedKFold(
        n_splits=K,
        n_repeats=cfg.n_repeats,
        random_state=cfg.seed,
    )

    fold_audit_rows = []
    geometry_fold_rows = []
    kdn_fold_rows = []
    removed_audit_rows = []
    oof_store = {}

    split_id = 0

    for train_idx, test_idx in cv.split(X, y):
        split_id += 1
        repeat_id = (split_id - 1) // K + 1
        fold_id = (split_id - 1) % K + 1

        Xtr_raw, Xte_raw = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        art_train = artifact_mask[train_idx]

        scaler = MedianMADScaler()
        Xtr = scaler.fit_transform(Xtr_raw)
        Xte = scaler.transform(Xte_raw)

        X_tomek, y_tomek, tomek_removed_idx = apply_tomek_majority(Xtr, ytr)
        X_random, y_random, random_removed_idx = random_matched_undersample(
            Xtr,
            ytr,
            n_remove=len(tomek_removed_idx),
            random_state=cfg.seed + split_id,
        )
        X_smote, y_smote = apply_smote_tomek(
            Xtr,
            ytr,
            random_state=cfg.seed + split_id,
        )

        # Auditoría detallada de eliminados Tomek y random
        for method_name, removed_idx in (
            ("tomek", tomek_removed_idx),
            ("random_matched", random_removed_idx),
        ):
            detail_rows = point_audit_metrics(
                Xtr,
                ytr,
                removed_idx,
                artifact_mask_local=art_train,
                k=cfg.k_kdn,
            )
            for row in detail_rows:
                row.update({
                    "split_id": split_id,
                    "repeat": repeat_id,
                    "fold": fold_id,
                    "method": method_name,
                    "global_index": int(train_idx[row["local_index"]]),
                })
                removed_audit_rows.append(row)

        method_data = {
            "baseline": (Xtr, ytr),
            "tomek": (X_tomek, y_tomek),
            "random_matched": (X_random, y_random),
            "smote_tomek": (X_smote, y_smote),
        }

        fold_audit_rows.append({
            "split_id": split_id,
            "repeat": repeat_id,
            "fold": fold_id,
            "n_train": len(ytr),
            "n_test": len(yte),
            "rare_train": int(np.sum(ytr == cfg.rare_class)),
            "rare_test": int(np.sum(yte == cfg.rare_class)),
            "class0_train": int(np.sum(ytr == 0)),
            "class1_train": int(np.sum(ytr == 1)),
            "class2_train": int(np.sum(ytr == 2)),
            "tomek_removed_total": int(len(tomek_removed_idx)),
            "tomek_removed_rare": (
                int(np.sum(ytr[tomek_removed_idx] == cfg.rare_class))
                if len(tomek_removed_idx) else 0
            ),
            "random_removed_total": int(len(random_removed_idx)),
        })

        for method, (Xm, ym) in method_data.items():
            kdn, _ = kdn_metrics(
                Xm,
                ym,
                k=cfg.k_kdn,
                rare_class=cfg.rare_class,
            )
            kdn_fold_rows.append({
                "split_id": split_id,
                "repeat": repeat_id,
                "fold": fold_id,
                "method": method,
                **kdn,
            })

            VI = inverse_covariance_ledoit(Xm)

            for metric in cfg.metrics:
                this_VI = VI if metric == "mahalanobis" else None

                geo = geometry_metrics(
                    Xm,
                    ym,
                    rare_class=cfg.rare_class,
                    metric=metric,
                    VI=this_VI,
                )

                geometry_fold_rows.append({
                    "split_id": split_id,
                    "repeat": repeat_id,
                    "fold": fold_id,
                    "method": method,
                    "metric": metric,
                    "n_train_after": len(ym),
                    "class0_after": int(np.sum(ym == 0)),
                    "class1_after": int(np.sum(ym == 1)),
                    "class2_after": int(np.sum(ym == 2)),
                    **geo,
                })

                clf = DMinClassifier(
                    metric=metric,
                    rare_class=cfg.rare_class,
                ).fit(Xm, ym)

                pred = clf.predict(Xte)
                score = clf.rare_score(Xte)

                key = (repeat_id, method, metric)
                if key not in oof_store:
                    oof_store[key] = {"y_true": [], "y_pred": [], "score": []}

                oof_store[key]["y_true"].append(yte)
                oof_store[key]["y_pred"].append(pred)
                oof_store[key]["score"].append(score)

    audit_df = pd.DataFrame(fold_audit_rows)
    geo_fold_df = pd.DataFrame(geometry_fold_rows)
    kdn_fold_df = pd.DataFrame(kdn_fold_rows)
    removed_audit_df = pd.DataFrame(removed_audit_rows)

    audit_df.to_csv(output_dir / "01_auditoria_folds.csv", index=False)
    geo_fold_df.to_csv(output_dir / "02_geometria_por_fold.csv", index=False)
    kdn_fold_df.to_csv(output_dir / "03_kdn_por_fold.csv", index=False)
    removed_audit_df.to_csv(output_dir / "03b_auditoria_puntos_eliminados.csv", index=False)

    classification_repeat_rows = []
    for (repeat_id, method, metric), store in oof_store.items():
        y_true = np.concatenate(store["y_true"])
        y_pred = np.concatenate(store["y_pred"])
        scores = np.concatenate(store["score"])
        mets = evaluation_metrics(
            y_true,
            y_pred,
            scores,
            rare_class=cfg.rare_class,
        )
        classification_repeat_rows.append({
            "repeat": repeat_id,
            "method": method,
            "metric": metric,
            **mets,
        })

    class_repeat_df = pd.DataFrame(classification_repeat_rows)
    class_repeat_df.to_csv(
        output_dir / "04_clasificacion_oof_por_repeticion.csv",
        index=False,
    )

    geo_repeat_df = (
        geo_fold_df
        .groupby(["repeat", "method", "metric"], as_index=False)[
            [
                "D_intra_bal",
                "D_intra_rare",
                "D_inter_macro",
                "D_inter_rare",
                "G_macro",
                "G_rare",
                "D_intra_class_0",
                "D_intra_class_1",
                "D_intra_class_2",
            ]
        ]
        .mean()
    )

    kdn_repeat_df = (
        kdn_fold_df
        .groupby(["repeat", "method"], as_index=False)[
            [
                "kdn_micro",
                "kdn_macro",
                "kdn_rare",
                "kdn_class_0",
                "kdn_class_1",
                "kdn_class_2",
            ]
        ]
        .mean()
    )

    geo_repeat_df.to_csv(
        output_dir / "05_geometria_por_repeticion.csv",
        index=False,
    )
    kdn_repeat_df.to_csv(
        output_dir / "06_kdn_por_repeticion.csv",
        index=False,
    )

    pairs = {
        "G_rare": _paired_geo(geo_repeat_df, "G_rare", cfg.primary_distance),
        "sensitivity": _paired_class(class_repeat_df, "sensitivity", cfg.primary_distance),
        "balanced_accuracy_rare": _paired_class(
            class_repeat_df,
            "balanced_accuracy_rare",
            cfg.primary_distance,
        ),
        "roc_auc_rare": _paired_class(
            class_repeat_df,
            "roc_auc_rare",
            cfg.primary_distance,
        ),
    }

    hyp_rows = []
    for variable, pair_df in pairs.items():
        stat, p = safe_wilcoxon_greater(pair_df["post"], pair_df["pre"])
        hyp_rows.append({
            "variable": variable,
            "n_pairs": len(pair_df),
            "mean_pre": pair_df["pre"].mean(),
            "mean_post": pair_df["post"].mean(),
            "mean_delta": pair_df["delta"].mean(),
            "median_delta": pair_df["delta"].median(),
            "wilcoxon_stat": stat,
            "p_raw": p,
        })

    hyp_df = holm_table(hyp_rows, alpha=0.05)
    hyp_df.to_csv(output_dir / "07_hipotesis_wilcoxon_holm.csv", index=False)

    boot_rows = []
    for variable, pair_df in pairs.items():
        ci = bootstrap_delta_ci(
            pair_df["delta"].values,
            n_boot=20000,
            seed=cfg.seed,
        )
        boot_rows.append({"variable": variable, **ci})
    boot_df = pd.DataFrame(boot_rows)

    final_hyp_df = hyp_df.merge(
        boot_df,
        on=["variable", "mean_delta"],
        how="left",
    )
    final_hyp_df["IC95_above_zero"] = final_hyp_df["IC95_low"] > 0
    final_hyp_df["supported_by_both_criteria"] = (
        (final_hyp_df["p_holm"] < 0.05)
        & (final_hyp_df["IC95_above_zero"])
    )
    final_hyp_df.to_csv(output_dir / "08_hipotesis_finales.csv", index=False)

    kdn_change_rows = []
    for variable in ["kdn_micro", "kdn_macro", "kdn_rare"]:
        p = _paired_kdn(kdn_repeat_df, variable)
        kdn_change_rows.append({
            "variable": variable,
            "mean_pre": p["pre"].mean(),
            "mean_post": p["post"].mean(),
            "mean_delta": p["delta"].mean(),
        })
    kdn_change_df = pd.DataFrame(kdn_change_rows)
    kdn_change_df.to_csv(output_dir / "09_cambios_kdn.csv", index=False)

    primary_results = class_repeat_df[
        class_repeat_df["metric"] == cfg.primary_distance
    ]
    method_summary = (
        primary_results
        .groupby("method")[
            [
                "sensitivity",
                "specificity",
                "precision",
                "npv",
                "balanced_accuracy_rare",
                "roc_auc_rare",
                "pr_auc_rare",
                "accuracy_multiclass",
                "balanced_accuracy_multiclass",
            ]
        ]
        .agg(["mean", "std", "median"])
        .round(6)
    )
    method_summary.to_csv(output_dir / "10_resumen_metodos_euclidiana.csv")

    metric_summary = (
        class_repeat_df
        .groupby(["method", "metric"])[
            [
                "sensitivity",
                "specificity",
                "balanced_accuracy_rare",
                "roc_auc_rare",
                "pr_auc_rare",
                "balanced_accuracy_multiclass",
            ]
        ]
        .agg(["mean", "std"])
        .round(6)
    )
    metric_summary.to_csv(output_dir / "11_resumen_por_metrica.csv")

    # Resumen descriptivo de puntos eliminados
    if not removed_audit_df.empty:
        removed_summary = (
            removed_audit_df
            .groupby("method")
            .agg(
                events=("global_index", "size"),
                unique_points=("global_index", "nunique"),
                artifacts=("is_injected_artifact", "sum"),
                mean_kdn=("kdn", "mean"),
                median_kdn=("kdn", "median"),
                prop_kdn_ge_020=("kdn", lambda s: float(np.mean(s >= 0.20))),
                mean_own_centroid=("distance_own_centroid", "mean"),
                mean_rival_centroid=("distance_nearest_rival_centroid", "mean"),
                mean_margin=("centroid_margin", "mean"),
                prop_margin_lt_0=("centroid_margin", lambda s: float(np.mean(s < 0))),
                mean_same_class_knn=("mean_same_class_knn_distance", "mean"),
                mean_density=("local_density_proxy", "mean"),
            )
            .reset_index()
        )
        removed_summary.to_csv(
            output_dir / "11b_resumen_puntos_eliminados.csv",
            index=False,
        )

    for name, pair_df in pairs.items():
        pair_df.to_csv(output_dir / f"pares_{name}.csv", index=False)

    # Figuras
    scaler_vis = MedianMADScaler()
    X_vis = scaler_vis.fit_transform(X)
    save_class_distribution(y, output_dir)
    save_2d_view(X_vis, y, cfg.rare_class, output_dir)
    save_delta_plot(pairs, output_dir)
    save_kdn_plot(kdn_repeat_df, output_dir)

    # Checks
    assert audit_df["rare_test"].min() >= 1
    assert audit_df["tomek_removed_rare"].max() == 0
    for name, p in pairs.items():
        assert len(p) == cfg.n_repeats
        assert not p[["pre", "post"]].isna().any().any(), f"{name} contiene NaN"

    # Texto resumen
    lines = [
        "RESUMEN AUTOMÁTICO DEL EXPERIMENTO",
        "=" * 60,
        f"N = {cfg.n_samples}",
        f"d = {cfg.n_features}",
        f"K = {K}",
        f"R = {cfg.n_repeats}",
        f"Casos raros = {n_rare}",
        f"Prevalencia rara = {n_rare/cfg.n_samples:.4%}",
        "",
        "AUDITORÍA TOMЕK",
        f"Media eliminados por fold = {audit_df['tomek_removed_total'].mean():.3f}",
        f"Máximo de raros eliminados = {audit_df['tomek_removed_rare'].max()}",
        "",
        "HIPÓTESIS CONFIRMATORIAS",
    ]
    for _, row in final_hyp_df.iterrows():
        lines.append(
            f"{row['variable']}: "
            f"pre={row['mean_pre']:.6f}, "
            f"post={row['mean_post']:.6f}, "
            f"delta={row['mean_delta']:.6f}, "
            f"p_Holm={row['p_holm']:.6f}, "
            f"IC95=[{row['IC95_low']:.6f}, {row['IC95_high']:.6f}], "
            f"respaldo={bool(row['supported_by_both_criteria'])}"
        )
    lines += ["", "kDN"]
    for _, row in kdn_change_df.iterrows():
        lines.append(
            f"{row['variable']}: "
            f"pre={row['mean_pre']:.6f}, "
            f"post={row['mean_post']:.6f}, "
            f"delta={row['mean_delta']:.6f}"
        )

    with open(output_dir / "12_resumen_automatico.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    zip_path = output_dir.parent / f"{output_dir.name}_completo.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for path in output_dir.rglob("*"):
            if path.is_file():
                z.write(path, arcname=path.name)

    return {
        "output_dir": output_dir,
        "zip_path": zip_path,
        "config": config_dict,
        "audit_df": audit_df,
        "geo_repeat_df": geo_repeat_df,
        "kdn_repeat_df": kdn_repeat_df,
        "class_repeat_df": class_repeat_df,
        "final_hyp_df": final_hyp_df,
        "removed_audit_df": removed_audit_df,
    }
