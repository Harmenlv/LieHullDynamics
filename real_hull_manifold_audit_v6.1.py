#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
REAL SAILBOAT HULL DATASET AUDIT V3.2
=====================================

Purpose
-------
V3.2 is the robustness stage built on the verified V2/V3.1 data logic.

Core data structure
-------------------
1018 experimental records
        ->
70 independent hulls identified by (Serie, Sysser)
        ->
16 geometric descriptors
        ->
robust intrinsic-dimension analysis

IMPORTANT
---------
1. The 1018 experimental records are NOT treated as 1018 independent hulls.
2. Fn and Rn are operating-condition variables, not geometry.
3. Rt*10^3/Delta is hydrodynamic response, not geometry.
4. Intrinsic dimension is estimated from the 70 independent hulls.
5. V3.1 naive bootstrap-with-replacement is NOT used for nearest-neighbour ID estimators because duplicate hulls create zero distances. V3.2 replaces it with leave-one-hull-out jackknife and subsampling without replacement.
6. PCA is a linear baseline, not evidence of nonlinear intrinsic dimension.
7. Isomap/LLE are used as nonlinear manifold diagnostics, not as sole proof.
8. No deep learning, GPU, UMAP, VAE, or autoencoder is required.

Input
-----
Default:
    Sailboat Hull Resistance Dataset V01.csv

The CSV is expected to be semicolon-delimited, as in the V2 dataset.

Outputs
-------
results_v32/
    00_summary.txt
    01_dataset_overview.csv
    02_series_summary.csv
    03_geometry_statistics_1018.csv
    04_hull_geometry.csv
    05_hull_geometry_consistency.csv
    06_hull_pearson_correlation.csv
    07_hull_spearman_correlation.csv
    08_hull_mutual_information.csv
    09_pca_variance.csv
    10_pca_loadings.csv
    11_intrinsic_dimension.csv
    12_bootstrap_intrinsic_dimension.csv
    13_bootstrap_summary.csv
    14_subsample_intrinsic_dimension.csv
    15_isomap_quality.csv
    16_lle_quality.csv
    17_geometry_distance_summary.csv
    18_hull_pca_coordinates.csv
    19_hull_response_summary.csv
    figures/
        01_pca_scree.png
        02_pca_cumulative_variance.png
        03_hull_pearson.png
        04_hull_spearman.png
        05_intrinsic_dimension.png
        06_bootstrap_dimension.png
        07_pca_2d.png
        08_pca_3d.png
        09_isomap_2d.png
        10_lle_2d.png
"""

from __future__ import annotations

import argparse
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# =============================================================================
# Fixed dataset definition from V2
# =============================================================================

EXPECTED_COLUMNS = [
    "Serie",
    "Sysser",
    "Rt*10^3/Delta",
    "Rn",
    "Fn",
    "Cp",
    "Cm",
    "Cb",
    "Cw",
    "Lwl/Bwl",
    "Bwl/Tc",
    "Lwl/Tc",
    "Lwl/Vol^(1/3)",
    "Lcb/Lwl",
    "Lcf/Lwl",
    "Lcb/Lcf",
    "Sc/Vol^(2/3)",
    "Aw/Vol^(2/3)",
    "Sc/Aw",
    "Sc/Ax",
    "Ax/Aw",
]

GEOMETRY_COLUMNS = [
    "Cp",
    "Cm",
    "Cb",
    "Cw",
    "Lwl/Bwl",
    "Bwl/Tc",
    "Lwl/Tc",
    "Lwl/Vol^(1/3)",
    "Lcb/Lwl",
    "Lcf/Lwl",
    "Lcb/Lcf",
    "Sc/Vol^(2/3)",
    "Aw/Vol^(2/3)",
    "Sc/Aw",
    "Sc/Ax",
    "Ax/Aw",
]

RESPONSE_COLUMN = "Rt*10^3/Delta"
OPERATING_COLUMNS = ["Fn", "Rn"]
ID_COLUMNS = ["Serie", "Sysser"]

EXPECTED_RECORDS = 1018
EXPECTED_HULLS = 70
EXPECTED_GEOMETRY_DIM = 16


# =============================================================================
# Utility
# =============================================================================

def safe_name(x: str) -> str:
    return (
        str(x)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def detect_separator(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        first = f.readline()

    candidates = {
        ";": first.count(";"),
        ",": first.count(","),
        "\t": first.count("\t"),
    }

    return max(candidates, key=candidates.get)


def save_heatmap(matrix, title, path, vmin=None, vmax=None):
    fig_w = max(10, min(18, 0.58 * matrix.shape[1]))
    fig_h = max(8, min(16, 0.58 * matrix.shape[0]))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        matrix.values,
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(matrix.columns, rotation=70, ha="right", fontsize=8)
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_title(title)

    if matrix.shape[0] <= 20 and matrix.shape[1] <= 20:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iloc[i, j]
                if np.isfinite(value):
                    ax.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                    )

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Intrinsic dimension estimators
# =============================================================================

def mle_intrinsic_dimension(X, k=10):
    """
    Levina-Bickel-style local MLE.

    The estimator is reported as a diagnostic estimate.
    For small n=70, results must be interpreted together with
    k-stability and bootstrap intervals.
    """
    X = np.asarray(X, dtype=float)

    if len(X) < k + 2:
        return np.nan

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(X)
    distances, _ = nbrs.kneighbors(X)

    distances = distances[:, 1:]
    distances = np.maximum(distances, 1e-12)

    Tk = distances[:, -1]
    logs = np.log(Tk[:, None] / distances[:, :-1])

    sums = np.sum(logs, axis=1)

    valid = np.isfinite(sums) & (sums > 0)

    if not np.any(valid):
        return np.nan

    local = (k - 1) / sums[valid]
    local = local[np.isfinite(local)]

    if len(local) == 0:
        return np.nan

    return float(np.mean(local))


def two_nn_dimension(X):
    """
    Two-NN estimator.

    For an ideal locally uniform manifold:
        E[log(r2/r1)] ~= 1/d

    A simple global estimator is:
        d = 1 / mean(log(r2/r1))

    It is used as a complementary diagnostic.
    """
    X = np.asarray(X, dtype=float)

    if len(X) < 5:
        return np.nan

    nbrs = NearestNeighbors(n_neighbors=3, metric="euclidean")
    nbrs.fit(X)
    dists, _ = nbrs.kneighbors(X)

    d1 = np.maximum(dists[:, 1], 1e-12)
    d2 = np.maximum(dists[:, 2], 1e-12)

    mu = d2 / d1
    logs = np.log(mu)

    logs = logs[np.isfinite(logs) & (logs > 0)]

    if len(logs) == 0:
        return np.nan

    return float(1.0 / np.mean(logs))


def correlation_dimension_proxy(
    X,
    max_pairs=50000,
    random_state=42,
):
    """
    Lightweight correlation-dimension diagnostic.

    This is deliberately labelled as a proxy rather than a definitive
    Grassberger-Procaccia estimator.
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(random_state)

    n = len(X)

    if n < 20:
        return np.nan

    total_pairs = n * (n - 1) // 2
    m = min(max_pairs, total_pairs)

    i = rng.integers(0, n, size=m)
    j = rng.integers(0, n, size=m)

    mask = i != j
    i = i[mask]
    j = j[mask]

    if len(i) < 100:
        return np.nan

    dist = np.linalg.norm(X[i] - X[j], axis=1)
    dist = dist[np.isfinite(dist) & (dist > 0)]

    if len(dist) < 100:
        return np.nan

    q10, q90 = np.quantile(dist, [0.10, 0.90])

    if q90 <= q10:
        return np.nan

    radii = np.geomspace(q10, q90, 30)
    C = np.array([(dist < r).mean() for r in radii])

    good = (C > 0.02) & (C < 0.80)

    if good.sum() < 5:
        return np.nan

    slope, _ = np.polyfit(
        np.log(radii[good]),
        np.log(C[good]),
        1,
    )

    return float(slope)


# =============================================================================
# Data loading
# =============================================================================

def load_dataset(input_path: Path):
    sep = detect_separator(input_path)

    print(f"Detected separator: {repr(sep)}")

    df = pd.read_csv(
        input_path,
        sep=sep,
        encoding="utf-8-sig",
    )

    df.columns = [str(c).strip() for c in df.columns]

    print(
        f"Dataset shape: "
        f"{df.shape[0]} rows × {df.shape[1]} columns"
    )

    if df.shape[0] != EXPECTED_RECORDS:
        print(
            f"WARNING: expected {EXPECTED_RECORDS} records, "
            f"found {df.shape[0]}."
        )

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(
            "Required columns are missing:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    return df


# =============================================================================
# Build independent-hull matrix
# =============================================================================

def build_independent_hulls(df, output_dir):
    print("\n" + "=" * 80)
    print("INDEPENDENT-HULL CONSTRUCTION")
    print("=" * 80)

    geometry = df[GEOMETRY_COLUMNS].copy()

    # Convert explicitly to numeric
    for c in GEOMETRY_COLUMNS:
        geometry[c] = pd.to_numeric(
            geometry[c],
            errors="coerce",
        )

    # Basic geometry statistics at record level
    stats = geometry.describe().T
    stats["missing"] = geometry.isna().sum()
    stats["missing_rate"] = geometry.isna().mean()
    stats["n_unique"] = geometry.nunique()

    stats.to_csv(
        output_dir / "03_geometry_statistics_1018.csv",
        encoding="utf-8-sig",
    )

    # Within-hull consistency
    consistency_rows = []

    grouped = df.groupby(
        ["Serie", "Sysser"],
        dropna=False,
    )

    for (serie, sysser), group in grouped:
        for c in GEOMETRY_COLUMNS:
            values = pd.to_numeric(
                group[c],
                errors="coerce",
            ).dropna().values

            consistency_rows.append(
                {
                    "Serie": serie,
                    "Sysser": sysser,
                    "geometry": c,
                    "n_records": len(values),
                    "min": float(np.min(values))
                    if len(values)
                    else np.nan,
                    "max": float(np.max(values))
                    if len(values)
                    else np.nan,
                    "mean": float(np.mean(values))
                    if len(values)
                    else np.nan,
                    "std": float(np.std(values, ddof=1))
                    if len(values) > 1
                    else 0.0,
                    "range": float(np.ptp(values))
                    if len(values)
                    else np.nan,
                }
            )

    consistency = pd.DataFrame(consistency_rows)

    consistency.to_csv(
        output_dir / "05_hull_geometry_consistency.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # One row per independent hull.
    # Geometry should be invariant across repeated experimental records.
    hull_df = (
        df.groupby(
            ["Serie", "Sysser"],
            dropna=False,
        )[GEOMETRY_COLUMNS]
        .first()
        .reset_index()
    )

    hull_df.to_csv(
        output_dir / "04_hull_geometry.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Experimental records: {len(df)}")
    print(f"Independent hulls:     {len(hull_df)}")
    print(f"Geometry dimensions:   {len(GEOMETRY_COLUMNS)}")

    if len(hull_df) != EXPECTED_HULLS:
        print(
            f"WARNING: expected {EXPECTED_HULLS} independent hulls, "
            f"found {len(hull_df)}."
        )

    return hull_df, consistency


# =============================================================================
# PCA
# =============================================================================

def run_pca(X_scaled, geometry_columns, fig_dir, output_dir):
    print("\n" + "=" * 80)
    print("PCA LINEAR BASELINE")
    print("=" * 80)

    pca = PCA()
    scores = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    pca_variance = pd.DataFrame(
        {
            "PC": np.arange(1, len(explained) + 1),
            "eigenvalue": pca.explained_variance_,
            "explained_variance_ratio": explained,
            "cumulative_variance": cumulative,
        }
    )

    pca_variance.to_csv(
        output_dir / "09_pca_variance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    loadings = pd.DataFrame(
        pca.components_.T,
        index=geometry_columns,
        columns=[f"PC{i+1}" for i in range(len(explained))],
    )

    loadings.to_csv(
        output_dir / "10_pca_loadings.csv",
        encoding="utf-8-sig",
    )

    for threshold in [0.80, 0.90, 0.95, 0.99]:
        dim = int(np.argmax(cumulative >= threshold) + 1)
        print(
            f"PCA dimensions for {threshold:.0%} variance: {dim}"
        )

    # Scree
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        np.arange(1, len(explained) + 1),
        explained,
        marker="o",
    )
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA Scree Plot: 70 Independent Hulls")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        fig_dir / "01_pca_scree.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Cumulative
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        np.arange(1, len(cumulative) + 1),
        cumulative,
        marker="o",
    )

    for threshold in [0.80, 0.90, 0.95, 0.99]:
        dim = int(np.argmax(cumulative >= threshold) + 1)
        ax.axhline(
            threshold,
            linestyle="--",
            alpha=0.4,
        )
        ax.axvline(
            dim,
            linestyle="--",
            alpha=0.4,
        )
        ax.text(
            dim,
            threshold,
            f" {threshold:.0%}: {dim}D",
        )

    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA Cumulative Variance: 70 Independent Hulls")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        fig_dir / "02_pca_cumulative_variance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 2D
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(scores[:, 0], scores[:, 1], alpha=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Independent Hulls in PCA 2D Space")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        fig_dir / "07_pca_2d.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 3D
    if scores.shape[1] >= 3:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            scores[:, 0],
            scores[:, 1],
            scores[:, 2],
            alpha=0.8,
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title("Independent Hulls in PCA 3D Space")
        fig.tight_layout()
        fig.savefig(
            fig_dir / "08_pca_3d.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    return pca, scores, pca_variance


# =============================================================================
# Correlations and mutual information
# =============================================================================

def run_dependence_analysis(X_hull, output_dir, fig_dir):
    pearson = X_hull.corr(method="pearson")
    spearman = X_hull.corr(method="spearman")

    pearson.to_csv(
        output_dir / "06_hull_pearson_correlation.csv",
        encoding="utf-8-sig",
    )

    spearman.to_csv(
        output_dir / "07_hull_spearman_correlation.csv",
        encoding="utf-8-sig",
    )

    save_heatmap(
        pearson,
        "Pearson Correlation: 70 Independent Hulls",
        fig_dir / "03_hull_pearson.png",
        vmin=-1,
        vmax=1,
    )

    save_heatmap(
        spearman,
        "Spearman Correlation: 70 Independent Hulls",
        fig_dir / "04_hull_spearman.png",
        vmin=-1,
        vmax=1,
    )

    # Mutual information
    print("Computing mutual information on 70 independent hulls...")

    values = X_hull.values
    n_features = X_hull.shape[1]

    mi = pd.DataFrame(
        np.eye(n_features),
        index=X_hull.columns,
        columns=X_hull.columns,
    )

    for j, target in enumerate(X_hull.columns):
        features = np.delete(values, j, axis=1)

        try:
            mi_values = mutual_info_regression(
                features,
                values[:, j],
                random_state=42,
                n_neighbors=min(5, len(X_hull) - 1),
            )
        except Exception as exc:
            warnings.warn(
                f"MI failed for {target}: {exc}"
            )
            continue

        other_cols = [
            c for c in X_hull.columns
            if c != target
        ]

        for c, value in zip(other_cols, mi_values):
            mi.loc[c, target] = value
            mi.loc[target, c] = value

    mi.to_csv(
        output_dir / "08_hull_mutual_information.csv",
        encoding="utf-8-sig",
    )

    return pearson, spearman, mi


# =============================================================================
# Intrinsic dimension
# =============================================================================

def estimate_intrinsic_dimension(X, random_state=42):
    rows = []

    for k in [5, 7, 10, 12, 15, 20]:
        if k < len(X) - 1:
            rows.append(
                {
                    "method": "Local MLE",
                    "parameter": f"k={k}",
                    "estimate": mle_intrinsic_dimension(X, k),
                    "n_samples": len(X),
                    "dimension": X.shape[1],
                }
            )

    rows.append(
        {
            "method": "Two-NN",
            "parameter": "k=2",
            "estimate": two_nn_dimension(X),
            "n_samples": len(X),
            "dimension": X.shape[1],
        }
    )

    rows.append(
        {
            "method": "Correlation-dimension proxy",
            "parameter": "pairwise diagnostic",
            "estimate": correlation_dimension_proxy(
                X,
                random_state=random_state,
            ),
            "n_samples": len(X),
            "dimension": X.shape[1],
        }
    )

    return rows


def run_intrinsic_dimension(X_scaled, output_dir, fig_dir):
    print("\n" + "=" * 80)
    print("INTRINSIC DIMENSION: 70 INDEPENDENT HULLS")
    print("=" * 80)

    rows = estimate_intrinsic_dimension(
        X_scaled,
        random_state=42,
    )

    result = pd.DataFrame(rows)

    print(result.to_string(index=False))

    result.to_csv(
        output_dir / "11_intrinsic_dimension.csv",
        index=False,
        encoding="utf-8-sig",
    )

    mle = result[result["method"] == "Local MLE"].copy()

    if not mle.empty:
        ks = [
            int(str(x).split("=")[1])
            for x in mle["parameter"]
        ]

        estimates = mle["estimate"].values

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            ks,
            estimates,
            marker="o",
            linewidth=2,
        )
        ax.axhline(
            16,
            linestyle="--",
            alpha=0.5,
            label="Original geometry dimension = 16",
        )
        ax.set_xlabel("Neighborhood size k")
        ax.set_ylabel("Estimated intrinsic dimension")
        ax.set_title(
            "Independent-Hull Intrinsic Dimension: Local MLE"
        )
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            fig_dir / "05_intrinsic_dimension.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    return result


# =============================================================================
# V3.2 uncertainty and stability: NO naive bootstrap with replacement
# =============================================================================

def summarize_values(values, method, dataset):
    values = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) == 0:
        return {
            "dataset": dataset, "method": method, "n": 0,
            "mean": np.nan, "std": np.nan, "median": np.nan,
            "q025": np.nan, "q975": np.nan, "q10": np.nan, "q90": np.nan,
        }
    return {
        "dataset": dataset,
        "method": method,
        "n": len(values),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "median": float(values.median()),
        "q025": float(values.quantile(0.025)),
        "q975": float(values.quantile(0.975)),
        "q10": float(values.quantile(0.10)),
        "q90": float(values.quantile(0.90)),
    }


def run_series_stability(X_scaled, hull_df, output_dir):
    """Estimate ID separately for Delft, Il Moro di Venezia and US.

    The small US subset (n=9) is retained as a diagnostic only; estimates are
    not interpreted as definitive evidence.
    """
    print("\n" + "=" * 80)
    print("SERIES-STRATIFIED INTRINSIC DIMENSION STABILITY")
    print("=" * 80)

    rows = []
    labels = []
    for serie in hull_df["Serie"].astype(str).unique():
        labels.append(serie)

    for serie in sorted(labels):
        mask = hull_df["Serie"].astype(str).values == serie
        Xs = X_scaled[mask]
        n = len(Xs)
        safe = safe_name(serie)
        if n < 5:
            rows.append({"dataset": safe, "n_samples": n, "method": "Insufficient sample", "parameter": "n<5", "estimate": np.nan})
            continue
        for k in [5, 7, 10, 12, 15, 20]:
            if k < n - 1:
                rows.append({
                    "dataset": safe, "n_samples": n,
                    "method": "Local MLE", "parameter": f"k={k}",
                    "estimate": mle_intrinsic_dimension(Xs, k=k),
                })
        rows.append({
            "dataset": safe, "n_samples": n,
            "method": "Two-NN", "parameter": "k=2",
            "estimate": two_nn_dimension(Xs),
        })

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "12_series_stratified_intrinsic_dimension.csv", index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))
    return result


def run_jackknife_hull(X_scaled, output_dir, ks=(10, 15)):
    """Leave-one-hull-out jackknife.

    Unlike ordinary bootstrap-with-replacement, every jackknife sample contains
    unique hulls, so nearest-neighbour distances cannot be made zero by
    duplicated observations.
    """
    print("\n" + "=" * 80)
    print("LEAVE-ONE-HULL-OUT JACKKNIFE")
    print("=" * 80)

    n = len(X_scaled)
    rows = []
    for i in range(n):
        sample = np.delete(X_scaled, i, axis=0)
        row = {"omitted_hull_index": i + 1}
        for k in ks:
            row[f"MLE_k{k}"] = mle_intrinsic_dimension(sample, k=k) if k < len(sample) - 1 else np.nan
        row["TWO_NN"] = two_nn_dimension(sample)
        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "13_leave_one_hull_out_jackknife.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for col in [f"MLE_k{k}" for k in ks] + ["TWO_NN"]:
        summary_rows.append(summarize_values(result[col], col, "LEAVE_ONE_HULL_OUT"))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "14_jackknife_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    return result, summary


def run_subsample_stability_v32(X_scaled, output_dir, n_subsamples=2000,
                                 fractions=(0.80, 0.90), random_state=20260910):
    """Repeated subsampling WITHOUT replacement.

    This replaces the invalid V3.1 naive bootstrap for nearest-neighbour ID
    estimators. It provides a finite-sample stability distribution without
    duplicated hulls.
    """
    print("\n" + "=" * 80)
    print("SUBSAMPLING-BASED UNCERTAINTY (WITHOUT REPLACEMENT)")
    print("=" * 80)

    rng = np.random.default_rng(random_state)
    n = len(X_scaled)
    rows = []

    for fraction in fractions:
        sample_size = max(20, int(round(n * fraction)))
        sample_size = min(sample_size, n - 1)
        for i in range(n_subsamples):
            idx = rng.choice(n, size=sample_size, replace=False)
            sample = X_scaled[idx]
            rows.append({
                "fraction": fraction,
                "iteration": i + 1,
                "sample_size": sample_size,
                "TWO_NN": two_nn_dimension(sample),
                "MLE_k10": mle_intrinsic_dimension(sample, k=10) if 10 < sample_size - 1 else np.nan,
                "MLE_k15": mle_intrinsic_dimension(sample, k=15) if 15 < sample_size - 1 else np.nan,
            })
        print(f"Subsampling fraction {fraction:.0%}: {n_subsamples}/{n_subsamples} complete")

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "15_subsample_intrinsic_dimension_v32.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for fraction in fractions:
        sub = result[result["fraction"] == fraction]
        for col in ["MLE_k10", "MLE_k15", "TWO_NN"]:
            summary_rows.append(summarize_values(sub[col], col, f"SUBSAMPLE_{fraction:.0%}"))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "16_subsample_summary_v32.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    return result, summary


def run_leave_one_variable_out(hull_df, geometry_columns, output_dir):
    """Remove each geometry descriptor once and recompute scaling + ID.

    Re-fitting the scaler for every reduced feature set avoids a hidden
    dependence on the omitted variable and makes this a genuine sensitivity
    analysis of the 16D representation.
    """
    print("\n" + "=" * 80)
    print("LEAVE-ONE-GEOMETRY-VARIABLE-OUT SENSITIVITY")
    print("=" * 80)

    base = hull_df[geometry_columns].copy()
    rows = []

    for omitted in geometry_columns:
        cols = [c for c in geometry_columns if c != omitted]
        X = base[cols].apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median(numeric_only=True))
        Xs = StandardScaler().fit_transform(X)
        rows.append({
            "omitted_variable": omitted,
            "remaining_dimensions": len(cols),
            "MLE_k10": mle_intrinsic_dimension(Xs, k=10),
            "MLE_k15": mle_intrinsic_dimension(Xs, k=15),
            "TWO_NN": two_nn_dimension(Xs),
        })

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "17_leave_one_variable_out.csv", index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))
    return result


def make_v32_stability_summary(intrinsic_df, series_df, jackknife_summary,
                               subsample_summary, lvo_df, output_dir):
    """Produce a compact, conservative decision table for the 3-4D hypothesis."""
    rows = []

    full = intrinsic_df[intrinsic_df["method"] == "Local MLE"].copy()
    for _, r in full.iterrows():
        rows.append({"test": "Full 70 hulls", "condition": r["parameter"], "estimate": r["estimate"], "status": "primary"})

    if not jackknife_summary.empty:
        for _, r in jackknife_summary.iterrows():
            rows.append({"test": "Leave-one-hull-out", "condition": r["method"], "estimate": r["median"], "status": "median of 70 fits"})

    if not subsample_summary.empty:
        for _, r in subsample_summary.iterrows():
            rows.append({"test": "Subsampling", "condition": r["dataset"] + " / " + r["method"], "estimate": r["median"], "status": "median; no duplicates"})

    if not lvo_df.empty:
        for col in ["MLE_k10", "MLE_k15"]:
            vals = lvo_df[col].dropna()
            rows.append({
                "test": "Leave-one-variable-out",
                "condition": col,
                "estimate": vals.median(),
                "status": f"median across {len(vals)} omitted variables",
            })

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "18_v32_stability_summary.csv", index=False, encoding="utf-8-sig")
    return result


# =============================================================================
# Isomap
# =============================================================================

def run_isomap(
    X_scaled,
    output_dir,
    fig_dir,
):
    print("\n" + "=" * 80)
    print("ISOMAP NONLINEAR MANIFOLD DIAGNOSTIC")
    print("=" * 80)

    rows = []

    for k in [5, 7, 10, 12, 15, 20]:
        if k >= len(X_scaled):
            continue

        for d in [2, 3, 4]:
            try:
                model = Isomap(
                    n_neighbors=k,
                    n_components=d,
                )

                embedding = model.fit_transform(X_scaled)

                # sklearn versions expose reconstruction_error().
                error = model.reconstruction_error()

                rows.append(
                    {
                        "method": "Isomap",
                        "neighbors": k,
                        "dimensions": d,
                        "reconstruction_error": error,
                    }
                )

                if k == 10 and d == 2:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(
                        embedding[:, 0],
                        embedding[:, 1],
                        alpha=0.8,
                    )
                    ax.set_xlabel("Isomap-1")
                    ax.set_ylabel("Isomap-2")
                    ax.set_title(
                        "Isomap 2D Embedding "
                        "(70 Independent Hulls, k=10)"
                    )
                    ax.grid(alpha=0.25)
                    fig.tight_layout()
                    fig.savefig(
                        fig_dir / "09_isomap_2d.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.close(fig)

            except Exception as exc:
                print(
                    f"Isomap failed: k={k}, d={d}: {exc}"
                )

    result = pd.DataFrame(rows)

    result.to_csv(
        output_dir / "15_isomap_quality.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return result


# =============================================================================
# LLE
# =============================================================================

def run_lle(
    X_scaled,
    output_dir,
    fig_dir,
):
    print("\n" + "=" * 80)
    print("LLE NONLINEAR MANIFOLD DIAGNOSTIC")
    print("=" * 80)

    rows = []

    for k in [5, 7, 10, 12, 15, 20]:
        if k >= len(X_scaled):
            continue

        for d in [2, 3, 4]:
            try:
                model = LocallyLinearEmbedding(
                    n_neighbors=k,
                    n_components=d,
                    method="standard",
                    random_state=42,
                )

                embedding = model.fit_transform(X_scaled)

                error = model.reconstruction_error_

                rows.append(
                    {
                        "method": "LLE",
                        "neighbors": k,
                        "dimensions": d,
                        "reconstruction_error": error,
                    }
                )

                if k == 10 and d == 2:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(
                        embedding[:, 0],
                        embedding[:, 1],
                        alpha=0.8,
                    )
                    ax.set_xlabel("LLE-1")
                    ax.set_ylabel("LLE-2")
                    ax.set_title(
                        "LLE 2D Embedding "
                        "(70 Independent Hulls, k=10)"
                    )
                    ax.grid(alpha=0.25)
                    fig.tight_layout()
                    fig.savefig(
                        fig_dir / "10_lle_2d.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.close(fig)

            except Exception as exc:
                print(
                    f"LLE failed: k={k}, d={d}: {exc}"
                )

    result = pd.DataFrame(rows)

    result.to_csv(
        output_dir / "16_lle_quality.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return result


# =============================================================================
# Distance analysis
# =============================================================================

def run_distance_analysis(
    X_scaled,
    output_dir,
):
    D = pairwise_distances(
        X_scaled,
        metric="euclidean",
    )

    upper = D[
        np.triu_indices_from(D, k=1)
    ]

    upper = upper[np.isfinite(upper)]

    summary = pd.DataFrame(
        {
            "statistic": [
                "mean",
                "std",
                "min",
                "median",
                "max",
                "q10",
                "q90",
            ],
            "value": [
                np.mean(upper),
                np.std(upper),
                np.min(upper),
                np.median(upper),
                np.max(upper),
                np.quantile(upper, 0.10),
                np.quantile(upper, 0.90),
            ],
        }
    )

    summary.to_csv(
        output_dir / "17_geometry_distance_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return D, summary


# =============================================================================
# Hydrodynamic response summary
# =============================================================================

def build_response_summary(df, hull_df):
    response = (
        df.groupby(
            ["Serie", "Sysser"],
            dropna=False,
        )[RESPONSE_COLUMN]
        .agg(
            mean_response="mean",
            median_response="median",
            min_response="min",
            max_response="max",
            std_response="std",
            n_records="size",
        )
        .reset_index()
    )

    return hull_df.merge(
        response,
        on=["Serie", "Sysser"],
        how="left",
    )


# =============================================================================
# Final report
# =============================================================================

def write_summary(
    output_dir,
    df,
    hull_df,
    X_hull,
    pca_variance,
    intrinsic_df,
    series_df,
    jackknife_summary,
    subsample_summary,
    lvo_df,
    consistency_df,
):
    lines = []

    lines.append("=" * 80)
    lines.append("REAL SAILBOAT HULL DATASET AUDIT V3.2")
    lines.append("=" * 80)
    lines.append("")

    lines.append("1. DATA STRUCTURE")
    lines.append("-" * 80)
    lines.append(
        f"Experimental records: {len(df)}"
    )
    lines.append(
        f"Independent hulls: {len(hull_df)}"
    )
    lines.append(
        f"Geometry dimensions: {X_hull.shape[1]}"
    )
    lines.append("")

    lines.append("2. GEOMETRY VARIABLES")
    lines.append("-" * 80)
    for i, c in enumerate(GEOMETRY_COLUMNS, 1):
        lines.append(f"{i:02d}. {c}")
    lines.append("")

    lines.append("3. WITHIN-HULL GEOMETRY CONSISTENCY")
    lines.append("-" * 80)

    if not consistency_df.empty:
        max_range = (
            consistency_df
            .groupby("geometry")["range"]
            .max()
            .sort_values(ascending=False)
        )

        lines.append(
            f"Largest observed within-hull geometry range: "
            f"{max_range.iloc[0]:.8g}"
            if len(max_range)
            else
            "Largest observed within-hull geometry range: unavailable"
        )

        nonzero = consistency_df[
            consistency_df["range"].fillna(0) > 1e-12
        ]

        lines.append(
            f"Hull/geometry combinations with non-zero range "
            f"(>1e-12): {len(nonzero)}"
        )

    lines.append("")

    lines.append("4. PCA BASELINE")
    lines.append("-" * 80)

    cumulative = pca_variance["cumulative_variance"].values

    for threshold in [0.80, 0.90, 0.95, 0.99]:
        dim = int(
            np.argmax(cumulative >= threshold) + 1
        )
        lines.append(
            f"{threshold:.0%} cumulative variance: {dim}D"
        )

    lines.append("")

    lines.append("5. INTRINSIC DIMENSION")
    lines.append("-" * 80)

    all_id = intrinsic_df.copy()

    for _, row in all_id.iterrows():
        estimate = row["estimate"]

        if pd.notna(estimate):
            lines.append(
                f"{row['method']} "
                f"({row['parameter']}): "
                f"{estimate:.6f}"
            )

    lines.append("")

    lines.append("6. V3.2 UNCERTAINTY / STABILITY")
    lines.append("-" * 80)
    lines.append("IMPORTANT: V3.1 naive bootstrap-with-replacement is discarded for kNN ID estimators because duplicated hulls generate zero nearest-neighbour distances.")
    lines.append("V3.2 therefore uses series-stratified analysis, leave-one-hull-out jackknife, and subsampling without replacement.")

    if jackknife_summary is not None and not jackknife_summary.empty:
        lines.append("Leave-one-hull-out summary:")
        for _, row in jackknife_summary.iterrows():
            lines.append(
                f"  {row['method']}: median={row['median']:.6f}, "
                f"95% interval=[{row['q025']:.6f}, {row['q975']:.6f}]"
            )

    if subsample_summary is not None and not subsample_summary.empty:
        lines.append("Subsampling summary:")
        for _, row in subsample_summary.iterrows():
            lines.append(
                f"  {row['dataset']} / {row['method']}: "
                f"median={row['median']:.6f}, "
                f"95% interval=[{row['q025']:.6f}, {row['q975']:.6f}]"
            )

    lines.append("")
    lines.append("7. SERIES-STRATIFIED ANALYSIS")
    lines.append("-" * 80)
    if series_df is not None and not series_df.empty:
        for _, row in series_df.iterrows():
            if pd.notna(row.get("estimate")):
                lines.append(f"{row['dataset']} {row['parameter']}: {row['estimate']:.6f}")

    lines.append("")
    lines.append("8. LEAVE-ONE-VARIABLE-OUT")
    lines.append("-" * 80)
    if lvo_df is not None and not lvo_df.empty:
        for col in ["MLE_k10", "MLE_k15", "TWO_NN"]:
            vals = lvo_df[col].dropna()
            if len(vals):
                lines.append(f"{col}: median={vals.median():.6f}, min={vals.min():.6f}, max={vals.max():.6f}")

    lines.append("")

    lines.append("9. INTERPRETATION RULE")
    lines.append("-" * 80)
    lines.append(
        "A low-dimensional-manifold claim should require "
        "convergence/stability across multiple estimators, "
        "neighborhood scales, bootstrap samples and subsamples."
    )
    lines.append(
        "A single intrinsic-dimension estimate is not treated as proof."
    )
    lines.append(
        "The present dataset contains only 70 independent hulls; "
        "therefore all intrinsic-dimension results are diagnostic evidence."
    )
    lines.append("")

    lines.append("10. RESEARCH DECISION")
    lines.append("-" * 80)
    lines.append(
        "If PCA indicates substantially more dimensions than nonlinear "
        "estimators while nonlinear estimates remain stable around a "
        "small value, the nonlinear low-dimensional structure becomes "
        "a stronger candidate for subsequent manifold modelling."
    )
    lines.append(
        "If estimates disagree strongly or vary substantially with k, "
        "bootstrap and subsampling, the 3-4D hypothesis should not be forced."
    )

    report_path = output_dir / "00_summary.txt"
    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return report_path




# =============================================================================\n
# =============================================================================
# V6 publication helpers
# =============================================================================

def _safe_numeric(df, cols):
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _paper_savefig(fig, path, dpi=360):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG OK] {path.name}")


def _paper_save_table(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[TABLE OK] {path.name}")


def _set_3d_clean_style(ax):
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.fill = True
        except Exception:
            pass
    ax.tick_params(labelsize=9, pad=2)
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass
    ax.view_init(elev=24, azim=-58)


def _hull_response_aligned(df, hull_df):
    rs = (
        df.groupby(["Serie", "Sysser"], dropna=False)[RESPONSE_COLUMN]
        .mean().reset_index(name="mean_response")
    )
    key = pd.MultiIndex.from_frame(hull_df[["Serie", "Sysser"]])
    rkey = pd.MultiIndex.from_frame(rs[["Serie", "Sysser"]])
    lookup = pd.Series(rs["mean_response"].to_numpy(dtype=float), index=rkey)
    return lookup.reindex(key).to_numpy(dtype=float)


def _condition_residual_aligned(df, hull_df):
    """Descriptive Fn/Rn adjustment; not a causal hydrodynamic model."""
    from sklearn.linear_model import LinearRegression
    hyd = _safe_numeric(df, ["Fn", "Rn", RESPONSE_COLUMN]).copy()
    hyd["Serie"] = df["Serie"].values
    hyd["Sysser"] = df["Sysser"].values
    hyd = hyd.dropna(subset=["Fn", "Rn", RESPONSE_COLUMN])
    if len(hyd) < 20:
        return None, None
    fn = hyd["Fn"].to_numpy(float)
    rn = hyd["Rn"].to_numpy(float)
    y = hyd[RESPONSE_COLUMN].to_numpy(float)
    Z = np.column_stack([fn, rn, fn**2, rn**2, fn*rn])
    model = LinearRegression().fit(Z, y)
    hyd["condition_residual"] = y - model.predict(Z)
    rs = (hyd.groupby(["Serie", "Sysser"], dropna=False)["condition_residual"]
          .agg(mean_residual="mean", median_residual="median", std_residual="std", n_records="size")
          .reset_index())
    key = pd.MultiIndex.from_frame(hull_df[["Serie", "Sysser"]])
    rkey = pd.MultiIndex.from_frame(rs[["Serie", "Sysser"]])
    lookup = pd.Series(rs["mean_residual"].to_numpy(float), index=rkey)
    return lookup.reindex(key).to_numpy(float), rs


# V6 publication-grade nonlinear manifold visualization
# =============================================================================

def _v6_recompute_embeddings(X_scaled, neighbors=10):
    """Recompute Isomap/LLE embeddings used by the V6 visualization layer."""
    out = {}
    n = len(X_scaled)
    k = min(neighbors, n - 2)
    for d in (2, 3, 4):
        try:
            out[("Isomap", d)] = Isomap(n_neighbors=k, n_components=d).fit_transform(X_scaled)
        except Exception as exc:
            print(f"[WARN] V6 Isomap {d}D failed: {exc}")
        try:
            out[("LLE", d)] = LocallyLinearEmbedding(
                n_neighbors=k, n_components=d, method="standard", random_state=42
            ).fit_transform(X_scaled)
        except Exception as exc:
            print(f"[WARN] V6 LLE {d}D failed: {exc}")
    return out


def _v6_mst_edges(X):
    """Return a sparse geometric backbone using the Euclidean MST."""
    from scipy.sparse.csgraph import minimum_spanning_tree
    D = pairwise_distances(X, metric="euclidean")
    T = minimum_spanning_tree(D)
    rows, cols = T.nonzero()
    return [(int(i), int(j)) for i, j in zip(rows, cols)]


def _v6_plot_3d_backbone(embedding, hull_df, path, title, color_values=None, color_label=None,
                         response_log=False, annotate=False):
    from matplotlib.colors import Normalize, LogNorm
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    emb = np.asarray(embedding, dtype=float)
    fig = plt.figure(figsize=(11.5, 8.8))
    ax = fig.add_subplot(111, projection="3d")
    _set_3d_clean_style(ax)

    edges = _v6_mst_edges(emb)
    segs = [(emb[a], emb[b]) for a, b in edges]
    if segs:
        lc = Line3DCollection(segs, linewidths=1.0, alpha=0.34)
        ax.add_collection3d(lc)

    series = hull_df["Serie"].astype(str).to_numpy()
    names = list(dict.fromkeys(series.tolist()))
    markers = ["o", "^", "s", "D", "P", "X"]
    marker_map = {s: markers[i % len(markers)] for i, s in enumerate(names)}

    if color_values is None:
        for s in names:
            m = series == s
            ax.scatter(emb[m,0], emb[m,1], emb[m,2], s=72, marker=marker_map[s], alpha=0.90,
                       edgecolors="white", linewidths=0.65, label=s, depthshade=True)
        ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.98), fontsize=8, frameon=True)
    else:
        v = np.asarray(color_values, dtype=float)
        finite = np.isfinite(v)
        if response_log and np.sum(finite & (v > 0)) >= 3:
            pos = v[finite & (v > 0)]
            norm = LogNorm(vmin=np.percentile(pos, 2), vmax=np.percentile(pos, 98))
        else:
            vv = v[finite]
            norm = Normalize(vmin=np.percentile(vv, 2), vmax=np.percentile(vv, 98))
        sc = ax.scatter(emb[finite,0], emb[finite,1], emb[finite,2], c=v[finite], norm=norm,
                        s=78, alpha=0.94, depthshade=True)
        cb = fig.colorbar(sc, ax=ax, pad=0.08, shrink=0.72)
        cb.set_label(color_label or "Response", fontsize=10)
        cb.ax.tick_params(labelsize=8)

    if annotate:
        for i in range(len(emb)):
            label = f"{hull_df.iloc[i]['Serie']} / {hull_df.iloc[i]['Sysser']}"
            ax.text(emb[i,0], emb[i,1], emb[i,2], str(i+1), fontsize=6, alpha=0.65)

    ax.set_xlabel("Isomap-1", fontsize=11, labelpad=8)
    ax.set_ylabel("Isomap-2", fontsize=11, labelpad=8)
    ax.set_zlabel("Isomap-3", fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=15, pad=18)
    ax.text2D(0.02, 0.025, "Nodes = independent hulls; thin lines = minimum-spanning geometric backbone",
              transform=ax.transAxes, fontsize=8.5)
    _paper_savefig(fig, path, dpi=420)


def _v6_plot_3d_density_envelope(embedding, hull_df, path, title, response=None, seed=42):
    """Create a transparent 3-D density envelope. It is explicitly a visual support envelope, not a fitted exact manifold."""
    from scipy.stats import gaussian_kde
    from skimage.measure import marching_cubes
    emb = np.asarray(embedding, dtype=float)
    rng = np.random.default_rng(seed)
    fig = plt.figure(figsize=(11.5, 8.8))
    ax = fig.add_subplot(111, projection="3d")
    _set_3d_clean_style(ax)

    # Robust plotting limits.
    lo = np.nanpercentile(emb, 1, axis=0)
    hi = np.nanpercentile(emb, 99, axis=0)
    pad = 0.10 * (hi - lo + 1e-9)
    lo -= pad; hi += pad
    grid_n = 42
    axes = [np.linspace(lo[j], hi[j], grid_n) for j in range(3)]
    Xg, Yg, Zg = np.meshgrid(*axes, indexing="ij")
    pts = np.vstack([Xg.ravel(), Yg.ravel(), Zg.ravel()])
    try:
        kde = gaussian_kde(emb.T, bw_method="scott")
        density = kde(pts).reshape((grid_n, grid_n, grid_n))
        level = np.percentile(density, 28)
        verts, faces, _, _ = marching_cubes(density, level=level)
        scale = np.array([axes[0][1]-axes[0][0], axes[1][1]-axes[1][0], axes[2][1]-axes[2][0]])
        origin = np.array([axes[0][0], axes[1][0], axes[2][0]])
        verts = verts * scale + origin
        ax.plot_trisurf(verts[:,0], verts[:,1], faces, verts[:,2], alpha=0.16, linewidth=0.15)
    except Exception as exc:
        print(f"[WARN] Density envelope could not be generated: {exc}")

    if response is None:
        ax.scatter(emb[:,0], emb[:,1], emb[:,2], s=62, alpha=0.82)
    else:
        v = np.asarray(response, dtype=float)
        m = np.isfinite(v)
        sc = ax.scatter(emb[m,0], emb[m,1], emb[m,2], c=v[m], s=70, alpha=0.92,
                        edgecolors="white", linewidths=0.55)
        cb = fig.colorbar(sc, ax=ax, pad=0.08, shrink=0.72)
        cb.set_label("Mean hull response", fontsize=10)
    ax.set_xlabel("Isomap-1", fontsize=11, labelpad=8)
    ax.set_ylabel("Isomap-2", fontsize=11, labelpad=8)
    ax.set_zlabel("Isomap-3", fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=15, pad=18)
    ax.text2D(0.02, 0.025, "Transparent envelope = 3-D sample-density support (visualization only)",
              transform=ax.transAxes, fontsize=8.5)
    _paper_savefig(fig, path, dpi=420)


def _v6_plot_3d_projection(embedding, values, path, title, ylabel, log_color=False):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    emb = np.asarray(embedding, dtype=float)
    v = np.asarray(values, dtype=float)
    m = np.isfinite(v) & np.all(np.isfinite(emb), axis=1)
    fig = plt.figure(figsize=(10.5, 8.2))
    ax = fig.add_subplot(111, projection="3d")
    _set_3d_clean_style(ax)
    if log_color and np.sum(m & (v > 0)) >= 3:
        from matplotlib.colors import LogNorm
        pos = v[m & (v > 0)]
        norm = LogNorm(vmin=np.percentile(pos, 2), vmax=np.percentile(pos, 98))
    else:
        from matplotlib.colors import Normalize
        vv = v[m]
        norm = Normalize(vmin=np.percentile(vv, 2), vmax=np.percentile(vv, 98))
    sc = ax.scatter(emb[m,0], emb[m,1], emb[m,2], c=v[m], norm=norm,
                    s=76, alpha=0.92)
    cb = fig.colorbar(sc, ax=ax, pad=0.08, shrink=0.72)
    cb.set_label(ylabel, fontsize=10)
    ax.set_xlabel("Isomap-1", fontsize=11); ax.set_ylabel("Isomap-2", fontsize=11); ax.set_zlabel("Isomap-3", fontsize=11)
    ax.set_title(title, fontsize=15, pad=18)
    _paper_savefig(fig, path, dpi=420)


def _v6_plot_embedding_quality(isomap_df, lle_df, path):
    q = pd.concat([isomap_df.assign(method="Isomap"), lle_df.assign(method="LLE")], ignore_index=True)
    q["neighbors"] = pd.to_numeric(q["neighbors"], errors="coerce")
    q["dimensions"] = pd.to_numeric(q["dimensions"], errors="coerce")
    q["reconstruction_error"] = pd.to_numeric(q["reconstruction_error"], errors="coerce")
    q = q.dropna(subset=["neighbors", "dimensions", "reconstruction_error"])
    if q.empty: return
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    for method in ["Isomap", "LLE"]:
        g = q[(q.method == method) & (q.neighbors == 10)]
        if not g.empty:
            g = g.sort_values("dimensions")
            ax.plot(g.dimensions, g.reconstruction_error, marker="o", linewidth=2.0, label=method)
    ax.set_xlabel("Embedding dimension")
    ax.set_ylabel("Reconstruction / embedding error")
    ax.set_title("Nonlinear embedding quality at k=10")
    ax.set_xticks([2,3,4])
    ax.grid(alpha=0.20)
    ax.legend(frameon=True)
    _paper_savefig(fig, path, dpi=360)


def _v6_interactive_isomap(embedding, hull_df, response, residual, out_html):
    try:
        import plotly.graph_objects as go
        emb = np.asarray(embedding, dtype=float)
        r = np.asarray(response, dtype=float)
        res = np.asarray(residual, dtype=float) if residual is not None else np.full(len(emb), np.nan)
        custom = []
        for i in range(len(hull_df)):
            custom.append([str(hull_df.iloc[i]["Serie"]), str(hull_df.iloc[i]["Sysser"]), float(r[i]) if np.isfinite(r[i]) else np.nan,
                           float(res[i]) if np.isfinite(res[i]) else np.nan])
        fig = go.Figure(data=[go.Scatter3d(
            x=emb[:,0], y=emb[:,1], z=emb[:,2], mode="markers",
            marker=dict(size=7, color=r, colorscale="Viridis", opacity=0.90, colorbar=dict(title="Mean response")),
            customdata=custom,
            hovertemplate="Hull: %{customdata[0]} / %{customdata[1]}<br>Response: %{customdata[2]:.4g}<br>Adjusted residual: %{customdata[3]:.4g}<extra></extra>"
        )])
        fig.update_layout(title="Interactive 3-D Isomap representation of independent hull geometry",
                          scene=dict(xaxis_title="Isomap-1", yaxis_title="Isomap-2", zaxis_title="Isomap-3"),
                          margin=dict(l=0,r=0,b=0,t=50), template="plotly_white")
        fig.write_html(out_html, include_plotlyjs="cdn")
        print(f"[HTML OK] {out_html.name}")
    except Exception as exc:
        print(f"[WARN] Interactive Plotly output skipped: {exc}")


def run_v6_paper_layer(df, hull_df, X_hull, X_scaled, pca, pca_scores, intrinsic_df, series_df,
                       jackknife_df, subsample_df, lvo_df, isomap_df, lle_df, response_summary,
                       output_dir, image_dir=None, seed=42, interactive=True):
    """V6 publication layer: nonlinear manifold first, PCA supplementary."""
    paper_dir = output_dir / "paper"
    table_dir = paper_dir / "tables"
    fig_dir = paper_dir / "figures"
    latex_dir = paper_dir / "latex"
    for d in (paper_dir, table_dir, fig_dir, latex_dir): d.mkdir(parents=True, exist_ok=True)

    print("\
" + "="*80)
    print("V6 PAPER OUTPUT — 3-D MANIFOLD VISUALIZATION FIRST")
    print("="*80)

    # Tables: reuse scientifically validated V5 tables.
    overview = pd.DataFrame([
        ["Experimental records", len(df)], ["Independent hulls", len(hull_df)], ["Geometry dimensions", X_hull.shape[1]],
        ["Delft hulls", int(hull_df["Serie"].astype(str).str.contains("DELFT", case=False, na=False).sum())],
        ["Il Moro di Venezia hulls", int(hull_df["Serie"].astype(str).str.contains("Il Moro", case=False, na=False).sum())],
        ["US hulls", int(hull_df["Serie"].astype(str).str.fullmatch("US Series", case=False, na=False).sum())],
    ], columns=["item","value"])
    _paper_save_table(overview, table_dir / "Table01_dataset_overview.csv")
    _paper_save_table(X_hull.describe().T.reset_index().rename(columns={"index":"variable"}), table_dir / "Table02_geometry_statistics.csv")
    for src,dst in {
        "03_hull_pearson_correlation.csv":"Table03_Pearson_correlation.csv",
        "04_hull_spearman_correlation.csv":"Table04_Spearman_correlation.csv",
        "05_hull_mutual_information.csv":"Table05_mutual_information.csv",
        "08_intrinsic_dimension.csv":"Table07_intrinsic_dimension.csv",
        "15_isomap_quality.csv":"Table12_Isomap_quality.csv",
        "16_lle_quality.csv":"Table13_LLE_quality.csv",
        "17_geometry_distance_summary.csv":"Table14_geometry_distance_summary.csv",
        "19_hull_response_summary.csv":"Table15_hull_response_summary.csv",
    }.items():
        p = output_dir / src
        if p.exists(): _paper_save_table(pd.read_csv(p), table_dir / dst)
    _paper_save_table(series_df, table_dir / "Table08_series_ID_stability.csv")
    _paper_save_table(jackknife_df, table_dir / "Table09_jackknife_hull.csv")
    _paper_save_table(subsample_df, table_dir / "Table10_subsampling_without_replacement.csv")
    _paper_save_table(lvo_df, table_dir / "Table11_leave_one_variable_out.csv")

    # Recompute embeddings and save coordinates.
    emb = _v6_recompute_embeddings(X_scaled, neighbors=10)
    for (method,d), arr in emb.items():
        cols = ["Serie","Sysser"] + [f"{method}-{i+1}" for i in range(d)]
        cdf = pd.DataFrame(arr, columns=cols[2:])
        cdf.insert(0, "Sysser", hull_df["Sysser"].astype(str).values)
        cdf.insert(0, "Serie", hull_df["Serie"].astype(str).values)
        _paper_save_table(cdf, table_dir / f"{method}_{d}D_coordinates.csv")

    iso3 = emb.get(("Isomap",3))
    iso2 = emb.get(("Isomap",2))
    iso4 = emb.get(("Isomap",4))
    response = _hull_response_aligned(df, hull_df)
    residual, residual_table = _condition_residual_aligned(df, hull_df)
    if residual_table is not None: _paper_save_table(residual_table, table_dir / "Table16_condition_residual_by_hull.csv")

    # Figure 00: real image contact sheet or explicit schematic.
    if image_dir is not None and Path(image_dir).exists():
        try:
            from PIL import Image, ImageDraw
            paths = sorted([p for p in Path(image_dir).glob("*") if p.suffix.lower() in {".png",".jpg",".jpeg",".webp"}])[:12]
            thumbs=[]
            for pth in paths:
                try:
                    im=Image.open(pth).convert("RGB"); im.thumbnail((360,230)); can=Image.new("RGB",(380,270),"white")
                    can.paste(im,((380-im.width)//2,8)); ImageDraw.Draw(can).text((10,242),pth.stem[:50],fill="black"); thumbs.append(can)
                except Exception: pass
            if thumbs:
                cols=3; rows=int(np.ceil(len(thumbs)/cols)); sheet=Image.new("RGB",(cols*380,rows*270),"white")
                for i,t in enumerate(thumbs): sheet.paste(t,((i%cols)*380,(i//cols)*270))
                sheet.save(fig_dir/"Fig00_hull_image_gallery.jpg",quality=95); print("[FIG OK] Fig00_hull_image_gallery.jpg")
        except Exception as exc: print(f"[WARN] image gallery failed: {exc}")
    if not (fig_dir/"Fig00_hull_image_gallery.jpg").exists():
        fig, ax=plt.subplots(figsize=(10,4.8)); ax.axis("off")
        ax.plot([0.08,0.30,0.70,0.92,0.70,0.30,0.08],[0.50,0.28,0.20,0.50,0.80,0.72,0.50],linewidth=3)
        ax.plot([0.18,0.50,0.82],[0.50,0.50,0.50],linewidth=1.3,linestyle="--")
        ax.text(0.50,0.90,"Schematic only — no real hull photograph supplied",ha="center",fontsize=13)
        ax.text(0.50,0.08,"70 independent hulls represented by 16 geometry descriptors",ha="center",fontsize=12)
        _paper_savefig(fig,fig_dir/"Fig00_hull_schematic_fallback.png",dpi=360)

    # Dataset composition.
    ssum=df.groupby("Serie").size().reset_index(name="records")
    fig,ax=plt.subplots(figsize=(8.8,5.5)); ax.bar(ssum["Serie"].astype(str),ssum["records"]); ax.set_ylabel("Experimental records"); ax.set_title("Experimental records by hull series"); ax.tick_params(axis="x",rotation=15); ax.grid(axis="y",alpha=.20); _paper_savefig(fig,fig_dir/"Fig01_dataset_series_records.png")

    # Core intrinsic dimension.
    g=intrinsic_df.copy(); g["k_numeric"]=g["parameter"].astype(str).str.extract(r"k\\s*=\\s*([0-9]+(?:\\.[0-9]+)?)")[0].astype(float)
    g=g[g["method"].eq("Local MLE")].dropna(subset=["k_numeric","estimate"])
    fig,ax=plt.subplots(figsize=(9.2,5.8)); ax.plot(g["k_numeric"],g["estimate"],marker="o",linewidth=2.3); ax.axhspan(3.3,3.7,alpha=.10,label="Reference band 3.3–3.7"); ax.set_xlabel("Neighborhood size, k"); ax.set_ylabel("Estimated intrinsic dimension"); ax.set_title("Intrinsic dimensionality of independent hull geometry"); ax.grid(alpha=.20); ax.legend(); _paper_savefig(fig,fig_dir/"Fig03_intrinsic_dimension.png")

    # Robustness.
    fig,ax=plt.subplots(figsize=(9.5,6.0));
    if not jackknife_df.empty and "MLE_k10" in jackknife_df.columns:
        vals=pd.to_numeric(jackknife_df["MLE_k10"],errors="coerce").dropna(); ax.boxplot(vals,positions=[1],widths=.45,patch_artist=False); ax.scatter(np.ones(len(vals))*1,vals,s=10,alpha=.22)
    if not subsample_df.empty and "MLE_k10" in subsample_df.columns:
        for pos,frac in [(2,.80),(3,.90)]:
            vals=subsample_df.loc[pd.to_numeric(subsample_df["fraction"],errors="coerce").eq(frac),"MLE_k10"]; vals=pd.to_numeric(vals,errors="coerce").dropna();
            if len(vals): ax.boxplot(vals,positions=[pos],widths=.45,patch_artist=False); ax.scatter(np.ones(len(vals))*pos,vals,s=5,alpha=.12)
    ax.axhspan(3.3,3.7,alpha=.10); ax.set_xticks([1,2,3],labels=["Jackknife","80% subsample","90% subsample"]); ax.set_ylabel("MLE intrinsic dimension, k=10"); ax.set_title("Robustness of intrinsic-dimension estimate"); ax.grid(axis="y",alpha=.20); _paper_savefig(fig,fig_dir/"Fig04_intrinsic_dimension_robustness.png")

    # Main V6 figure: sparse geometric backbone.
    if iso3 is not None:
        _v6_plot_3d_backbone(iso3,hull_df,fig_dir/"Fig05_Isomap_3D_geometric_backbone.png",
                              "Independent Hulls in a 3-D Nonlinear Isomap Representation")
        _v6_plot_3d_backbone(iso3,hull_df,fig_dir/"Fig06_Isomap_3D_hydrodynamic_response.png",
                              "Hydrodynamic Response on the Nonlinear Hull Geometry",response,"Mean hull response",True)
        if residual is not None:
            _v6_plot_3d_backbone(iso3,hull_df,fig_dir/"Fig07_Isomap_3D_condition_adjusted_response.png",
                                  "Condition-Adjusted Response on the Nonlinear Hull Geometry",residual,"Mean response residual",False)
        _v6_plot_3d_density_envelope(iso3,hull_df,fig_dir/"Fig08_Isomap_3D_density_envelope.png",
                                     "3-D Sample-Density Support of the Nonlinear Embedding",response,seed)
        if residual is not None:
            _v6_plot_3d_projection(iso3,residual,fig_dir/"Fig09_Isomap_3D_residual_field.png",
                                   "Condition-Adjusted Hydrodynamic Response in Isomap Space",
                                   "Mean response residual",False)

    # Isomap 2D auxiliary.
    if iso2 is not None:
        fig,ax=plt.subplots(figsize=(8.5,6.3)); series=hull_df["Serie"].astype(str).to_numpy();
        for s in list(dict.fromkeys(series.tolist())):
            m=series==s; ax.scatter(iso2[m,0],iso2[m,1],s=55,alpha=.84,label=s,edgecolors="white",linewidths=.5)
        ax.set_xlabel("Isomap-1"); ax.set_ylabel("Isomap-2"); ax.set_title("2-D nonlinear embedding of independent hull geometry"); ax.grid(alpha=.20); ax.legend(fontsize=8); _paper_savefig(fig,fig_dir/"Fig10_Isomap_2D_series.png")

    # LLE diagnostic.
    lle3=emb.get(("LLE",3))
    if lle3 is not None: _v6_plot_3d_backbone(lle3,hull_df,fig_dir/"Fig11_LLE_3D_diagnostic.png","LLE 3-D Embedding — Supplementary Diagnostic")

    # Coordinate-response plots.
    if iso3 is not None and residual is not None:
        for j in range(3):
            fig,ax=plt.subplots(figsize=(7.8,5.3)); ax.scatter(iso3[:,j],residual,s=45,alpha=.78,edgecolors="white",linewidths=.4); ax.axhline(0,linestyle="--",linewidth=1,alpha=.55); ax.set_xlabel(f"Isomap-{j+1}"); ax.set_ylabel("Condition-adjusted response residual"); ax.set_title(f"Isomap-{j+1} and adjusted hydrodynamic response"); ax.grid(alpha=.20); _paper_savefig(fig,fig_dir/f"Fig12_Isomap_coord{j+1}_response.png")

    _v6_plot_embedding_quality(isomap_df,lle_df,fig_dir/"Fig13_embedding_quality_3D_vs_4D.png")

    # PCA only as supplementary baseline.
    evr=np.asarray(pca.explained_variance_ratio_); cum=np.cumsum(evr); fig,ax=plt.subplots(figsize=(7.8,5.0)); x=np.arange(1,len(evr)+1); ax.plot(x,cum,marker="o",linewidth=1.8);
    for t in (.80,.90,.95): ax.axhline(t,linestyle="--",linewidth=1,alpha=.5);
    ax.set_xlabel("Number of principal components"); ax.set_ylabel("Cumulative explained variance"); ax.set_ylim(.45,1.03); ax.set_title("Supplementary linear baseline: PCA"); ax.grid(alpha=.20); _paper_savefig(fig,fig_dir/"FigS1_PCA_linear_baseline.png")

    # Quality comparison table with 3D/4D decision helper.
    q=pd.concat([isomap_df.assign(method_family="Isomap"),lle_df.assign(method_family="LLE")],ignore_index=True); _paper_save_table(q,table_dir/"Table17_embedding_quality_recomputed.csv")
    if iso3 is not None and iso4 is not None:
        d3=pd.DataFrame(iso3,columns=["d1","d2","d3"]); d4=pd.DataFrame(iso4,columns=["d1","d2","d3","d4"]);
        extra=pd.DataFrame({"metric":["Isomap 3D coordinate variance sum","Isomap 4D coordinate variance sum"],"value":[float(d3.var().sum()),float(d4.var().sum())]}); _paper_save_table(extra,table_dir/"Table18_3D_4D_coordinate_diagnostic.csv")

    if interactive and iso3 is not None:
        _v6_interactive_isomap(iso3,hull_df,response,residual,paper_dir/"Isomap_3D_interactive.html")

    # Manifest.
    manifest=[]
    for p in sorted(fig_dir.glob("*")):
        if p.suffix.lower() in {".png",".jpg",".jpeg",".webp"}: manifest.append({"type":"figure","file":str(p.relative_to(paper_dir))})
    for p in sorted(table_dir.glob("*.csv")): manifest.append({"type":"table_csv","file":str(p.relative_to(paper_dir))})
    for p in sorted(paper_dir.glob("*.html")): manifest.append({"type":"interactive_html","file":str(p.relative_to(paper_dir))})
    _paper_save_table(pd.DataFrame(manifest),paper_dir/"paper_output_manifest.csv")

    tex=["% Auto-generated by real_hull_manifold_audit_v6.py",""]
    for p in sorted(fig_dir.glob("*")):
        if p.suffix.lower() in {".png",".jpg",".jpeg",".webp"}:
            stem=p.stem; tex += ["\\begin{figure}[htbp]","  \\centering",f"  \\includegraphics[width=0.90\\linewidth]{{figures/{p.name}}}",f"  \\caption{{{stem.replace('_',' ')}}}",f"  \\label{{fig:{stem.lower()}}}","\\end{figure}",""]
    (latex_dir/"figures.tex").write_text("\
".join(tex),encoding="utf-8"); print("[FILE OK] figures.tex")

    fig_files=[p for p in fig_dir.glob("*") if p.suffix.lower() in {".png",".jpg",".jpeg",".webp"}]; table_files=list(table_dir.glob("*.csv")); html_files=list(paper_dir.glob("*.html"))
    report=["REAL SAILBOAT HULL MANIFOLD — V6 PAPER OUTPUT REPORT","="*78,f"Experimental records: {len(df)}",f"Independent hulls: {len(hull_df)}",f"Geometry dimensions: {X_hull.shape[1]}","",
            "Primary framework: intrinsic-dimension-first nonlinear manifold analysis.","PCA is supplementary only.","3-D visualization uses a sparse minimum-spanning geometric backbone; it does not assert that the data lie on an exact 2-D surface.",
            "The transparent 3-D envelope is a density-support visualization, not a fitted exact manifold.","Hydrodynamic residual adjustment is descriptive rather than causal.","",
            f"Figures generated: {len(fig_files)}",f"Tables generated: {len(table_files)}",f"Interactive HTML generated: {len(html_files)}",
            "",f"Figures directory: {fig_dir}",f"Tables directory: {table_dir}",f"LaTeX directory: {latex_dir}"]
    (paper_dir/"V6_paper_report.txt").write_text("\
".join(report),encoding="utf-8"); print("[FILE OK] V6_paper_report.txt")
    print(f"[COUNT] Figures generated: {len(fig_files)}")
    print(f"[COUNT] Tables generated: {len(table_files)}")
    print(f"[COUNT] Interactive HTML generated: {len(html_files)}")
    return paper_dir

# =============================================================================
# V6 MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="V6 intrinsic-dimension-first real sailboat hull manifold analysis with publication-grade 3D visualization."
    )
    parser.add_argument("input_csv", nargs="?", default="Sailboat Hull Resistance Dataset V01.csv")
    parser.add_argument("--output", default="results_v6", help="Output directory. Default: results_v6")
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="Legacy compatibility argument; no kNN bootstrap-with-replacement is used.")
    parser.add_argument("--subsamples", type=int, default=500,
                        help="Number of hull-level subsampling repetitions.")
    parser.add_argument("--seed", type=int, default=20260909, help="Random seed.")
    parser.add_argument("--image-dir", default=None,
                        help="Optional folder containing real hull photographs/images.")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Skip Plotly interactive HTML output.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    input_path = Path(args.input_csv)
    output_dir = Path(args.output)
    fig_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{input_path.resolve()}\n"
        )

    print("\n" + "=" * 80)
    print("REAL SAILBOAT HULL DATASET AUDIT V6")
    print("=" * 80)
    print(f"Input : {input_path.resolve()}")
    print(f"Output: {output_dir.resolve()}")

    df = load_dataset(input_path)

    overview_rows = []
    for c in df.columns:
        overview_rows.append({
            "column": c,
            "dtype": str(df[c].dtype),
            "n_total": len(df[c]),
            "n_missing": int(df[c].isna().sum()),
            "missing_rate": float(df[c].isna().mean()),
            "n_unique": int(df[c].nunique(dropna=True)),
        })
    pd.DataFrame(overview_rows).to_csv(
        output_dir / "01_dataset_overview.csv", index=False, encoding="utf-8-sig"
    )

    series_summary = (
        df.groupby("Serie", dropna=False)
        .agg(
            records=("Serie", "size"),
            unique_sysser=("Sysser", "nunique"),
            fn_min=("Fn", "min"), fn_max=("Fn", "max"),
            rn_min=("Rn", "min"), rn_max=("Rn", "max"),
            response_min=(RESPONSE_COLUMN, "min"),
            response_max=(RESPONSE_COLUMN, "max"),
        )
        .reset_index()
    )
    series_summary.to_csv(
        output_dir / "02_series_summary.csv", index=False, encoding="utf-8-sig"
    )
    print("\nSeries summary:")
    print(series_summary.to_string(index=False))

    hull_df, consistency_df = build_independent_hulls(df, output_dir)

    X_hull = hull_df[GEOMETRY_COLUMNS].copy()
    for c in GEOMETRY_COLUMNS:
        X_hull[c] = pd.to_numeric(X_hull[c], errors="coerce")

    if X_hull.isna().sum().sum() > 0:
        print("\nWARNING: missing hull-level geometry detected; median imputation applied where necessary.")
        X_hull = X_hull.fillna(X_hull.median(numeric_only=True))

    usable_columns = [
        c for c in X_hull.columns if X_hull[c].nunique(dropna=True) > 1
    ]
    X_hull = X_hull[usable_columns]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_hull)

    print("\nIndependent hull matrix")
    print("-" * 80)
    print(f"Hulls:       {len(hull_df)}")
    print(f"Dimensions:  {X_hull.shape[1]}")
    print(f"Matrix:      {X_scaled.shape}")

    pearson, spearman, mi = run_dependence_analysis(
        X_hull, output_dir, fig_dir
    )
    pca, pca_scores, pca_variance = run_pca(
        X_scaled, list(X_hull.columns), fig_dir, output_dir
    )
    intrinsic_df = run_intrinsic_dimension(
        X_scaled, output_dir, fig_dir
    )
    series_df = run_series_stability(
        X_scaled, hull_df, output_dir
    )
    jackknife_df, jackknife_summary = run_jackknife_hull(
        X_scaled, output_dir, ks=(10, 15)
    )
    subsample_df, subsample_summary = run_subsample_stability_v32(
        X_scaled,
        output_dir,
        n_subsamples=args.subsamples,
        fractions=(0.80, 0.90),
        random_state=args.seed + 1,
    )
    lvo_df = run_leave_one_variable_out(
        hull_df, GEOMETRY_COLUMNS, output_dir
    )
    isomap_df = run_isomap(
        X_scaled, output_dir, fig_dir
    )
    lle_df = run_lle(
        X_scaled, output_dir, fig_dir
    )
    D, distance_summary = run_distance_analysis(
        X_scaled, output_dir
    )

    pca_coordinates = hull_df[["Serie", "Sysser"]].copy()
    for i in range(min(6, pca_scores.shape[1])):
        pca_coordinates[f"PC{i+1}"] = pca_scores[:, i]
    pca_coordinates.to_csv(
        output_dir / "18_hull_pca_coordinates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    response_summary = build_response_summary(df, hull_df)
    response_summary.to_csv(
        output_dir / "19_hull_response_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    make_v32_stability_summary(
        intrinsic_df, series_df, jackknife_summary,
        subsample_summary, lvo_df, output_dir
    )

    write_summary(
        output_dir, df, hull_df, X_hull, pca_variance,
        intrinsic_df, series_df, jackknife_summary,
        subsample_summary, lvo_df, consistency_df,
    )

    paper_dir = run_v6_paper_layer(
        df, hull_df, X_hull, X_scaled, pca, pca_scores,
        intrinsic_df, series_df, jackknife_df, subsample_df,
        lvo_df, isomap_df, lle_df, response_summary,
        output_dir, image_dir=args.image_dir,
        seed=args.seed, interactive=not args.no_interactive,
    )

    print("\n" + "=" * 80)
    print("V6 PAPER ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Experimental records: {len(df)}")
    print(f"Independent hulls:    {len(hull_df)}")
    print(f"Geometry dimensions:  {X_hull.shape[1]}")
    print(f"Output directory:     {output_dir.resolve()}")
    print(f"Paper directory:      {paper_dir.resolve()}")
    print("PCA is supplementary only; nonlinear manifold analysis is the main framework.")
    print("3-D visualization uses a sparse geometric backbone rather than a dense kNN spider-web.")


if __name__ == "__main__":
    main()
