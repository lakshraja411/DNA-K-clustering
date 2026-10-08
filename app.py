
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from core import (
    FEATURE_NAMES,
    read_uploaded_bytes,
    load_dataset,
    dataset_table,
    npz_to_dict,
    map_fitting_to_dataset,
    map_fitting_to_raw,
    minmax_scale,
    kmeans_numpy,
    clustering_diagnostics,
    scan_k,
    reorder_labels_by_height,
    pca_numpy,
    eta_squared,
    centered_profiles,
    pointwise_cluster_medians,
    segment_metrics,
    export_many_groups_zip,
    parse_edges,
    gate_rows,
)

st.set_page_config(page_title="DNA K-Means Workbench", layout="wide")
st.title("DNA K-Means Workbench")
st.caption(
    "NanoSense scalar features → reproducible K-means → waveform families → physical plots → current gates → segment-weighted analysis"
)

# -----------------------------
# session state
# -----------------------------
for key, default in {
    "loaded": None,
    "cluster_result": None,
    "multi_compare": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# -----------------------------
# helpers
# -----------------------------
def load_recording(dataset_blob, eventdata_blob, fitting_blob, dataset_name, eventdata_name, fitting_name, tolerance):
    X, dataset_arrays = load_dataset(dataset_blob)
    eventdata_arrays = npz_to_dict(eventdata_blob, allow_pickle=True)
    fitting_arrays = npz_to_dict(fitting_blob, allow_pickle=True)

    df = dataset_table(X)

    fit_to_dataset, ds_status = map_fitting_to_dataset(
        fitting_arrays, X, tolerance=float(tolerance)
    )
    fit_to_raw, raw_status = map_fitting_to_raw(
        fitting_arrays, eventdata_arrays, tolerance=float(tolerance)
    )

    row_to_fit = {int(row): int(fid) for fid, row in fit_to_dataset.items()}
    df["event_id"] = [row_to_fit.get(int(i), np.nan) for i in df["dataset_row"]]

    return dict(
        X=X,
        df=df,
        dataset_arrays=dataset_arrays,
        eventdata_arrays=eventdata_arrays,
        fitting_arrays=fitting_arrays,
        fit_to_dataset=fit_to_dataset,
        fit_to_raw=fit_to_raw,
        ds_status=ds_status,
        raw_status=raw_status,
        source_names=dict(
            dataset=dataset_name,
            eventdata=eventdata_name,
            fitting=fitting_name,
        ),
    )


def run_kmeans_pipeline(recording, mode, manual_k, kmax, n_init, random_state):
    df = recording["df"].copy()
    X_features = df[FEATURE_NAMES].to_numpy(float)
    finite_mask = np.all(np.isfinite(X_features), axis=1)

    X_valid = X_features[finite_mask]
    valid_df = df.loc[finite_mask].copy().reset_index(drop=True)
    if len(valid_df) < 3:
        raise ValueError("Too few valid events for clustering.")

    X_scaled, feature_min, feature_max, constant = minmax_scale(X_valid)
    scan = scan_k(
        X_scaled,
        k_min=2,
        k_max=int(kmax),
        n_init=int(n_init),
        max_iter=2000,
        random_state=int(random_state),
    )
    if len(scan) == 0:
        raise ValueError("k scan failed. Check that enough events remain after filtering.")

    if mode == "Silhouette":
        k = int(scan.loc[scan["silhouette"].idxmax(), "k"])
    else:
        k = int(manual_k)

    labels, centers, inertia = kmeans_numpy(
        X_scaled,
        k,
        n_init=int(n_init),
        max_iter=2000,
        random_state=int(random_state),
    )

    labels, label_mapping = reorder_labels_by_height(valid_df, labels)
    diag = clustering_diagnostics(X_scaled, labels)

    assignments = df.copy()
    assignments["cluster"] = -1
    assignments.loc[finite_mask, "cluster"] = labels

    scores, explained, loadings = pca_numpy(X_scaled)

    return dict(
        k=k,
        labels=labels,
        assignments=assignments,
        valid_mask=finite_mask,
        valid_df=valid_df,
        X_scaled=X_scaled,
        scan=scan,
        diagnostics=diag,
        inertia=inertia,
        scores=scores,
        explained=explained,
        loadings=loadings,
        feature_min=feature_min,
        feature_max=feature_max,
        constant=constant,
        n_init=int(n_init),
        random_state=int(random_state),
        mode=mode,
    )


def cluster_summary_table(assignments):
    valid = assignments[assignments["cluster"] >= 0].copy()
    summary = (
        valid.groupby("cluster")
        .agg(
            n=("cluster", "size"),
            median_blockade_nA=("height_nA", "median"),
            median_fwhm_ms=("fwhm_ms", "median"),
            median_dwell_ms=("width_ms", "median"),
            median_area_nA_ms=("area_nA_ms", "median"),
            median_fwhm_fraction=("fwhm_fraction", "median"),
            median_skew=("skew", "median"),
            median_kurtosis=("kurtosis", "median"),
        )
        .reset_index()
    )
    summary["population_pct"] = 100 * summary["n"] / summary["n"].sum()
    return summary


def build_labels_by_event(recording, assignments):
    row_to_fit = {int(row): int(fid) for fid, row in recording["fit_to_dataset"].items()}
    valid = assignments[assignments["cluster"] >= 0].copy()
    labels_by_event = {}
    for _, row in valid.iterrows():
        dsrow = int(row["dataset_row"])
        if dsrow in row_to_fit:
            labels_by_event[row_to_fit[dsrow]] = int(row["cluster"])
    return labels_by_event


def get_waveform_family_data(recording, assignments, window_samples=850, min_coverage_fraction=0.5):
    labels_by_event = build_labels_by_event(recording, assignments)
    if not labels_by_event:
        return None, None
    info = centered_profiles(
        recording["fitting_arrays"],
        sorted(labels_by_event),
        labels_by_event,
        window_samples=int(window_samples),
    )
    medians = pointwise_cluster_medians(info, float(min_coverage_fraction)) if info is not None else {}
    return info, medians


def plot_waveform_family_panels(info, medians, x_mode="time", max_members_per_cluster=50):
    if info is None or not medians:
        return None, None

    x = info["time_ms"] if x_mode == "time" else info["data_index"]
    xlabel = "Time relative to event midpoint (ms)" if x_mode == "time" else "Centered data index"
    labels = info["labels"]
    clusters = sorted(medians)
    n = len(clusters)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.6 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax in axes[n:]:
        ax.axis("off")

    for ax, c in zip(axes, clusters):
        item = medians[c]
        members = item["members"]
        show_n = min(max_members_per_cluster, len(members))
        if show_n > 0:
            step = max(1, len(members) // show_n)
            sample = members[::step][:show_n]
            for prof in sample:
                ax.plot(x, prof, color="gray", alpha=0.10, linewidth=0.8)
        ax.plot(x, item["median"], color="red", linewidth=2.0, label="Median representative")
        ax.set_title(f"Cluster {c} · n={item['n']}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Blockade (nA)")
        ax.legend(frameon=False)

    fig.tight_layout()

    overlay, ax = plt.subplots(figsize=(10, 4.8))
    for c in clusters:
        ax.plot(x, medians[c]["median"], linewidth=2, label=f"Cluster {c} (n={medians[c]['n']})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Median blockade (nA)")
    ax.set_title("Median representative profiles overlaid")
    ax.legend()
    overlay.tight_layout()
    return fig, overlay


def plot_histograms_by_cluster(valid, column, xlabel, bins=30):
    clusters = sorted(valid["cluster"].unique())
    n = len(clusters)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.4 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.axis("off")
    for ax, c in zip(axes, clusters):
        vals = valid.loc[valid["cluster"] == c, column].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(vals, bins=bins)
        ax.axvline(np.median(vals), linestyle="--", linewidth=1)
        ax.set_title(f"Cluster {c} · n={len(vals)}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Event count")
    fig.tight_layout()
    return fig


def group_uploaded_triplets(files):
    grouped = {}
    pat = re.compile(r"(.+?)(\.dataset|\.event_data|\.event_fitting)\.npz$", flags=re.IGNORECASE)
    for f in files:
        m = pat.match(f.name)
        if not m:
            continue
        key, kind = m.group(1), m.group(2).lower()
        grouped.setdefault(key, {})
        grouped[key][kind] = f
    return grouped


def guess_salt_label(group_key):
    token = re.split(r"[_\-]", group_key)[0]
    return token


def lineplot_summary_across_salts(summary_df, ycol, ylabel, title):
    families = sorted(summary_df["cluster"].unique())
    salts = list(dict.fromkeys(summary_df["salt"].tolist()))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for fam in families:
        g = summary_df[summary_df["cluster"] == fam]
        g = g.set_index("salt").reindex(salts).reset_index()
        ax.plot(g["salt"], g[ycol], marker="o", label=f"Family {chr(65 + int(fam))}")
    ax.set_xlabel("Electrolyte")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def compare_waveform_profiles(results_by_salt, x_mode="time", min_common_families=None):
    salts = list(results_by_salt.keys())
    ks = []
    for salt, item in results_by_salt.items():
        med = item.get("medians") or {}
        ks.append(len(med))
    if not ks or min(ks) == 0:
        return None
    n_fam = min(ks) if min_common_families is None else min(min_common_families, min(ks))
    ncols = 2 if n_fam > 1 else 1
    nrows = int(np.ceil(n_fam / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.5 * nrows), squeeze=False, sharey=True)
    axes = axes.ravel()
    for ax in axes[n_fam:]:
        ax.axis("off")
    for fam in range(n_fam):
        ax = axes[fam]
        for salt, item in results_by_salt.items():
            info = item["profile_info"]
            med = item["medians"]
            if fam not in med:
                continue
            x = info["time_ms"] if x_mode == "time" else info["data_index"]
            ax.plot(x, med[fam]["median"], linewidth=2, label=salt)
        ax.set_title(f"Family {chr(65 + fam)}")
        ax.set_xlabel("Time relative to event midpoint (ms)" if x_mode == "time" else "Centered data index")
        ax.set_ylabel("Median blockade (nA)")
        ax.legend(frameon=False)
    fig.suptitle("Matched DNA event families across salts", y=1.01)
    fig.tight_layout()
    return fig


def cross_salt_cluster_stats(assignments, salt):
    """Per-cluster medians + event-level IQRs for cross-salt comparison."""
    d = assignments[assignments["cluster"] >= 0].copy()
    rows = []
    for c, g in d.groupby("cluster"):
        rows.append(
            dict(
                salt=salt,
                cluster=int(c),
                n=int(len(g)),
                population_pct=100.0 * len(g) / len(d),
                median_dwell_ms=float(np.nanmedian(g["width_ms"])),
                q1_dwell_ms=float(np.nanpercentile(g["width_ms"], 25)),
                q3_dwell_ms=float(np.nanpercentile(g["width_ms"], 75)),
                median_blockade_nA=float(np.nanmedian(g["height_nA"])),
                q1_blockade_nA=float(np.nanpercentile(g["height_nA"], 25)),
                q3_blockade_nA=float(np.nanpercentile(g["height_nA"], 75)),
                median_fwhm_ms=float(np.nanmedian(g["fwhm_ms"])),
                median_area_nA_ms=float(np.nanmedian(g["area_nA_ms"])),
            )
        )
    return pd.DataFrame(rows)


def mapped_cross_salt_stats(results_by_salt, mapping):
    rows = []
    for salt, item in results_by_salt.items():
        stats = item["stats"].copy()
        m = mapping.get(salt, {})
        stats["family"] = stats["cluster"].map(lambda c: m.get(int(c), "Unmapped"))
        rows.append(stats)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def cross_salt_profile_figure(results_by_salt, mapping, x_mode="time"):
    families = sorted(
        {
            fam
            for salt, m in mapping.items()
            for fam in m.values()
            if fam != "Unmapped"
        }
    )
    if not families:
        return None

    ncols = 2 if len(families) > 1 else 1
    nrows = int(np.ceil(len(families) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7 * ncols, 3.5 * nrows),
        squeeze=False,
        sharey=True,
    )
    axes = axes.ravel()

    for ax in axes[len(families):]:
        ax.axis("off")

    for ax, fam in zip(axes, families):
        for salt, item in results_by_salt.items():
            inverse = {
                family: int(cluster)
                for cluster, family in mapping.get(salt, {}).items()
                if family != "Unmapped"
            }
            if fam not in inverse:
                continue
            cluster = inverse[fam]
            info = item.get("profile_info")
            medians = item.get("medians") or {}
            if info is None or cluster not in medians:
                continue

            x = info["time_ms"] if x_mode == "time" else info["data_index"]
            ax.plot(
                x,
                medians[cluster]["median"],
                linewidth=2,
                label=salt,
            )

        ax.set_title(f"Family {fam}")
        ax.set_xlabel(
            "Time relative to event midpoint (ms)"
            if x_mode == "time"
            else "Centered data index"
        )
        ax.set_ylabel("Median blockade (nA)")
        ax.legend(frameon=False)

    fig.suptitle("Matched DNA event families across salts", y=1.01)
    fig.tight_layout()
    return fig


def cross_salt_metric_figure(
    stats,
    metric,
    q1,
    q3,
    ylabel,
    title,
):
    families = sorted([f for f in pd.unique(stats["family"]) if f != "Unmapped"])
    salts = [s for s in ["LiCl", "NaCl", "KCl", "RbCl", "CsCl"] if s in set(stats["salt"])]
    salts += [s for s in pd.unique(stats["salt"]) if s not in salts]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for fam in families:
        xs, ys, loerr, hierr = [], [], [], []
        for i, salt in enumerate(salts):
            q = stats[(stats["salt"] == salt) & (stats["family"] == fam)]
            if not len(q):
                continue
            row = q.iloc[0]
            value = float(row[metric])
            low = float(row[q1])
            high = float(row[q3])
            xs.append(i)
            ys.append(value)
            loerr.append(max(0.0, value - low))
            hierr.append(max(0.0, high - value))
        if xs:
            ax.errorbar(
                xs,
                ys,
                yerr=[loerr, hierr],
                marker="o",
                linewidth=1.5,
                capsize=3,
                label=f"Family {fam}",
            )

    ax.set_xticks(range(len(salts)), salts)
    ax.set_xlabel("Electrolyte")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def cross_salt_population_figure(stats):
    salts = [s for s in ["LiCl", "NaCl", "KCl", "RbCl", "CsCl"] if s in set(stats["salt"])]
    salts += [s for s in pd.unique(stats["salt"]) if s not in salts]
    families = sorted([f for f in pd.unique(stats["family"]) if f != "Unmapped"])
    if "Unmapped" in set(stats["family"]):
        families += ["Unmapped"]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bottom = np.zeros(len(salts))
    for fam in families:
        vals = []
        for salt in salts:
            q = stats[(stats["salt"] == salt) & (stats["family"] == fam)]
            vals.append(float(q["population_pct"].sum()) if len(q) else 0.0)
        ax.bar(salts, vals, bottom=bottom, label=("Unmapped" if fam == "Unmapped" else f"Family {fam}"))
        bottom += np.asarray(vals)

    ax.set_ylim(0, 100)
    ax.set_xlabel("Electrolyte")
    ax.set_ylabel("Population (%)")
    ax.set_title("Event-family population fractions across salts")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    return fig


def cross_salt_heatmap(stats, metric, title, value_label):
    families = sorted([f for f in pd.unique(stats["family"]) if f != "Unmapped"])
    salts = [s for s in ["LiCl", "NaCl", "KCl", "RbCl", "CsCl"] if s in set(stats["salt"])]
    salts += [s for s in pd.unique(stats["salt"]) if s not in salts]

    matrix = np.full((len(families), len(salts)), np.nan)
    for i, fam in enumerate(families):
        for j, salt in enumerate(salts):
            q = stats[(stats["salt"] == salt) & (stats["family"] == fam)]
            if len(q):
                matrix[i, j] = float(q.iloc[0][metric])

    fig, ax = plt.subplots(figsize=(1.3 * max(5, len(salts)), 1.0 * max(3, len(families) + 1)))
    im = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(salts)), salts)
    ax.set_yticks(range(len(families)), [f"Family {f}" for f in families])
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_label)

    for i in range(len(families)):
        for j in range(len(salts)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")

    fig.tight_layout()
    return fig


def cross_salt_blockade_dwell_panels(results_by_salt):
    salts = list(results_by_salt.keys())
    if not salts:
        return None

    all_valid = [
        item["result"]["assignments"].query("cluster >= 0")
        for item in results_by_salt.values()
    ]
    xmin = min(float(d["width_ms"].min()) for d in all_valid)
    xmax = max(float(d["width_ms"].max()) for d in all_valid)
    ymin = min(float(d["height_nA"].min()) for d in all_valid)
    ymax = max(float(d["height_nA"].max()) for d in all_valid)

    ncols = 2 if len(salts) > 1 else 1
    nrows = int(np.ceil(len(salts) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.2 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax in axes[len(salts):]:
        ax.axis("off")

    for ax, salt in zip(axes, salts):
        valid = results_by_salt[salt]["result"]["assignments"]
        valid = valid[valid["cluster"] >= 0]
        for c, g in valid.groupby("cluster"):
            ax.scatter(g["width_ms"], g["height_nA"], s=12, alpha=0.40, label=f"Cluster {c}")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(salt)
        ax.set_xlabel("Event width / dwell-like duration (ms)")
        ax.set_ylabel("Blockade height (nA)")
        ax.legend(frameon=False)

    fig.suptitle("Blockade–dwell distributions with shared axes", y=1.01)
    fig.tight_layout()
    return fig


# -----------------------------
# UI tabs
# -----------------------------
tabs = st.tabs(
    [
        "1 · Load files",
        "2 · K-means clustering",
        "3 · ΔI gates",
        "4 · Segment-weighted analysis",
        "5 · Export",
        "6 · Multi-salt compare",
    ]
)

# -------------------------------------------------------------------
# 1. LOAD
# -------------------------------------------------------------------
with tabs[0]:
    st.header("Load one recording")

    c1, c2, c3 = st.columns(3)
    dataset_file = c1.file_uploader("dataset.npz", type=["npz"], key="dataset_upload")
    eventdata_file = c2.file_uploader("event_data.npz", type=["npz"], key="eventdata_upload")
    fitting_file = c3.file_uploader("event_fitting.npz", type=["npz"], key="fitting_upload")

    tolerance = st.number_input(
        "Timestamp matching tolerance (s)",
        min_value=1e-10,
        max_value=1e-3,
        value=1e-7,
        format="%.1e",
    )

    if st.button("Load and validate", type="primary"):
        if not (dataset_file and eventdata_file and fitting_file):
            st.error("Upload all three files first.")
        else:
            try:
                recording = load_recording(
                    read_uploaded_bytes(dataset_file),
                    read_uploaded_bytes(eventdata_file),
                    read_uploaded_bytes(fitting_file),
                    dataset_file.name,
                    eventdata_file.name,
                    fitting_file.name,
                    float(tolerance),
                )
                st.session_state.loaded = recording
                st.session_state.cluster_result = None
                st.success("Files loaded.")
            except Exception as exc:
                st.exception(exc)

    p = st.session_state.loaded
    if p:
        df = p["df"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Dataset events", len(df))
        m2.metric("Dataset ↔ fitting matched", len(p["fit_to_dataset"]))
        m3.metric("Fitting ↔ eventdata matched", len(p["fit_to_raw"]))
        m4.metric("Dataset columns", p["X"].shape[1])

        st.subheader("NanoSense feature table")
        st.dataframe(df.head(30), use_container_width=True)

        with st.expander("Dataset ↔ event-fitting matching audit"):
            st.dataframe(p["ds_status"], use_container_width=True)

        with st.expander("Event-fitting ↔ event-data matching audit"):
            st.dataframe(p["raw_status"], use_container_width=True)

        st.info(
            "Clustering uses dataset.npz X[:,0:7]: height, FWHM, height-at-FWHM, "
            "area, width, skew and kurtosis. Baseline/time bookkeeping columns are not clustering coordinates."
        )

# -------------------------------------------------------------------
# 2. K-MEANS
# -------------------------------------------------------------------
with tabs[1]:
    st.header("Reproducible K-means clustering")
    p = st.session_state.loaded

    if not p:
        st.info("Load the three files in Step 1 first.")
    else:
        df = p["df"].copy()
        X_features = df[FEATURE_NAMES].to_numpy(float)
        finite_mask = np.all(np.isfinite(X_features), axis=1)
        X_valid = X_features[finite_mask]
        valid_df = df.loc[finite_mask].copy().reset_index(drop=True)
        X_scaled, feature_min, feature_max, constant = minmax_scale(X_valid)

        if np.any(constant):
            names = [FEATURE_NAMES[i] for i in np.flatnonzero(constant)]
            st.warning("Constant feature(s): " + ", ".join(names))

        c1, c2, c3, c4 = st.columns(4)
        mode = c1.radio("Choose k", ["Manual", "Silhouette"], horizontal=False)
        manual_k = c2.number_input("Manual k", min_value=2, max_value=12, value=4, step=1)
        kmax = c3.number_input("Maximum k to test", min_value=3, max_value=12, value=8, step=1)
        n_init = c4.number_input("K-means initialisations", min_value=10, max_value=300, value=100, step=10)
        random_state = st.number_input("Random seed", min_value=0, max_value=1_000_000, value=42, step=1)

        if st.button("Run K-means", type="primary"):
            try:
                st.session_state.cluster_result = run_kmeans_pipeline(
                    p, mode, int(manual_k), int(kmax), int(n_init), int(random_state)
                )
                st.success(f"K-means complete: k={st.session_state.cluster_result['k']}")
            except Exception as exc:
                st.exception(exc)

        r = st.session_state.cluster_result
        if r:
            assignments = r["assignments"]
            valid = assignments[assignments["cluster"] >= 0].copy()
            scan = r["scan"]

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Selected k", r["k"])
            d2.metric("Silhouette", f'{r["diagnostics"]["silhouette"]:.3f}')
            d3.metric("Davies–Bouldin", f'{r["diagnostics"]["davies_bouldin"]:.3f}')
            d4.metric("Calinski–Harabasz", f'{r["diagnostics"]["calinski_harabasz"]:.1f}')

            if np.isfinite(r["diagnostics"]["silhouette"]) and r["diagnostics"]["silhouette"] < 0.25:
                st.warning(
                    "Silhouette is below 0.25. Treat these as candidate signal families rather than proven discrete populations."
                )

            st.subheader("k diagnostics")
            st.dataframe(scan.round(4), use_container_width=True)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(scan["k"], scan["silhouette"], marker="o", label="Silhouette")
            ax.set_xlabel("Number of clusters, k")
            ax.set_ylabel("Silhouette")
            ax2 = ax.twinx()
            ax2.plot(scan["k"], scan["inertia"], marker="s", label="Inertia")
            ax2.set_ylabel("Within-cluster dispersion")
            ax.set_title("K selection")
            st.pyplot(fig, use_container_width=True)

            st.subheader("Cluster summary")
            summary = cluster_summary_table(valid)
            st.dataframe(summary.round(4), use_container_width=True)

            st.subheader("Physical plots")
            plot_choice = st.selectbox(
                "Plot",
                [
                    "Blockade vs dwell",
                    "Dwell-time histograms by cluster",
                    "Blockade histograms by cluster",
                    "FWHM vs total width",
                    "FWHM / width",
                    "Area vs dwell",
                    "Population fraction",
                    "Waveform family panels + overlay",
                ],
            )

            if plot_choice == "Blockade vs dwell":
                fig, ax = plt.subplots(figsize=(8, 5))
                for c, g in valid.groupby("cluster"):
                    ax.scatter(g["width_ms"], g["height_nA"], s=16, alpha=0.45, label=f"Cluster {c}")
                    ax.scatter(g["width_ms"].median(), g["height_nA"].median(), marker="X", s=100)
                ax.set_xlabel("Event width / dwell-like duration (ms)")
                ax.set_ylabel("Blockade height (nA)")
                ax.legend()
                st.pyplot(fig, use_container_width=True)

            elif plot_choice == "Dwell-time histograms by cluster":
                fig = plot_histograms_by_cluster(valid, "width_ms", "Event width / dwell-like duration (ms)", bins=30)
                st.pyplot(fig, use_container_width=True)

            elif plot_choice == "Blockade histograms by cluster":
                fig = plot_histograms_by_cluster(valid, "height_nA", "Blockade height (nA)", bins=30)
                st.pyplot(fig, use_container_width=True)

            elif plot_choice == "FWHM vs total width":
                fig, ax = plt.subplots(figsize=(8, 5))
                for c, g in valid.groupby("cluster"):
                    ax.scatter(g["width_ms"], g["fwhm_ms"], s=16, alpha=0.45, label=f"Cluster {c}")
                mx = max(valid["width_ms"].max(), valid["fwhm_ms"].max())
                ax.plot([0, mx], [0, mx], "--", linewidth=1)
                ax.set_xlabel("Total event width (ms)")
                ax.set_ylabel("FWHM (ms)")
                ax.legend()
                st.pyplot(fig, use_container_width=True)

            elif plot_choice == "FWHM / width":
                fig, ax = plt.subplots(figsize=(8, 5))
                groups = sorted(valid["cluster"].unique())
                arr = [
                    valid.loc[valid.cluster == c, "fwhm_fraction"]
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                    .to_numpy()
                    for c in groups
                ]
                ax.boxplot(arr, labels=[f"Cluster {c}" for c in groups], showfliers=False)
                ax.set_ylabel("FWHM / event width")
                ax.set_title("Fraction of event spent above half maximum")
                st.pyplot(fig, use_container_width=True)

            elif plot_choice == "Area vs dwell":
                fig, ax = plt.subplots(figsize=(8, 5))
                for c, g in valid.groupby("cluster"):
                    ax.scatter(g["width_ms"], g["area_nA_ms"], s=16, alpha=0.45, label=f"Cluster {c}")
                ax.set_xlabel("Event width / dwell-like duration (ms)")
                ax.set_ylabel("Integrated area (nA·ms)")
                ax.legend()
                st.pyplot(fig, use_container_width=True)

            elif plot_choice == "Population fraction":
                pop = valid["cluster"].value_counts().sort_index()
                pct = 100 * pop / pop.sum()
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.bar([f"Cluster {c}" for c in pct.index], pct.values)
                ax.set_ylabel("Population (%)")
                for i, value in enumerate(pct.values):
                    ax.text(i, value, f"{value:.1f}%", ha="center", va="bottom")
                st.pyplot(fig, use_container_width=True)

            else:
                c1, c2 = st.columns([1, 1])
                x_mode = c1.radio("Waveform x-axis", ["time", "data_index"], horizontal=True)
                win = c2.slider("Centered window (samples)", 200, 3000, 850, 50)
                info, medians = get_waveform_family_data(p, assignments, window_samples=win)
                if not medians:
                    st.warning("No matched event-fitting traces available for family waveforms.")
                else:
                    fig_panels, fig_overlay = plot_waveform_family_panels(info, medians, x_mode=x_mode)
                    st.pyplot(fig_panels, use_container_width=True)
                    st.caption(
                        "Grey curves are a deterministic sample of real member traces. The red curve is the pointwise median representative profile."
                    )
                    st.pyplot(fig_overlay, use_container_width=True)

            st.subheader("PCA contribution analysis")
            scores = r["scores"]
            explained = r["explained"]
            loadings = r["loadings"]

            c1, c2 = st.columns(2)

            fig, ax = plt.subplots(figsize=(7, 5))
            for c in sorted(np.unique(r["labels"])):
                mask = r["labels"] == c
                ax.scatter(scores[mask, 0], scores[mask, 1], s=15, alpha=0.5, label=f"Cluster {c}")
            ax.set_xlabel(f"PC1 ({100*explained[0]:.1f}% variance)")
            ax.set_ylabel(f"PC2 ({100*explained[1]:.1f}% variance)")
            ax.legend()
            c1.pyplot(fig, use_container_width=True)

            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(len(FEATURE_NAMES))
            w = 0.35
            ax.bar(x - w / 2, loadings[0], width=w, label="PC1")
            ax.bar(x + w / 2, loadings[1], width=w, label="PC2")
            ax.axhline(0, linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(FEATURE_NAMES, rotation=35, ha="right")
            ax.set_ylabel("PCA loading")
            ax.legend()
            c2.pyplot(fig, use_container_width=True)

            contribution = explained[0] * loadings[0] ** 2 + explained[1] * loadings[1] ** 2
            contribution_pct = 100 * contribution / contribution.sum()
            eta = [
                eta_squared(valid[col].to_numpy(), valid["cluster"].to_numpy())
                for col in FEATURE_NAMES
            ]

            contribution_table = pd.DataFrame(
                {
                    "feature": FEATURE_NAMES,
                    "PC1_PC2_contribution_pct": contribution_pct,
                    "eta_squared_cluster_separation": eta,
                }
            ).sort_values("eta_squared_cluster_separation", ascending=False)

            st.dataframe(contribution_table.round(4), use_container_width=True)

            st.download_button(
                "Download cluster assignments CSV",
                assignments.to_csv(index=False),
                "cluster_assignments.csv",
                "text/csv",
            )

# -------------------------------------------------------------------
# 3. DELTA-I GATES
# -------------------------------------------------------------------
with tabs[2]:
    st.header("ΔI histogram, current gates and dwell-time histograms")
    p = st.session_state.loaded

    if not p:
        st.info("Load the files first.")
    else:
        df = p["df"].copy()
        metric_options = {
            "NanoSense height / ΔI": "height_nA",
            "Height at FWHM": "height_at_fwhm_nA",
        }

        segment_df = segment_metrics(p["fitting_arrays"], p["fit_to_dataset"])
        if len(segment_df):
            weighted_map = segment_df.dropna(subset=["dataset_row"]).set_index("dataset_row")
            df["weighted_mean_deltaI_nA"] = df["dataset_row"].map(
                weighted_map["duration_weighted_mean_blockade_nA"]
            )
            metric_options["Segment-duration-weighted mean ΔI"] = "weighted_mean_deltaI_nA"

        metric_label = st.selectbox("Quantity used to gate events", list(metric_options))
        metric = metric_options[metric_label]
        values = df[metric].to_numpy(float)

        finite = np.isfinite(values)
        bins = st.slider("Histogram bins", 10, 100, 45, 5)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(values[finite], bins=bins)
        ax.set_xlabel(metric_label + " (nA)")
        ax.set_ylabel("Event count")
        ax.set_title("ΔI distribution")
        st.pyplot(fig, use_container_width=True)

        if np.any(finite):
            vmin = float(np.nanmin(values))
            vmax = float(np.nanmax(values))
            default_edges = np.linspace(vmin, vmax, 5)
            default_text = ", ".join(f"{x:.3g}" for x in default_edges)
        else:
            default_text = "0, 1, 2, 3"

        edge_text = st.text_input(
            "Gate edges (nA)",
            value=default_text,
            help="Example: 0, 1, 2, 3, 4 creates four current ranges.",
        )

        try:
            edges = parse_edges(edge_text)
            groups = gate_rows(values, edges)

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.hist(values[finite], bins=bins)
            for edge in edges:
                ax.axvline(edge, linestyle="--", linewidth=1)
            ax.set_xlabel(metric_label + " (nA)")
            ax.set_ylabel("Event count")
            ax.set_title("ΔI gates")
            st.pyplot(fig, use_container_width=True)

            gate_rows_table = []
            for label, rows in groups.items():
                sub = df.iloc[rows]
                gate_rows_table.append(
                    dict(
                        gate=label,
                        n=len(sub),
                        median_deltaI_nA=float(np.nanmedian(sub[metric])) if len(sub) else np.nan,
                        median_dwell_ms=float(np.nanmedian(sub["width_ms"])) if len(sub) else np.nan,
                    )
                )
            gate_summary = pd.DataFrame(gate_rows_table)
            st.dataframe(gate_summary.round(4), use_container_width=True)

            selected_gate = st.selectbox("Inspect gate", list(groups))
            selected_rows = groups[selected_gate]
            gate_df = df.iloc[selected_rows].copy()

            c1, c2 = st.columns(2)

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(gate_df[metric].dropna(), bins=min(bins, max(5, len(gate_df) // 5 + 1)))
            ax.set_xlabel(metric_label + " (nA)")
            ax.set_ylabel("Event count")
            ax.set_title(selected_gate)
            c1.pyplot(fig, use_container_width=True)

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(gate_df["width_ms"].dropna(), bins=min(bins, max(5, len(gate_df) // 5 + 1)))
            ax.set_xlabel("Event width / dwell-like duration (ms)")
            ax.set_ylabel("Event count")
            ax.set_title(f"Dwell-time distribution · {selected_gate}")
            c2.pyplot(fig, use_container_width=True)

            st.session_state["gate_groups"] = groups
            st.session_state["gate_table"] = df

        except Exception as exc:
            st.warning(str(exc))

# -------------------------------------------------------------------
# 4. SEGMENT WEIGHTED
# -------------------------------------------------------------------
with tabs[3]:
    st.header("Segment-weighted analysis on the original undivided recording")
    p = st.session_state.loaded

    if not p:
        st.info("Load the files first.")
    else:
        seg = segment_metrics(p["fitting_arrays"], p["fit_to_dataset"])

        if not len(seg):
            st.warning(
                "No SEGMENT_INFO_*_segment_mean_diffs / segment_widths_time arrays were found."
            )
        else:
            st.write(
                "For event segments j with blockade ΔIⱼ and duration τⱼ, the primary weighted blockade is "
                "Σ(ΔIⱼ τⱼ) / Στⱼ. This is the duration-weighted mean blockade and is equivalent to "
                "integrated segmented blockade divided by total segmented dwell."
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Events with segment data", len(seg))
            m2.metric("Median segment count", f'{np.nanmedian(seg["n_segments"]):.0f}')
            m3.metric(
                "Median total segmented dwell",
                f'{np.nanmedian(seg["total_segment_dwell_ms"]):.3f} ms',
            )

            st.dataframe(seg.head(50).round(5), use_container_width=True)

            time_metric_map = {
                "Total segmented dwell time": "total_segment_dwell_ms",
                "Duration-weighted mean segment duration": "duration_weighted_segment_duration_ms",
                "Blockade-weighted mean segment duration": "blockade_weighted_segment_duration_ms",
            }

            time_choice = st.selectbox("Time quantity for the weighted ΔI–time plot", list(time_metric_map))
            time_col = time_metric_map[time_choice]

            c1, c2 = st.columns(2)

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(
                seg[time_col],
                seg["duration_weighted_mean_blockade_nA"],
                s=18,
                alpha=0.5,
            )
            ax.set_xlabel(time_choice + " (ms)")
            ax.set_ylabel("Duration-weighted mean ΔI (nA)")
            ax.set_title("Weighted blockade vs time")
            c1.pyplot(fig, use_container_width=True)

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(
                seg["total_segment_dwell_ms"],
                seg["segment_ecd_signed_nA_ms"],
                s=18,
                alpha=0.5,
            )
            ax.set_xlabel("Total segmented dwell (ms)")
            ax.set_ylabel("Segment ECD (nA·ms)")
            ax.set_title("Integrated segmented blockade")
            c2.pyplot(fig, use_container_width=True)

            c1, c2 = st.columns(2)

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(seg["duration_weighted_mean_blockade_nA"].dropna(), bins=45)
            ax.set_xlabel("Duration-weighted mean ΔI (nA)")
            ax.set_ylabel("Event count")
            c1.pyplot(fig, use_container_width=True)

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(seg["total_segment_dwell_ms"].dropna(), bins=45)
            ax.set_xlabel("Total segmented dwell (ms)")
            ax.set_ylabel("Event count")
            c2.pyplot(fig, use_container_width=True)

            st.caption(
                "The default time quantity is total dwell Στⱼ because it has the clearest physical meaning. "
                "The other two weighted time summaries are provided as exploratory shape descriptors."
            )

            st.download_button(
                "Download segment-weighted event table CSV",
                seg.to_csv(index=False),
                "segment_weighted_event_metrics.csv",
                "text/csv",
            )

# -------------------------------------------------------------------
# 5. EXPORT
# -------------------------------------------------------------------
with tabs[4]:
    st.header("Export grouped files in the same three-file NPZ categories")
    p = st.session_state.loaded

    if not p:
        st.info("Load the files first.")
    else:
        source_names = p["source_names"]

        st.subheader("A. Export K-means clusters")
        r = st.session_state.cluster_result
        if not r:
            st.info("Run K-means first.")
        else:
            assignments = r["assignments"]
            groups = {}
            for c in sorted(assignments.loc[assignments.cluster >= 0, "cluster"].unique()):
                rows = assignments.loc[assignments.cluster == c, "dataset_row"].astype(int).tolist()
                groups[f"cluster_{int(c)}"] = rows

            if st.button("Prepare all cluster files ZIP"):
                blob = export_many_groups_zip(
                    groups,
                    p["dataset_arrays"],
                    p["fitting_arrays"],
                    p["eventdata_arrays"],
                    p["fit_to_dataset"],
                    p["fit_to_raw"],
                    assignments,
                    source_names=source_names,
                )
                st.session_state["cluster_export_blob"] = blob

            if "cluster_export_blob" in st.session_state:
                st.download_button(
                    "Download cluster files",
                    st.session_state["cluster_export_blob"],
                    "kmeans_cluster_files.zip",
                    "application/zip",
                )

        st.divider()
        st.subheader("B. Export ΔI gates")

        gate_groups = st.session_state.get("gate_groups")
        gate_table = st.session_state.get("gate_table")
        if not gate_groups or gate_table is None:
            st.info("Create ΔI gates in Step 3 first.")
        else:
            if st.button("Prepare all gate files ZIP"):
                blob = export_many_groups_zip(
                    gate_groups,
                    p["dataset_arrays"],
                    p["fitting_arrays"],
                    p["eventdata_arrays"],
                    p["fit_to_dataset"],
                    p["fit_to_raw"],
                    gate_table,
                    source_names=source_names,
                )
                st.session_state["gate_export_blob"] = blob

            if "gate_export_blob" in st.session_state:
                st.download_button(
                    "Download current-gated files",
                    st.session_state["gate_export_blob"],
                    "deltaI_gate_files.zip",
                    "application/zip",
                )

        st.info(
            "Exports preserve the original dataset X rows and preserve event-indexed arrays from the uploaded "
            "event-fitting/event-data files wherever they can be matched. Original event IDs are retained rather "
            "than renumbered. A provenance JSON and CSV are included in every group ZIP."
        )

# -------------------------------------------------------------------
# 6. MULTI-SALT COMPARE
# -------------------------------------------------------------------
with tabs[5]:
    st.header("Multiple salts: individual plots first, then combined comparison")
    st.write(
        "Upload the three matching files separately for each salt. "
        "This avoids relying on identical timestamps in the filenames."
    )

    salts = ["LiCl", "NaCl", "KCl", "RbCl", "CsCl"]
    salt_uploads = {}

    for salt in salts:
        with st.expander(f"{salt} files", expanded=False):
            c1, c2, c3 = st.columns(3)
            ds = c1.file_uploader(
                f"{salt} · dataset.npz",
                type=["npz"],
                key=f"multi_{salt}_dataset",
            )
            raw = c2.file_uploader(
                f"{salt} · event_data.npz",
                type=["npz"],
                key=f"multi_{salt}_eventdata",
            )
            fit = c3.file_uploader(
                f"{salt} · event_fitting.npz",
                type=["npz"],
                key=f"multi_{salt}_fitting",
            )
            salt_uploads[salt] = dict(dataset=ds, eventdata=raw, fitting=fit)

    st.subheader("Use the same clustering settings for every salt")
    c1, c2, c3, c4, c5 = st.columns(5)
    cmp_mode = c1.radio(
        "Choose k",
        ["Manual", "Silhouette"],
        horizontal=False,
        key="cmp_mode",
    )
    cmp_manual_k = c2.number_input(
        "Manual k",
        min_value=2,
        max_value=12,
        value=4,
        step=1,
        key="cmp_manual_k",
    )
    cmp_kmax = c3.number_input(
        "Maximum k to test",
        min_value=3,
        max_value=12,
        value=8,
        step=1,
        key="cmp_kmax",
    )
    cmp_n_init = c4.number_input(
        "K-means initialisations",
        min_value=10,
        max_value=300,
        value=100,
        step=10,
        key="cmp_n_init",
    )
    cmp_seed = c5.number_input(
        "Random seed",
        min_value=0,
        max_value=1_000_000,
        value=42,
        step=1,
        key="cmp_seed",
    )

    cmp_tol = st.number_input(
        "Timestamp matching tolerance (s)",
        min_value=1e-10,
        max_value=1e-3,
        value=1e-7,
        format="%.1e",
        key="cmp_tol",
    )
    cmp_window = st.slider(
        "Centered waveform window (samples)",
        200,
        3000,
        850,
        50,
        key="cmp_window",
    )
    cmp_x_mode = st.radio(
        "Waveform x-axis",
        ["time", "data_index"],
        horizontal=True,
        key="cmp_x_mode",
    )

    if st.button("Run salt analyses", type="primary"):
        results_by_salt = {}
        errors = []

        for salt in salts:
            files = salt_uploads[salt]
            present = [files["dataset"] is not None, files["eventdata"] is not None, files["fitting"] is not None]

            if not any(present):
                continue
            if not all(present):
                errors.append(f"{salt}: upload all three files or leave all three empty.")
                continue

            try:
                rec = load_recording(
                    read_uploaded_bytes(files["dataset"]),
                    read_uploaded_bytes(files["eventdata"]),
                    read_uploaded_bytes(files["fitting"]),
                    files["dataset"].name,
                    files["eventdata"].name,
                    files["fitting"].name,
                    float(cmp_tol),
                )

                result = run_kmeans_pipeline(
                    rec,
                    cmp_mode,
                    int(cmp_manual_k),
                    int(cmp_kmax),
                    int(cmp_n_init),
                    int(cmp_seed),
                )

                stats = cross_salt_cluster_stats(result["assignments"], salt)
                info, medians = get_waveform_family_data(
                    rec,
                    result["assignments"],
                    window_samples=int(cmp_window),
                )

                results_by_salt[salt] = dict(
                    recording=rec,
                    result=result,
                    stats=stats,
                    profile_info=info,
                    medians=medians,
                )

            except Exception as exc:
                errors.append(f"{salt}: {exc}")

        for err in errors:
            st.error(err)

        if results_by_salt:
            st.session_state.multi_compare = dict(
                results_by_salt=results_by_salt,
                x_mode=cmp_x_mode,
                window=int(cmp_window),
            )
            st.success(f"Analysed {len(results_by_salt)} salt recording(s).")
        elif not errors:
            st.warning("Upload at least one complete three-file salt recording.")

    mc = st.session_state.multi_compare

    if mc:
        results_by_salt = mc["results_by_salt"]

        # -------------------------------------------------------
        # Individual salt results
        # -------------------------------------------------------
        st.divider()
        st.header("A · Individual salt results")
        st.caption(
            "Each salt is clustered independently, but with the same global K-means settings."
        )

        for salt, item in results_by_salt.items():
            with st.expander(f"{salt} · individual results", expanded=True):
                result = item["result"]
                assignments = result["assignments"]
                valid = assignments[assignments["cluster"] >= 0].copy()

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("k", result["k"])
                m2.metric("Silhouette", f'{result["diagnostics"]["silhouette"]:.3f}')
                m3.metric("Davies–Bouldin", f'{result["diagnostics"]["davies_bouldin"]:.3f}')
                m4.metric("Events", len(valid))

                st.dataframe(item["stats"].round(4), use_container_width=True)

                wtab, bdtab, htab, ptab = st.tabs(
                    [
                        "Waveform families",
                        "Blockade–dwell",
                        "Histograms",
                        "Population",
                    ]
                )

                with wtab:
                    if item["medians"]:
                        family_panels, overlay = plot_waveform_family_panels(
                            item["profile_info"],
                            item["medians"],
                            x_mode=mc["x_mode"],
                        )
                        st.pyplot(family_panels, use_container_width=True)
                        st.caption(
                            "Grey traces are real cluster members; red is the pointwise median representative."
                        )
                        st.pyplot(overlay, use_container_width=True)
                    else:
                        st.warning("No matched event-fitting traces were available.")

                with bdtab:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    for c, g in valid.groupby("cluster"):
                        ax.scatter(
                            g["width_ms"],
                            g["height_nA"],
                            s=16,
                            alpha=0.45,
                            label=f"Cluster {c}",
                        )
                        ax.scatter(
                            g["width_ms"].median(),
                            g["height_nA"].median(),
                            marker="X",
                            s=100,
                        )
                    ax.set_xlabel("Event width / dwell-like duration (ms)")
                    ax.set_ylabel("Blockade height (nA)")
                    ax.set_title(f"{salt} · blockade vs dwell")
                    ax.legend()
                    st.pyplot(fig, use_container_width=True)

                with htab:
                    st.pyplot(
                        plot_histograms_by_cluster(
                            valid,
                            "width_ms",
                            "Event width / dwell-like duration (ms)",
                            bins=30,
                        ),
                        use_container_width=True,
                    )
                    st.pyplot(
                        plot_histograms_by_cluster(
                            valid,
                            "height_nA",
                            "Blockade height (nA)",
                            bins=30,
                        ),
                        use_container_width=True,
                    )

                with ptab:
                    pop = valid["cluster"].value_counts().sort_index()
                    pct = 100 * pop / pop.sum()
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.bar([f"Cluster {c}" for c in pct.index], pct.values)
                    ax.set_ylabel("Population (%)")
                    ax.set_title(f"{salt} · cluster population")
                    for i, value in enumerate(pct.values):
                        ax.text(i, value, f"{value:.1f}%", ha="center", va="bottom")
                    st.pyplot(fig, use_container_width=True)

        # -------------------------------------------------------
        # Family mapping
        # -------------------------------------------------------
        st.divider()
        st.header("B · Match salt-specific clusters to common families")
        st.info(
            "Cluster IDs are not assumed homologous across salts. "
            "The default suggestion follows increasing median blockade, but you can change any assignment."
        )

        family_options = ["Unmapped"] + [chr(65 + i) for i in range(10)]
        mapping = {}

        for salt, item in results_by_salt.items():
            clusters = sorted(item["stats"]["cluster"].astype(int).tolist())
            st.markdown(f"**{salt}**")
            cols = st.columns(min(5, max(1, len(clusters))))
            mapping[salt] = {}
            for j, cluster in enumerate(clusters):
                default_family = chr(65 + j)
                default_index = family_options.index(default_family)
                family = cols[j % len(cols)].selectbox(
                    f"Cluster {cluster}",
                    family_options,
                    index=default_index,
                    key=f"family_map_{salt}_{cluster}",
                )
                mapping[salt][int(cluster)] = family

        stats = mapped_cross_salt_stats(results_by_salt, mapping)

        st.subheader("Matched family summary")
        st.dataframe(
            stats[
                [
                    "salt",
                    "cluster",
                    "family",
                    "n",
                    "population_pct",
                    "median_dwell_ms",
                    "q1_dwell_ms",
                    "q3_dwell_ms",
                    "median_blockade_nA",
                    "q1_blockade_nA",
                    "q3_blockade_nA",
                ]
            ].round(4),
            use_container_width=True,
        )

        # -------------------------------------------------------
        # Combined plots
        # -------------------------------------------------------
        st.divider()
        st.header("C · Combined cross-salt plots")

        st.subheader("1 · Blockade–dwell distributions for all salts")
        st.caption("Every panel uses the same x and y limits, so the recordings can be compared directly.")
        st.pyplot(
            cross_salt_blockade_dwell_panels(results_by_salt),
            use_container_width=True,
        )

        st.subheader("2 · Matched median event-family shapes across salts")
        family_fig = cross_salt_profile_figure(
            results_by_salt,
            mapping,
            x_mode=mc["x_mode"],
        )
        if family_fig is not None:
            st.pyplot(family_fig, use_container_width=True)
        else:
            st.warning("No mapped waveform families are available.")

        st.subheader("3 · Family population fractions")
        st.pyplot(
            cross_salt_population_figure(stats),
            use_container_width=True,
        )

        st.subheader("4 · Family-resolved dwell time")
        st.caption(
            "Points are event-level medians; vertical bars show the event-level IQR (Q1–Q3), not replicate uncertainty."
        )
        st.pyplot(
            cross_salt_metric_figure(
                stats,
                "median_dwell_ms",
                "q1_dwell_ms",
                "q3_dwell_ms",
                "Median dwell time (ms)",
                "Family-resolved dwell time across salts",
            ),
            use_container_width=True,
        )

        st.subheader("5 · Family-resolved blockade")
        st.pyplot(
            cross_salt_metric_figure(
                stats,
                "median_blockade_nA",
                "q1_blockade_nA",
                "q3_blockade_nA",
                "Median blockade (nA)",
                "Family-resolved blockade across salts",
            ),
            use_container_width=True,
        )

        st.subheader("6 · Cross-salt heatmaps")
        c1, c2 = st.columns(2)
        c1.pyplot(
            cross_salt_heatmap(
                stats,
                "median_dwell_ms",
                "Median dwell time",
                "ms",
            ),
            use_container_width=True,
        )
        c2.pyplot(
            cross_salt_heatmap(
                stats,
                "median_blockade_nA",
                "Median blockade",
                "nA",
            ),
            use_container_width=True,
        )

        st.download_button(
            "Download cross-salt matched-family summary CSV",
            stats.to_csv(index=False),
            "cross_salt_family_summary.csv",
            "text/csv",
        )
