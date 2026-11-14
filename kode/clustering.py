"""Reproduce customer segmentation, stability checks, profiles and cluster.csv."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
from itertools import combinations
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score, silhouette_score, pairwise_distances_argmin
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
ID, SEED = "id_pelanggan", 42
K_RANGE = range(2, 9)
METHODS = ("KMeans", "GMM diagonal", "Ward")
MAIN = {
    "pembelian": (["recency_hari", "frekuensi_90h", "rata2_keranjang"], 1.0),
    "promosi_pembayaran": (["persen_voucher", "persen_flash_sale", "persen_paylater", "persen_cod"], .70),
    "aktivitas": (["sesi_per_minggu", "durasi_sesi_menit", "rasio_konversi"], .65),
}
LOG = set(MAIN["pembelian"][0] + ["sesi_per_minggu", "durasi_sesi_menit"])
CAT = ["kota_tier", "perangkat"]


def blocks(data):
    time = [c for c in data if c.startswith("porsi_jam_")]
    category = [c for c in data if c.startswith("porsi_") and c not in time]
    crm = [c for c in data if c[1:].isdigit() and c[0] == "n"]
    assert len(time) == 4 and len(category) == 16 and len(crm) >= 10
    return category, time, crm


def audit(data):
    category, time, crm = blocks(data)
    info = {
        "rows": len(data), "columns": len(data.columns),
        "missing_ids": int(data[ID].isna().sum()), "duplicated_ids": int(data[ID].duplicated().sum()),
        "missing_basket": int(data.rata2_keranjang.isna().sum()),
        "no_orders": int(data.frekuensi_90h.eq(0).sum()),
        "basket_missing_matches_no_orders": bool(data.rata2_keranjang.isna().equals(data.frekuensi_90h.eq(0))),
        "conversion_out_of_bounds": int((data.rasio_konversi.notna() & ~data.rasio_konversi.between(0, 1)).sum()),
        "spend_30_gt_90": int(data.belanja_30h.gt(data.belanja_90h).sum()),
        "crm_actual": crm,
        "spend_median": float(data.belanja_90h.median()),
        "spend_p99": float(data.belanja_90h.quantile(.99)),
        "spend_max": float(data.belanja_90h.max()),
        "spearman_30_90_spend": float(data[["belanja_30h", "belanja_90h"]].corr(method="spearman").iloc[0, 1]),
        "spearman_30_90_count": float(data[["frekuensi_30h", "frekuensi_90h"]].corr(method="spearman").iloc[0, 1]),
        "composition": {},
    }
    for name, cols in [("kategori", category), ("waktu", time)]:
        x = data[cols]
        info["composition"][name] = {
            "zeros_rows": int(x.eq(0).any(axis=1).sum()),
            "max_sum_error": float(x.sum(axis=1).sub(1).abs().max()),
        }
        assert x.notna().all().all() and x.ge(0).all().all() and x.le(1).all().all()
        assert info["composition"][name]["max_sum_error"] < .001
    assert info["missing_ids"] == info["duplicated_ids"] == 0
    return info


def transform(data, fit, variant):
    """Learn imputation, scaling and clipping only from the supplied fit rows."""
    category, time, crm = blocks(data)
    parts, tails = [], []
    groups = dict(MAIN)
    if variant == "tambah_crm":
        groups["crm"] = (crm, 1.0)
    for cols, weight in groups.values():
        x = data[cols].copy()
        for col in cols:
            if col in LOG:
                x[col] = np.log1p(x[col].where(x[col].ge(0)))
        if "rasio_konversi" in cols:
            x["rasio_konversi"] = x.rasio_konversi.where(x.rasio_konversi.between(0, 1))
        # The basket is undefined when there are no purchases; use the learned median
        # to avoid pretending that the price of an observed basket was zero.
        med = x.iloc[fit].median()
        x = x.fillna(med)
        lo, hi = x.iloc[fit].quantile(.005), x.iloc[fit].quantile(.995)
        clipped = x.clip(lo, hi, axis=1)
        spread = (clipped.iloc[fit].quantile(.75) - clipped.iloc[fit].quantile(.25)).replace(0, 1)
        scale = weight / np.sqrt(len(cols))
        parts.append(((clipped - med) / spread).to_numpy() * scale)
        tails.append(((x - med) / spread).to_numpy() * scale)
    if variant != "tanpa_minat_dan_konteks":
        for cols, weight in [(category, .48), (time, .28)]:
            raw = data[cols].to_numpy().copy()
            zero_orders = data.frekuensi_90h.eq(0).to_numpy()
            # With no purchases, don't interpret generated purchase shares as preference.
            raw[zero_orders] = np.median(raw[np.asarray(fit)[~zero_orders[fit]]], axis=0)
            root = np.sqrt(raw / raw.sum(axis=1, keepdims=True))
            parts.append(root * weight)
            tails.append(root * weight)
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        cats = enc.fit(data.iloc[fit][CAT]).transform(data[CAT]) * (.16 / np.sqrt(2))
        parts.append(cats)
        tails.append(cats)
    x, tail = (np.column_stack(items).astype("float32") for items in [parts, tails])
    assert np.isfinite(x).all() and np.isfinite(tail).all()
    return x, tail


def fit_method(method, x, k, seed, ward_tree=None):
    if method == "KMeans":
        return KMeans(n_clusters=k, n_init=5, random_state=seed).fit(x)
    if method == "GMM diagonal":
        model = GaussianMixture(n_components=k, covariance_type="diag", reg_covar=.003,
                                n_init=2, max_iter=200, tol=.001, random_state=seed).fit(x)
        if not model.converged_:
            raise RuntimeError(f"GMM belum konvergen untuk k={k}, seed={seed}.")
        return model
    tree = ward_tree if ward_tree is not None else linkage(x, method="ward")
    labels = cut_tree(tree, n_clusters=[k])[:, 0]
    return np.stack([x[labels == i].mean(axis=0) for i in range(k)])


def predict(model, x):
    return pairwise_distances_argmin(x, model) if isinstance(model, np.ndarray) else model.predict(x)


def centers(model):
    if isinstance(model, np.ndarray):
        return model
    return model.means_ if isinstance(model, GaussianMixture) else model.cluster_centers_


def choose_indices(pool, n, seed):
    return np.sort(np.random.default_rng(seed).choice(pool, size=n, replace=False))


def tune(data, reference, pool):
    fit = choose_indices(pool, 9000, SEED)
    limit = {"KMeans": len(fit), "GMM diagonal": len(fit), "Ward": 3500}
    ward_fit = choose_indices(fit, limit["Ward"], SEED)
    x, _ = transform(data, fit, "utama")
    ward_tree = linkage(x[ward_fit], method="ward")
    seeds = [101, 102, 103]
    # Refit preprocessing as well as each model for every subsample.
    samples = {method: [] for method in METHODS}
    for seed in seeds:
        sample = choose_indices(fit, 7200, seed)
        z, _ = transform(data, sample, "utama")
        small = choose_indices(sample, 2800, seed)
        samples["Ward"].append((z[small], z[reference], linkage(z[small], method="ward")))
        for method in ("KMeans", "GMM diagonal"):
            samples[method].append((z[sample], z[reference], None))
    rows, details = [], []
    for method in METHODS:
        print("Comparing", method, flush=True)
        for k in K_RANGE:
            xf = x[ward_fit if method == "Ward" else fit]
            model = fit_method(method, xf, k, SEED, ward_tree if method == "Ward" else None)
            ref = x[reference]
            lab = predict(model, ref)
            fraction = np.bincount(lab, minlength=k) / len(lab)
            sil = silhouette_score(ref, lab) if len(np.unique(lab)) > 1 else -1.0
            predictions = [predict(fit_method(method, sub_x, k, seed, tree), sub_ref)
                           for seed, (sub_x, sub_ref, tree) in zip(seeds, samples[method])]
            ari = []
            for i, j in combinations(range(len(seeds)), 2):
                value = float(adjusted_rand_score(predictions[i], predictions[j]))
                details.append(dict(method=method, k=k, seed_a=seeds[i], seed_b=seeds[j], ari=value))
                ari.append(value)
            eligible = fraction.min() >= .03 and fraction.max() <= .80
            rows.append(dict(method=method, k=k, n_fit=len(xf), n_reference=len(ref),
                             silhouette=sil, ari_mean=float(np.mean(ari)), ari_min=min(ari),
                             min_fraction=float(fraction.min()), max_fraction=float(fraction.max()),
                             bic=float(model.bic(xf)) if method == "GMM diagonal" else None,
                             eligible=eligible, selection_score=sil - .25*(1-np.mean(ari))))
    table = pd.DataFrame(rows)
    chosen = table.loc[table.eligible].sort_values(["selection_score", "k", "method"], ascending=[False, True, True]).iloc[0]
    return table, pd.DataFrame(details), str(chosen.method), int(chosen.k), fit


def feature_sensitivity(data, method, k, fit, reference):
    rows, base = [], None
    model_fit = choose_indices(fit, 3500, SEED) if method == "Ward" else fit
    for name in ["utama", "tanpa_minat_dan_konteks", "tambah_crm"]:
        x, _ = transform(data, fit, name)
        model = fit_method(method, x[model_fit], k, SEED)
        lab = predict(model, x[reference])
        if base is None:
            base = lab
        rows.append(dict(representation=name, features=x.shape[1],
                         silhouette=silhouette_score(x[reference], lab),
                         ari_vs_main=adjusted_rand_score(base, lab)))
    return pd.DataFrame(rows)


def label_all(data, method, k, pool, reference):
    all_idx = np.arange(len(data))
    x, tail = transform(data, all_idx, "utama")
    sample = choose_indices(pool, 7000, SEED) if method == "Ward" else all_idx
    model = fit_method(method, x[sample], k, SEED)
    groups = predict(model, x)
    assert set(groups) == set(range(k))
    distance = np.linalg.norm(tail - centers(model)[groups], axis=1)
    # Consensus: very distant from assigned group AND in the extreme global novelty tail.
    detector_idx = choose_indices(all_idx, 25000, SEED)
    iso = IsolationForest(n_estimators=160, max_samples=512, random_state=SEED, n_jobs=1).fit(tail[detector_idx])
    novelty = -iso.score_samples(tail)
    anomaly_summary = []
    for percentile in [.975, .985, .99]:
        cutoff = np.array([np.quantile(distance[groups == i], percentile) for i in range(k)])
        anomaly = (distance > cutoff[groups]) & (novelty > np.quantile(novelty, .975))
        anomaly_summary.append({"distance_percentile_per_cluster": percentile, "global_isolation_percentile": .975,
                                "n": int(anomaly.sum()), "fraction": float(anomaly.mean())})
        if percentile == .985:
            selected = anomaly.copy()
    post = groups.copy()
    # Stable output labels, ordered by median purchase frequency then spending.
    by_group = data.assign(cluster=groups).groupby("cluster")[["frekuensi_90h", "belanja_90h"]].median()
    ordered = by_group.sort_values(["frekuensi_90h", "belanja_90h"], ascending=False).index
    for i, old in enumerate(ordered):
        post[groups == old] = i
    post[selected] = -1
    return post, {
        "n_fit_final": len(sample), "features": x.shape[1], "n_anomalies": int(selected.sum()),
        "silhouette_final_before_anomaly": float(silhouette_score(x[reference], groups[reference])),
        "anomaly_sensitivity": anomaly_summary,
    }


def profiles(data, labels):
    category, time, _ = blocks(data)
    rows = []
    for cluster, g in data.assign(cluster=labels).groupby("cluster", sort=True):
        active = g[g.frekuensi_90h.gt(0)]
        spend = active[category].mean().nlargest(3)
        timing = active[time].mean().nlargest(2)
        rows.append({
            "cluster": int(cluster), "n": len(g), "share": len(g)/len(data),
            "no_orders_share": float(g.frekuensi_90h.eq(0).mean()),
            "median_recency_days": float(g.recency_hari.median()),
            "median_order_count": float(g.frekuensi_90h.median()),
            "median_spend_90d": float(g.belanja_90h.median()),
            "median_basket_if_active": float(active.rata2_keranjang.median()),
            "median_voucher": float(g.persen_voucher.median()),
            "median_flash_sale": float(g.persen_flash_sale.median()),
            "median_paylater": float(g.persen_paylater.median()),
            "median_cod": float(g.persen_cod.median()),
            "median_weekly_sessions": float(g.sesi_per_minggu.median()),
            "median_session_minutes": float(g.durasi_sesi_menit.median()),
            "median_valid_conversion": float(g.loc[g.rasio_konversi.between(0,1), "rasio_konversi"].median()),
            "top_category_and_mean_shares_active": "; ".join(f"{c.removeprefix('porsi_')} {v:.4f}" for c,v in spend.items()),
            "top_times_and_mean_shares_active": "; ".join(f"{c.removeprefix('porsi_jam_')} {v:.4f}" for c,v in timing.items()),
            "top_device": g.perangkat.mode().iat[0], "top_city_tier": int(g.kota_tier.mode().iat[0]),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data_uts_ml")
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    input_path = args.data / "clustering.csv.gz"
    data = pd.read_csv(input_path)
    info = audit(data)
    all_idx = np.arange(len(data))
    reference = choose_indices(all_idx, 2600, SEED + 1)
    pool = np.setdiff1d(all_idx, reference)
    evidence = args.output / "hasil/clustering"
    evidence.mkdir(parents=True, exist_ok=True)
    table, stability, method, k, fit = tune(data, reference, pool)
    table.to_csv(evidence / "perbandingan_model.csv", index=False)
    stability.to_csv(evidence / "stabilitas_ari.csv", index=False)
    feature_sensitivity(data, method, k, fit, reference).to_csv(evidence / "sensitivitas_fitur.csv", index=False)
    print(f"Chosen {method}, k={k}. Fitting whole dataset and screening anomalies.", flush=True)
    labels, final = label_all(data, method, k, pool, reference)
    output = pd.DataFrame({ID: data[ID], "cluster": labels})
    assert output[ID].equals(data[ID]) and len(output) == len(data)
    assert set(labels).issubset({-1, *range(k)}) and len(set(labels) - {-1}) == k
    assert output[ID].is_unique and np.issubdtype(labels.dtype, np.integer)
    output.to_csv(args.output / "cluster.csv", index=False)
    profile = profiles(data, labels)
    assert profile.n.sum() == len(data)
    profile.to_csv(evidence / "profil_segmen.csv", index=False)
    with input_path.open("rb") as f:
        input_hash = hashlib.file_digest(f, "sha256").hexdigest()
    run = {"audit": info, "method": method, "k": k, "seed": SEED, "final": final,
           "n_dev_fit": len(fit), "n_reference": len(reference), "input_sha256": input_hash,
           "python": platform.python_version(),
           "versions": {p: metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]}}
    (evidence / "metrics.json").write_text(json.dumps(run, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"PASS {len(data):,} IDs; {k} segments; {final['n_anomalies']:,} anomalies.", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
