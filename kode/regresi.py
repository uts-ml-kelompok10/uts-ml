"""Reproduce the delivery-time experiment and export P50/P90 for the test set."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
ID, TARGET = "id_paket", "durasi_jam"
SCAN = "jumlah_scan"
CAT = ["layanan", "mitra_kurir", "hub_asal", "kota_asal", "pulau_tujuan", "kota_tujuan"]
MEASURES = [
    "jarak_km", "berat_kg", "volume_liter", "jumlah_item", "nilai_barang", "asuransi",
    "cod", "alamat_skor", "pengalaman_kurir_bulan", "rating_penjual",
    "umur_akun_penjual_hari", "volume_penjual_30h", "antrean_hub_asal",
    "indeks_kemacetan", "hujan_asal_mm", "hujan_tujuan_mm",
]
LOG_FEATURES = [
    "jarak_km", "berat_kg", "volume_liter", "nilai_barang", "umur_akun_penjual_hari",
    "volume_penjual_30h", "antrean_hub_asal", "hujan_asal_mm", "hujan_tujuan_mm",
]
INTERNAL = [f"f{i:02d}" for i in range(1, 51)]
CANDIDATES = [
    ("baseline", {}),
    ("ridge_log", {"alpha": 1.0}),
    ("ridge_log", {"alpha": 25.0}),
    ("lightgbm", {"num_leaves": 31, "n_estimators": 160}),
    ("lightgbm", {"num_leaves": 63, "n_estimators": 240}),
]
QUANTILES = (0.5, 0.9)


def asof(value):
    return pd.Timestamp(value)


def load_data(path):
    train, test = (pd.read_csv(path / f"regresi_{split}.csv.gz") for split in ("train", "test"))
    if set(train) - set(test) != {TARGET} or set(test) - set(train):
        raise ValueError("Train dan test hanya boleh berbeda pada kolom target.")
    for frame in (train, test):
        if frame[ID].isna().any() or not frame[ID].is_unique:
            raise ValueError("ID paket kosong atau tidak unik.")
        if not frame["jam_pickup"].between(0, 23).all():
            raise ValueError("Jam pickup di luar 0–23.")
    if set(train[ID]) & set(test[ID]):
        raise ValueError("ID train dan test beririsan.")
    if not train[TARGET].gt(0).all() or not np.isfinite(train[TARGET]).all():
        raise ValueError("Target durasi harus positif dan finite.")
    train_ts, test_ts = [
        pd.to_datetime(frame["tgl_pickup"], format="%Y-%m-%d")
        + pd.to_timedelta(frame["jam_pickup"], unit="h") for frame in (train, test)
    ]
    if not (train_ts.notna().all() and test_ts.notna().all()):
        raise ValueError("Tanggal pickup kosong.")
    if train_ts.ge(asof("2025-09-01")).any() or not (
        test_ts.ge(asof("2025-09-01")) & test_ts.lt(asof("2025-11-01"))
    ).all():
        raise ValueError("Periode pickup tidak sesuai pembagian soal.")
    return train, test, train_ts, test_ts


def features(frame, ts):
    x = frame[CAT + MEASURES + INTERNAL].copy()
    x["jarak_tidak_positif"] = x["jarak_km"].le(0).astype("int8")
    x["jarak_km"] = x["jarak_km"].where(x["jarak_km"].gt(0))
    x["berat_kosong"] = x["berat_kg"].isna().astype("int8")
    for col in LOG_FEATURES:
        x[col] = np.log1p(x[col].where(x[col].ge(0)))
    for angle, values in [("jam", ts.dt.hour / 24), ("pekan", ts.dt.dayofweek / 7)]:
        x[f"{angle}_sin"] = np.sin(2 * np.pi * values)
        x[f"{angle}_cos"] = np.cos(2 * np.pi * values)
    x[CAT] = x[CAT].astype("category")
    return x


def destination_categories(x_fit, x_other):
    """Treat a category unseen in this fit as missing, without learning from test."""
    prepared = x_other.copy()
    for col in CAT:
        levels = x_fit[col].cat.categories
        known = prepared[col].astype("str")
        known = known.where(known.isin(levels))
        prepared[col] = pd.Categorical(known, categories=levels)
    return prepared


def fit_predict(candidate, x, y, *eval_frames):
    family, params = candidate
    if family == "baseline":
        return [np.broadcast_to(np.quantile(y, QUANTILES), (len(test), 2)).copy() for test in eval_frames]
    if family == "ridge_log":
        numeric = [c for c in x if c not in CAT]
        prep = ColumnTransformer([
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            ("category", OneHotEncoder(min_frequency=25, handle_unknown="infrequent_if_exist"), CAT),
        ])
        model = make_pipeline(prep, Ridge(**params, solver="lsqr"))
        log_y = np.log(y)
        model.fit(x, log_y)
        offsets = np.quantile(log_y - model.predict(x), QUANTILES)
        return [np.exp(model.predict(test)[:, None] + offsets[None, :])
                for test in eval_frames]
    x_fit = x.copy()
    for c in CAT:
        x_fit[c] = x_fit[c].cat.remove_unused_categories()
    inputs = [destination_categories(x_fit, test) for test in eval_frames]
    predictions = [[] for _ in inputs]
    for q in QUANTILES:
        estimator = LGBMRegressor(
            objective="quantile", alpha=q, learning_rate=0.065, min_child_samples=70,
            reg_lambda=1.0, cat_smooth=20, min_data_per_group=30,
            random_state=SEED, n_jobs=1, deterministic=True, force_col_wise=True,
            verbosity=-1, **params,
        )
        estimator.fit(x_fit, y, categorical_feature=CAT)
        for preds, test in zip(predictions, inputs):
            preds.append(estimator.predict(test))
    return [np.column_stack(preds) for preds in predictions]


def noncross(pred):
    pred = np.maximum(pred, 0)
    return np.column_stack([pred[:, 0], np.maximum(pred[:, 0], pred[:, 1])])


def scores(y, pred):
    actual = np.asarray(y)
    p50, p90 = pred.T
    return {
        "n": len(actual), "pinball_p50": mean_pinball_loss(actual, p50, alpha=.5),
        "pinball_p90": mean_pinball_loss(actual, p90, alpha=.9),
        "coverage_p90": np.mean(actual <= p90),
        "mae_p50": np.mean(np.abs(actual-p50)),
        "bias_p50": np.mean(p50-actual),
    }


def upper_offset(y, pred):
    # Finite-sample split conformal upper quantile of signed residuals.
    residual = np.asarray(y) - noncross(pred)[:, 1]
    rank = math.ceil((len(residual) + 1) * 0.9)
    if not 1 <= rank <= len(residual):
        raise ValueError("Sampel kalibrasi tidak cukup.")
    return float(np.partition(residual, rank-1)[rank-1])


def calibrate(pred, offset):
    pred = noncross(pred)
    pred[:, 1] += offset
    return noncross(pred)


def comparison_splits(ts, label_ready):
    for start, end in [("2025-05-01", "2025-06-01"), ("2025-06-01", "2025-07-01")]:
        fit = (ts < asof(start)) & (label_ready < asof(start))
        validation = (ts >= asof(start)) & (ts < asof(end))
        # Both tuning folds must be fully observable by the July fitting date.
        validation &= label_ready.lt(asof("2025-07-01"))
        if not ts[fit].max() < ts[validation].min():
            raise ValueError("Fold tuning melanggar urutan waktu.")
        yield start[:7], np.flatnonzero(fit), np.flatnonzero(validation)


def deployment_split(ts, label_ready, calibration_start, final_cutoff):
    start, cutoff = asof(calibration_start), asof(final_cutoff)
    fit = ts.lt(start) & label_ready.lt(start)
    candidates = ts.ge(start) & ts.lt(start + pd.Timedelta(days=20))
    cal = candidates & label_ready.lt(cutoff)
    if not (label_ready[fit].lt(start).all() and label_ready[cal].lt(cutoff).all()):
        raise ValueError("Label belum diketahui pada cutoff simulasi.")
    return np.flatnonzero(fit), np.flatnonzero(cal), int((candidates & ~cal).sum())


def candidate_name(candidate):
    family, params = candidate
    return " ".join([family, *(f"{k}={v}" for k, v in params.items())])


def segment_scores(train_holdout, pred):
    rows = []
    for field in ["layanan", "pulau_tujuan", "mitra_kurir"]:
        for label, group in train_holdout.groupby(field, dropna=False, sort=True).indices.items():
            rows.append({"dimensi": field, "segmen": str(label), **scores(train_holdout.iloc[group][TARGET], pred[group])})
    return pd.DataFrame(rows)


def partial_dependence(candidate, x_fit, y_fit, x_holdout, raw_holdout):
    # Re-evaluate predictions for the same 500 August packages while changing one feature.
    sample = x_holdout.sample(n=min(500, len(x_holdout)), random_state=SEED)
    varied, keys = [], []
    for col in ["jarak_km", "berat_kg"]:
        values = raw_holdout[col].where(raw_holdout[col].gt(0)).quantile([.1, .25, .5, .75, .9])
        for value in sorted(set(values)):
            trial = sample.copy()
            trial[col] = np.log1p(value)
            for indicator in {"jarak_km": ["jarak_tidak_positif"], "berat_kg": ["berat_kosong"]}[col]:
                trial[indicator] = 0
            varied.append(trial)
            keys.append({"fitur": col, "nilai": float(value), "n_sampel": len(sample)})
    combined = pd.concat(varied, ignore_index=True)
    pred, = fit_predict(candidate, x_fit, y_fit, combined)
    for i, row in enumerate(keys):
        row["rerata_p50_jam"] = float(np.mean(noncross(pred[i*len(sample):(i+1)*len(sample)])[:, 0]))
    return pd.DataFrame(keys)


def audit(train, test, ts, test_ts, ready):
    rows = {}
    for name, frame, timestamp in [("train", train, ts), ("test", test, test_ts)]:
        rows[name] = {
            "n": len(frame), "from": str(timestamp.min()), "to": str(timestamp.max()),
            "missing_weight": int(frame["berat_kg"].isna().sum()),
            "nonpositive_distance": int(frame["jarak_km"].le(0).sum()),
            "scan_median": float(frame[SCAN].median()),
        }
    rows["unseen_categories"] = {}
    for field in CAT:
        unseen = set(test[field].dropna()) - set(train[field].dropna())
        rows["unseen_categories"][field] = {
            "train_levels": int(train[field].nunique()), "new_levels": sorted(unseen),
            "test_rows": int(test[field].isin(unseen).sum()),
        }
    rows["scan_spearman"] = train[[SCAN, TARGET]].corr(method="spearman").iloc[0, 1]
    rows["labels_after_august"] = int(ready.ge(asof("2025-09-01")).sum())
    rows["target_p99_hours"] = float(train[TARGET].quantile(.99))
    rows["target_max_hours"] = float(train[TARGET].max())
    return rows


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data_uts_ml")
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    out = args.output
    evidence = out / "hasil/regresi"
    evidence.mkdir(parents=True, exist_ok=True)
    train, test, ts, test_ts = load_data(args.data)
    ready = ts + pd.to_timedelta(train[TARGET], unit="h")
    x, test_x = features(train, ts), features(test, test_ts)
    y = train[TARGET]
    assert not {ID, TARGET, SCAN, "tgl_pickup"}.intersection(x.columns)
    result = {
        "audit": audit(train, test, ts, test_ts, ready), "seed": SEED,
        "python": platform.python_version(),
        "versions": {p: metadata.version(p) for p in ["lightgbm", "numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]},
        "features": list(x.columns),
    }
    result["input_sha256"] = {}
    for part in ("train", "test"):
        with (args.data / f"regresi_{part}.csv.gz").open("rb") as source:
            result["input_sha256"][part] = hashlib.file_digest(source, "sha256").hexdigest()

    folds = list(comparison_splits(ts, ready))
    tuning = []
    for i, recipe in enumerate(CANDIDATES):
        print("Tuning", candidate_name(recipe), flush=True)
        for month, fit, val in folds:
            p, = fit_predict(recipe, x.iloc[fit], y.iloc[fit], x.iloc[val])
            tuning.append({"candidate": i, "model": candidate_name(recipe), "bulan": month,
                           "n_fit": len(fit), **scores(y.iloc[val], noncross(p))})
    tuning_table = pd.DataFrame(tuning)
    tuning_table.to_csv(evidence / "tuning_waktu.csv", index=False)
    ranking = tuning_table.groupby("candidate")[["pinball_p50", "pinball_p90"]].mean()
    best = int(ranking.sum(axis=1).idxmin())
    chosen = CANDIDATES[best]
    result["chosen_model"] = candidate_name(chosen)
    result["tuning_average"] = ranking.assign(model=[candidate_name(c) for c in CANDIDATES]).to_dict(orient="records")
    result["time_validation"] = tuning_table[tuning_table.candidate.eq(best)].to_dict(orient="records")
    print("Selected", result["chosen_model"], flush=True)

    # Random shuffle splits use the same development pool and per-fold sample budgets.
    pool = np.flatnonzero(ts.lt(asof("2025-07-01")) & ready.lt(asof("2025-07-01")))
    rand = []
    for fold_no, (month, fit, val) in enumerate(folds):
        li, ri = next(ShuffleSplit(n_splits=1, train_size=len(fit), test_size=len(val),
                                   random_state=SEED + fold_no).split(pool))
        p, = fit_predict(chosen, x.iloc[pool[li]], y.iloc[pool[li]], x.iloc[pool[ri]])
        rand.append({"paired_month": month, "n_fit": len(fit), **scores(y.iloc[pool[ri]], noncross(p))})
    result["random_validation"] = rand

    fit, cal, late = deployment_split(ts, ready, "2025-07-01", "2025-08-01")
    val = np.flatnonzero(ts.ge(asof("2025-08-01")) & ts.lt(asof("2025-09-01")))
    print("August holdout and July calibration", flush=True)
    p_cal, p_val = fit_predict(chosen, x.iloc[fit], y.iloc[fit], x.iloc[cal], x.iloc[val])
    offset = upper_offset(y.iloc[cal], p_cal)
    validation = calibrate(p_val, offset)
    result["august"] = {
        "n_fit": len(fit), "n_cal": len(cal), "labels_not_ready_in_cal": late,
        "offset_hours": offset, "calibration_raw": scores(y.iloc[cal], noncross(p_cal)),
        "raw": scores(y.iloc[val], noncross(p_val)), "calibrated": scores(y.iloc[val], validation),
        "crossings_before_repair": int((p_val[:, 1] < p_val[:, 0]).sum()),
        "baseline": scores(y.iloc[val], noncross(fit_predict(CANDIDATES[0], x.iloc[fit], y.iloc[fit], x.iloc[val])[0])),
        "outcomes_after_september_cutoff": int(ready.iloc[val].ge(asof("2025-09-01")).sum()),
    }
    result["august"]["by_half"] = []
    for start, end in [("2025-08-01", "2025-08-16"), ("2025-08-16", "2025-09-01")]:
        mask = (ts.iloc[val].ge(asof(start)) & ts.iloc[val].lt(asof(end))).to_numpy()
        result["august"]["by_half"].append({"start": start, "end_exclusive": end,
                                           **scores(y.iloc[val][mask], validation[mask])})

    print("Measuring the scan leakage risk on the same August holdout", flush=True)
    if chosen[0] == "baseline":
        diagnostic_recipe = min((i for i in range(1, len(CANDIDATES))), key=lambda i: ranking.loc[i].sum())
        diagnostic_recipe = CANDIDATES[diagnostic_recipe]
        p_without, = fit_predict(diagnostic_recipe, x.iloc[fit], y.iloc[fit], x.iloc[val])
    else:
        diagnostic_recipe, p_without = chosen, p_val
    x_fit_scan = x.iloc[fit].assign(**{SCAN: train[SCAN].iloc[fit]})
    x_val_scan = x.iloc[val].assign(**{SCAN: train[SCAN].iloc[val]})
    scan_low = int(test[SCAN].median())
    p_with, p_scan_low = fit_predict(
        diagnostic_recipe, x_fit_scan, y.iloc[fit], x_val_scan, x_val_scan.assign(**{SCAN: scan_low}),
    )
    result["scan_ablation"] = {
        "model": candidate_name(diagnostic_recipe),
        "without_scan": scores(y.iloc[val], noncross(p_without)),
        "with_recorded_scan": scores(y.iloc[val], noncross(p_with)),
        "with_scan_fixed_at_test_median": scores(y.iloc[val], noncross(p_scan_low)),
        "fixed_value": scan_low,
    }
    segment_scores(train.iloc[val], validation).to_csv(evidence / "error_per_segmen.csv", index=False)
    partial_dependence(chosen, x.iloc[fit], y.iloc[fit], x.iloc[val], train.iloc[val]).to_csv(
        evidence / "partial_dependence.csv", index=False)

    print("Final fit: August calibration, September–October predictions", flush=True)
    fit, cal, late = deployment_split(ts, ready, "2025-08-01", "2025-09-01")
    p_cal, p_test = fit_predict(chosen, x.iloc[fit], y.iloc[fit], x.iloc[cal], test_x)
    offset = upper_offset(y.iloc[cal], p_cal)
    final = calibrate(p_test, offset)
    prediction = pd.DataFrame({ID: test[ID], "p50": final[:, 0], "p90": final[:, 1]})
    if not (
        len(prediction) == len(test) and prediction[ID].equals(test[ID])
        and np.isfinite(final).all() and (final >= 0).all() and (final[:, 1] >= final[:, 0]).all()
    ):
        raise ValueError("Prediksi akhir tidak memenuhi kontrak soal.")
    prediction.to_csv(out / "prediksi_regresi.csv", index=False, float_format="%.6f")
    result["final"] = {"n_fit": len(fit), "n_cal": len(cal), "labels_not_ready_in_cal": late,
                       "offset_hours": offset, "n_test": len(test),
                       "calibration_raw": scores(y.iloc[cal], noncross(p_cal)),
                       "crossings_before_repair": int((p_test[:, 1] < p_test[:, 0]).sum())}
    (evidence / "metrics.json").write_text(
        json.dumps(result, indent=2, default=json_default, allow_nan=False) + "\n", encoding="utf-8")
    print(f"PASS: {len(test):,} predictions; see {evidence}", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
