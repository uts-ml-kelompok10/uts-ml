"""Reproduce the fraud experiment and its timestamp-safe test predictions."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
ID, TARGET, REVIEW = "id_transaksi", "is_fraud", "flag_review_manual"
SEED, REVIEW_COST, GAP = 42, 25_000, 14
CAT = ["kategori_merchant", "kanal", "tipe_perangkat", "email_domain"]
NUM = [
    "nominal", "jam", "hari_libur", "umur_akun_hari", "level_kyc", "umur_pengguna_tahun",
    "umur_perangkat_hari", "perangkat_baru", "jumlah_akun_per_perangkat", "ganti_password_7h",
    "gagal_login_24j", "jarak_lokasi_km", "negara_ip_beda", "kartu_tidak_hadir",
    "menit_sejak_trx_terakhir", "trx_1j", "trx_24j", "trx_7h", "nominal_24j",
    "rasio_nominal_rata2", "saldo_sebelum", "persen_saldo", "umur_merchant_hari",
]
APPROACHES = [
    ("tanpa_bobot", None), ("bobot_positif", None), ("undersampling", None),
    ("tanpa_bobot", 90), ("bobot_positif", 90), ("undersampling", 90),
]
MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]


def build_history(frame, entity, prefix):
    """Exclude all events at the same second, including events from another split."""
    keys = [entity, "waktu"]
    batch = frame.groupby(keys, sort=False, observed=True)["nominal"].agg(
        count="size", amount="sum").reset_index().sort_values("waktu", kind="stable")
    by_entity = batch.groupby(entity, sort=False, observed=True)
    batch["cum_count"] = by_entity["count"].cumsum()
    batch["cum_amount"] = by_entity["amount"].cumsum()
    recent = batch["cum_count"] - batch["count"]
    old_amount = batch["cum_amount"] - batch["amount"]
    result = batch[keys].copy()
    result[prefix + "_hist_total"] = recent
    result[prefix + "_hist_rerata"] = old_amount / recent.where(recent.gt(0))
    result[prefix + "_hist_jeda_menit"] = (batch["waktu"] - by_entity["waktu"].shift()).dt.total_seconds() / 60
    for days in (1, 7):
        boundary = batch[[entity, "waktu"]].rename(columns={"waktu": "batas"})
        boundary["batas"] -= pd.Timedelta(days=days)
        prior = pd.merge_asof(
            boundary, batch[keys + ["cum_count", "cum_amount"]],
            left_on="batas", right_on="waktu", by=entity, allow_exact_matches=False,
        )
        result[f"{prefix}_hist_jumlah_{days}h"] = recent.to_numpy() - prior.cum_count.fillna(0).to_numpy()
        result[f"{prefix}_hist_nominal_{days}h"] = old_amount.to_numpy() - prior.cum_amount.fillna(0).to_numpy()
    aligned = frame[keys].merge(result, on=keys, how="left", sort=False, validate="many_to_one")
    return aligned.drop(columns=keys).astype("float32")


def make_features(frame):
    x = frame[CAT + NUM].copy()
    x[CAT] = x[CAT].astype("category")
    x[NUM] = x[NUM].astype("float32")
    x["hari_pekan"] = frame.waktu.dt.dayofweek.astype("int8")
    pieces = [
        x, build_history(frame, "id_pengguna", "pengguna"), build_history(frame, "id_merchant", "merchant"),
    ]
    out = pd.concat(pieces, axis=1)
    for entity in ("pengguna", "merchant"):
        mean = out[f"{entity}_hist_rerata"]
        out[f"{entity}_hist_rasio_nominal"] = out.nominal / mean.where(mean.gt(0))
    return out


def load_data(directory):
    train, test = [pd.read_csv(directory / f"klasifikasi_{part}.csv.gz", low_memory=False)
                   for part in ("train", "test")]
    assert set(train) - set(test) == {TARGET} and not set(test) - set(train)
    for frame in (train, test):
        assert frame[ID].notna().all() and frame[ID].is_unique
        assert frame["id_pengguna"].notna().all() and frame["id_merchant"].notna().all()
        assert frame.nominal.gt(0).all() and np.isfinite(frame.nominal).all()
        frame["waktu"] = pd.to_datetime(frame.waktu, format="%Y-%m-%d %H:%M:%S")
        assert frame.waktu.notna().all()
    assert not set(train[ID]) & set(test[ID])
    assert train[TARGET].isin([0, 1]).all() and train[TARGET].nunique() == 2
    assert train.waktu.lt("2025-09-01").all()
    assert test.waktu.ge("2025-09-01").all() and test.waktu.lt("2025-11-01").all()
    return train, test


def subset(time, lo, hi):
    return np.flatnonzero(time.ge(pd.Timestamp(lo)) & time.lt(pd.Timestamp(hi)))


def fit_indices(time, score_start, days, gap=GAP):
    start = pd.Timestamp(score_start)
    end = start - pd.Timedelta(days=gap)
    lower = end - pd.Timedelta(days=days) if days else time.min()
    return subset(time, lower, end)


def model_name(recipe):
    strategy, days = recipe
    return f"{strategy} | " + (f"{days} hari" if days else "semua riwayat")


def fit_model(x, y, fit, strategy, *inputs):
    # Resampling/weighting changes training only; evaluation stays at natural prevalence.
    y_fit = y[fit]
    weight = None
    if strategy == "undersampling":
        positive, negative = fit[y_fit == 1], fit[y_fit == 0]
        keep = np.random.default_rng(SEED).choice(negative, size=round(.30 * len(negative)), replace=False)
        fit = np.sort(np.r_[positive, keep])
    elif strategy == "bobot_positif":
        weight = np.where(y_fit == 1, 8.0, 1.0)
    xf = x.iloc[fit].copy()
    for col in CAT:
        xf[col] = xf[col].cat.remove_unused_categories()
    model = LGBMClassifier(
        n_estimators=115, learning_rate=.055, num_leaves=24, min_child_samples=100,
        reg_lambda=2.0, verbosity=-1, n_jobs=1, random_state=SEED,
        deterministic=True, force_col_wise=True,
    )
    model.fit(xf, y[fit], sample_weight=weight, categorical_feature=CAT)
    out = []
    for idx in inputs:
        xx = x.iloc[idx].copy()
        for col in CAT:
            text = xx[col].astype("str")
            levels = xf[col].cat.categories
            xx[col] = pd.Categorical(text.where(text.isin(levels)), categories=levels)
        out.append(model.predict(xx, raw_score=True))
    return out


def sigmoid(raw):
    return 1.0 / (1.0 + np.exp(-np.clip(raw, -35, 35)))


def calibrator(raw, target):
    # Platt-style logistic fit on natural-prevalence scores from a separate period.
    return LogisticRegression(C=10.0, max_iter=1000).fit(raw.reshape(-1, 1), target)


def probabilities(cal, raw):
    return cal.predict_proba(raw.reshape(-1, 1))[:, 1]


def flags(prob, nominal, multiplier):
    return prob * nominal > REVIEW_COST * multiplier


def decision_stats(y, nominal, flag):
    pos = y == 1
    alerts, tp, fn = int(flag.sum()), int((flag & pos).sum()), int((~flag & pos).sum())
    missed = float(nominal[~flag & pos].sum())
    return {
        "n": len(y), "fraud": int(pos.sum()), "alerts": alerts, "true_positive": tp,
        "false_positive": alerts - tp, "false_negative": fn, "missed_nominal": missed,
        "review_cost": alerts * REVIEW_COST, "total_cost": missed + alerts * REVIEW_COST,
        "precision": tp / alerts if alerts else None, "recall": tp / int(pos.sum()) if pos.any() else None,
    }


def probability_stats(y, p):
    return {
        "n": len(y), "positives": int(y.sum()), "observed_rate": float(y.mean()),
        "mean_probability": float(p.mean()), "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def risk_patterns(frame):
    # Exclusive descriptions of observed features, not true fraud typologies.
    return np.select(
        [frame.negara_ip_beda.eq(1), frame.perangkat_baru.eq(1),
         frame.gagal_login_24j.gt(0), frame.kartu_tidak_hadir.eq(1)],
        ["IP negara berbeda", "perangkat baru", "ada gagal login", "kartu tidak hadir"], default="lainnya",
    )


def training_audit(train, test):
    crosstab = pd.crosstab(train[REVIEW], train[TARGET])
    near_first = train[["menit_sejak_trx_terakhir", "rasio_nominal_rata2"]].isna()
    result = {
        "train": len(train), "test": len(test), "positives": int(train[TARGET].sum()),
        "review_table": crosstab.to_dict(orient="index"),
        "manual_review_alone_ap": float(average_precision_score(train[TARGET], train[REVIEW])),
        "missing_history_fields": near_first.sum().to_dict(),
        "anonymous_features_excluded": [f"v{i:02d}" for i in range(1, 51)],
        "new_entities_test": {},
    }
    for entity in ("id_pengguna", "id_merchant"):
        unknown = ~test[entity].isin(train[entity])
        result["new_entities_test"][entity] = {"distinct": int(test.loc[unknown, entity].nunique()), "rows": int(unknown.sum())}
    return result


def monthly_audit(train, test):
    blocks = []
    for name, data in [("train", train), ("test", test)]:
        for month, block in data.groupby(data.waktu.dt.strftime("%Y-%m")):
            observed = block[TARGET].eq(1) if TARGET in block else None
            blocks.append({
                "split": name, "bulan": month, "n": len(block),
                "fraud_terkonfirmasi": int(observed.sum()) if observed is not None else None,
                "proporsi_fraud_terkonfirmasi": float(observed.mean()) if observed is not None else None,
                "median_nominal": float(block.nominal.median()),
                "proporsi_perangkat_baru": float(block.perangkat_baru.eq(1).mean()),
                "proporsi_ip_negara_beda": float(block.negara_ip_beda.eq(1).mean()),
                "perangkat_baru_di_fraud": float(block.loc[observed, "perangkat_baru"].eq(1).mean()) if observed is not None else None,
                "ip_negara_beda_di_fraud": float(block.loc[observed, "negara_ip_beda"].eq(1).mean()) if observed is not None else None,
                "gagal_login_di_fraud": float(block.loc[observed, "gagal_login_24j"].gt(0).mean()) if observed is not None else None,
            })
    return pd.DataFrame(blocks)


def select_model(x, train, y):
    rows = []
    # The two forward score windows end at least 14 days before the next month.
    for start, next_month in [("2025-05-01", "2025-06-01"), ("2025-06-01", "2025-07-01")]:
        idx = subset(train.waktu, start, pd.Timestamp(next_month) - pd.Timedelta(days=GAP))
        for recipe in APPROACHES:
            strategy, days = recipe
            fit = fit_indices(train.waktu, start, days)
            print("Tuning", start[:7], model_name(recipe), flush=True)
            margin, = fit_model(x, y, fit, strategy, idx)
            rows.append({"bulan": start[:7], "model": model_name(recipe), "n_fit_sebelum_sampling": len(fit),
                         **probability_stats(y[idx], sigmoid(margin))})
        fit = fit_indices(train.waktu, start, None)
        p = np.full(len(idx), y[fit].mean())
        rows.append({"bulan": start[:7], "model": "baseline prevalensi", "n_fit_sebelum_sampling": len(fit),
                     **probability_stats(y[idx], p)})
    table = pd.DataFrame(rows)
    averages = table.groupby("model").average_precision.mean()
    chosen = max(APPROACHES, key=lambda recipe: averages[model_name(recipe)])
    return chosen, table, averages.to_dict()


def run_period(x, time, y, nominal, recipe, prior_month, score_month, score_indices):
    day = pd.Timestamp(prior_month)
    end = pd.Timestamp(score_month) - pd.Timedelta(days=GAP)
    cut = day + (end - day) // 2
    fit = fit_indices(time, day, recipe[1])
    cal_idx, rule_idx = subset(time, day, cut), subset(time, cut, end)
    assert time.iloc[fit].max() < time.iloc[cal_idx].min() <= time.iloc[cal_idx].max() < time.iloc[rule_idx].min()
    raw_cal, raw_rule, raw_score = fit_model(x, y, fit, recipe[0], cal_idx, rule_idx, score_indices)
    cal = calibrator(raw_cal, y[cal_idx])
    p_rule = probabilities(cal, raw_rule)
    rules = [
        {"multiplier": k, **decision_stats(y[rule_idx], nominal[rule_idx], flags(p_rule, nominal[rule_idx], k))}
        for k in MULTIPLIERS
    ]
    picked = min(rules, key=lambda row: (row["total_cost"], abs(row["multiplier"] - 1), row["multiplier"]))
    return raw_score, probabilities(cal, raw_score), {
        "n_fit_sebelum_sampling": len(fit), "n_cal": len(cal_idx), "positives_cal": int(y[cal_idx].sum()),
        "n_decision": len(rule_idx), "positives_decision": int(y[rule_idx].sum()),
        "fit_cutoff_exclusive": str(day - pd.Timedelta(days=GAP)),
        "cal_start": str(day), "cal_end_exclusive": str(cut), "decision_end_exclusive": str(end),
        "calibration_slope": float(cal.coef_[0, 0]), "calibration_intercept": float(cal.intercept_[0]),
        "decision_multiplier": picked["multiplier"], "decision_grid": rules,
    }


def segment_stats(frame, p, alert):
    rows = []
    values = {
        "kanal": frame.kanal.to_numpy(),
        "kategori_merchant": frame.kategori_merchant.to_numpy(),
        "pola_fitur": risk_patterns(frame),
    }
    y, nominal = frame[TARGET].to_numpy(), frame.nominal.to_numpy()
    for dimension, labels in values.items():
        for label in sorted(np.unique(labels)):
            idx = labels == label
            metrics = decision_stats(y[idx], nominal[idx], alert[idx])
            ap = float(average_precision_score(y[idx], p[idx])) if metrics["fraud"] else None
            rows.append({"dimensi": dimension, "segmen": str(label), "average_precision": ap, **metrics})
    return pd.DataFrame(rows)


def jsonify(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data_uts_ml")
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    out = args.output / "hasil/klasifikasi"
    out.mkdir(parents=True, exist_ok=True)
    train, test = load_data(args.data)
    audit = training_audit(train, test)
    monthly_audit(train, test).to_csv(out / "audit_bulanan.csv", index=False)
    print("Building as-of user and merchant history", flush=True)
    combined = pd.concat([train, test], ignore_index=True)
    x = make_features(combined)
    del combined
    assert not {TARGET, REVIEW, ID, "id_pengguna", "id_merchant"}.intersection(x.columns)
    y, nominal, time = train[TARGET].to_numpy(), train.nominal.to_numpy(), train.waktu
    chosen, tuning, averages = select_model(x, train, y)
    tuning.to_csv(out / "tuning_waktu.csv", index=False)
    print("Selected", model_name(chosen), flush=True)

    aug = subset(time, "2025-08-01", "2025-09-01")
    print("August holdout, calibration and decision cost", flush=True)
    raw, p, stage = run_period(x, time, y, nominal, chosen, "2025-07-01", "2025-08-01", aug)
    mature = time.iloc[aug].lt(pd.Timestamp("2025-09-01") - pd.Timedelta(days=GAP)).to_numpy()
    alert = flags(p, nominal[aug], stage["decision_multiplier"])
    holdout = {}
    for name, mask in [("early_mature_14d", mature), ("late_less_mature", ~mature), ("all_august", np.ones(len(aug), dtype=bool))]:
        idx = aug[mask]
        obs, transaction = y[idx], nominal[idx]
        holdout[name] = {
            "raw": probability_stats(obs, sigmoid(raw[mask])),
            "calibrated": probability_stats(obs, p[mask]),
            "baseline_none": decision_stats(obs, transaction, np.zeros(len(idx), dtype=bool)),
            "baseline_all": decision_stats(obs, transaction, np.ones(len(idx), dtype=bool)),
            "theoretical_cost_rule": decision_stats(obs, transaction, flags(p[mask], transaction, 1)),
            "selected_cost_rule": decision_stats(obs, transaction, alert[mask]),
        }
    mature_idx = aug[mature]
    segment_stats(train.iloc[mature_idx], p[mature], alert[mature]).to_csv(out / "error_per_segmen.csv", index=False)
    sensitivity = []
    for gap in (7, 14, 21):
        mask = time.iloc[aug].lt(pd.Timestamp("2025-09-01") - pd.Timedelta(days=gap)).to_numpy()
        idx = aug[mask]
        sensitivity.append({"gap_days": gap, **probability_stats(y[idx], p[mask]),
                            **decision_stats(y[idx], nominal[idx], alert[mask])})
    pd.DataFrame(sensitivity).to_csv(out / "sensitivitas_label.csv", index=False)
    print("Leakage and constructed-history ablations", flush=True)
    fit = fit_indices(time, "2025-07-01", chosen[1])
    # Review is run once as a diagnostic on labeled rows only, never on submission data.
    review_x = x.iloc[:len(train)].assign(**{REVIEW: train[REVIEW].astype("int8")})
    with_review, = fit_model(review_x, y, fit, chosen[0], mature_idx)
    hist_cols = [col for col in x if "_hist_" in col]
    no_history, = fit_model(x.drop(columns=hist_cols), y, fit, chosen[0], mature_idx)
    diagnostics = {
        "safe": probability_stats(y[mature_idx], sigmoid(raw[mature])),
        "with_post_transaction_review": probability_stats(y[mature_idx], sigmoid(with_review)),
        "without_constructed_history": probability_stats(y[mature_idx], sigmoid(no_history)),
    }
    del review_x

    print("Final forward fit and September–October predictions", flush=True)
    inds = np.arange(len(train), len(train) + len(test))
    _, test_prob, final = run_period(x, time, y, nominal, chosen, "2025-08-01", "2025-09-01", inds)
    # Round first so the decisions agree with the exported probabilities as read.
    test_prob = np.round(test_prob, 12)
    test_flags = flags(test_prob, test.nominal.to_numpy(), final["decision_multiplier"])
    prediction = pd.DataFrame({ID: test[ID], "prob": test_prob, "tandai": test_flags.astype("int8")})
    assert prediction[ID].equals(test[ID]) and prediction[ID].is_unique
    assert np.isfinite(prediction.prob).all() and prediction.prob.between(0, 1).all()
    assert prediction.tandai.isin([0, 1]).all()
    prediction.to_csv(args.output / "prediksi_klasifikasi.csv", index=False, float_format="%.12f")
    final.update(n_test=len(test), n_flagged=int(test_flags.sum()), review_cost_if_deployed=int(test_flags.sum()) * REVIEW_COST)
    info = {
        "source_audit": audit, "seed": SEED, "gap_days_assumed": GAP, "python": platform.python_version(),
        "versions": {pkg: metadata.version(pkg) for pkg in ("lightgbm", "numpy", "pandas", "scikit-learn", "threadpoolctl")},
        "chosen_model": model_name(chosen), "tuning_mean_ap": averages, "features": list(x.columns),
        "august_design": stage, "august": holdout, "ablation": diagnostics, "final_design": final,
        "input_sha256": {},
    }
    for part in ("train", "test"):
        with (args.data / f"klasifikasi_{part}.csv.gz").open("rb") as file:
            info["input_sha256"][part] = hashlib.file_digest(file, "sha256").hexdigest()
    (out / "metrics.json").write_text(json.dumps(info, indent=2, default=jsonify, allow_nan=False) + "\n", encoding="utf-8")
    print(f"PASS: {len(test):,} predictions, {int(test_flags.sum()):,} flagged.", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
