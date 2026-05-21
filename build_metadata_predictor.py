"""
Build a metadata-based duration predictor for non-Oracle scheduling.

The simulator's view of a job is just (job_class_id, job_duration).  The
scheduler-side oracle approach (SRTF) uses job_duration directly — that's
ground truth and would not be available in production.

This script:
  1. Reads the raw Alibaba 2026 trace metadata (num_inference_steps,
     checkpoint_model, predict_type, num_images_per_prompt, num_lora).
  2. Trains a simple linear regression (numpy) on the FIRST 3000 jobs
     (training set) to predict exec_time_seconds.
  3. Applies the model to ALL jobs and saves `{jobid: predicted_dur}`
     as JSON.  The scheduler can look up this file at init time, giving
     each job a non-Oracle predicted remaining time.

Result file: `metadata_pred.json` in CWD.
"""
import json
import math
import datetime
import numpy as np

SRC = "cluster_job_log"
OUT = "metadata_pred.json"
TRAIN_END = 3000   # use jobs 0-3000 for training (the tracked range starts here)


def fmt(t):
    return datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S")


def main():
    print(f"Reading {SRC} ...")
    data = json.load(open(SRC))

    rows = []
    for j in data:
        for a in j.get("attempts", []):
            try:
                dur = (fmt(a["end_time"]) - fmt(a["start_time"])).total_seconds()
                if dur <= 0:
                    continue
                rows.append({
                    "jobid": j["jobid"],
                    "dur": dur,
                    "steps": float(j.get("num_inference_steps", 30)),
                    "imgs": float(j.get("num_images_per_prompt", 1)),
                    "plen": float(j.get("prompt_length", 0)),
                    "ptype": str(j.get("predict_type", "?")),
                    "model": str(j.get("checkpoint_model", "?")),
                    "nlora": float(j.get("num_lora", 0)),
                    "idx": int(j["jobid"].rsplit("_", 1)[1]),
                })
                break
            except Exception:
                continue

    print(f"  {len(rows)} jobs with valid duration")

    # Feature engineering: numeric + one-hot for ptype (limited categories).
    ptypes = sorted({r["ptype"] for r in rows})
    models = sorted({r["model"] for r in rows})
    # Use only top-N models to avoid explosion.
    from collections import Counter
    top_models = [m for m, _ in Counter(r["model"] for r in rows).most_common(10)]

    def featurize(r):
        feats = [1.0, r["steps"], r["imgs"], r["plen"], r["nlora"],
                 r["steps"] * r["imgs"]]   # interaction
        for p in ptypes:
            feats.append(1.0 if r["ptype"] == p else 0.0)
        for m in top_models:
            feats.append(1.0 if r["model"] == m else 0.0)
        return feats

    feat_names = (["intercept", "steps", "imgs", "plen", "nlora", "steps*imgs"]
                  + [f"ptype={p}" for p in ptypes]
                  + [f"model={m}" for m in top_models])

    X = np.array([featurize(r) for r in rows], dtype=float)
    y = np.array([r["dur"] for r in rows], dtype=float)

    train_mask = np.array([r["idx"] < TRAIN_END for r in rows])
    Xtr, ytr = X[train_mask], y[train_mask]
    Xte, yte = X[~train_mask], y[~train_mask]

    print(f"  train n={len(ytr)},  test n={len(yte)}")

    # Linear regression (least squares).
    coef, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    yhat_tr = Xtr @ coef
    yhat_te = Xte @ coef

    def metrics(y, yh):
        mae = np.mean(np.abs(y - yh))
        mape = np.mean(np.abs(y - yh) / np.maximum(y, 1.0))
        r2 = 1 - np.sum((y - yh)**2) / np.sum((y - np.mean(y))**2)
        return mae, mape, r2

    tr_mae, tr_mape, tr_r2 = metrics(ytr, yhat_tr)
    te_mae, te_mape, te_r2 = metrics(yte, yhat_te)
    print(f"  TRAIN: MAE={tr_mae:.2f}s  MAPE={tr_mape*100:.1f}%  R²={tr_r2:.3f}")
    print(f"  TEST : MAE={te_mae:.2f}s  MAPE={te_mape*100:.1f}%  R²={te_r2:.3f}")

    # Compare to category-mean baseline.
    # job_class_id is 0..4 based on round-robin in workload.py — approx model index.
    # For honest comparison we use the per-predict-type mean as a proxy.
    type_means = {}
    for p in ptypes:
        durs = [r["dur"] for r in rows if r["ptype"] == p
                and r["idx"] < TRAIN_END]
        type_means[p] = float(np.mean(durs)) if durs else 23.0
    yhat_mean = np.array([type_means.get(r["ptype"], 23.0) for r in rows])
    bm_mae, bm_mape, bm_r2 = metrics(y, yhat_mean)
    print(f"  ptype-mean baseline: MAE={bm_mae:.2f}s  MAPE={bm_mape*100:.1f}%  R²={bm_r2:.3f}")

    # Top contributing features
    importance = sorted(
        zip(feat_names, coef, np.std(X, axis=0)),
        key=lambda x: abs(x[1] * x[2]), reverse=True,
    )
    print("\n  Top 8 features (by |coef × std|):")
    for name, c, s in importance[:8]:
        print(f"    {name:24s}  coef={c:+8.3f}  std={s:6.2f}  effect={c*s:+7.2f}s")

    # Save predictions {jobid: predicted_duration}
    yhat_all = X @ coef
    # Floor at 1s to avoid zero/negative predictions
    yhat_all = np.maximum(yhat_all, 1.0)
    preds = {r["jobid"]: float(yhat_all[i]) for i, r in enumerate(rows)}
    with open(OUT, "w") as f:
        json.dump({
            "model_type": "linear_regression",
            "features": feat_names,
            "coef": coef.tolist(),
            "train_metrics": {"mae": tr_mae, "mape": tr_mape, "r2": tr_r2},
            "test_metrics": {"mae": te_mae, "mape": te_mape, "r2": te_r2},
            "ptype_mean_baseline_metrics": {"mae": bm_mae, "mape": bm_mape, "r2": bm_r2},
            "predictions": preds,
        }, f)
    print(f"\nSaved {OUT}  ({len(preds)} predictions)")


if __name__ == "__main__":
    main()
