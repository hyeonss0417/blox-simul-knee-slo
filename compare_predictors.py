"""
Compare 4 prediction models on the same train/test split:
  1. Mean baseline (constant predicting train mean)
  2. Linear regression (current)
  3. CatBoost
  4. LightGBM

Goal: see if heavier ML meaningfully reduces residual error on the
metadata→duration task, and whether the residual is fundamental noise.

Writes:
  - metadata_pred_catboost.json (predictions ready for scheduler use)
  - docs/report_v2/figures/predictor_comparison.png
"""
import json
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = [
    "Apple SD Gothic Neo", "AppleGothic", "Helvetica", "Arial", "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False

SRC = "cluster_job_log"
TRAIN_END = 3000
OUT_PRED = "metadata_pred_catboost.json"
OUT_FIG = "docs/report_v2/figures/predictor_comparison.png"


def fmt(t):
    return datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S")


def main():
    print("Loading trace...")
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
                    "idx": int(j["jobid"].rsplit("_", 1)[1]),
                    "dur": dur,
                    "steps": float(j.get("num_inference_steps", 30)),
                    "imgs": float(j.get("num_images_per_prompt", 1)),
                    "plen": float(j.get("prompt_length", 0)),
                    "nlora": float(j.get("num_lora", 0)),
                    "ptype": str(j.get("predict_type", "?")),
                    "model": str(j.get("checkpoint_model", "?")),
                })
                break
            except Exception:
                continue
    print(f"  {len(rows)} jobs with valid duration")

    train = [r for r in rows if r["idx"] < TRAIN_END]
    test  = [r for r in rows if r["idx"] >= TRAIN_END]
    print(f"  train n={len(train)}, test n={len(test)}")

    y_train = np.array([r["dur"] for r in train])
    y_test  = np.array([r["dur"] for r in test])

    def metrics(y, yh, name):
        mae = float(np.mean(np.abs(y - yh)))
        mape = float(np.mean(np.abs(y - yh) / np.maximum(y, 1.0)))
        r2 = 1 - float(np.sum((y - yh)**2) / np.sum((y - np.mean(y))**2))
        return {"name": name, "mae": mae, "mape": mape, "r2": r2}

    results = []

    # ── 1. Mean baseline ─────────────────────────────────────
    mean_pred = np.full(len(y_test), y_train.mean())
    results.append(metrics(y_test, mean_pred, "Mean baseline"))

    # ── 2. Linear regression (numpy lstsq, same as v1) ───────
    ptypes = sorted({r["ptype"] for r in rows})
    from collections import Counter
    top_models = [m for m, _ in Counter(r["model"] for r in rows).most_common(10)]

    def featurize(r):
        feats = [1.0, r["steps"], r["imgs"], r["plen"], r["nlora"],
                 r["steps"] * r["imgs"]]
        for p in ptypes:
            feats.append(1.0 if r["ptype"] == p else 0.0)
        for m in top_models:
            feats.append(1.0 if r["model"] == m else 0.0)
        return feats

    Xtr = np.array([featurize(r) for r in train])
    Xte = np.array([featurize(r) for r in test])
    coef, *_ = np.linalg.lstsq(Xtr, y_train, rcond=None)
    yhat_lin = Xte @ coef
    results.append(metrics(y_test, yhat_lin, "Linear regression"))

    # ── 3. CatBoost ─────────────────────────────────────────
    from catboost import CatBoostRegressor
    # Native categorical support — no need for one-hot
    def make_X(records):
        # numerical + categorical (string)
        X = []
        for r in records:
            X.append([
                r["steps"], r["imgs"], r["plen"], r["nlora"],
                r["ptype"], r["model"],
            ])
        return X
    cat_features = [4, 5]   # ptype, model are categorical indices

    cb = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        cat_features=cat_features,
        verbose=False,
        random_seed=42,
    )
    cb.fit(make_X(train), y_train)
    yhat_cb = cb.predict(make_X(test))
    yhat_cb = np.maximum(yhat_cb, 1.0)
    results.append(metrics(y_test, yhat_cb, "CatBoost"))

    # ── 4. LightGBM ─────────────────────────────────────────
    from lightgbm import LGBMRegressor
    # LightGBM needs numeric encoding; use pandas category dtype trick
    import pandas as pd
    train_df = pd.DataFrame(make_X(train),
                            columns=["steps","imgs","plen","nlora","ptype","model"])
    train_df["ptype"] = train_df["ptype"].astype("category")
    train_df["model"] = train_df["model"].astype("category")
    test_df = pd.DataFrame(make_X(test),
                           columns=["steps","imgs","plen","nlora","ptype","model"])
    # use SAME categories as train
    test_df["ptype"] = pd.Categorical(test_df["ptype"],
                                       categories=train_df["ptype"].cat.categories)
    test_df["model"] = pd.Categorical(test_df["model"],
                                       categories=train_df["model"].cat.categories)
    lgb = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        verbosity=-1,
        random_state=42,
    )
    lgb.fit(train_df, y_train, categorical_feature=["ptype", "model"])
    yhat_lgb = lgb.predict(test_df)
    yhat_lgb = np.maximum(yhat_lgb, 1.0)
    results.append(metrics(y_test, yhat_lgb, "LightGBM"))

    # ── Print ───────────────────────────────────────────────
    print(f"\n{'Model':25s} {'MAE (s)':>10s} {'MAPE':>8s} {'R²':>8s}")
    print("-" * 55)
    for r in results:
        print(f"{r['name']:25s} {r['mae']:10.3f} {r['mape']*100:7.2f}% {r['r2']:8.3f}")

    # ── Save CatBoost predictions for scheduler use ─────────
    all_X = make_X(rows)
    yhat_all = cb.predict(all_X)
    yhat_all = np.maximum(yhat_all, 1.0)
    preds = {r["jobid"]: float(yhat_all[i]) for i, r in enumerate(rows)}
    with open(OUT_PRED, "w") as f:
        json.dump({
            "model_type": "catboost",
            "iterations": 500, "depth": 6,
            "test_metrics": {"mae": results[2]["mae"],
                              "mape": results[2]["mape"],
                              "r2": results[2]["r2"]},
            "predictions": preds,
        }, f)
    print(f"\nSaved {OUT_PRED}  ({len(preds)} predictions)")

    # ── Plot ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    names = [r["name"] for r in results]
    colors = ["#bdc3c7", "#3498db", "#27ae60", "#9b59b6"]

    # MAE
    ax = axes[0]
    bars = ax.bar(names, [r["mae"] for r in results], color=colors,
                  edgecolor="white", linewidth=1.5)
    for bar, r in zip(bars, results):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{r['mae']:.2f}s", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("MAE (seconds)")
    ax.set_title("Mean Absolute Error (낮을수록 좋음)")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right", fontsize=9)

    # R²
    ax = axes[1]
    bars = ax.bar(names, [r["r2"] for r in results], color=colors,
                  edgecolor="white", linewidth=1.5)
    for bar, r in zip(bars, results):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                y + 0.02 if y > 0 else y - 0.05,
                f"{r['r2']:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("R²")
    ax.set_title("R² (높을수록 좋음, 음수는 random 보다 나쁨)")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right", fontsize=9)

    # Predicted vs True scatter (CatBoost)
    ax = axes[2]
    ax.scatter(y_test, yhat_cb, alpha=0.3, s=10, color="#27ae60", label="CatBoost")
    ax.scatter(y_test, yhat_lin, alpha=0.3, s=10, color="#3498db", label="Linear")
    mx = max(y_test.max(), yhat_cb.max(), yhat_lin.max())
    ax.plot([0, mx], [0, mx], "k--", lw=1, label="perfect")
    ax.set_xlabel("True duration (s)")
    ax.set_ylabel("Predicted (s)")
    ax.set_title("Predicted vs True (test set, CatBoost vs Linear)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, min(200, mx))
    ax.set_ylim(0, min(200, mx))

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {OUT_FIG}")


if __name__ == "__main__":
    main()
