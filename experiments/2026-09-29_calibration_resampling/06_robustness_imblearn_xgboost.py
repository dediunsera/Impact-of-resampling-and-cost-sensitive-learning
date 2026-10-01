"""
06_robustness_imblearn_xgboost.py  -- JALANKAN DI GOOGLE COLAB (butuh imbalanced-learn, xgboost, lightgbm)

Uji robustness untuk menjawab dua pertanyaan reviewer:
 A. Apakah implementasi ROS/RUS/SMOTE buatan sendiri (02_train_predict.py) setara dengan imbalanced-learn?
    -> model LR & HGB dilatih dengan kedua implementasi pada split yang SAMA, lalu metrik dibandingkan.
 B. Apakah temuan berlaku juga untuk XGBoost dan LightGBM (bukan hanya HGB scikit-learn)?
    -> XGBoost & LightGBM x 5 strategi x 4 kalibrasi, pipeline identik.
Split, pra-pemrosesan, calibration set, dan fungsi metrik diambil langsung dari 02_ dan 03_ agar identik.

Output:
  results/robustness_imblearn_equivalence.csv
  results/robustness_xgb_lgbm.csv          (per fold)
  results/robustness_xgb_lgbm_summary.csv  (mean ± SD lintas fold)
  results/robustness_smote_detectability_imblearn.csv
  results/robustness_versions.json
  figures/fig10_robustness_xgb_lgbm.png
Jalankan:  python 06_robustness_imblearn_xgboost.py --folds 0          (cepat, ~10-15 menit)
           python 06_robustness_imblearn_xgboost.py --folds 0 1 2 3 4  (lengkap)
"""
import argparse, json, sys, time, itertools
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--folds", type=int, nargs="+", default=[0])
args = ap.parse_args()
FOLDS = args.folds

# ---- ambil definisi data, pra-pemrosesan, resampling buatan sendiri, dan kalibrator dari 02_ ----
sys.argv = ["x"]
exec(open("02_train_predict.py").read().split("MODELS, STRATS")[0])
# ---- ambil fungsi metrik dari 03_ ----
exec(open("03_evaluate.py").read().split("folds = [k for k")[0])

import imblearn, xgboost, lightgbm, sklearn
from imblearn.over_sampling import SMOTE as ImbSMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

VERS = {"python": sys.version.split()[0], "sklearn": sklearn.__version__, "imblearn": imblearn.__version__,
        "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__, "numpy": np.__version__, "pandas": pd.__version__}
print(VERS, flush=True)
json.dump(VERS, open("results/robustness_versions.json", "w"), indent=1)

def imb_resample(name, X, y):
    if name == "ROS": return RandomOverSampler(random_state=SEED).fit_resample(X, y)
    if name == "RUS": return RandomUnderSampler(random_state=SEED).fit_resample(X, y)
    if name == "SMOTE": return ImbSMOTE(k_neighbors=5, random_state=SEED).fit_resample(X, y)
    return X, y

def own_resample(name, X, y, rng):
    if name == "ROS": return ros(X, y, rng)
    if name == "RUS": return rus(X, y, rng)
    if name == "SMOTE": return smote(X, y, rng)
    return X, y

def make_boost(name, strategy, y):
    spw = float((y == 0).sum() / (y == 1).sum()) if strategy == "CSL" else 1.0
    if name == "XGB":
        return XGBClassifier(n_estimators=300, learning_rate=0.08, max_depth=6, subsample=1.0, colsample_bytree=1.0,
                             tree_method="hist", scale_pos_weight=spw, eval_metric="logloss", n_jobs=-1, random_state=SEED)
    return LGBMClassifier(n_estimators=300, learning_rate=0.08, num_leaves=31, scale_pos_weight=spw,
                          n_jobs=-1, random_state=SEED, verbose=-1)

outer = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(X_all, y_all, g_all))
rowsA, rowsB, rowsD = [], [], []
for k in FOLDS:
    dev_idx, te_idx = outer[k]
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED + k)
    tr_rel, ca_rel = next(inner.split(X_all.iloc[dev_idx], y_all[dev_idx], g_all[dev_idx]))
    tr_idx, ca_idx = dev_idx[tr_rel], dev_idx[ca_rel]
    pre = make_pre().fit(X_all.iloc[tr_idx])
    Xtr = pre.transform(X_all.iloc[tr_idx]).astype(np.float32)
    Xca = pre.transform(X_all.iloc[ca_idx]).astype(np.float32)
    Xte = pre.transform(X_all.iloc[te_idx]).astype(np.float32)
    ytr, yca, yte = y_all[tr_idx], y_all[ca_idx], y_all[te_idx]
    pi_tr = ytr.mean()
    print(f"[fold {k}] train {len(ytr)} calib {len(yca)} test {len(yte)} p={Xtr.shape[1]}", flush=True)

    # ---------------- A. setara imbalanced-learn? ----------------
    for s in ["ROS", "RUS", "SMOTE"]:
        rng = np.random.default_rng(SEED + k)
        for impl in ["own", "imblearn"]:
            Xs, ys = own_resample(s, Xtr, ytr, rng) if impl == "own" else imb_resample(s, Xtr, ytr)
            for m in ["LR", "HGB"]:
                t0 = time.time()
                mdl = make_model(m, balanced=False).fit(Xs, ys)
                p_ca, p_te = mdl.predict_proba(Xca)[:, 1], mdl.predict_proba(Xte)[:, 1]
                cals, _ = calibrators(p_ca, yca, pi_tr, 0.5)
                for c in ["UNCAL", "PLATT", "ISO"]:
                    r = all_metrics(np.clip(cals[c](p_te), 0, 1), yte, pi_tr)
                    rowsA.append({"fold": k, "model": m, "strategy": s, "implementation": impl, "calibration": c,
                                  "n_train_fit": len(ys), **{x: r[x] for x in ["AUC", "Brier", "ECE", "Cal_slope", "Cal_intercept", "Mean_pred", "Sens_t05"]}})
                print(f"   A {s:5s} {impl:8s} {m:3s} {time.time()-t0:5.1f}s", flush=True)
        if s == "SMOTE":   # deteksi sampel sintetis imblearn (replikasi Tabel S1)
            Xs, ys = imb_resample("SMOTE", Xtr, ytr)
            synth = Xs[len(ytr):]; real_min = Xtr[ytr == 1]
            bin_cols = [i for i in range(Xtr.shape[1]) if set(np.unique(Xtr[:, i])) <= {0.0, 1.0}]
            nonbin_rows = (~np.isin(synth[:, bin_cols], [0.0, 1.0])).any(axis=1).mean()
            rr = np.random.default_rng(SEED); n = min(len(real_min), 10000)
            A = np.vstack([real_min[rr.choice(len(real_min), n, False)], synth[rr.choice(len(synth), n, False)]]); b = np.r_[np.zeros(n), np.ones(n)]
            cvk = StratifiedKFold(5, shuffle=True, random_state=SEED)
            rowsD.append({"fold": k, "implementation": "imblearn", "synthetic_rows": len(synth), "share_rows_nonbinary": nonbin_rows,
                          "AUC_detect_HGB": cross_val_score(make_model("HGB", False), A, b, cv=cvk, scoring="roc_auc").mean(),
                          "AUC_detect_LR": cross_val_score(make_model("LR", False), A, b, cv=cvk, scoring="roc_auc").mean()})

    # ---------------- B. XGBoost & LightGBM ----------------
    for s in ["NONE", "CSL", "ROS", "RUS", "SMOTE"]:
        Xs, ys = imb_resample(s, Xtr, ytr) if s in ("ROS", "RUS", "SMOTE") else (Xtr, ytr)
        pi_seen = pi_tr if s == "NONE" else 0.5
        for m in ["XGB", "LGBM"]:
            t0 = time.time()
            mdl = make_boost(m, s, ys).fit(Xs, ys)
            p_ca, p_te = mdl.predict_proba(Xca)[:, 1], mdl.predict_proba(Xte)[:, 1]
            cals, _ = calibrators(p_ca, yca, pi_tr, pi_seen)
            for c in ["UNCAL", "PRIOR", "PLATT", "ISO"]:
                r = all_metrics(np.clip(cals[c](p_te), 0, 1), yte, pi_tr)
                rowsB.append({"fold": k, "model": m, "strategy": s, "calibration": c, **r})
            print(f"   B {m:4s} {s:5s} {time.time()-t0:5.1f}s", flush=True)
    pd.DataFrame(rowsA).to_csv("results/robustness_imblearn_equivalence.csv", index=False)
    pd.DataFrame(rowsB).to_csv("results/robustness_xgb_lgbm.csv", index=False)
    pd.DataFrame(rowsD).to_csv("results/robustness_smote_detectability_imblearn.csv", index=False)

# ---------------- ringkasan ----------------
A = pd.DataFrame(rowsA)
w = A.pivot_table(index=["fold", "model", "strategy", "calibration"], columns="implementation", values=["AUC", "Brier", "ECE", "Mean_pred"])
diff = pd.DataFrame({mt: (w[(mt, "imblearn")] - w[(mt, "own")]).abs() for mt in ["AUC", "Brier", "ECE", "Mean_pred"]})
print("\n=== A. |imblearn - own| (maks per strategi) ===")
print(diff.groupby(level="strategy").max().round(4))
B = pd.DataFrame(rowsB)
summ = B.groupby(["model", "strategy", "calibration"], sort=False)[["AUC", "Brier", "BSS", "ECE", "Cal_slope", "Cal_intercept", "Mean_pred", "Sens_t05", "Sens_tprev"]].agg(["mean", "std"])
summ.columns = [f"{a}_{b}" for a, b in summ.columns]; summ = summ.reset_index()
summ.to_csv("results/robustness_xgb_lgbm_summary.csv", index=False)
print("\n=== B. XGBoost / LightGBM (mean lintas fold) ===")
print(summ[["model", "strategy", "calibration", "AUC_mean", "Brier_mean", "ECE_mean", "Mean_pred_mean"]].round(4).to_string())
print("\n=== D. SMOTE imblearn: deteksi sampel sintetis ==="); print(pd.DataFrame(rowsD).round(4).to_string())

# ---------------- gambar ----------------
import matplotlib.pyplot as plt
from plot_style import apply_style
apply_style(9)
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
col = {"UNCAL": "#D55E00", "PRIOR": "#E69F00", "PLATT": "#0072B2", "ISO": "#009E73"}
S = ["NONE", "CSL", "ROS", "RUS", "SMOTE"]; x = np.arange(len(S)); wd = 0.2
for ax, m in zip(axs, ["XGB", "LGBM"]):
    for i, c in enumerate(["UNCAL", "PRIOR", "PLATT", "ISO"]):
        d = summ[(summ.model == m) & (summ.calibration == c)].set_index("strategy").loc[S]
        ax.bar(x + (i - 1.5) * wd, d.ECE_mean, wd, yerr=d.ECE_std.fillna(0), color=col[c], label=c)
    ax.set_xticks(x); ax.set_xticklabels(S); ax.set_title(m); ax.set_ylabel("ECE")
axs[0].legend(fontsize=7)
fig.savefig("figures/fig10_robustness_xgb_lgbm.png"); print("selesai")
