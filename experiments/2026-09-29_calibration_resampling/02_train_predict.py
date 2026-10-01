"""
02_train_predict.py
Tahap 2 - Pelatihan model + strategi penanganan ketidakseimbangan kelas + kalibrasi probabilitas.

Desain (untuk setiap outer fold k = 0..4, split berbasis RUMAH TANGGA):
  test  = fold k (20%)             -> hanya untuk evaluasi akhir
  calib = 1/4 sisa (~20%)          -> distribusi ASLI, untuk fitting Platt / Isotonic
  train = 3/4 sisa (~60%)          -> satu-satunya bagian yang di-resampling

Faktor eksperimen:
  Model     : LR (regresi logistik L2), RF (random forest), HGB (histogram gradient boosting)
  Strategi  : NONE, CSL (cost-sensitive, class_weight=balanced), ROS, RUS, SMOTE (k=5)
  Kalibrasi : UNCAL, PRIOR (koreksi prior analitik), PLATT (sigmoid), ISO (isotonic)
Output: data/preds_fold{k}.npz  (probabilitas test untuk 60 kondisi) + results/train_log.csv
Jalankan:  python 02_train_predict.py --folds 0 1 2 3 4
"""
import argparse, json, time, os
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

SEED = 42
ap = argparse.ArgumentParser()
ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
ap.add_argument("--out", default=".")
a = ap.parse_args()

d = pd.read_pickle(f"{a.out}/data/analytic.pkl")
ft = json.load(open(f"{a.out}/data/feature_types.json"))
NUM, CAT = ft["num"], ft["cat"]
y_all = d["stunting"].values.astype(int); g_all = d["household_id"].values
X_all = d[NUM + CAT].copy()
for c in CAT: X_all[c] = X_all[c].astype("object").where(X_all[c].notna(), "MISSING").astype(str)

def make_pre():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50, sparse_output=False), CAT)],
        sparse_threshold=0.0)

# ---------------- resampling (diimplementasikan eksplisit agar portabel; setara imbalanced-learn) -------------
def ros(X, y, rng):
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    add = rng.choice(i1, len(i0) - len(i1), replace=True)
    idx = np.concatenate([i0, i1, add]); return X[idx], y[idx]

def rus(X, y, rng):
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    keep = rng.choice(i0, len(i1), replace=False)
    idx = np.concatenate([keep, i1]); return X[idx], y[idx]

def smote(X, y, rng, k=5):
    """SMOTE (Chawla et al.): sampel sintetis = x_i + u * (x_nn - x_i), nn dari k tetangga minoritas."""
    Xm = X[y == 1]; n_new = int((y == 0).sum() - (y == 1).sum())
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xm)
    _, nbr = nn.kneighbors(Xm)
    base = rng.integers(0, len(Xm), n_new)
    pick = nbr[base, rng.integers(1, k + 1, n_new)]
    u = rng.random((n_new, 1)).astype(X.dtype)
    Xs = Xm[base] + u * (Xm[pick] - Xm[base])
    return np.vstack([X, Xs]), np.concatenate([y, np.ones(n_new, int)])

def make_model(name, balanced):
    cw = "balanced" if balanced else None
    if name == "LR":
        return LogisticRegression(C=1.0, max_iter=3000, class_weight=cw)
    if name == "RF":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_features="sqrt", max_samples=0.5,
                                      class_weight=cw, n_jobs=-1, random_state=SEED)
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, early_stopping=True,
                                          validation_fraction=0.1, class_weight=cw, random_state=SEED)

EPS = 1e-6
def logit(p): p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))

def calibrators(p_cal, y_cal, pi_train, pi_seen):
    """Kembalikan dict fungsi kalibrasi yang di-fit pada calibration set (distribusi asli)."""
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p_cal).reshape(-1, 1), y_cal)
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p_cal, y_cal)
    r = (pi_train / (1 - pi_train)) / (pi_seen / (1 - pi_seen))   # rasio odds prior asli / prior yang dilihat model
    return {
        "UNCAL": lambda p: p,
        "PRIOR": lambda p: (r * p) / (r * p + (1 - p)),
        "PLATT": lambda p: platt.predict_proba(logit(p).reshape(-1, 1))[:, 1],
        "ISO": lambda p: iso.predict(p),
    }, {"platt_a": float(platt.intercept_[0]), "platt_b": float(platt.coef_[0, 0]), "prior_r": float(r)}

MODELS, STRATS = ["LR", "RF", "HGB"], ["NONE", "CSL", "ROS", "RUS", "SMOTE"]
outer = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(X_all, y_all, g_all))
log = []
for k in a.folds:
    rng = np.random.default_rng(SEED + k)
    dev_idx, te_idx = outer[k]
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED + k)
    tr_rel, ca_rel = next(inner.split(X_all.iloc[dev_idx], y_all[dev_idx], g_all[dev_idx]))
    tr_idx, ca_idx = dev_idx[tr_rel], dev_idx[ca_rel]
    assert not (set(g_all[tr_idx]) & set(g_all[te_idx])) and not (set(g_all[ca_idx]) & set(g_all[te_idx]))
    assert not (set(g_all[tr_idx]) & set(g_all[ca_idx]))
    pre = make_pre().fit(X_all.iloc[tr_idx])
    Xtr = pre.transform(X_all.iloc[tr_idx]).astype(np.float32)
    Xca = pre.transform(X_all.iloc[ca_idx]).astype(np.float32)
    Xte = pre.transform(X_all.iloc[te_idx]).astype(np.float32)
    ytr, yca, yte = y_all[tr_idx], y_all[ca_idx], y_all[te_idx]
    pi_tr = ytr.mean()
    print(f"[fold {k}] train {len(tr_idx)} calib {len(ca_idx)} test {len(te_idx)} | p={Xtr.shape[1]} | prev {pi_tr:.4f}", flush=True)
    store = {"y_test": yte, "g_test": g_all[te_idx], "idx_test": te_idx, "y_cal": yca}
    for s in STRATS:
        if s == "ROS": Xs, ys = ros(Xtr, ytr, rng)
        elif s == "RUS": Xs, ys = rus(Xtr, ytr, rng)
        elif s == "SMOTE": Xs, ys = smote(Xtr, ytr, rng)
        else: Xs, ys = Xtr, ytr
        pi_seen = 0.5 if s != "NONE" else pi_tr     # CSL 'balanced' setara prior efektif 0.5
        for m in MODELS:
            t0 = time.time()
            mdl = make_model(m, balanced=(s == "CSL")).fit(Xs, ys)
            p_ca, p_te = mdl.predict_proba(Xca)[:, 1], mdl.predict_proba(Xte)[:, 1]
            cals, info = calibrators(p_ca, yca, pi_tr, pi_seen)
            for c, f in cals.items():
                store[f"{m}|{s}|{c}"] = np.clip(f(p_te), 0, 1).astype(np.float64)
            dt = time.time() - t0
            log.append({"fold": k, "model": m, "strategy": s, "n_train_fit": len(ys), "prev_train_fit": ys.mean(),
                        "n_features": Xtr.shape[1], "fit_seconds": round(dt, 1), **info})
            print(f"   {m:3s} {s:5s} n={len(ys):6d} prev={ys.mean():.3f} {dt:6.1f}s", flush=True)
    np.savez_compressed(f"{a.out}/data/preds_fold{k}.npz", **store)
    pd.DataFrame(log).to_csv(f"{a.out}/results/train_log.csv", index=False)
print("selesai")
