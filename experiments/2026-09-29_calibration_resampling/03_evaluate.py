"""
03_evaluate.py
Tahap 3 - Evaluasi kalibrasi & diskriminasi untuk 60 kondisi (3 model x 5 strategi x 4 kalibrasi).
 - Metrik per fold (5 outer folds berbasis rumah tangga)
 - Fold 0 (analisis utama): CI 95% dengan cluster bootstrap rumah tangga (B=1000) + delta berpasangan
 - Uji Friedman + Nemenyi lintas fold
 - Tabel "prediksi tinggi vs kejadian nyata", decision curve analysis, data kurva kalibrasi
Output di results/
"""
import json, itertools
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression

SEED, B, NB = 42, 1000, 10
MODELS, STRATS, CALS = ["LR", "RF", "HGB"], ["NONE", "CSL", "ROS", "RUS", "SMOTE"], ["UNCAL", "PRIOR", "PLATT", "ISO"]
EPS = 1e-6
def logit(p): p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))
def expit(z): return 1 / (1 + np.exp(-z))

def ece_mce(p, y, nb=NB, adaptive=False):
    if adaptive:
        edges = np.unique(np.quantile(p, np.linspace(0, 1, nb + 1)))
        b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    else:
        b = np.minimum((p * nb).astype(int), nb - 1)
    n = np.bincount(b, minlength=nb); sp = np.bincount(b, p, nb); sy = np.bincount(b, y, nb)
    m = n > 0; gap = np.abs(sp[m] / n[m] - sy[m] / n[m])
    return float((n[m] * gap).sum() / len(p)), float(gap.max())

def murphy(p, y, nb=NB):
    b = np.minimum((p * nb).astype(int), nb - 1)
    n = np.bincount(b, minlength=nb); sp = np.bincount(b, p, nb); sy = np.bincount(b, y, nb)
    m = n > 0; ob = y.mean()
    rel = (n[m] * (sp[m] / n[m] - sy[m] / n[m]) ** 2).sum() / len(p)
    res = (n[m] * (sy[m] / n[m] - ob) ** 2).sum() / len(p)
    return rel, res, ob * (1 - ob)

def cal_slope_itl(p, y):
    lp = logit(p)
    slope = LogisticRegression(C=1e8, max_iter=2000).fit(lp.reshape(-1, 1), y).coef_[0, 0]
    a = 0.0
    for _ in range(50):                     # Newton: intercept dengan offset logit(p)
        q = expit(a + lp); g = (y - q).sum(); h = (q * (1 - q)).sum(); a += g / h
        if abs(g / h) < 1e-10: break
    return float(slope), float(a)

def spiegelhalter(p, y):
    z = ((y - p) * (1 - 2 * p)).sum() / np.sqrt(((1 - 2 * p) ** 2 * p * (1 - p)).sum())
    return float(z), float(2 * stats.norm.sf(abs(z)))

def thr_metrics(p, y, t):
    yh = (p >= t).astype(int); tp = ((yh == 1) & (y == 1)).sum(); tn = ((yh == 0) & (y == 0)).sum()
    fp = ((yh == 1) & (y == 0)).sum(); fn = ((yh == 0) & (y == 1)).sum()
    sens = tp / (tp + fn); spec = tn / (tn + fp); prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    return sens, spec, prec, f1, mcc

def all_metrics(p, y, pi_tr):
    brier = np.mean((p - y) ** 2); bref = np.mean((pi_tr - y) ** 2)
    ece, mce = ece_mce(p, y); ecea, _ = ece_mce(p, y, adaptive=True)
    slope, citl = cal_slope_itl(p, y); z, zp = spiegelhalter(p, y); rel, res, unc = murphy(p, y)
    pc = np.clip(p, EPS, 1 - EPS)
    r = dict(AUC=roc_auc_score(y, p), PR_AUC=average_precision_score(y, p), Brier=brier, BSS=1 - brier / bref,
             LogLoss=-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)), ECE=ece, ECE_adaptive=ecea, MCE=mce,
             Cal_slope=slope, Cal_intercept=citl, OE_ratio=y.sum() / p.sum(), Mean_pred=p.mean(), Obs_prev=y.mean(),
             Spiegelhalter_Z=z, Spiegelhalter_p=zp, Reliability=rel, Resolution=res, Uncertainty=unc)
    for t, tag in [(0.5, "t05"), (pi_tr, "tprev")]:
        s, sp, pr, f1, mcc = thr_metrics(p, y, t)
        r.update({f"Sens_{tag}": s, f"Spec_{tag}": sp, f"Prec_{tag}": pr, f"F1_{tag}": f1, f"MCC_{tag}": mcc})
    return r

folds = [k for k in range(5)]
P = {k: dict(np.load(f"data/preds_fold{k}.npz")) for k in folds}
tl = pd.read_csv("results/train_log.csv")
rows = []
for k in folds:
    y = P[k]["y_test"]; pi_tr = float(tl[tl.fold == k].query("strategy=='NONE'").prev_train_fit.iloc[0])
    for m, s, c in itertools.product(MODELS, STRATS, CALS):
        rows.append({"fold": k, "model": m, "strategy": s, "calibration": c, **all_metrics(P[k][f"{m}|{s}|{c}"], y, pi_tr)})
    print("fold", k, "ok", flush=True)
met = pd.DataFrame(rows); met.to_csv("results/metrics_all_folds.csv", index=False)

# ringkasan lintas fold (mean ± SD)
key = ["model", "strategy", "calibration"]
cols = ["AUC", "PR_AUC", "Brier", "BSS", "LogLoss", "ECE", "ECE_adaptive", "MCE", "Cal_slope", "Cal_intercept", "OE_ratio",
        "Mean_pred", "Sens_t05", "Spec_t05", "F1_t05", "MCC_t05", "Sens_tprev", "Spec_tprev", "F1_tprev", "MCC_tprev", "Reliability", "Resolution"]
summ = met.groupby(key, sort=False)[cols].agg(["mean", "std"])
summ.columns = [f"{a}_{b}" for a, b in summ.columns]; summ = summ.reset_index()
summ.to_csv("results/metrics_cv_summary.csv", index=False)

# ---------------- bootstrap (fold 0) ----------------
rng = np.random.default_rng(SEED)
y0 = P[0]["y_test"]; g0 = P[0]["g_test"]; pi0 = float(tl[(tl.fold == 0) & (tl.strategy == "NONE")].prev_train_fit.iloc[0])
_, inv = np.unique(g0, return_inverse=True); members = pd.Series(np.arange(len(g0))).groupby(inv).apply(np.array).values
H = len(members)
BOOT = [np.concatenate(members[rng.integers(0, H, H)]) for _ in range(B)]
def boot_vec(p, fn): return np.array([fn(p[i], y0[i]) for i in BOOT])
brier_f = lambda p, y: np.mean((p - y) ** 2)
ece_f = lambda p, y: ece_mce(p, y)[0]
bs = {}
for m, s, c in itertools.product(MODELS, STRATS, CALS):
    p = P[0][f"{m}|{s}|{c}"]
    bs[(m, s, c, "Brier")] = boot_vec(p, brier_f); bs[(m, s, c, "ECE")] = boot_vec(p, ece_f)
    if c in ("UNCAL", "ISO"): bs[(m, s, c, "AUC")] = boot_vec(p, lambda pp, yy: roc_auc_score(yy, pp))
print("bootstrap ok", flush=True)
def ci(v): return np.percentile(v, [2.5, 97.5])
m0 = met[met.fold == 0].set_index(key)
out = []
for m, s, c in itertools.product(MODELS, STRATS, CALS):
    r = {"model": m, "strategy": s, "calibration": c}
    for mt in ["Brier", "ECE", "AUC"]:
        kk = (m, s, c if (mt != "AUC" or c in ("UNCAL", "ISO")) else "UNCAL", mt)
        r[mt] = m0.loc[(m, s, c), mt]; lo, hi = ci(bs[kk]); r[f"{mt}_lo"], r[f"{mt}_hi"] = lo, hi
    # delta vs UNCAL (strategi sama) dan vs NONE-UNCAL (model sama)
    for ref, tag in [((m, s, "UNCAL"), "vsUNCAL"), ((m, "NONE", "UNCAL"), "vsBASE")]:
        for mt in ["Brier", "ECE"]:
            dv = bs[(m, s, c, mt)] - bs[(*ref, mt)]
            r[f"d{mt}_{tag}"] = m0.loc[(m, s, c), mt] - m0.loc[ref, mt]
            r[f"d{mt}_{tag}_lo"], r[f"d{mt}_{tag}_hi"] = ci(dv)
            r[f"d{mt}_{tag}_p"] = min(1.0, 2 * min((dv >= 0).mean(), (dv <= 0).mean())) if ref != (m, s, c) else np.nan
    out.append(r)
boot = pd.DataFrame(out); boot.to_csv("results/fold0_bootstrap_ci.csv", index=False)
met[met.fold == 0].drop(columns="fold").to_csv("results/metrics_fold0.csv", index=False)

# ---------------- Friedman + Nemenyi ----------------
QA = {4: 2.569, 5: 2.728}
fr = []
def friedman(df, treat, blocks, metric, label):
    w = df.pivot_table(index=blocks, columns=treat, values=metric)
    ranks = w.rank(axis=1).mean()
    st, pv = stats.friedmanchisquare(*[w[c] for c in w.columns])
    k, n = w.shape[1], w.shape[0]; cd = QA[k] * np.sqrt(k * (k + 1) / (6 * n))
    fr.append({"comparison": label, "metric": metric, "k": k, "N_blocks": n, "chi2": st, "p_value": pv, "CD_nemenyi": cd,
               **{f"rank_{c}": ranks[c] for c in w.columns}})
for mt in ["Brier", "ECE", "LogLoss"]:
    friedman(met[met.calibration == "UNCAL"], "strategy", ["fold", "model"], mt, "Strategies (uncalibrated)")
    friedman(met[met.strategy != "NONE"], "calibration", ["fold", "model", "strategy"], mt, "Calibration methods (resampled/CSL models)")
    friedman(met[met.calibration == "ISO"], "strategy", ["fold", "model"], mt, "Strategies (after isotonic)")
pd.DataFrame(fr).to_csv("results/friedman_nemenyi.csv", index=False)

# ---------------- prediksi tinggi vs kejadian nyata (fold 0) ----------------
hr = []
for m, s, c in itertools.product(MODELS, STRATS, CALS):
    p = P[0][f"{m}|{s}|{c}"]
    for lo in [0.5, 0.7, 0.8]:
        sel = p >= lo
        hr.append({"model": m, "strategy": s, "calibration": c, "threshold": lo, "n_flagged": int(sel.sum()),
                   "pct_flagged": sel.mean() * 100, "mean_pred": p[sel].mean() if sel.any() else np.nan,
                   "observed_rate": y0[sel].mean() if sel.any() else np.nan})
pd.DataFrame(hr).to_csv("results/high_risk_observed.csv", index=False)

# decile table (HGB, SMOTE) untuk ilustrasi
dec = []
for (m, s) in [("HGB", "NONE"), ("HGB", "SMOTE"), ("LR", "SMOTE"), ("RF", "SMOTE")]:
    for c in CALS:
        p = P[0][f"{m}|{s}|{c}"]; b = np.minimum((p * 10).astype(int), 9)
        for i in range(10):
            sel = b == i
            dec.append({"model": m, "strategy": s, "calibration": c, "bin": f"{i/10:.1f}-{(i+1)/10:.1f}", "n": int(sel.sum()),
                        "mean_pred": p[sel].mean() if sel.any() else np.nan, "observed": y0[sel].mean() if sel.any() else np.nan})
pd.DataFrame(dec).to_csv("results/reliability_bins_fold0.csv", index=False)

# ---------------- decision curve (fold 0) ----------------
th = np.round(np.arange(0.05, 0.61, 0.01), 2); dca = []
N = len(y0); prev = y0.mean()
for t in th:
    dca.append({"threshold": t, "curve": "Treat all", "net_benefit": prev - (1 - prev) * t / (1 - t)})
    dca.append({"threshold": t, "curve": "Treat none", "net_benefit": 0.0})
    for m, s, c in itertools.product(MODELS, STRATS, CALS):
        p = P[0][f"{m}|{s}|{c}"]; yh = p >= t
        tp = (yh & (y0 == 1)).sum() / N; fp = (yh & (y0 == 0)).sum() / N
        dca.append({"threshold": t, "curve": f"{m}|{s}|{c}", "net_benefit": tp - fp * t / (1 - t)})
pd.DataFrame(dca).to_csv("results/decision_curve_fold0.csv", index=False)
json.dump({"B": B, "n_test_fold0": int(N), "households_test_fold0": int(H), "prev_test_fold0": float(prev), "pi_train_fold0": pi0},
          open("results/eval_config.json", "w"), indent=1)
print("selesai")
