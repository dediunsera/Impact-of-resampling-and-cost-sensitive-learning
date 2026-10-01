"""
05_smote_diagnostic.py
Diagnostik: mengapa SMOTE berperilaku berbeda pada model pohon?
Uji 'adversarial validation': seberapa mudah sampel sintetis SMOTE dibedakan dari sampel minoritas asli
(di ruang fitur setelah one-hot + indikator missing), serta proporsi nilai non-biner pada kolom dummy.
Output: results/smote_diagnostic.csv
"""
import json, numpy as np, pandas as pd, importlib.util, sys
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
src = open("02_train_predict.py").read().split("MODELS, STRATS")[0]   # hanya definisi fungsi & data
sys.argv = ["x"]; exec(src)
outer = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(X_all, y_all, g_all))
dev_idx, te_idx = outer[0]
tr_rel, ca_rel = next(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED).split(X_all.iloc[dev_idx], y_all[dev_idx], g_all[dev_idx]))
tr_idx = dev_idx[tr_rel]
pre = make_pre().fit(X_all.iloc[tr_idx]); Xtr = pre.transform(X_all.iloc[tr_idx]).astype(np.float32); ytr = y_all[tr_idx]
rng = np.random.default_rng(SEED)
Xs, ys = smote(Xtr, ytr, rng)
n0 = len(ytr); real_min = Xtr[ytr == 1]; synth = Xs[n0:]
n_num = len(NUM); names = pre.get_feature_names_out()
binary_cols = [i for i in range(Xtr.shape[1]) if set(np.unique(Xtr[:, i])) <= {0.0, 1.0}]
frac_nonbin = np.mean([(~np.isin(synth[:, i], [0.0, 1.0])).mean() for i in binary_cols])
rows_nonbin = (~np.isin(synth[:, binary_cols], [0.0, 1.0])).any(axis=1).mean()
# adversarial validation: asli (0) vs sintetis (1), subsample seimbang
k = min(len(real_min), 10000); A = np.vstack([real_min[rng.choice(len(real_min), k, False)], synth[rng.choice(len(synth), k, False)]])
b = np.r_[np.zeros(k), np.ones(k)]
cvk = StratifiedKFold(5, shuffle=True, random_state=SEED)
auc_hgb = cross_val_score(HistGradientBoostingClassifier(random_state=SEED), A, b, cv=cvk, scoring="roc_auc")
auc_lr = cross_val_score(LogisticRegression(max_iter=3000), A, b, cv=cvk, scoring="roc_auc")
res = pd.DataFrame([
    {"quantity": "binary (dummy/indicator) columns in design matrix", "value": len(binary_cols)},
    {"quantity": "share of synthetic cells in binary columns with non-0/1 value", "value": round(frac_nonbin, 4)},
    {"quantity": "share of synthetic rows with >=1 non-0/1 binary cell", "value": round(rows_nonbin, 4)},
    {"quantity": "AUC separating synthetic vs real minority, HGB (5-fold, mean)", "value": round(auc_hgb.mean(), 4)},
    {"quantity": "AUC separating synthetic vs real minority, HGB (SD)", "value": round(auc_hgb.std(), 4)},
    {"quantity": "AUC separating synthetic vs real minority, LR (5-fold, mean)", "value": round(auc_lr.mean(), 4)},
    {"quantity": "AUC separating synthetic vs real minority, LR (SD)", "value": round(auc_lr.std(), 4)}])
res.to_csv("results/smote_diagnostic.csv", index=False); print(res.to_string())
