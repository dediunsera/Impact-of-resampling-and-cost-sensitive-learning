"""04_figures.py - Seluruh gambar & tabel pembanding untuk manuskrip (membaca data/ dan results/)."""
import itertools, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from plot_style import apply_style, COLORBLIND_SAFE_PALETTE as PAL
apply_style(9)
MODELS, STRATS, CALS = ["LR", "RF", "HGB"], ["NONE", "CSL", "ROS", "RUS", "SMOTE"], ["UNCAL", "PRIOR", "PLATT", "ISO"]
CCOL = {"UNCAL": "#D55E00", "PRIOR": "#E69F00", "PLATT": "#0072B2", "ISO": "#009E73"}
SCOL = dict(zip(STRATS, ["#000000", "#CC79A7", "#56B4E9", "#E69F00", "#D55E00"]))
P0 = dict(np.load("data/preds_fold0.npz")); y0 = P0["y_test"]
met = pd.read_csv("results/metrics_all_folds.csv"); cv = pd.read_csv("results/metrics_cv_summary.csv")
boot = pd.read_csv("results/fold0_bootstrap_ci.csv"); tl = pd.read_csv("results/train_log.csv")
hr = pd.read_csv("results/high_risk_observed.csv"); dca = pd.read_csv("results/decision_curve_fold0.csv")
fr = pd.read_csv("results/friedman_nemenyi.csv")

def rel_curve(p, y, nb=10):
    edges = np.unique(np.quantile(p, np.linspace(0, 1, nb + 1)))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    df = pd.DataFrame({"b": b, "p": p, "y": y}).groupby("b").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    se = np.sqrt(df.y * (1 - df.y) / df.n)
    return df.p.values, df.y.values, 1.96 * se.values

# ---------- Fig 2: distribusi kelas sebelum/sesudah resampling ----------
t0 = tl[(tl.fold == 0) & (tl.model == "LR")].set_index("strategy").loc[STRATS]
fig, ax = plt.subplots(figsize=(6.4, 2.8))
pos = t0.n_train_fit * t0.prev_train_fit; neg = t0.n_train_fit - pos
ax.bar(STRATS, neg, color="#56B4E9", label="Not stunted"); ax.bar(STRATS, pos, bottom=neg, color="#D55E00", label="Stunted")
for i, s in enumerate(STRATS):
    ax.text(i, t0.n_train_fit[s] + 800, f"n={int(t0.n_train_fit[s]):,}\n{t0.prev_train_fit[s]*100:.1f}% stunted", ha="center", fontsize=7.5)
ax.text(1, 2000, "reweighted\n(no new rows)", ha="center", fontsize=7, color="white")
ax.set_ylabel("Training rows (fold 0)"); ax.set_ylim(0, 92000); ax.legend(loc="upper left", ncol=2)
fig.savefig("figures/fig02_class_distribution.png"); plt.close()

# ---------- Fig 3: reliability diagrams 3x5 ----------
fig, axs = plt.subplots(3, 5, figsize=(11, 6.8), sharex=True, sharey=True)
for i, m in enumerate(MODELS):
    for j, s in enumerate(STRATS):
        ax = axs[i, j]; ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=0.9)
        for c in ["UNCAL", "PLATT", "ISO"]:
            px, py, e = rel_curve(P0[f"{m}|{s}|{c}"], y0)
            ax.errorbar(px, py, yerr=e, marker="o", ms=2.6, lw=1.1, capsize=0, color=CCOL[c], label=c)
        e_u = boot.query("model==@m and strategy==@s and calibration=='UNCAL'").ECE.iloc[0]
        e_i = boot.query("model==@m and strategy==@s and calibration=='ISO'").ECE.iloc[0]
        ax.text(0.03, 0.93, f"ECE raw {e_u:.3f}\nECE iso {e_i:.3f}", fontsize=6.8, va="top", transform=ax.transAxes)
        if i == 0: ax.set_title(s, fontsize=10)
        if j == 0: ax.set_ylabel(f"{m}\nobserved stunting rate")
        if i == 2: ax.set_xlabel("mean predicted probability")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
axs[0, 0].legend(loc="lower right", fontsize=7)
fig.savefig("figures/fig03_reliability_grid.png"); plt.close()

# ---------- Fig 4: distribusi probabilitas (HGB) ----------
fig, axs = plt.subplots(1, 5, figsize=(11, 2.6), sharey=True)
bins = np.linspace(0, 1, 41)
for j, s in enumerate(STRATS):
    ax = axs[j]
    for c in ["UNCAL", "PLATT"]:
        ax.hist(P0[f"HGB|{s}|{c}"], bins=bins, density=True, histtype="step", lw=1.4, color=CCOL[c], label=c)
    ax.axvline(y0.mean(), color="k", ls="--", lw=0.8); ax.set_title(f"HGB · {s}", fontsize=9); ax.set_xlabel("predicted probability")
axs[0].set_ylabel("density"); axs[0].legend(fontsize=7)
fig.savefig("figures/fig04_probability_distributions.png"); plt.close()

# ---------- Fig 5: heatmap Brier & ECE (mean over 5 folds) ----------
fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.6))
for ax, mt, cm in zip(axs, ["Brier", "ECE"], ["YlOrRd", "YlOrRd"]):
    M = cv.pivot_table(index=["model", "strategy"], columns="calibration", values=f"{mt}_mean").reindex(
        pd.MultiIndex.from_product([MODELS, STRATS]))[CALS]
    im = ax.imshow(M.values, cmap=cm, aspect="auto")
    for (r, c), v in np.ndenumerate(M.values):
        ax.text(c, r, f"{v:.3f}", ha="center", va="center", fontsize=7, color="k" if v < np.nanpercentile(M.values, 80) else "white")
    ax.set_xticks(range(4)); ax.set_xticklabels(CALS); ax.set_yticks(range(15)); ax.set_yticklabels([f"{a} · {b}" for a, b in M.index])
    ax.set_title(f"{mt} (mean of 5 household-grouped folds)", fontsize=9.5); ax.grid(False); plt.colorbar(im, ax=ax, shrink=0.8)
fig.savefig("figures/fig05_heatmap_brier_ece.png"); plt.close()

# ---------- Fig 6: forest plot Brier dengan CI bootstrap (fold 0) ----------
fig, axs = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True)
off = dict(zip(CALS, [-0.27, -0.09, 0.09, 0.27]))
for ax, m in zip(axs, MODELS):
    for c in CALS:
        b = boot[(boot.model == m) & (boot.calibration == c)].set_index("strategy").loc[STRATS]
        yy = np.arange(len(STRATS)) + off[c]
        ax.errorbar(b.Brier, yy, xerr=[b.Brier - b.Brier_lo, b.Brier_hi - b.Brier], fmt="o", ms=3.5, color=CCOL[c], label=c, capsize=2, lw=1)
    ref = boot.query("model==@m and strategy=='NONE' and calibration=='UNCAL'").Brier.iloc[0]
    ax.axvline(ref, color="grey", ls="--", lw=0.8); ax.set_title(m); ax.set_xlabel("Brier score (95% cluster-bootstrap CI)")
    ax.set_yticks(range(len(STRATS))); ax.set_yticklabels(STRATS); ax.invert_yaxis()
axs[0].legend(fontsize=7, loc="lower right")
fig.savefig("figures/fig06_forest_brier.png"); plt.close()

# ---------- Fig 7: (a) AUC vs ECE panah UNCAL->ISO ; (b) slope vs intercept ----------
f0 = met[met.fold == 0]
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
mk = dict(zip(MODELS, ["o", "s", "^"]))
for m, s in itertools.product(MODELS, STRATS):
    a = f0.query("model==@m and strategy==@s and calibration=='UNCAL'").iloc[0]; b = f0.query("model==@m and strategy==@s and calibration=='ISO'").iloc[0]
    axs[0].annotate("", xy=(b.ECE, b.AUC), xytext=(a.ECE, a.AUC), arrowprops=dict(arrowstyle="->", color=SCOL[s], lw=0.9))
    axs[0].scatter(a.ECE, a.AUC, marker=mk[m], color=SCOL[s], s=28, zorder=3)
    axs[0].scatter(b.ECE, b.AUC, marker=mk[m], facecolor="white", edgecolor=SCOL[s], s=28, zorder=3)
axs[0].set_xlabel("ECE (filled: uncalibrated → open: isotonic)"); axs[0].set_ylabel("AUC"); axs[0].set_title("(a) Discrimination vs calibration error")
from matplotlib.lines import Line2D
h = [Line2D([], [], color=SCOL[s], marker="o", ls="", label=s) for s in STRATS] + [Line2D([], [], color="k", marker=mk[m], ls="", mfc="none", label=m) for m in MODELS]
axs[0].legend(handles=h, fontsize=7, ncol=2, loc="center right")
for c in CALS:
    d = f0[f0.calibration == c]
    axs[1].scatter(d.Cal_intercept, d.Cal_slope, color=CCOL[c], label=c, s=[{"LR": 20, "RF": 20, "HGB": 20}[x] for x in d.model],
                   marker="o", alpha=0.85)
axs[1].axhline(1, color="grey", ls="--", lw=0.8); axs[1].axvline(0, color="grey", ls="--", lw=0.8)
axs[1].set_xlabel("calibration intercept (ideal 0)"); axs[1].set_ylabel("calibration slope (ideal 1)"); axs[1].set_title("(b) Calibration intercept and slope, 60 conditions")
axs[1].legend(fontsize=7)
fig.savefig("figures/fig07_auc_ece_slope.png"); plt.close()

# ---------- Fig 8: (a) prediksi >=0.8 vs kejadian nyata; (b) decision curve HGB ----------
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.0))
h8 = hr[(hr.threshold == 0.5)]
w = 0.2; x = np.arange(len(STRATS))
for k, c in enumerate(["UNCAL", "PLATT"]):
    d = h8[(h8.model == "HGB") & (h8.calibration == c)].set_index("strategy").reindex(STRATS)
    axs[0].bar(x + (k - 0.5) * 2 * w - w / 2, d.mean_pred, w, color=CCOL[c], alpha=0.45, label=f"{c}: mean predicted")
    axs[0].bar(x + (k - 0.5) * 2 * w + w / 2, d.observed_rate, w, color=CCOL[c], label=f"{c}: observed")
    for i, s in enumerate(STRATS):
        axs[0].text(x[i] + (k - 0.5) * 2 * w, (d.mean_pred.fillna(0)[s]) + 0.03, f"n={int(d.n_flagged[s]):,}", ha="center", fontsize=6, rotation=90)
axs[0].set_xticks(x); axs[0].set_xticklabels(STRATS); axs[0].set_ylim(0, 1.05)
axs[0].set_ylabel("probability / rate"); axs[0].set_title("(a) HGB, children flagged with p ≥ 0.5"); axs[0].legend(fontsize=6.5, loc="upper left")
curves = {"HGB|NONE|UNCAL": ("NONE, uncalibrated", "#000000", "-"), "HGB|SMOTE|UNCAL": ("SMOTE, uncalibrated", "#D55E00", "-"),
          "HGB|SMOTE|ISO": ("SMOTE + isotonic", "#009E73", "--"), "HGB|RUS|UNCAL": ("RUS, uncalibrated", "#E69F00", "-"),
          "Treat all": ("Treat all", "grey", ":"), "Treat none": ("Treat none", "grey", "-")}
for k, (lab, col, ls) in curves.items():
    d = dca[dca.curve == k]; axs[1].plot(d.threshold, d.net_benefit, ls=ls, color=col, label=lab, lw=1.3)
axs[1].set_ylim(-0.03, 0.20); axs[1].set_xlabel("threshold probability"); axs[1].set_ylabel("net benefit"); axs[1].set_title("(b) Decision curve analysis (HGB, fold 0)")
axs[1].legend(fontsize=7)
fig.savefig("figures/fig08_highrisk_dca.png"); plt.close()

# ---------- Fig 9: critical difference (Nemenyi) ----------
def cd_plot(ax, row, labels, title):
    ranks = {l: row[f"rank_{l}"] for l in labels}; cd = row.CD_nemenyi; k = len(labels)
    ax.set_xlim(1, k); ax.set_ylim(0, 1); ax.invert_xaxis(); ax.set_yticks([]); ax.grid(False)
    for sp in ["left", "right", "top"]: ax.spines[sp].set_visible(False)
    srt = sorted(ranks.items(), key=lambda t: t[1])
    for i, (l, r) in enumerate(srt):
        yy = 0.85 - i * 0.15
        ax.plot([r, r], [0, yy], color="k", lw=0.8); ax.text(r, yy + 0.02, f"{l} ({r:.2f})", ha="center", fontsize=7.5)
    ax.plot([k, k - cd], [0.97, 0.97], color="#D55E00", lw=3); ax.text(k - cd / 2, 0.9, f"CD = {cd:.2f}", ha="center", fontsize=7, color="#D55E00")
    ax.set_title(f"{title}\nFriedman χ² = {row.chi2:.1f}, p = {row.p_value:.1e}", fontsize=8.5); ax.set_xlabel("mean rank (1 = best)")
fig, axs = plt.subplots(1, 3, figsize=(11, 2.9))
r1 = fr.query("comparison=='Strategies (uncalibrated)' and metric=='Brier'").iloc[0]
r2 = fr.query("comparison=='Calibration methods (resampled/CSL models)' and metric=='Brier'").iloc[0]
r3 = fr.query("comparison=='Strategies (after isotonic)' and metric=='Brier'").iloc[0]
cd_plot(axs[0], r1, STRATS, "(a) Strategies, uncalibrated (Brier)")
cd_plot(axs[1], r2, CALS, "(b) Calibration methods (Brier)")
cd_plot(axs[2], r3, STRATS, "(c) Strategies after isotonic (Brier)")
fig.savefig("figures/fig09_critical_difference.png"); plt.close()

# ================= TABEL =================
def fmt(v, d=3): return f"{v:.{d}f}"
# Tabel A: perbandingan strategi (uncalibrated) fold 0 + CI
rows = []
for m, s in itertools.product(MODELS, STRATS):
    b = boot.query("model==@m and strategy==@s and calibration=='UNCAL'").iloc[0]
    f = f0.query("model==@m and strategy==@s and calibration=='UNCAL'").iloc[0]
    rows.append({"Model": m, "Strategy": s, "AUC (95% CI)": f"{b.AUC:.3f} ({b.AUC_lo:.3f}–{b.AUC_hi:.3f})",
                 "Brier (95% CI)": f"{b.Brier:.4f} ({b.Brier_lo:.4f}–{b.Brier_hi:.4f})", "ECE (95% CI)": f"{b.ECE:.3f} ({b.ECE_lo:.3f}–{b.ECE_hi:.3f})",
                 "Slope": fmt(f.Cal_slope, 2), "Intercept": fmt(f.Cal_intercept, 2), "Mean p": fmt(f.Mean_pred), "O/E": fmt(f.OE_ratio, 2),
                 "Sens@0.5": fmt(f.Sens_t05), "Spec@0.5": fmt(f.Spec_t05)})
pd.DataFrame(rows).to_csv("results/table_strategies_uncalibrated.csv", index=False)
# Tabel B: efek kalibrasi (mean ± SD 5 fold) per model x strategi
rows = []
for m, s in itertools.product(MODELS, STRATS):
    r = {"Model": m, "Strategy": s}
    for c in CALS:
        q = cv.query("model==@m and strategy==@s and calibration==@c").iloc[0]
        r[f"Brier {c}"] = f"{q.Brier_mean:.4f}±{q.Brier_std:.4f}"; r[f"ECE {c}"] = f"{q.ECE_mean:.3f}±{q.ECE_std:.3f}"
    rows.append(r)
pd.DataFrame(rows).to_csv("results/table_calibration_effect_cv.csv", index=False)
# Tabel C: delta Brier & ECE vs uncalibrated (fold 0, bootstrap)
rows = []
for m, s, c in itertools.product(MODELS, STRATS, ["PRIOR", "PLATT", "ISO"]):
    b = boot.query("model==@m and strategy==@s and calibration==@c").iloc[0]
    rows.append({"Model": m, "Strategy": s, "Calibration": c,
                 "ΔBrier vs UNCAL (95% CI)": f"{b.dBrier_vsUNCAL:+.4f} ({b.dBrier_vsUNCAL_lo:+.4f} to {b.dBrier_vsUNCAL_hi:+.4f})",
                 "ΔECE vs UNCAL (95% CI)": f"{b.dECE_vsUNCAL:+.3f} ({b.dECE_vsUNCAL_lo:+.3f} to {b.dECE_vsUNCAL_hi:+.3f})",
                 "ΔBrier vs NONE-UNCAL (95% CI)": f"{b.dBrier_vsBASE:+.4f} ({b.dBrier_vsBASE_lo:+.4f} to {b.dBrier_vsBASE_hi:+.4f})"})
pd.DataFrame(rows).to_csv("results/table_delta_bootstrap.csv", index=False)
# comparison.csv (ringkas, siap naskah): mean 5 fold
comp = cv[["model", "strategy", "calibration", "AUC_mean", "Brier_mean", "BSS_mean", "ECE_mean", "Cal_slope_mean", "Cal_intercept_mean", "Mean_pred_mean", "Sens_t05_mean", "Sens_tprev_mean", "Spec_tprev_mean"]]
comp.round(4).to_csv("results/comparison.csv", index=False)
comp.round(4).to_csv("results/metrics.csv", index=False)
print("figures & tables ok")
