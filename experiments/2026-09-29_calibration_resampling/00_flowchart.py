"""00_flowchart.py - Diagram alur metodologi penelitian (Figure 1 manuskrip)."""
import json
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from plot_style import apply_style
apply_style(10)
plt.rcParams["axes.grid"] = False

flow = {r["step"]: r["n"] for r in __import__("pandas").read_csv("results/cohort_flow.csv").to_dict("records")}
cs = json.load(open("results/cohort_summary.json"))
fig, ax = plt.subplots(figsize=(7.2, 9.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 132); ax.axis("off")
C = {"data": "#DCEAF7", "prep": "#E8F4EA", "split": "#FFF1D6", "exp": "#F9E0E0", "cal": "#EDE3F5", "eval": "#E6E6E6"}
def box(x, y, w, h, text, col, fs=8.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=col, ec="#333333", lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", wrap=True)
def arr(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#333333"))

box(20, 122, 60, 8, f"SKI 2023 child module (0–59 months)\nn = {flow['Catatan balita di modul SKI 2023']:,} children, 171 variables", C["data"], bold=True)
arr(50, 122, 50, 118.5)
box(6, 106, 88, 12, "Outcome construction: age in days; length/height harmonised (±0.7 cm)\nHAZ from WHO Child Growth Standards (LMS, lenanthro)\n"
    f"exclude missing height/position and |HAZ| > 6  →  n = {cs['n']:,}; stunting = HAZ < −2 ({cs['prev_unweighted']*100:.1f}%)", C["prep"])
arr(50, 106, 50, 102.5)
box(6, 92, 88, 10.5, f"Leakage-free feature set: {cs['n_features']} upstream determinants\n(current anthropometry, outcome-derived items, IDs and weights removed)\n"
    "median imputation + missing indicators, scaling, one-hot (fit on training only)", C["prep"])
arr(50, 92, 50, 88.5)
box(6, 79, 88, 9.5, "Household-grouped stratified 5-fold outer split (no household shared)\nper fold:  Training 60%  |  Calibration 20% (original prevalence)  |  Test 20%", C["split"], bold=False)
arr(50, 79, 50, 77.3)
box(18, 73.2, 64, 4, "Class-imbalance handling (applied to the training part only)", "#FFFFFF", fs=8.2, bold=True)
labs = ["NONE\n(reference)", "CSL\nclass weights", "ROS\nrandom over", "RUS\nrandom under", "SMOTE\nk = 5"]
for i, l in enumerate(labs):
    box(4 + i * 19, 62, 16, 9, l, C["exp"], fs=7.8)
    arr(50, 72.8, 12 + i * 19, 71.3)
    arr(12 + i * 19, 62, 50, 58.3)
box(6, 50.5, 88, 7.5, "Models: logistic regression (LR) · random forest (RF) · histogram gradient boosting (HGB)\n→ raw predicted probabilities for calibration and test sets", C["exp"])
arr(50, 50.5, 50, 48.8)
box(18, 44.6, 64, 4, "Post-hoc probability calibration (fitted on the calibration set)", "#FFFFFF", fs=8.2, bold=True)
labs = ["UNCAL\nraw output", "PRIOR\nanalytic prior shift", "PLATT\nlogistic on logit(p)", "ISO\nisotonic regression"]
for i, l in enumerate(labs):
    box(5 + i * 23.5, 34, 20, 9, l, C["cal"], fs=7.8)
    arr(50, 44.2, 15 + i * 23.5, 43.3)
    arr(15 + i * 23.5, 34, 50, 30.3)
box(3, 18, 94, 12, "Evaluation on unseen households (3 models × 5 strategies × 4 calibrations = 60 conditions)\n"
    "Calibration: Brier score, Brier skill score, ECE, calibration slope & intercept, O/E, reliability diagrams\n"
    "Discrimination & utility: AUC, PR-AUC, sensitivity/specificity, decision-curve net benefit",
    C["eval"], fs=7.6)
arr(50, 18, 50, 14.5)
box(3, 3, 94, 11.5, "Statistical inference: household cluster bootstrap (B = 1,000) with paired ΔBrier/ΔECE\n"
    "and Friedman–Nemenyi tests across 5 folds  →  recommendations for probability-based stunting risk tools", C["eval"], fs=7.6)
fig.savefig("figures/fig01_flowchart.png", dpi=300); print("ok")
