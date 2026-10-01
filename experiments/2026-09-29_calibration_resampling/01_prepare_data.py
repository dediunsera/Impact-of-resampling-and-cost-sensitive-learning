"""
01_prepare_data.py
Tahap 1 - Persiapan data SKI 2023 (balita 0-59 bulan) untuk eksperimen kalibrasi.
 - Menghitung umur (hari) dari tanggal lahir dan tanggal wawancara
 - Menghitung HAZ dengan tabel resmi WHO (lenanthro.txt, metode LMS)
 - Label stunting: HAZ < -2
 - Menyusun set fitur determinan hulu (BEBAS kebocoran antropometri)
Output: data/analytic.pkl, results/feature_list.csv, results/exclusion_list.csv,
        results/cohort_flow.csv
Jalankan:  python 01_prepare_data.py --xlsx "<path SKI xlsx>" --lms "<path lenanthro.txt>"
"""
import argparse, json, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--xlsx", default="SKI 2023 Balita 0-59 Bulan_asli labels.xlsx")
ap.add_argument("--lms", default="lenanthro.txt")
ap.add_argument("--pkl", default=None, help="cache pickle of raw xlsx (optional)")
ap.add_argument("--out", default=".")
a = ap.parse_args()
os.makedirs(f"{a.out}/data", exist_ok=True); os.makedirs(f"{a.out}/results", exist_ok=True)

if a.pkl and os.path.exists(a.pkl):
    df = pd.read_pickle(a.pkl)
else:
    df = pd.read_excel(a.xlsx)
flow = [("Catatan balita di modul SKI 2023", len(df))]

# ---------- umur (hari) ----------
def to_date(x):
    s = str(int(x)).zfill(8)
    return pd.to_datetime(s, format="%d%m%Y", errors="coerce")
dob = df["6. Tanggal Lahir"].map(to_date)
doi = df["2. Tanggal Pengumpulan data: (tgl-bln)"].map(to_date)
df["age_days"] = (doi - dob).dt.days
df["sex"] = df["4. Jenis Kelamin"].map({"Laki-laki": 1, "Prempuan": 2})

# ---------- HAZ (WHO LMS; L=1 untuk length/height-for-age) ----------
H = "J02.b.Tinggi/Panjang Badan (cm)"; POS = "J02.c.KHUSUS UNTUK BALITA, (Posisi pengukuran TB/PB)"
lms = pd.read_csv(a.lms, sep=r"\s+")
ok = df[H].notna() & df[POS].notna() & df["age_days"].between(0, 1856) & df["sex"].notna()
flow.append(("Tinggi/panjang, posisi ukur, umur valid tersedia", int(ok.sum())))
d = df[ok].copy()
h = d[H].astype(float).copy()
h[(d[POS] == "Berdiri") & (d["age_days"] < 731)] += 0.7     # konversi ke panjang badan
h[(d[POS] == "Telentang") & (d["age_days"] >= 731)] -= 0.7  # konversi ke tinggi badan
ref = lms.set_index(["sex", "age"])
key = list(zip(d["sex"].astype(int), d["age_days"].astype(int)))
L = ref.loc[key, "l"].values; M = ref.loc[key, "m"].values; S = ref.loc[key, "s"].values
d["HAZ"] = (np.power(h.values / M, L) - 1) / (L * S)
d = d[d["HAZ"].abs() <= 6]
flow.append(("Setelah membuang HAZ tidak plausibel (|HAZ|>6)", len(d)))
d["stunting"] = (d["HAZ"] < -2).astype(int)

# ---------- set fitur (bebas kebocoran) ----------
ID_HH = "ID RT"
excl = {}
def X(cols, why):
    for c in cols:
        if c in d.columns: excl[c] = why
X(["ID ART", "ID RT", "ID Ibu", "ID Anak Terakhir", "Primary Sampling Unit", "STRATA", "1. No Urut ART"], "Identifier / desain sampling")
X(["Penimbang Populasi Individu", "Penimbang Populasi RT"], "Bobot sampling (dipakai hanya untuk prevalensi tertimbang)")
X(["6. Tanggal Lahir", "2. Tanggal Pengumpulan data: (tgl-bln)", "Kode umur", "7. Umur hari", "7. Umur Bulan"], "Tanggal/umur mentah (digantikan age_days)")
X(["2. Kabupaten/Kota"], "Kardinalitas >500 level")
X(["4. Jenis Kelamin"], "Digantikan kode sex")
X([c for c in d.columns if c.startswith("J0")], "Antropometri saat ini / proses pengukuran (kebocoran target)")
X([c for c in d.columns if c.startswith("I49.") and any(k in c for k in ["Gizi buruk", "Gizi Kurang", "Kurus", "tidak naik"])], "Alasan PMT karena status gizi (turunan outcome)")
X([c for c in d.columns if c.startswith("G03.") and c[4:5] in "b123456789" and "G03.a" not in c and "G03.c" not in c], "Persepsi ibu tentang arti stunting (dapat dipengaruhi outcome)")
X(["HAZ", "stunting"], "Outcome")
for c in d.columns:
    if c not in excl and d[c].isna().mean() > 0.95: excl[c] = "Missing >95%"
    elif c not in excl and d[c].dtype == object and d[c].nunique() > 60: excl[c] = "Teks bebas / kardinalitas tinggi"
feats = [c for c in d.columns if c not in excl]
# sentinel -> NaN
for c in ["I05.a.\tBerapa berat badan [NAMA] saat dilahirkan", "I07.Berapa panjang badan [NAMA] saat dilahirkan", "I04.Usia kehamilan saat [NAMA] dilahirkan", "I10.Salin dari catatan/dokumen lingkar kepala [NAMA]", "H01.Berapa umur [NAMA] ketika pertama kali hamil?"]:
    if c in d.columns:
        d.loc[d[c] >= 88, c] = np.nan if "berat" not in c else d.loc[d[c] >= 88, c]
bw = "I05.a.\tBerapa berat badan [NAMA] saat dilahirkan"
d.loc[d[bw] >= 8000, bw] = np.nan; d.loc[d[bw] < 500, bw] = np.nan
bl = "I07.Berapa panjang badan [NAMA] saat dilahirkan"; d.loc[d[bl] < 30, bl] = np.nan
for c in [c for c in feats if "berapa kali" in c]: d.loc[d[c] > 7, c] = np.nan      # 88 = tidak tahu
tt = "1.a.Waktu yang diperlukan dari Rumah ke faskes - Puskesmas"; d.loc[d[tt] >= 990, tt] = np.nan
# kode sumber informasi & alat transportasi adalah kode kategorik
for c in ["G03.c.Selama ini darimana [NAMA] mengetahui informasi tentang stunting?", "1.a.Alat Transportasi yang di gunakan - Puskesmas"]:
    d[c] = d[c].map(lambda v: np.nan if pd.isna(v) else str(int(v)))
num = [c for c in feats if pd.api.types.is_numeric_dtype(d[c])]
cat = [c for c in feats if c not in num]

out = d[feats + ["stunting", "HAZ", ID_HH, "Penimbang Populasi Individu"]].rename(columns={ID_HH: "household_id", "Penimbang Populasi Individu": "weight"})
out.to_pickle(f"{a.out}/data/analytic.pkl")
json.dump({"num": num, "cat": cat}, open(f"{a.out}/data/feature_types.json", "w"), indent=1)
pd.DataFrame({"feature": feats, "type": ["numeric" if c in num else "categorical" for c in feats],
              "missing_rate": [round(d[c].isna().mean(), 4) for c in feats]}).to_csv(f"{a.out}/results/feature_list.csv", index=False)
pd.DataFrame(sorted(excl.items(), key=lambda x: x[1]), columns=["variable", "reason"]).to_csv(f"{a.out}/results/exclusion_list.csv", index=False)
wprev = np.average(out["stunting"], weights=out["weight"])
flow += [("Balita stunting (HAZ<-2)", int(out["stunting"].sum())), ("Rumah tangga unik", int(out["household_id"].nunique()))]
pd.DataFrame(flow, columns=["step", "n"]).to_csv(f"{a.out}/results/cohort_flow.csv", index=False)
print(pd.DataFrame(flow)); print("prevalensi tak tertimbang %.4f | tertimbang %.4f" % (out.stunting.mean(), wprev))
print("fitur:", len(feats), "numerik", len(num), "kategorik", len(cat))
json.dump({"n": len(out), "prev_unweighted": out.stunting.mean(), "prev_weighted": wprev, "n_features": len(feats)}, open(f"{a.out}/results/cohort_summary.json", "w"), indent=1)
