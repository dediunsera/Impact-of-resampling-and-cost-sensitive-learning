# Eksperimen: Dampak Resampling & Cost-Sensitive Learning terhadap Kalibrasi Probabilitas (Stunting, SKI 2023)

**Tanggal run:** 2026-09-29 · **Seed:** 42 · **Data:** SKI 2023 modul balita (n = 84.205 setelah eksklusi; prevalensi stunting 23,5% tak tertimbang, 22,0% tertimbang) · **Fitur:** 122 determinan hulu (bebas kebocoran antropometri).

## Apa yang diuji
3 model (LR, RF, HGB) × 5 strategi ketidakseimbangan (NONE, CSL, ROS, RUS, SMOTE) × 4 kalibrasi (UNCAL, PRIOR, PLATT, ISO) = 60 kondisi, dievaluasi pada rumah tangga yang tidak pernah dilihat model (5 outer fold berbasis rumah tangga; train 60% / calibration 20% / test 20%). Metrik utama: Brier score, ECE, calibration slope & intercept; pendukung: AUC, PR-AUC, sensitivitas/spesifisitas, decision curve.

## Hasil utama (angka dari `results/`)
1. Semua koreksi ketidakseimbangan yang *efektif* (CSL, ROS, RUS; SMOTE pada LR) merusak kalibrasi: rata-rata prediksi naik dari ≈0,235 menjadi ≈0,44–0,48 (hampir 2× prevalensi nyata), ECE ≈0,20–0,25, Brier naik dari ≈0,168 menjadi ≈0,21–0,23 dan Brier skill score menjadi **negatif** (lebih buruk dari sekadar menebak prevalensi).
2. Contoh konkret (HGB + RUS, fold 0): 44,2% anak diberi probabilitas ≥0,5 (rata-rata prediksi 0,61), tetapi hanya 33,3% di antara mereka yang benar-benar stunting.
3. AUC hampir tidak berubah (0,645–0,673); SMOTE justru menurunkan AUC pada LR/RF.
4. Platt scaling (peringkat 1,03 dari 4; Friedman p < 10⁻³⁰) dan isotonic mengembalikan kalibrasi (ECE < 0,01; Brier kembali ≈ baseline), tetapi tidak ada kombinasi resampling+kalibrasi yang lebih baik secara bermakna daripada model tanpa koreksi (HGB-NONE Brier 0,1671).
5. Koreksi prior analitik bekerja untuk LR dan HGB-RUS/CSL, tetapi gagal (overkoreksi) pada RF dan pada SMOTE berbasis pohon.
6. Diagnostik: HGB dapat membedakan sampel sintetis SMOTE dari minoritas asli dengan AUC 1,000 (LR: 0,554) karena 99,9% baris sintetis memiliki nilai dummy non-0/1 — SMOTE "dinetralkan" oleh model pohon.
7. "Kenaikan deteksi" dari resampling setara dengan menurunkan ambang: model terkalibrasi dengan ambang = prevalensi mencapai sensitivitas ≈0,58–0,64, setara dengan model resampling pada ambang 0,5 (0,55–0,63 untuk CSL/ROS/RUS pada LR dan HGB).

**Artinya:** untuk alat skrining risiko stunting yang menampilkan probabilitas, jangan menilai model dari akurasi/recall saja; resampling tidak menambah informasi, hanya menggeser skala probabilitas. Jika resampling tetap dipakai, kalibrasi ulang (Platt/isotonic) pada data dengan prevalensi asli wajib dilakukan, dan ambang keputusan dipilih secara eksplisit.

## Cara reproduksi
```
python 01_prepare_data.py --xlsx "SKI 2023 Balita 0-59 Bulan_asli labels.xlsx" --lms lenanthro.txt
python 02_train_predict.py --folds 0 1 2 3 4
python 03_evaluate.py
python 00_flowchart.py && python 04_figures.py && python 05_smote_diagnostic.py
```
Atau buka `colab_crosscheck.ipynb` di Google Colab (sel terakhir membandingkan angka dengan `results/reference_comparison_run1.csv`).

## Isi folder
- `results/metrics.csv`, `results/comparison.csv` – ringkasan 60 kondisi (rerata 5 fold)
- `results/metrics_all_folds.csv`, `metrics_cv_summary.csv`, `fold0_bootstrap_ci.csv`, `friedman_nemenyi.csv`, `high_risk_observed.csv`, `decision_curve_fold0.csv`, `smote_diagnostic.csv`, `table_*.csv`
- `figures/fig01…fig09*.png` (300 dpi)
- `data/` – dataset analitik turunan & prediksi per fold (jangan dibagikan publik; turunan data SKI berlisensi)

## Robustness (Google Colab, 2026-09-30; fold 0)
Versi: Python 3.13.15, scikit-learn 1.6.1, imbalanced-learn 0.14.2, XGBoost 3.4.1, LightGBM 4.7.0.
- **Implementasi resampling vs imbalanced-learn** (split identik, LR & HGB, 18 pasangan): selisih maks |ΔAUC| 0,003, |ΔBrier| 0,001, |ΔECE| 0,004, |Δmean p̂| 0,002 → setara.
- **XGBoost & LightGBM**: pola sama — CSL/ROS/RUS: mean p̂ 0,43–0,47, ECE 0,19–0,23, BSS negatif; SMOTE hampir terkalibrasi (ECE 0,013); PRIOR overkoreksi SMOTE (ECE 0,143); Platt → ECE 0,008–0,011; tidak ada model terkoreksi+terkalibrasi yang lebih baik >0,0001 Brier dari NONE+Platt.
- **Deteksi SMOTE imblearn**: 99,96% baris sintetis non-biner; AUC deteksi HGB 1,000, LR 0,557.
- Catatan: karena versi library Colab berbeda, pembagian rumah tangga (split) sedikit berbeda dari analisis utama (prevalensi test 0,239). Untuk reproduksi angka utama persis, pasang `scikit-learn==1.8.0` (dan sebaiknya numpy 2.4.4, pandas 3.0.2) di runtime baru.
- File: `results/robustness_*.csv`, `results/robustness_versions.json`, `figures/fig10_robustness_xgb_lgbm.png`.
