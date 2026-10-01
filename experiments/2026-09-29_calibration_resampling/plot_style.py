"""
Gaya visualisasi konsisten untuk semua figure eksperimen dalam proyek ini.
Panggil apply_style() sekali di awal skrip, sebelum membuat plot apa pun.

Kenapa ini penting: tanpa gaya konsisten, tiap eksperimen menghasilkan
figure dengan font/warna/ukuran berbeda-beda -- menyulitkan pembaca
membandingkan lintas eksperimen dan menyulitkan proses "tempel ke naskah
jurnal" karena harus dirapikan ulang satu-satu. Palet warna di bawah
dipilih agar tetap bisa dibedakan oleh pembaca buta warna (deuteranopia/
protanopia-safe, diadaptasi dari palet kualitatif Okabe-Ito).
"""

import matplotlib.pyplot as plt

# Palet kualitatif aman-buta-warna (Okabe & Ito, 2008) -- urutan dipakai
# berulang untuk kategori ke-1, ke-2, dst.
COLORBLIND_SAFE_PALETTE = [
    "#0072B2",  # biru
    "#D55E00",  # oranye kemerahan
    "#009E73",  # hijau
    "#CC79A7",  # magenta
    "#E69F00",  # kuning-oranye
    "#56B4E9",  # biru muda
    "#F0E442",  # kuning
    "#000000",  # hitam (aksen/baseline)
]


def apply_style(base_font_size: int = 11) -> None:
    """Terapkan rcParams matplotlib yang konsisten untuk seluruh proyek."""
    plt.rcParams.update({
        "figure.dpi": 150,          # tampilan di layar
        "savefig.dpi": 300,          # cukup tajam untuk cetak/jurnal
        "font.size": base_font_size,
        "axes.titlesize": base_font_size + 2,
        "axes.labelsize": base_font_size,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "figure.autolayout": True,
        "axes.prop_cycle": plt.cycler(color=COLORBLIND_SAFE_PALETTE),
        "savefig.bbox": "tight",
    })


def annotate_source(fig, text: str) -> None:
    """Tambahkan caption kecil (sumber data, catatan cara baca grafik, dll.)
    di pojok kiri bawah figure -- praktik baik untuk figure yang akan dipakai
    di bab hasil/naskah jurnal, supaya pembaca (dan diri sendiri 6 bulan lagi)
    tahu data ini dari mana atau bagaimana cara membacanya. Gunakan fungsi ini
    (bukan menulis fig.text(...) manual) supaya posisi & gaya caption konsisten
    di semua figure proyek ini -- lihat pemakaiannya di comparison_table.py."""
    fallback_size = max(plt.rcParams.get("font.size", 11) - 3, 7)
    fig.text(0.01, -0.03, text, fontsize=fallback_size, color="#555555", ha="left", va="top")
