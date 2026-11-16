"""Grafik untuk thread (post/gambar/). Semua angka dibaca dari results/final/.

Jalankan setelah final_summary.py dan momentum_vs_indicators.py:
    uv run python scripts/make_figures.py
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from talab import evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final"
OUT = ROOT / "post" / "gambar"

INK, MUTED, GRID, BG = "#16202a", "#66727f", "#dfe4ea", "#ffffff"
RED, GREEN, BLUE, AMBER, GREY = "#c8442f", "#1f8a5b", "#2f6fb0", "#d99a1c", "#aab3bd"
SOURCE = "Data: Yahoo Finance & Binance, 2010–2026. Kode dan semua hasil terbuka."

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15, "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": GRID, "xtick.color": MUTED, "ytick.color": INK, "figure.facecolor": BG,
    "axes.facecolor": BG, "axes.spines.top": False, "axes.spines.right": False,
})

GROUPS = [
    ("indikator", "Indikator (MA, RSI, MACD, dll.)"), ("candlestick", "Pola candlestick"),
    ("pola_chart_B", "Pola chart (H&S, triangle, dll.)"), ("pola_chart_A", "Pola chart versi riset 2000"),
    ("support_resistance", "Support / resistance"), ("trendline", "Trendline"), ("fibonacci", "Fibonacci"),
    ("supply_demand", "Supply & demand"), ("smc_ict", "SMC / ICT"), ("elliott", "Elliott Wave"),
    ("influencer", "Strategi lengkap ala kursus"),
]
MARKETS = [("idx", "Saham\nIndonesia"), ("us", "Saham AS"), ("forex", "Forex\n& emas"), ("crypto", "Crypto")]


def finish(fig, title, subtitle, name):
    fig.text(0.04, 0.955, title, fontsize=25, fontweight="bold", va="top")
    fig.text(0.04, 0.885, textwrap.fill(subtitle, 125), fontsize=15, color=MUTED, va="top", linespacing=1.45)
    fig.text(0.04, 0.025, SOURCE, fontsize=10.5, color=MUTED)
    fig.savefig(OUT / name, dpi=100)
    plt.close(fig)
    print("tulis", name)


def fig_matrix(v):
    ta = v[~v.control]
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0.30, 0.08, 0.66, 0.66])
    ax.set_xlim(0, len(MARKETS))
    ax.set_ylim(len(GROUPS), 0)
    for r, (g, _) in enumerate(GROUPS):
        for c, (m, _) in enumerate(MARKETS):
            d = ta[(ta.group == g) & (ta.market == m)]
            k, n = int(d.passes.sum()), len(d)
            color = "#f6e3df" if k == 0 else "#cfe9dc"
            ax.add_patch(plt.Rectangle((c + 0.03, r + 0.06), 0.94, 0.88, color=color, lw=0))
            ax.text(c + 0.5, r + 0.5, f"{k} dari {n}", ha="center", va="center", fontsize=17,
                    fontweight="bold" if k else "normal", color=GREEN if k else RED)
    ax.set_xticks(np.arange(len(MARKETS)) + 0.5, [m[1] for m in MARKETS], fontsize=15)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(GROUPS)) + 0.5, [g[1] for g in GROUPS], fontsize=15)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    tot = ta.groupby(ta.market == "crypto").passes.agg(["sum", "size"])
    finish(fig, "Berapa strategi technical analysis yang benar-benar terbukti?",
           f"Lolos = mengalahkan entry acak, untung setelah biaya, dan bekerja di mayoritas instrumen; di seluruh periode dan di 2022–2026.  "
           f"Di luar crypto: {int(tot.loc[False, 'sum'])} dari {int(tot.loc[False, 'size']):,}.".replace(",", "."),
           "01_tabel_lolos.png")


def fig_profit_vs_proof(v):
    rows = [("zodiak", "Zodiak sungguhan\n(fase bulan, Merkurius\nretrograde, elemen)")] + [
        ("candlestick", "Pola candlestick"), ("fibonacci", "Fibonacci"), ("smc_ict", "SMC / ICT"),
        ("supply_demand", "Supply & demand"), ("support_resistance", "Support / resistance")]
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0.27, 0.13, 0.68, 0.62])
    y = np.arange(len(rows))
    untung = [100 * v[v.group == g].profitable.mean() for g, _ in rows]
    bukti = [100 * v[v.group == g].beats_random.mean() for g, _ in rows]
    ax.barh(y - 0.19, untung, 0.36, color=GREY, label='"Untung" di backtest')
    ax.barh(y + 0.19, bukti, 0.36, color=BLUE, label="Mengalahkan entry acak")
    for i, (u, b) in enumerate(zip(untung, bukti)):
        ax.text(u + 0.8, i - 0.19, f"{u:.0f}%", va="center", fontsize=14, color=MUTED)
        ax.text(b + 0.8, i + 0.19, f"{b:.0f}%", va="center", fontsize=14, color=BLUE, fontweight="bold")
    ax.set_yticks(y, [r[1] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, max(untung) * 1.15)
    ax.set_xlabel("% kombinasi strategi × pasar × timeframe")
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=14)
    finish(fig, '"Backtest saya untung" tidak membuktikan apa-apa',
           "Ramalan zodiak pun sering \"untung\", karena pasar memang cenderung naik. "
           "Yang penting: apakah lebih baik dari entry acak?", "02_untung_vs_terbukti.png")


def fig_fib(v):
    d = v[v.group.isin(["fibonacci", "fibonacci_palsu"])].copy()
    d["level"] = d.strategy.str.extract(r"_(\d{3})")[0].astype(int) / 10
    g = d.groupby(["level", "group"]).agg(z=("z", "mean"), n=("z", "size")).reset_index().sort_values("level")
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0.08, 0.21, 0.87, 0.55])
    x = np.arange(len(g))
    colors = [AMBER if grp == "fibonacci" else GREY for grp in g.group]
    ax.bar(x, g.z, 0.62, color=colors)
    ax.axhline(0, color=INK, lw=1)
    ax.axhline(1.96, color=RED, lw=1, ls="--")
    ax.text(len(g) - 0.5, 2.0, "batas \"lebih baik dari acak\"", color=RED, fontsize=12, ha="right", va="bottom")
    ax.set_xticks(x, [f"{lv:.1f}%" for lv in g.level], fontsize=15)
    for i, (grp, lv) in enumerate(zip(g.group, g.level)):
        ax.text(i, -0.10, "Fibonacci" if grp == "fibonacci" else "angka\nasal", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=12, color=AMBER if grp == "fibonacci" else MUTED,
                fontweight="bold" if grp == "fibonacci" else "normal")
    ax.set_ylim(min(-1.2, g.z.min() - 0.3), 2.6)
    ax.set_ylabel("Keunggulan atas entry acak (skor z, rata-rata 10 pasar/timeframe)", fontsize=12.5)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    finish(fig, "Level Fibonacci tidak lebih sakti dari angka asal",
           "Beli di retracement 38,2 / 50 / 61,8 / 78,6% dibandingkan dengan level karangan 30 / 44 / 56 / 70 / 85%. "
           "Tidak ada yang mengalahkan entry acak.", "03_fibonacci_vs_angka_asal.png")


def fig_era(allt):
    # Era 2017 dibuang: data Binance baru mulai Agustus 2017, transaksinya terlalu sedikit
    era = allt[allt.era.isin(["B_2018_2021", "C_2022_2026"]) & (allt.market == "crypto") & (allt.tf == "4h") & (allt.group == "indikator")
               & (allt.kind == "state")]
    trend = ["sma_20_50", "golden_cross_50_200", "ema_9_21", "price_vs_ema200", "macd_signal", "macd_zero",
             "ichimoku", "supertrend_10_3", "parabolic_sar", "adx_di_25", "donchian_20_10", "bollinger_breakout"]
    era = era[era.strategy.str[:-3].isin(trend)]
    g = era.groupby("era").apply(lambda d: (d.mean_rg - d.null_mean_rg).mean(), include_groups=False)
    labels = {"A_2010_2017": "2017", "B_2018_2021": "2018–2021", "C_2022_2026": "2022–2026"}
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0.09, 0.12, 0.86, 0.64])
    x = np.arange(len(g))
    ax.bar(x, g.values, 0.45, color=[BLUE, RED])
    for i, val in enumerate(g.values):
        ax.text(i, val + 0.04, f"{val:.2f}".replace(".", ","), ha="center", fontsize=22, fontweight="bold")
    ax.set_xticks(x, [labels[k] for k in g.index], fontsize=18)
    ax.set_ylabel("Keunggulan atas entry acak per transaksi\n(dalam satuan volatilitas)", fontsize=13)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    drop = 100 * (1 - g.values[-1] / g.values[0])
    finish(fig, f"Satu-satunya yang bekerja pun sudah menyusut {drop:.0f}%",
           "Strategi ikut-tren di crypto 4 jam (12 indikator tren, 12 koin): keunggulan atas entry acak, sebelum biaya.",
           "04_keunggulan_menyusut.png")
    return g


def fig_momentum():
    s = pd.read_csv(FINAL / "momentum_vs_indikator_sharpe.csv")
    s = s[(s.market == "crypto") & (s.sisi == "lo") & s.tf.isin(["1d", "4h"])].set_index("tf").loc[["1d", "4h"]]
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0.09, 0.12, 0.86, 0.58])
    x = np.arange(2)
    series = [("sharpe_buyhold", "Beli lalu diamkan", GREY), ("sharpe_indikator", "14 indikator tren (EMA, golden cross, Ichimoku, …)", AMBER),
              ("sharpe_momentum", "Aturan satu kalimat, tanpa indikator", BLUE)]
    for k, (col, lab, color) in enumerate(series):
        ax.bar(x + (k - 1) * 0.26, s[col].values, 0.24, color=color, label=lab)
        for i, val in enumerate(s[col].values):
            ax.text(i + (k - 1) * 0.26, val + 0.012, f"{val:.2f}".replace(".", ","), ha="center", fontsize=17,
                    fontweight="bold")
    ax.set_xticks(x, ["Crypto harian", "Crypto 4 jam"], fontsize=18)
    ax.set_ylabel("Sharpe ratio setelah biaya (median 12 koin)", fontsize=13)
    ax.set_ylim(0, 0.98)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=13, bbox_to_anchor=(0.5, 1.12))
    finish(fig, "Yang bekerja momentumnya, bukan indikatornya",
           'Aturan "beli kalau harga lebih tinggi dari N candle lalu, selain itu keluar" hasilnya setara dengan indikator tren.',
           "05_momentum_vs_indikator.png")


def fig_buyhold(v):
    t = v[~v.control & v.instrumen.notna()]
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0.2, 0.14, 0.5, 0.56])
    rows = [(lab.replace("\n", " "), t[t.market == m]) for m, lab in MARKETS]
    y = np.arange(len(rows))
    win = [100 * d.beats_bh.mean() for _, d in rows]
    real = [100 * (d.beats_bh & (d.cagr > 0)).mean() for _, d in rows]
    ax.barh(y, [100] * len(rows), 0.55, color="#eef1f6")
    ax.barh(y, win, 0.55, color=GREY, label="Menang, tetapi tetap rugi (hanya rugi lebih sedikit)")
    ax.barh(y, real, 0.55, color=BLUE, label="Menang dan untung")
    for i, (lab, d) in enumerate(rows):
        n_win, n_real = int(d.beats_bh.sum()), int((d.beats_bh & (d.cagr > 0)).sum())
        ax.text(102, i - 0.1, f"{n_win} dari {len(d):,} menang".replace(",", "."), va="center", fontsize=17,
                fontweight="bold")
        ax.text(102, i + 0.17, f"{n_real} di antaranya untung", va="center", fontsize=13, color=MUTED)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=17)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Kombinasi strategi yang hasil akhirnya mengalahkan beli lalu diamkan")
    ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=13, bbox_to_anchor=(0.5, 1.13))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    total = 100 * (1 - t.beats_bh.mean())
    finish(fig, f"Beli lalu diamkan mengalahkan {total:.0f}% strategi technical analysis",
           "Modal yang menjalankan strategi terus-menerus dibanding modal yang membeli instrumen yang sama di awal data lalu "
           "mendiamkannya, setelah biaya. Menang = lebih tinggi di lebih dari separuh instrumen.", "06_beli_lalu_diamkan.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v = pd.read_csv(FINAL / "verdicts.csv")
    allt = pd.read_csv(FINAL / "all_tests.csv")
    fig_matrix(v)
    fig_profit_vs_proof(v)
    fig_fib(v)
    print(fig_era(allt).round(3).to_string())
    fig_momentum()
    fig_buyhold(v)


if __name__ == "__main__":
    main()
