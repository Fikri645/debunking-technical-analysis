"""Audit visual pola chart definisi B.

Untuk tiap pola, ambil sampel acak hasil deteksi dari data nyata dan gambar
chart-nya DIPOTONG TEPAT di candle sinyal (kelanjutan harga tidak terlihat),
supaya penilaian bentuk tidak terpengaruh hasilnya.

Jalankan: uv run python scripts/audit_patterns.py [tingkat toleransi 0-2] [timeframe]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from talab import evaluate, patterns  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "audit"
N_SAMPLE = 12


def candles(ax, df):
    x = np.arange(len(df))
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    up = c >= o
    ax.vlines(x, l, h, color="#555", linewidth=0.6)
    ax.bar(x[up], (c - o)[up], bottom=o[up], width=0.7, color="#2a9d8f")
    ax.bar(x[~up], (o - c)[~up], bottom=c[~up], width=0.7, color="#e76f51")


def draw(ax, df, row, label):
    pts = [(int(row[f"pi{q}"]), row[f"pp{q}"]) for q in range(patterns.NPTS) if row[f"pi{q}"] >= 0]
    sig = int(row["idx"])
    first = min(p[0] for p in pts)
    start = max(0, first - max(20, (sig - first) // 2))
    view = df.iloc[start: sig + 1]
    candles(ax, view)
    xs = [p[0] - start for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color="#1d3557", linewidth=1.3, marker="o", markersize=4)
    ax.axhline(row["sl"], color="#d62828", linestyle="--", linewidth=0.9)
    ax.axhline(row["tp"], color="#2b9348", linestyle="--", linewidth=0.9)
    ax.axvline(sig - start, color="#888", linestyle=":", linewidth=0.8)
    arah = "LONG" if row["dir"] == 1 else "SHORT"
    ax.set_title(f"{label} | {arah} | skala {row['scale']:.0f} ATR", fontsize=8)
    ax.set_xticks([])
    ax.tick_params(labelsize=6)


def main() -> None:
    tol = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    tf = sys.argv[2] if len(sys.argv) > 2 else "1d"
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    cells = {m: evaluate.load_cell(m, tf) for m in ("idx", "us", "forex", "crypto")}
    counts = []
    for name in patterns.DETECTORS:
        found = []
        total = 0
        bars = 0
        for market, cell in cells.items():
            for sym, df in cell.items():
                det = patterns.detect(df, name, tol)
                total += len(det)
                bars += len(df)
                for _, row in det.iterrows():
                    found.append((market, sym, row))
        counts.append(dict(pola=name, toleransi=patterns.TOL_NAMES[tol], tf=tf, deteksi=total,
                           per_1000_candle=round(1000 * total / bars, 2),
                           bullish=sum(1 for f in found if f[2]["dir"] == 1)))
        if not found:
            continue
        pick = rng.choice(len(found), size=min(N_SAMPLE, len(found)), replace=False)
        fig, axes = plt.subplots(3, 4, figsize=(16, 9.5))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, i in zip(axes.ravel(), pick):
            market, sym, row = found[i]
            ax.axis("on")
            draw(ax, cells[market][sym], row, sym)
        fig.suptitle(f"{name} — toleransi {patterns.TOL_NAMES[tol]} — {tf} — "
                     f"garis putus merah = SL, hijau = TP, chart berakhir di candle sinyal", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / f"{name}_{patterns.TOL_NAMES[tol]}_{tf}.png", dpi=85)
        plt.close(fig)
    out = pd.DataFrame(counts)
    out.to_csv(OUT / f"_jumlah_{patterns.TOL_NAMES[tol]}_{tf}.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
