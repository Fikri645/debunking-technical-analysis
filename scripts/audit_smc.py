"""Audit visual tahap 4: supply-demand, FVG, order block, model ICT 2022, Elliott Wave.

Chart dipotong di candle entry (zona) atau candle sinyal (Elliott), sehingga
kelanjutan harga tidak terlihat.

Jalankan: uv run python scripts/audit_smc.py [timeframe]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_patterns import candles  # noqa: E402
from talab import evaluate, strat_smc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "audit"
N_SAMPLE = 12

ZONES = {
    "supply_demand_semua": lambda df: strat_smc.sd_raw(df, 1.5, 6, 2.0, 0),
    "supply_demand_kuat": lambda df: strat_smc.sd_raw(df, 2.5, 3, 2.0, 0),
    "fair_value_gap": lambda df: strat_smc.fvg_raw(df, False, 2.0),
    "order_block": lambda df: strat_smc.ob_raw(df, 2.0),
    "ict_2022": lambda df: strat_smc.ict_raw(df, 3.0),
}
WAVES = {"elliott_wave3_fib": (0, True), "elliott_wave5_aturan": (1, False), "elliott_abc_aturan": (2, False)}


def sheet(name, tf, items, draw):
    fig, axes = plt.subplots(3, 4, figsize=(16, 9.5))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, item in zip(axes.ravel(), items):
        ax.axis("on")
        draw(ax, *item)
        ax.set_xticks([])
        ax.tick_params(labelsize=6)
    fig.suptitle(f"{name} — {tf} — merah = SL, hijau = TP, chart berakhir di candle entry/sinyal", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_{tf}.png", dpi=85)
    plt.close(fig)


def draw_zone(ax, df, sym, row):
    e, px, d, sl, tp, z0, zlo, zhi = row
    e, z0 = int(e), int(z0)
    start = max(0, z0 - 25)
    candles(ax, df.iloc[start: e + 1])
    ax.add_patch(Rectangle((z0 - start - 0.5, zlo), e - z0 + 1, zhi - zlo, color="#457b9d", alpha=0.25))
    ax.axhline(sl, color="#d62828", linestyle="--", linewidth=0.9)
    ax.axhline(tp, color="#2b9348", linestyle="--", linewidth=0.9)
    ax.set_title(f"{sym} | {'LONG' if d == 1 else 'SHORT'}", fontsize=8)


def draw_wave(ax, df, sym, row):
    pts = [(int(row[f"pi{q}"]), row[f"pp{q}"]) for q in range(6) if row[f"pi{q}"] >= 0]
    sig = int(row["idx"])
    start = max(0, pts[0][0] - 25)
    candles(ax, df.iloc[start: sig + 1])
    ax.plot([p[0] - start for p in pts], [p[1] for p in pts], color="#1d3557", linewidth=1.3, marker="o", markersize=4)
    for q, p in enumerate(pts):
        ax.annotate(str(q), (p[0] - start, p[1]), fontsize=8, color="#1d3557",
                    textcoords="offset points", xytext=(3, 3))
    ax.axhline(row["sl"], color="#d62828", linestyle="--", linewidth=0.9)
    ax.axhline(row["tp"], color="#2b9348", linestyle="--", linewidth=0.9)
    ax.set_title(f"{sym} | {'LONG' if row['dir'] == 1 else 'SHORT'} | skala {row['scale']:.0f} ATR", fontsize=8)


def main() -> None:
    tf = sys.argv[1] if len(sys.argv) > 1 else "1d"
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    cells = {m: evaluate.load_cell(m, tf) for m in ("idx", "us", "forex", "crypto")}
    frames = [(sym, df) for cell in cells.values() for sym, df in cell.items()]
    bars = sum(len(df) for _, df in frames)
    for name, fn in ZONES.items():
        found = []
        for sym, df in frames:
            res = fn(df)
            found += [(df, sym, tuple(x[i] for x in res)) for i in range(len(res[0]))]
        print(f"{name}: {len(found)} deteksi ({1000 * len(found) / bars:.2f} per 1000 candle)")
        if found:
            pick = rng.choice(len(found), size=min(N_SAMPLE, len(found)), replace=False)
            sheet(name, tf, [found[i] for i in pick], draw_zone)
    for name, (setup, fib) in WAVES.items():
        found = []
        for sym, df in frames:
            det = strat_smc.ew_detect(df, setup, fib)
            found += [(df, sym, row) for _, row in det.iterrows()]
        print(f"{name}: {len(found)} deteksi ({1000 * len(found) / bars:.2f} per 1000 candle)")
        if found:
            pick = rng.choice(len(found), size=min(N_SAMPLE, len(found)), replace=False)
            sheet(name, tf, [found[i] for i in pick], draw_wave)


if __name__ == "__main__":
    main()
