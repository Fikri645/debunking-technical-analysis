"""Audit visual strategi tambahan (rounding dan channel), dipotong di candle sinyal.

Jalankan: uv run python scripts/audit_extra.py [timeframe]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_patterns import candles  # noqa: E402
from talab import evaluate, strat_extra  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "figures" / "audit"


def sheet(name, tf, items, draw):
    fig, axes = plt.subplots(3, 4, figsize=(16, 9.5))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, item in zip(axes.ravel(), items):
        ax.axis("on")
        draw(ax, *item)
        ax.set_xticks([])
        ax.tick_params(labelsize=6)
    fig.suptitle(f"{name} — {tf} — merah = SL, hijau = TP, chart berakhir di candle sinyal", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_{tf}.png", dpi=85)
    plt.close(fig)


def draw_round(ax, df, sym, r):
    idx, d, sl, tp, start, rim = r
    a = max(0, int(start) - 20)
    candles(ax, df.iloc[a: int(idx) + 1])
    ax.axvspan(start - a - 0.5, idx - a + 0.5, color="#6a45d9", alpha=0.10)
    ax.axhline(rim, color="#6a45d9", linewidth=1.0)
    ax.axhline(sl, color="#d62828", linestyle="--", linewidth=0.9)
    ax.axhline(tp, color="#2b9348", linestyle="--", linewidth=0.9)
    ax.set_title(f"{sym} | {'LONG' if d == 1 else 'SHORT'} | {int(idx - start + 1)} candle", fontsize=8)


def draw_channel(ax, df, sym, r):
    idx, d, sl, tp, g = r
    idx = int(idx)
    first = int(min(g[0], g[3]))
    a = max(0, first - 15)
    candles(ax, df.iloc[a: idx + 1])
    ax.plot([g[0] - a, idx - a], [g[1], g[2]], color="#6a45d9", linewidth=1.3)
    ax.plot([g[3] - a, idx - a], [g[4], g[5]], color="#6a45d9", linewidth=1.3)
    ax.axhline(sl, color="#d62828", linestyle="--", linewidth=0.9)
    ax.axhline(tp, color="#2b9348", linestyle="--", linewidth=0.9)
    ax.set_title(f"{sym} | {'LONG' if d == 1 else 'SHORT'}", fontsize=8)


def main() -> None:
    tf = sys.argv[1] if len(sys.argv) > 1 else "1d"
    rng = np.random.default_rng(3)
    frames = [(s, df) for m in ("idx", "us", "forex", "crypto") for s, df in evaluate.load_cell(m, tf).items()]
    bars = sum(len(df) for _, df in frames)
    jobs = {"rounding": (lambda df: strat_extra.rounding_raw(df), draw_round),
            "channel_bounce": (lambda df: strat_extra.channel_raw(df, 0), draw_channel),
            "channel_break": (lambda df: strat_extra.channel_raw(df, 1), draw_channel)}
    for name, (fn, draw) in jobs.items():
        found = []
        for sym, df in frames:
            res = fn(df)
            found += [(df, sym, tuple(x[i] for x in res)) for i in range(len(res[0]))]
        print(f"{name}: {len(found)} deteksi ({1000 * len(found) / bars:.2f} per 1000 candle)")
        if found:
            pick = rng.choice(len(found), size=min(12, len(found)), replace=False)
            sheet(name, tf, [found[i] for i in pick], draw)


if __name__ == "__main__":
    main()
