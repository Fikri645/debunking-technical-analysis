"""Apakah indikator tren menambah sesuatu di atas momentum polos?

Membandingkan kontrol "harga lebih tinggi dari N candle lalu" (strat_momentum)
dengan indikator ikut-tren tahap 2, di sel yang sama:
1. keunggulan atas entry acak (mean_rg - null_mean_rg, z, p), semua periode dan 2022-2026;
2. Sharpe per instrumen (berpasangan): indikator vs momentum polos.

Jalankan setelah final_summary.py: uv run python scripts/momentum_vs_indicators.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "final"
TREND = ["sma_20_50", "golden_cross_50_200", "ema_9_21", "price_vs_ema200", "macd_signal", "macd_zero",
         "ichimoku", "supertrend_10_3", "parabolic_sar", "adx_di_25", "donchian_20_10", "bollinger_breakout",
         "obv_ema20", "confluence_ema200_macd_rsi"]
LAST_ERA = "C_2022_2026"


def kind_of(s: str) -> str | None:
    base = s[:-3]
    if s.startswith("momentum_polos_"):
        return "momentum polos"
    if base in TREND:
        return "indikator tren"
    return None


def main() -> None:
    pd.set_option("display.width", 250, "display.max_rows", 300)
    v = pd.read_csv(OUT / "verdicts.csv")
    v["jenis"] = v.strategy.map(kind_of)
    v = v[v.jenis.notna()].copy()
    v["sisi"] = v.strategy.str[-2:]
    v["edge"] = v.mean_rg - v.null_mean_rg

    allt = pd.read_csv(OUT / "all_tests.csv")
    c = allt[allt.era == LAST_ERA].set_index(["strategy", "market", "tf"])
    v = v.join((c.mean_rg - c.null_mean_rg).rename("edge_C"), on=["strategy", "market", "tf"])

    g = v.groupby(["market", "tf", "sisi", "jenis"]).agg(
        strategi=("p", "size"), lolos=("passes", "sum"), kalahkan_acak=("beats_random", "sum"),
        edge_median=("edge", "median"), edge_maks=("edge", "max"),
        edge_C_median=("edge_C", "median"), z_median=("z", "median"),
        net_median=("mean_net", "median")).round(4)
    g.to_csv(OUT / "momentum_vs_indikator.csv")
    print("Keunggulan atas entry acak (edge = mean_rg - null_mean_rg), per sel:")
    print(g.to_string())

    print("\nMomentum polos di crypto, rinci:")
    m = v[(v.jenis == "momentum polos") & (v.market == "crypto")]
    print(m[["strategy", "tf", "n", "win_rate", "mean_net", "edge", "p", "edge_C", "p_C", "passes"]]
          .sort_values(["tf", "strategy"]).round(5).to_string(index=False))

    # Sharpe per instrumen, berpasangan
    frames = []
    for st in ("stage2", "stage6_momentum"):
        for f in sorted((ROOT / "results" / st).glob("*_buyhold.csv")):
            d = pd.read_csv(f)
            d["tf"] = f.name.split("_")[1]
            frames.append(d)
    b = pd.concat(frames, ignore_index=True)
    b["jenis"] = b.strategy.map(kind_of)
    b = b[b.jenis.notna()].copy()
    b["sisi"] = b.strategy.str[-2:]
    rows = []
    for (mk, tf, sisi), d in b.groupby(["market", "tf", "sisi"]):
        med = d.groupby(["symbol", "jenis"]).strat_sharpe.median().unstack()
        best = d.groupby(["symbol", "jenis"]).strat_sharpe.max().unstack()
        bh = d.groupby("symbol").bh_sharpe.first()
        diff = (med["indikator tren"] - med["momentum polos"]).dropna()
        p = stats.wilcoxon(diff).pvalue if len(diff) >= 6 and (diff != 0).any() else np.nan
        rows.append(dict(market=mk, tf=tf, sisi=sisi, instrumen=len(diff),
                         sharpe_indikator=med["indikator tren"].median(),
                         sharpe_momentum=med["momentum polos"].median(),
                         sharpe_buyhold=bh.median(),
                         indikator_lebih_baik=int((diff > 0).sum()),
                         p_wilcoxon=p,
                         terbaik_indikator=best["indikator tren"].median(),
                         terbaik_momentum=best["momentum polos"].median()))
    s = pd.DataFrame(rows).round(3)
    s.to_csv(OUT / "momentum_vs_indikator_sharpe.csv", index=False)
    print("\nSharpe per instrumen (median antar-strategi dalam tiap jenis, lalu median antar-instrumen):")
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
