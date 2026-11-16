"""Gabungkan hasil satu tahap, terapkan koreksi uji berganda, dan cetak ringkasan.

Jalankan: uv run python scripts/summarize.py stage2
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from talab import evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LAST_ERA = "C_2022_2026"


def load(stage: str) -> pd.DataFrame:
    files = [f for f in (ROOT / "results" / stage).glob("*.csv") if "buyhold" not in f.name
             and not f.name.startswith("_")]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["tested"] = df["p"].notna()
    df["fdr_pass"] = evaluate.benjamini_hochberg(df["p"].values, 0.05)
    return df


def verdicts(df: pd.DataFrame) -> pd.DataFrame:
    """Kriteria lolos, per strategi x pasar x timeframe:
    signifikan pada FDR 5% di seluruh periode DAN di era terakhir, dan untung bersih."""
    key = ["strategy", "market", "tf"]
    a = df[df.era == "ALL"].set_index(key)
    c = df[df.era == LAST_ERA].set_index(key)
    out = a[["group", "kind", "n", "win_rate", "mean_net", "null_mean_net", "p", "z", "fdr_pass"]].copy()
    out = out.join(c[["n", "mean_net", "p", "fdr_pass"]].add_suffix("_C"))
    out["beats_random"] = out["fdr_pass"] & out["fdr_pass_C"].fillna(False).astype(bool)
    out["profitable"] = (out["mean_net"] > 0) & (out["mean_net_C"] > 0)
    out["passes"] = out["beats_random"] & out["profitable"]
    return out.reset_index()


def main() -> None:
    stage = sys.argv[1]
    df = load(stage)
    df.to_csv(ROOT / "results" / stage / "_all.csv", index=False)
    v = verdicts(df)
    v.to_csv(ROOT / "results" / stage / "_verdicts.csv", index=False)
    t = df[df.tested]
    print(f"Uji statistik: {len(t)} (dari {len(df)} baris; sisanya < {evaluate.MIN_TRADES} transaksi)")
    print(f"p < 0,05 tanpa koreksi: {(t.p < 0.05).sum()} = {(t.p < 0.05).mean():.1%} (harapan kebetulan 5%)")
    print(f"p > 0,95 (lebih buruk dari acak): {(t.p > 0.95).sum()} = {(t.p > 0.95).mean():.1%}")
    print(f"Lolos FDR 5%: {t.fdr_pass.sum()}")
    print(f"Median p: {t.p.median():.3f}")
    print(f"\nKombinasi strategi x pasar x timeframe: {len(v)}")
    print(f"  untung bersih (rata-rata per transaksi > 0, seluruh periode): {(v.mean_net > 0).sum()}")
    print(f"  mengalahkan acak (FDR, seluruh periode + era terakhir): {v.beats_random.sum()}")
    print(f"  LOLOS semua kriteria: {v.passes.sum()}")
    pd.set_option("display.width", 250, "display.max_rows", 200, "display.max_columns", 30)
    cols = ["strategy", "market", "tf", "n", "win_rate", "mean_net", "null_mean_net", "p", "n_C", "mean_net_C", "p_C", "passes"]
    fmt = {"win_rate": "{:.2f}".format, "mean_net": "{:.4f}".format, "null_mean_net": "{:.4f}".format,
           "mean_net_C": "{:.4f}".format, "p": "{:.5f}".format, "p_C": "{:.5f}".format}
    print("\nMengalahkan acak (FDR) di seluruh periode:")
    print(v[v.fdr_pass].sort_values("p")[cols].to_string(index=False, formatters=fmt))
    by = v.groupby(["market", "tf"]).agg(n=("passes", "size"), profitable=("profitable", "sum"),
                                         beats_random=("beats_random", "sum"), passes=("passes", "sum"))
    print("\nPer pasar:")
    print(by.to_string())
    strat = v.groupby("strategy").agg(cells=("passes", "size"), fdr_all=("fdr_pass", "sum"),
                                      passes=("passes", "sum"), median_p=("p", "median"))
    print("\nStrategi yang lolos di lebih dari satu pasar/timeframe:")
    print(strat[strat.passes > 1].to_string())


if __name__ == "__main__":
    main()
