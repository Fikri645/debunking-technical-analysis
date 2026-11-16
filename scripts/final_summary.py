"""Gabungan semua tahap dengan satu koreksi uji berganda global.

Kontrol (tembus swing tanpa pola, Fibonacci palsu, zodiak) dilaporkan terpisah
dan tidak ikut dihitung sebagai strategi TA.

Jalankan: uv run python scripts/final_summary.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from talab import evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "final"
STAGES = ["stage2", "stage3", "stage3_kontrol", "stage4", "stage5", "stage6_momentum"]
CONTROL_GROUPS = {"kontrol", "fibonacci_palsu", "zodiak", "kontrol_momentum"}
LAST_ERA = "C_2022_2026"
GROUP_LABEL = {
    "indikator": "Indikator", "candlestick": "Pola candlestick",
    "pola_chart_B": "Pola chart (definisi kita)", "pola_chart_A": "Pola chart (Lo dkk. 2000)",
    "support_resistance": "Support/resistance", "trendline": "Trendline", "fibonacci": "Fibonacci",
    "supply_demand": "Supply & demand", "smc_ict": "SMC / ICT", "elliott": "Elliott Wave",
    "influencer": "Strategi lengkap ala influencer",
    "kontrol": "Kontrol: tembus swing tanpa pola", "fibonacci_palsu": "Kontrol: Fibonacci palsu",
    "zodiak": "Kontrol: zodiak sungguhan",
    "kontrol_momentum": "Kontrol: momentum polos tanpa indikator",
}
MARKET_LABEL = {"idx": "Saham Indonesia", "us": "Saham AS", "forex": "Forex & emas", "crypto": "Crypto"}


def load() -> pd.DataFrame:
    frames = []
    for st in STAGES:
        for f in sorted((ROOT / "results" / st).glob("*.csv")):
            if "buyhold" in f.name or f.name.startswith("_"):
                continue
            d = pd.read_csv(f)
            d["stage"] = st
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["group"] = df["group"].fillna("kontrol")
    df["control"] = df["group"].isin(CONTROL_GROUPS)
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    ta = ~df["control"]
    df["fdr_pass"] = False
    # koreksi global untuk strategi TA; kontrol dikoreksi sendiri
    df.loc[ta, "fdr_pass"] = evaluate.benjamini_hochberg(df.loc[ta, "p"].values, 0.05)
    df.loc[~ta, "fdr_pass"] = evaluate.benjamini_hochberg(df.loc[~ta, "p"].values, 0.05)
    df.to_csv(OUT / "all_tests.csv", index=False)

    key = ["strategy", "market", "tf"]
    a = df[df.era == "ALL"].set_index(key)
    c = df[df.era == LAST_ERA].set_index(key)
    v = a[["group", "control", "kind", "n", "win_rate", "mean_gross", "mean_net", "null_mean_net",
           "mean_rg", "null_mean_rg", "p", "z", "fdr_pass"]].copy()
    v = v.join(c[["n", "mean_net", "p", "fdr_pass"]].add_suffix("_C"))
    v["beats_random"] = v["fdr_pass"] & v["fdr_pass_C"].fillna(False).astype(bool)
    v["profitable"] = (v["mean_net"] > 0) & (v["mean_net_C"] > 0)
    v["passes"] = v["beats_random"] & v["profitable"]
    v = v.reset_index()
    v.to_csv(OUT / "verdicts.csv", index=False)

    t = df[df.p.notna()]
    print(f"Total uji statistik: {len(t)} | strategi TA: {(~t.control).sum()} | kontrol: {t.control.sum()}")
    n_strat = v[~v.control].strategy.nunique()
    print(f"Strategi TA unik: {n_strat} | kombinasi strategi x pasar x timeframe: {(~v.control).sum()}")

    tab = v.groupby(["group", "market"]).agg(kombinasi=("p", "size"), untung=("profitable", "sum"),
                                             kalahkan_acak=("beats_random", "sum"), lolos=("passes", "sum"))
    wide = tab.unstack("market").fillna(0).astype(int)
    wide.to_csv(OUT / "tabel_kelompok_pasar.csv")
    rows = []
    for g in GROUP_LABEL:
        if g not in tab.index.get_level_values(0):
            continue
        r = {"Kelompok": GROUP_LABEL[g]}
        for m in MARKET_LABEL:
            if (g, m) in tab.index:
                x = tab.loc[(g, m)]
                r[MARKET_LABEL[m]] = f"{x.lolos} / {x.kombinasi}"
            else:
                r[MARKET_LABEL[m]] = "-"
        rows.append(r)
    ringkas = pd.DataFrame(rows)
    ringkas.to_csv(OUT / "ringkasan_lolos.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 200)
    print("\nLolos semua kriteria / jumlah kombinasi:")
    print(ringkas.to_string(index=False))

    tot = v[~v.control].groupby("market").agg(kombinasi=("p", "size"), untung=("profitable", "sum"),
                                              kalahkan_acak=("beats_random", "sum"), lolos=("passes", "sum"))
    print("\nTotal strategi TA per pasar:")
    print(tot.to_string())
    print("Total:", tot.sum().to_dict())

    print("\nSebaran nilai p per kelompok (semua era):")
    dist = t.groupby("group").p.agg(uji="size", p_di_bawah_005=lambda x: (x < .05).mean(),
                                    p_di_atas_095=lambda x: (x > .95).mean(), median="median").round(3)
    dist.to_csv(OUT / "sebaran_p.csv")
    print(dist.to_string())
    nc = t[(t.market != "crypto") & ~t.control]
    print(f"\nDi luar crypto (strategi TA): {len(nc)} uji, {(nc.p < .05).mean():.1%} di bawah 0,05, "
          f"{nc.fdr_pass.sum()} lolos FDR global")

    passed = v[v.passes].sort_values(["market", "tf", "group", "strategy"])
    passed.to_csv(OUT / "lolos.csv", index=False)
    print(f"\nLOLOS ({len(passed)}):")
    print(passed[["group", "strategy", "market", "tf", "n", "win_rate", "mean_net", "null_mean_net", "p",
                  "mean_net_C", "p_C"]].round(5).to_string(index=False))

    # Tren per era untuk strategi yang lolos
    ids = set(map(tuple, passed[key].values))
    era = df[df.era != "ALL"].copy()
    era["lolos"] = [tuple(x) in ids for x in era[key].values]
    e = era[era.lolos].groupby("era").agg(sel=("p", "size"), mean_rg=("mean_rg", "mean"),
                                          null_rg=("null_mean_rg", "mean"), mean_net=("mean_net", "mean"),
                                          win=("win_rate", "mean"), median_p=("p", "median")).round(4)
    e.to_csv(OUT / "era_strategi_lolos.csv")
    print("\nStrategi yang lolos, per era (catatan: lolos dipilih dari hasil yang sama, jadi era lama bias ke atas):")
    print(e.to_string())
    trend_all = era[(era.market == "crypto") & era.tf.isin(["1d", "4h"]) & (era.group == "indikator")
                    & (era.kind == "state")]
    e2 = trend_all.groupby(["tf", "era"]).agg(sel=("p", "size"), mean_rg=("mean_rg", "mean"),
                                              null_rg=("null_mean_rg", "mean"), median_p=("p", "median")).round(4)
    print("\nSemua strategi indikator posisi di crypto (tanpa seleksi), per era:")
    print(e2.to_string())


if __name__ == "__main__":
    main()
