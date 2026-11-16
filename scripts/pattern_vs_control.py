"""Apakah bentuk pola menambah sesuatu di atas "tembus swing terakhir"?

Untuk tiap pola chart (keluar setelah 10 candle), return per transaksi dalam satuan
volatilitas dibandingkan dengan kontrol tanpa-pola (kontrol_swing_break_h10) di sel
yang sama. Uji t Welch dua sisi atas selisih rata-rata.

Catatan: transaksi dalam satu sel tidak sepenuhnya saling bebas (koin bergerak
bersama), jadi nilai p di sini cenderung terlalu kecil; yang ditafsirkan adalah
arah dan besar selisihnya.

Jalankan: uv run python scripts/pattern_vs_control.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from talab import engine, evaluate, patterns, strat_extra, strat_patterns  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "final"
CELLS = [("crypto", "4h"), ("crypto", "1d"), ("crypto", "1h"), ("us", "1d"), ("idx", "1d"), ("forex", "1d")]


def rg_of(fn, market, cell) -> np.ndarray:
    out = []
    for sym, df in cell.items():
        tr = engine.run(df, fn(df), evaluate.cost_for(market, sym))
        out.append(tr["rg"].values)
    return np.concatenate(out) if out else np.array([])


def welch_p(a: np.ndarray, b: np.ndarray) -> float:
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def main() -> None:
    reg = {**strat_patterns.build(), **strat_extra.build()}
    control = strat_patterns.build_controls()["kontrol_swing_break_h10"]["fn"]
    names = [f"{p}_{t}_h10" for p in patterns.DETECTORS for t in patterns.TOL_NAMES] + ["rounding_h10"]
    rows = []
    for market, tf in CELLS:
        cell = evaluate.load_cell(market, tf)
        ctrl = rg_of(control, market, cell)
        for name in names:
            x = rg_of(reg[name]["fn"], market, cell)
            if len(x) < 30:
                continue
            rows.append(dict(strategy=name, market=market, tf=tf, n=len(x), mean_rg=x.mean(),
                             n_kontrol=len(ctrl), kontrol_rg=ctrl.mean(), selisih=x.mean() - ctrl.mean(),
                             p=welch_p(x, ctrl)))
        print(f"selesai {market} {tf}", flush=True)
    out = pd.DataFrame(rows)
    out["lebih_baik"] = (out.selisih > 0) & (out.p < 0.05)
    out["lebih_buruk"] = (out.selisih < 0) & (out.p < 0.05)
    out.round(4).to_csv(OUT / "pola_vs_kontrol.csv", index=False)
    pd.set_option("display.width", 220, "display.max_rows", 300)
    print(out.round(4).to_string(index=False))
    print(out.groupby(["market", "tf"]).agg(pola=("p", "size"), lebih_baik=("lebih_baik", "sum"),
                                            lebih_buruk=("lebih_buruk", "sum"),
                                            selisih_median=("selisih", "median")).round(4).to_string())


if __name__ == "__main__":
    main()
