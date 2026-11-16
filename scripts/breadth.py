"""Kriteria ketiga: strategi yang lolos harus bekerja di
lebih dari satu instrumen. Untuk tiap kombinasi yang lolos, hitung berapa
instrumen yang punya transaksi dan berapa yang rata-rata return bersihnya positif.

Jalankan: uv run python scripts/breadth.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from talab import engine, evaluate, strat_candles, strat_indicators, strat_influencer, strat_patterns, strat_smc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    reg = {}
    for b in (strat_indicators.build, strat_candles.build, strat_patterns.build,
              strat_patterns.build_controls, strat_smc.build, strat_influencer.build):
        reg.update(b())
    passed = pd.read_csv(ROOT / "results" / "final" / "lolos.csv")
    rows = []
    for _, r in passed.iterrows():
        cell = evaluate.load_cell(r.market, r.tf)
        n_inst = pos = n_c = pos_c = 0
        for sym, df in cell.items():
            tr = engine.run(df, reg[r.strategy]["fn"](df), evaluate.cost_for(r.market, sym))
            if len(tr) < 5:
                continue
            n_inst += 1
            pos += tr["net"].mean() > 0
            last = tr[tr["entry_time"] >= pd.Timestamp("2022-01-01")]
            if len(last) >= 5:
                n_c += 1
                pos_c += last["net"].mean() > 0
        rows.append(dict(strategy=r.strategy, market=r.market, tf=r.tf, instrumen=n_inst,
                         positif=int(pos), instrumen_2022=n_c, positif_2022=int(pos_c)))
    out = pd.DataFrame(rows)
    out["luas"] = ((out.instrumen >= 2) & (out.positif >= 0.6 * out.instrumen)
                   & (out.positif_2022 >= 0.6 * out.instrumen_2022))
    out.to_csv(ROOT / "results" / "final" / "lolos_per_instrumen.csv", index=False)
    pd.set_option("display.width", 200, "display.max_rows", 100)
    print(out.to_string(index=False))
    print("Positif di >= 60% instrumen (seluruh periode dan 2022-2026):", int(out.luas.sum()), "dari", len(out))


if __name__ == "__main__":
    main()
