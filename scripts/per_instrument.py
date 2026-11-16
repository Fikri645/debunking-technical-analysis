"""Hasil tiap strategi PER INSTRUMEN, untuk dua keperluan:

1. Kriteria lolos ketiga: strategi harus bekerja di banyak instrumen, bukan satu saja.
2. Pembanding beli-lalu-diamkan (buy and hold): modal yang menjalankan strategi itu
   terus-menerus di satu instrumen, dibandingkan dengan modal yang membeli instrumen
   yang sama di awal data dan mendiamkannya. Keduanya setelah biaya.

Hasil per sel disimpan di results/per_instrumen/<pasar>_<tf>.csv (bisa dilanjutkan).

Jalankan: uv run python scripts/per_instrument.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tests import CELLS, STAGES, applicable  # noqa: E402
from talab import engine, evaluate  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "per_instrumen"
ERA_C = pd.Timestamp("2022-01-01")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reg = {}
    for build in STAGES.values():
        reg.update(build())
    only = sys.argv[1:]
    for market, tf in CELLS:
        out = OUT / f"{market}_{tf}.csv"
        if out.exists() or (only and f"{market}_{tf}" not in only):
            continue
        cell = evaluate.load_cell(market, tf)
        t0 = time.time()
        rows = []
        for sym, df in cell.items():
            cost = evaluate.cost_for(market, sym)
            o, c = df["open"].values.astype(float), df["close"].values.astype(float)
            years = (df.index[-1] - df.index[0]).total_seconds() / (365.25 * 86400)
            bh_log = float(np.log(c[-1] / o[0]) + np.log1p(-cost))
            for name, spec in reg.items():
                if not applicable(spec, market, tf):
                    continue
                tr = engine.run(df, spec["fn"](df), cost)
                last = tr[tr["entry_time"] >= ERA_C]
                rows.append(dict(
                    strategy=name, market=market, tf=tf, symbol=sym, n=len(tr),
                    mean_net=tr["net"].mean() if len(tr) else np.nan,
                    n_C=len(last), mean_net_C=last["net"].mean() if len(last) else np.nan,
                    strat_log=float(np.log1p(np.clip(tr["net"].values, -0.99, None)).sum()),
                    bh_log=bh_log, years=years,
                    exposure=float((tr["exit_idx"] - tr["entry_idx"]).sum()) / len(df)))
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"selesai {market} {tf}: {len(rows)} baris, {time.time() - t0:.0f} dtk", flush=True)


if __name__ == "__main__":
    main()
