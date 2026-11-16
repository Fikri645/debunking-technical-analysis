"""Seberapa sering hitungan Elliott saling bertentangan?

Untuk tiap sinyal Elliott (kejar wave 3, kejar wave 5, koreksi ABC; semua skala),
cek apakah pada instrumen yang sama, dalam +/- 10 candle, ada hitungan Elliott
lain yang sah tetapi arahnya berlawanan.

Jalankan: uv run python scripts/elliott_conflicts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from talab import evaluate, strat_smc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WINDOW = 10


def main() -> None:
    rows = []
    for market, tf in [("idx", "1d"), ("us", "1d"), ("forex", "1d"), ("crypto", "1d"), ("crypto", "4h")]:
        total = conflict = 0
        for sym, df in evaluate.load_cell(market, tf).items():
            det = pd.concat([strat_smc.ew_detect(df, s, False) for s in (0, 1, 2)], ignore_index=True)
            if det.empty:
                continue
            idx, d = det["idx"].values.astype(np.int64), det["dir"].values
            up, dn = np.sort(idx[d == 1]), np.sort(idx[d == -1])
            for i, di in zip(idx, d):
                other = dn if di == 1 else up
                lo = np.searchsorted(other, i - WINDOW)
                hi = np.searchsorted(other, i + WINDOW, side="right")
                conflict += hi > lo
                total += 1
        rows.append(dict(pasar=market, tf=tf, sinyal=total, bertentangan=conflict,
                         persen=round(100 * conflict / max(total, 1), 1)))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "stage4" / "_elliott_konflik.csv", index=False)
    print(out.to_string(index=False))
    print("Total:", out.bertentangan.sum(), "dari", out.sinyal.sum(),
          f"= {100 * out.bertentangan.sum() / out.sinyal.sum():.1f}%")


if __name__ == "__main__":
    main()
