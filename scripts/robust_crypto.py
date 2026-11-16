"""Uji ketahanan temuan crypto: jalankan semua strategi indikator pada kelompok
koin lama yang sebagian besar jatuh (bukan pemenang masa kini).

Jalankan: uv run python scripts/robust_crypto.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from talab import data, evaluate, strat_indicators  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COINS = ["XLMUSDT", "BCHUSDT", "ETCUSDT", "ZECUSDT", "DASHUSDT", "NEOUSDT", "IOTAUSDT", "QTUMUSDT",
         "ONTUSDT", "VETUSDT", "ATOMUSDT", "EOSUSDT", "XMRUSDT", "ALGOUSDT", "FILUSDT", "ICPUSDT"]


def main() -> None:
    reg = {k: v for k, v in strat_indicators.build().items() if v["kind"] == "state" and not v.get("intraday")}
    out = ROOT / "results" / "stage2_robust"
    out.mkdir(parents=True, exist_ok=True)
    for tf in ("1d", "4h"):
        cell = {s: data.load("crypto_old", s, tf) for s in COINS}
        rows, bh = [], []
        for name, spec in reg.items():
            res = evaluate.evaluate(spec["fn"], "crypto_old", tf, K=1000, seed=1, cell=cell, name=name)
            if any(r.get("p", 1.0) < 0.01 for r in res):
                res = evaluate.evaluate(spec["fn"], "crypto_old", tf, K=20000, seed=2, cell=cell, name=name)
            rows += res
            bh += evaluate.vs_buy_hold_cell(spec["pos"], "crypto_old", cell, name)
        pd.DataFrame(rows).to_csv(out / f"crypto_old_{tf}.csv", index=False)
        pd.DataFrame(bh).to_csv(out / f"crypto_old_{tf}_buyhold.csv", index=False)
        print("selesai", tf, flush=True)


if __name__ == "__main__":
    main()
