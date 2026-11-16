"""Jalankan sekelompok strategi di semua sel uji (pasar x timeframe).

Hasil per sel disimpan di results/<stage>/<pasar>_<tf>.csv, jadi proses yang
terhenti bisa dilanjutkan. Strategi yang tampak menjanjikan (p < 0,01 pada 1.000
geseran) diuji ulang dengan 20.000 geseran supaya nilai p-nya cukup halus untuk
koreksi uji berganda.

Jalankan: uv run python scripts/run_tests.py stage2
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from talab import (evaluate, strat_candles, strat_indicators, strat_influencer, strat_extra, strat_momentum,  # noqa: E402
                   strat_patterns, strat_smc)

ROOT = Path(__file__).resolve().parents[1]

CELLS = [
    ("idx", "1d"), ("us", "1d"), ("forex", "1d"), ("crypto", "1d"),
    ("crypto", "4h"), ("idx", "1h"), ("us", "1h"), ("forex", "1h"),
    ("crypto", "1h"), ("crypto", "15m"),
]

STAGES = {
    "stage2": lambda: {**strat_indicators.build(), **strat_candles.build()},
    "stage3": strat_patterns.build,
    "stage3_kontrol": strat_patterns.build_controls,
    "stage4": strat_smc.build,
    "stage5": strat_influencer.build,
    "stage6_momentum": strat_momentum.build,
    "stage7_tambahan": strat_extra.build,
}

K_FAST, K_FINE = 1000, 20000


def applicable(spec: dict, market: str, tf: str) -> bool:
    if spec.get("needs_volume") and market == "forex":
        return False
    if spec.get("intraday") and tf == "1d":
        return False
    return True


def run_cell(stage: str, reg: dict, market: str, tf: str) -> None:
    out = ROOT / "results" / stage / f"{market}_{tf}.csv"
    if out.exists():
        print(f"lewati {market} {tf} (sudah ada)", flush=True)
        return
    cell = evaluate.load_cell(market, tf)
    if not cell:
        print(f"lewati {market} {tf} (data belum ada)", flush=True)
        return
    rows, bh_rows = [], []
    t0 = time.time()
    for i, (name, spec) in enumerate(reg.items(), 1):
        if not applicable(spec, market, tf):
            continue
        res = evaluate.evaluate(spec["fn"], market, tf, K=K_FAST, seed=1, cell=cell, name=name)
        if any(r.get("p", 1.0) < 0.01 for r in res):
            res = evaluate.evaluate(spec["fn"], market, tf, K=K_FINE, seed=2, cell=cell, name=name)
        for r in res:
            r.update(group=spec["group"], kind=spec["kind"])
        rows += res
        if spec["pos"] is not None:
            bh_rows += evaluate.vs_buy_hold_cell(spec["pos"], market, cell, name)
        if i % 10 == 0:
            print(f"  {market} {tf}: {i}/{len(reg)} strategi, {time.time() - t0:.0f} dtk", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    if bh_rows:
        pd.DataFrame(bh_rows).to_csv(out.with_name(f"{market}_{tf}_buyhold.csv"), index=False)
    print(f"selesai {market} {tf}: {len(rows)} baris, {time.time() - t0:.0f} dtk", flush=True)


def main() -> None:
    stage = sys.argv[1]
    only = sys.argv[2:]  # opsional: "us_1d crypto_4h"
    reg = STAGES[stage]()
    for market, tf in CELLS:
        if only and f"{market}_{tf}" not in only:
            continue
        run_cell(stage, reg, market, tf)


if __name__ == "__main__":
    main()
