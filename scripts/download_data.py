"""Tahap 0: unduh semua data, bersihkan, dan tulis laporan kualitas.

Jalankan: uv run python scripts/download_data.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from talab import data, universe  # noqa: E402


def main() -> None:
    jobs = []
    for market, symbols in universe.YF_MARKETS.items():
        for tf in universe.YF_TIMEFRAMES:
            jobs += [(market, s, tf) for s in symbols]
    for tf in universe.CRYPTO_TIMEFRAMES:
        jobs += [("crypto", s, tf) for s in universe.CRYPTO]

    reports = []
    for i, (market, symbol, tf) in enumerate(jobs, 1):
        try:
            fresh = not data.path_for(data.RAW, market, symbol, tf).exists()
            raw_path = data.download(market, symbol, tf, universe.START)
            if fresh and market != "crypto":
                time.sleep(0.4)
            # forex: sumbu > 5% di luar badan candle adalah tick rusak, bukan pergerakan nyata
            df, rep = data.clean(pd.read_parquet(raw_path), allow_extreme=(market == "crypto"),
                                 wick_ratio=1.05 if market == "forex" else 1.5)
            out = data.path_for(data.CLEAN, market, symbol, tf)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out)
            rep.update(market=market, symbol=symbol, tf=tf, error="")
        except Exception as e:  # catat dan lanjut; kegagalan terlihat di laporan
            rep = dict(market=market, symbol=symbol, tf=tf, error=repr(e)[:200])
        reports.append(rep)
        print(f"[{i}/{len(jobs)}] {market} {symbol} {tf}: "
              f"{rep.get('rows_clean', 'GAGAL')} {rep.get('error', '')}", flush=True)

    out = data.ROOT / "results" / "data_quality.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(reports).to_csv(out, index=False)
    print("Laporan:", out)


if __name__ == "__main__":
    main()
