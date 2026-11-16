"""Tahap 1: uji plasebo mesin.

Strategi yang sinyalnya murni acak tidak punya keunggulan apa pun. Jika mesin
uji benar, nilai p-nya harus tersebar rata: sekitar 5% di bawah 0,05, bukan lebih.

Dua jenis sinyal acak diuji:
- "independen": tiap instrumen punya tanggal sinyal acak sendiri.
- "serentak": semua instrumen memberi sinyal di tanggal yang sama. Ini kasus
  sulit, karena transaksi yang bergerombol membuat hasil tampak lebih meyakinkan
  daripada kenyataannya.

Jalankan: uv run python scripts/validate_engine.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from talab import engine, evaluate  # noqa: E402

N_STRATEGIES = 200
K = 300


def random_strategy(rng: np.random.Generator, common_dates: pd.DatetimeIndex | None):
    rate = rng.choice([0.01, 0.03])
    p_long = rng.choice([0.5, 1.0])
    bracket = rng.random() < 0.5
    hold = int(rng.choice([5, 10, 20]))
    seed = int(rng.integers(1 << 31))
    if common_dates is not None:
        days = set(common_dates[rng.random(len(common_dates)) < rate])

    def strategy(df: pd.DataFrame) -> pd.DataFrame:
        r = np.random.default_rng(seed + len(df))
        if common_dates is None:
            mask = r.random(len(df)) < rate
        else:
            mask = df.index.isin(days)
        is_long = r.random(len(df)) < p_long
        a = engine.atr(df)
        if bracket:
            return engine.signals_from_events(mask & is_long, mask & ~is_long, hold=50,
                                              sl_dist=1.5 * a, tp_dist=3.0 * a)
        return engine.signals_from_events(mask & is_long, mask & ~is_long, hold=hold)

    return strategy


def main() -> None:
    rows = []
    for market, tf in [("us", "1d"), ("crypto", "4h")]:
        cell = dict(list(evaluate.load_cell(market, tf).items())[:15])
        dates = pd.DatetimeIndex(sorted(set().union(*[set(d.index) for d in cell.values()])))
        for kind in ("independen", "serentak"):
            rng = np.random.default_rng(42)
            ps = []
            for i in range(N_STRATEGIES):
                strat = random_strategy(rng, dates if kind == "serentak" else None)
                res = evaluate.evaluate(strat, market, tf, K=K, seed=i, cell=cell)
                ps += [r["p"] for r in res if r["era"] == "ALL" and "p" in r]
            ps = np.array(ps)
            rows.append(dict(pasar=market, tf=tf, sinyal=kind, n_strategi=len(ps),
                             frac_p_below_0_05=float((ps < 0.05).mean()),
                             frac_p_below_0_10=float((ps < 0.10).mean()),
                             frac_p_above_0_95=float((ps > 0.95).mean()),
                             median_p=float(np.median(ps))))
            print(rows[-1], flush=True)
    out = Path(__file__).resolve().parents[1] / "results" / "placebo_engine.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Tersimpan:", out)


if __name__ == "__main__":
    main()
