"""Penurunan modal terdalam (max drawdown) strategi indikator yang LOLOS di crypto,
per koin, dibanding beli-lalu-diamkan. Sumber: results/stage2/crypto_*_buyhold.csv
(dihasilkan run_tests.py) dan results/final/verdicts.csv.

Jalankan: uv run python scripts/drawdown.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    v = pd.read_csv(ROOT / "results" / "final" / "verdicts.csv")
    ok = v[v.passes & ~v.control & (v.market == "crypto")][["strategy", "tf"]]
    parts = []
    for tf, d in ok.groupby("tf"):
        f = ROOT / "results" / "stage2" / f"crypto_{tf}_buyhold.csv"
        if f.exists():
            b = pd.read_csv(f)
            parts.append(b[b.strategy.isin(set(d.strategy))].assign(tf=tf))
    d = pd.concat(parts, ignore_index=True)
    d["strat_max_dd"] = d.strat_max_dd.clip(lower=-1.0)   # short yang rugi lebih dari 100% = modal habis
    out = d[["strategy", "tf", "symbol", "strat_max_dd", "bh_max_dd", "strat_cagr", "bh_cagr"]]
    out.to_csv(ROOT / "results" / "final" / "drawdown_lolos_crypto.csv", index=False)
    q = d.strat_max_dd.quantile([.1, .5, .9])
    lines = [
        f"Strategi lolos di crypto yang punya data posisi: {d.groupby(['strategy', 'tf']).ngroups} dari {len(ok)}",
        f"Pasangan strategi x koin: {len(d)}",
        f"Drawdown terdalam strategi: median {q[.5]:.1%}, 80% pasangan di antara {q[.9]:.1%} dan {q[.1]:.1%}",
        f"Lebih dalam dari -50%: {(d.strat_max_dd < -.5).mean():.1%} pasangan",
        f"Beli-lalu-diamkan pada pasangan yang sama: median {d.bh_max_dd.median():.1%}",
        f"Drawdown strategi lebih dangkal dari beli-lalu-diamkan: {(d.strat_max_dd > d.bh_max_dd).mean():.1%} pasangan",
    ]
    (ROOT / "results" / "final" / "_drawdown.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
