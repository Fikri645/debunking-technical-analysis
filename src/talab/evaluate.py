"""Evaluasi satu strategi pada satu sel uji (pasar x timeframe), per era,
melawan pembanding acak. Lihat bagian 6."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import numpy as np
import pandas as pd

from . import data, engine, universe

# Biaya pulang-pergi (pecahan dari nilai transaksi). Asumsi, lihat bagian 5.
COST = {"idx": 0.0040, "us": 0.0010, "crypto": 0.0020, "crypto_old": 0.0020, "forex": 0.00015}
COST_OVERRIDE = {"USDIDR=X": 0.0010, "GC=F": 0.0003, "USDJPY=X": 0.00015}

MIN_TRADES = 30


def cost_for(market: str, symbol: str) -> float:
    return COST_OVERRIDE.get(symbol, COST[market])


def symbols_for(market: str) -> list[str]:
    return universe.CRYPTO if market == "crypto" else universe.YF_MARKETS[market]


@lru_cache(maxsize=None)
def load_cell(market: str, tf: str) -> dict[str, pd.DataFrame]:
    out = {}
    for s in symbols_for(market):
        p = data.path_for(data.CLEAN, market, s, tf)
        if p.exists():
            df = pd.read_parquet(p)
            if len(df) >= 300:
                out[s] = df
    return out


def _ns(x) -> int:
    return int(pd.Timestamp(x, tz="UTC").value)


def era_windows(cell: dict[str, pd.DataFrame]) -> dict[str, tuple[int, int]]:
    """Jendela tiap era, dipotong ke rentang data yang benar-benar ada di sel ini."""
    lo = min(df.index[0].value for df in cell.values())
    hi = max(df.index[-1].value for df in cell.values())
    out = {}
    for name, (a, b) in universe.ERAS.items():
        a2, b2 = max(_ns(a), lo), min(_ns(b) + 86_400_000_000_000, hi)
        if b2 > a2:
            out[name] = (a2, b2)
    return out


def summarize(trades: pd.DataFrame) -> dict:
    n = len(trades)
    if n == 0:
        return dict(n=0)
    net = trades["net"].values
    wins, losses = net[net > 0].sum(), -net[net < 0].sum()
    return dict(
        n=n,
        win_rate=float((net > 0).mean()),
        mean_gross=float(trades["gross"].mean()),
        mean_net=float(net.mean()),
        mean_rg=float(trades["rg"].mean()),
        profit_factor=float(wins / losses) if losses > 0 else float("inf"),
    )


def evaluate(strategy: Callable[[pd.DataFrame], pd.DataFrame], market: str, tf: str,
             K: int = 1000, seed: int = 0, cell: dict[str, pd.DataFrame] | None = None,
             name: str = "") -> list[dict]:
    """Jalankan strategi di semua instrumen sel, lalu uji lawan K geseran acak.
    Mengembalikan satu baris per era ditambah baris 'ALL'."""
    if cell is None:
        cell = load_cell(market, tf)
    wins = era_windows(cell)
    shifts = np.random.default_rng(seed).uniform(0.02, 0.98, K)

    all_trades = []
    null = {e: [np.zeros(K), np.zeros(K), np.zeros(K)] for e in wins}
    for sym, df in cell.items():
        a = engine.atr(df)
        cost = cost_for(market, sym)
        tr = engine.run(df, strategy(df), cost, a)
        if tr.empty:
            continue
        t = tr["entry_time"].values.astype("datetime64[ns]").astype(np.int64)
        tr["era"] = ""
        for e, (wa, wb) in wins.items():
            m = (t >= wa) & (t < wb)
            if not m.any():
                continue
            tr.loc[m, "era"] = e
            sub = tr[m]
            s_rn, s_net, cnt = engine.null_sums(
                df, sub, cost, shifts, np.full(len(sub), wa), np.full(len(sub), wb), a)
            null[e][0] += s_rn
            null[e][1] += s_net
            null[e][2] += cnt
        tr["symbol"] = sym
        all_trades.append(tr[tr["era"] != ""])

    rows = []
    if not all_trades:
        return rows
    trades = pd.concat(all_trades, ignore_index=True)

    def row(era: str, sub: pd.DataFrame, s_rn, s_net, cnt) -> dict:
        r = dict(strategy=name, market=market, tf=tf, era=era, **summarize(sub))
        ok = cnt > 0
        if r["n"] >= MIN_TRADES and ok.sum() >= K // 2:
            nm = s_rn[ok] / cnt[ok]
            r["null_mean_rg"] = float(nm.mean())
            r["null_sd_rg"] = float(nm.std(ddof=1))
            r["null_mean_net"] = float((s_net[ok] / cnt[ok]).mean())
            r["p"] = float((1 + (nm >= r["mean_rg"]).sum()) / (1 + len(nm)))
            r["z"] = float((r["mean_rg"] - nm.mean()) / nm.std(ddof=1)) if nm.std() > 0 else np.nan
        return r

    tot = [np.zeros(K), np.zeros(K), np.zeros(K)]
    for e in wins:
        sub = trades[trades["era"] == e]
        if len(sub):
            rows.append(row(e, sub, *null[e]))
        for j in range(3):
            tot[j] += null[e][j]
    rows.append(row("ALL", trades, *tot))
    return rows


def benjamini_hochberg(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Mengembalikan mask hipotesis yang lolos pada false discovery rate q."""
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    out = np.zeros(len(p), dtype=bool)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return out
    order = np.argsort(pv)
    thresh = q * (np.arange(1, m + 1) / m)
    passed = pv[order] <= thresh
    if passed.any():
        kmax = np.flatnonzero(passed).max()
        sel = np.zeros(m, dtype=bool)
        sel[order[: kmax + 1]] = True
        out[np.flatnonzero(ok)] = sel
    return out


def permute_bars(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Data plasebo: urutan candle diacak. Bentuk dan ukuran candle sama persis
    dengan data asli, tetapi urutannya murni kebetulan, sehingga tidak ada pola
    yang bisa diramal."""
    rng = np.random.default_rng(seed)
    o, h, l, c = (np.log(df[k].values.astype(float)) for k in ("open", "high", "low", "close"))
    gap = np.empty(len(o))
    gap[0] = 0.0
    gap[1:] = o[1:] - c[:-1]
    rel_h, rel_l, rel_c = h - o, l - o, c - o
    perm = rng.permutation(len(o))
    gap, rel_h, rel_l, rel_c = gap[perm], rel_h[perm], rel_l[perm], rel_c[perm]
    vol = df["volume"].values[perm]
    no = np.empty(len(o))
    prev_c = o[0]
    for i in range(len(o)):
        no[i] = prev_c + gap[i]
        prev_c = no[i] + rel_c[i]
    return pd.DataFrame({
        "open": np.exp(no), "high": np.exp(no + rel_h), "low": np.exp(no + rel_l),
        "close": np.exp(no + rel_c), "volume": vol,
    }, index=df.index)


def vs_buy_hold(df: pd.DataFrame, pos: np.ndarray, cost: float) -> dict | None:
    """Bandingkan strategi posisi dengan buy-and-hold pada satu instrumen.
    Posisi yang diputuskan di penutupan candle i berlaku dari open i+1 ke open i+2."""
    o = df["open"].values.astype(float)
    if len(o) < 300:
        return None
    r = o[2:] / o[1:-1] - 1.0
    p = np.asarray(pos, dtype=float)[:-2]
    turnover = np.abs(np.diff(p, prepend=0.0))
    strat = p * r - turnover * cost / 2.0
    years = (df.index[-1] - df.index[0]).total_seconds() / (365.25 * 86400)
    bars_per_year = len(r) / max(years, 1e-9)

    def stats(x):
        sd = x.std()
        total = float(np.prod(1.0 + x) - 1.0)
        cagr = (1.0 + total) ** (1.0 / years) - 1.0 if total > -1 else -1.0
        eq = np.cumprod(1.0 + x)
        dd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
        return dict(sharpe=float(x.mean() / sd * np.sqrt(bars_per_year)) if sd > 0 else 0.0,
                    cagr=float(cagr), max_dd=dd)

    s, b = stats(strat), stats(r)
    return dict(strat_sharpe=s["sharpe"], bh_sharpe=b["sharpe"], strat_cagr=s["cagr"],
                bh_cagr=b["cagr"], strat_max_dd=s["max_dd"], bh_max_dd=b["max_dd"],
                exposure=float(np.mean(p != 0)))


def vs_buy_hold_cell(pos_fn, market: str, cell: dict[str, pd.DataFrame], name: str) -> list[dict]:
    rows = []
    for sym, df in cell.items():
        r = vs_buy_hold(df, pos_fn(df), cost_for(market, sym))
        if r:
            r.update(strategy=name, market=market, symbol=sym)
            rows.append(r)
    return rows
