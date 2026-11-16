"""Mesin backtest dan pembanding acak.

Aturan yang ditegakkan di sini:
- Sinyal dihitung di penutupan candle i, transaksi dibuka di pembukaan candle i+1.
- Jika SL dan TP kena di candle yang sama, SL dianggap kena lebih dulu.
- Jika harga pembukaan melompati SL/TP, keluar di harga pembukaan.
- Satu posisi dalam satu waktu per instrumen.
- Pembanding acak memakai aturan yang persis sama.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit, prange

NAN = np.nan
# Transaksi saat ATR di bawah 0,01% harga diabaikan (di strategi maupun pembanding
# acak): itu terjadi pada saham yang lama tidak bergerak, dan return per satuan
# volatilitas menjadi tak berarti.
MIN_ATR_PCT = 1e-4


def ts_ns(index: pd.DatetimeIndex) -> np.ndarray:
    """Waktu candle sebagai int64 nanodetik UTC."""
    return index.tz_convert(None).to_numpy().astype("datetime64[ns]").astype(np.int64)


def atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    """ATR Wilder. Nilai di indeks i hanya memakai data sampai candle i."""
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    pc = np.roll(c, 1)
    pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    out = pd.Series(tr).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().values
    return out


@njit(cache=True)
def _run_one(o, h, l, c, ei, d, ep, s, t, hold, is_limit):
    """Jalankan satu transaksi. Mengembalikan (indeks keluar, harga keluar, keluar_di_open)."""
    n = len(o)
    end = min(ei + hold, n)
    for j in range(ei, end):
        if j > ei:
            if not np.isnan(s) and d * (o[j] - s) <= 0:
                return j, o[j], True
            if not np.isnan(t) and d * (o[j] - t) >= 0:
                return j, o[j], True
        if d > 0:
            if not np.isnan(s) and l[j] <= s:
                return j, s, False
            if not np.isnan(t) and h[j] >= t and not (is_limit and j == ei):
                return j, t, False
        else:
            if not np.isnan(s) and h[j] >= s:
                return j, s, False
            if not np.isnan(t) and l[j] <= t and not (is_limit and j == ei):
                return j, t, False
    if end < n:
        return end, o[end], True
    return n - 1, c[n - 1], False


@njit(cache=True)
def _simulate(o, h, l, c, entry_idx, direction, sl, tp, sl_dist, tp_dist, hold,
              entry_price, is_limit):
    m = len(entry_idx)
    n = len(o)
    taken = np.zeros(m, dtype=np.bool_)
    e_px = np.full(m, NAN)
    x_idx = np.full(m, -1, dtype=np.int64)
    x_px = np.full(m, NAN)
    sl_out = np.full(m, NAN)
    tp_out = np.full(m, NAN)
    free_from = 0  # candle pertama yang boleh dipakai untuk entry baru
    for k in range(m):
        ei = entry_idx[k]
        if ei < 1 or ei >= n or ei < free_from:
            continue
        d = direction[k]
        ep = entry_price[k]
        if np.isnan(ep):
            ep = o[ei]
        s = sl[k]
        if np.isnan(s) and not np.isnan(sl_dist[k]):
            s = ep - d * sl_dist[k]
        t = tp[k]
        if np.isnan(t) and not np.isnan(tp_dist[k]):
            t = ep + d * tp_dist[k]
        # bracket tidak sah (SL/TP di sisi yang salah): lewati
        if not np.isnan(s) and d * (ep - s) <= 0:
            continue
        if not np.isnan(t) and d * (t - ep) <= 0:
            continue
        xi, xp, at_open = _run_one(o, h, l, c, ei, d, ep, s, t, hold[k], is_limit[k])
        taken[k] = True
        e_px[k] = ep
        x_idx[k] = xi
        x_px[k] = xp
        sl_out[k] = s
        tp_out[k] = t
        free_from = xi if at_open else xi + 1
    return taken, e_px, x_idx, x_px, sl_out, tp_out


def empty_signals() -> pd.DataFrame:
    return make_signals(np.array([], dtype=np.int64), np.array([], dtype=np.int64))


def make_signals(entry_idx, direction, sl=None, tp=None, sl_dist=None, tp_dist=None,
                 hold=50, entry_price=None, is_limit=None) -> pd.DataFrame:
    """Bentuk standar sinyal. entry_idx = candle tempat transaksi dibuka."""
    m = len(entry_idx)

    def col(x, fill, dtype=float):
        if x is None:
            return np.full(m, fill, dtype=dtype)
        if np.isscalar(x):
            return np.full(m, x, dtype=dtype)
        return np.asarray(x, dtype=dtype)

    df = pd.DataFrame({
        "entry_idx": np.asarray(entry_idx, dtype=np.int64),
        "dir": np.asarray(direction, dtype=np.int64),
        "sl": col(sl, NAN), "tp": col(tp, NAN),
        "sl_dist": col(sl_dist, NAN), "tp_dist": col(tp_dist, NAN),
        "hold": col(hold, 50, np.int64),
        "entry_price": col(entry_price, NAN),
        "is_limit": col(is_limit, False, bool),
    })
    return df.sort_values("entry_idx", kind="stable").reset_index(drop=True)


def signals_from_events(long_mask, short_mask, hold, sl_dist=None, tp_dist=None) -> pd.DataFrame:
    """Sinyal kejadian di candle i -> entry market di candle i+1.
    sl_dist/tp_dist: array jarak harga per candle (mis. k * ATR), atau None."""
    long_mask = np.asarray(long_mask, dtype=bool)
    short_mask = np.asarray(short_mask, dtype=bool)
    li, si = np.flatnonzero(long_mask), np.flatnonzero(short_mask)
    idx = np.concatenate([li, si])
    d = np.concatenate([np.ones(len(li), dtype=np.int64), -np.ones(len(si), dtype=np.int64)])
    sd = None if sl_dist is None else np.asarray(sl_dist, dtype=float)[idx]
    td = None if tp_dist is None else np.asarray(tp_dist, dtype=float)[idx]
    return make_signals(idx + 1, d, sl_dist=sd, tp_dist=td, hold=hold)


def signals_from_position(pos) -> pd.DataFrame:
    """Posisi {-1,0,1} yang diputuskan di penutupan candle i, dipegang mulai open i+1.
    Tiap rentang posisi yang sama menjadi satu transaksi tanpa SL/TP."""
    pos = np.nan_to_num(np.asarray(pos, dtype=float)).astype(np.int64)
    if len(pos) == 0:
        return empty_signals()
    change = np.flatnonzero(np.diff(pos) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(pos)]])
    keep = pos[starts] != 0
    starts, ends = starts[keep], ends[keep]
    return make_signals(starts + 1, pos[starts], hold=ends - starts)


def run(df: pd.DataFrame, signals: pd.DataFrame, cost: float, atr_arr=None) -> pd.DataFrame:
    """Jalankan sinyal pada satu instrumen. cost = biaya pulang-pergi (pecahan)."""
    if atr_arr is None:
        atr_arr = atr(df)
    o, h, l, c = (df[k].values.astype(float) for k in ("open", "high", "low", "close"))
    s = signals
    taken, e_px, x_idx, x_px, sl, tp = _simulate(
        o, h, l, c, s["entry_idx"].values, s["dir"].values, s["sl"].values, s["tp"].values,
        s["sl_dist"].values, s["tp_dist"].values, s["hold"].values,
        s["entry_price"].values, s["is_limit"].values)
    ei = s["entry_idx"].values[taken]
    d = s["dir"].values[taken]
    ep, xi, xp = e_px[taken], x_idx[taken], x_px[taken]
    a = atr_arr[ei - 1]
    times = df.index.tz_convert(None).to_numpy()
    gross = d * (xp / ep - 1.0)
    out = pd.DataFrame({
        "entry_idx": ei, "exit_idx": xi, "dir": d,
        "entry_time": times[ei], "exit_time": times[xi],
        "entry_price": ep, "exit_price": xp,
        "gross": gross, "net": gross - cost,
        "atr_pct": a / ep,
        "sl_atr": np.abs(ep - sl[taken]) / a,
        "tp_atr": np.abs(tp[taken] - ep) / a,
        "hold": s["hold"].values[taken],
    })
    out = out[np.isfinite(out["atr_pct"]) & (out["atr_pct"] >= MIN_ATR_PCT)].reset_index(drop=True)
    # Return KOTOR dalam satuan volatilitas: ukuran daya ramal sinyal. Biaya sengaja
    # tidak ikut dibagi volatilitas; kalau ikut, strategi yang kebetulan aktif saat
    # pasar ramai terlihat lebih baik tanpa punya daya ramal apa pun.
    out["rg"] = out["gross"] / out["atr_pct"]
    return out


@njit(parallel=True, cache=True)
def _null(ts, o, h, l, c, atr_arr, t_entry, win_a, win_b, d, sl_atr, tp_atr, hold, cost, shifts):
    """Untuk tiap geseran waktu, pindahkan semua transaksi lalu jalankan ulang.
    Mengembalikan jumlah rg (return kotor per volatilitas), jumlah return bersih, dan banyaknya transaksi per geseran."""
    K = len(shifts)
    n = len(o)
    m = len(t_entry)
    sum_rn = np.zeros(K)
    sum_net = np.zeros(K)
    cnt = np.zeros(K)
    for k in prange(K):
        srn = 0.0
        snet = 0.0
        cn = 0.0
        for i in range(m):
            length = win_b[i] - win_a[i]
            if length <= 0:
                continue
            t2 = win_a[i] + (t_entry[i] - win_a[i] + np.int64(shifts[k] * length)) % length
            ei = np.searchsorted(ts, t2)
            if ei < 1 or ei >= n:
                continue
            a = atr_arr[ei - 1]
            if np.isnan(a) or a <= 0:
                continue
            ep = o[ei]
            if a / ep < MIN_ATR_PCT:
                continue
            s = NAN if np.isnan(sl_atr[i]) else ep - d[i] * sl_atr[i] * a
            t = NAN if np.isnan(tp_atr[i]) else ep + d[i] * tp_atr[i] * a
            if not np.isnan(s) and s <= 0:
                continue
            xi, xp, _ = _run_one(o, h, l, c, ei, d[i], ep, s, t, hold[i], False)
            r = d[i] * (xp / ep - 1.0)
            srn += r / (a / ep)
            snet += r - cost
            cn += 1.0
        sum_rn[k] = srn
        sum_net[k] = snet
        cnt[k] = cn
    return sum_rn, sum_net, cnt


def null_sums(df: pd.DataFrame, trades: pd.DataFrame, cost: float, shifts: np.ndarray,
              win_a: np.ndarray, win_b: np.ndarray, atr_arr=None):
    """Pembanding acak untuk satu instrumen. win_a/win_b: jendela waktu (ns) tiap
    transaksi, tempat transaksi itu digeser secara melingkar. Semua instrumen dalam
    satu sel uji memakai `shifts` yang sama, sehingga transaksi yang bergerombol
    di waktu yang sama tetap bergerombol."""
    if atr_arr is None:
        atr_arr = atr(df)
    o, h, l, c = (df[k].values.astype(float) for k in ("open", "high", "low", "close"))
    ts = ts_ns(df.index)
    return _null(
        ts, o, h, l, c, atr_arr,
        trades["entry_time"].values.astype("datetime64[ns]").astype(np.int64),
        np.asarray(win_a, dtype=np.int64), np.asarray(win_b, dtype=np.int64),
        trades["dir"].values.astype(np.int64),
        trades["sl_atr"].values.astype(float), trades["tp_atr"].values.astype(float),
        trades["hold"].values.astype(np.int64), float(cost), np.asarray(shifts, dtype=float))
