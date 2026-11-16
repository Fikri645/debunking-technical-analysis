"""Indikator teknikal. Semua nilai di indeks i hanya memakai data sampai candle i."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit


def _s(x) -> pd.Series:
    return pd.Series(np.asarray(x, dtype=float))


def sma(x, n: int) -> np.ndarray:
    return _s(x).rolling(n).mean().values


def ema(x, n: int) -> np.ndarray:
    return _s(x).ewm(span=n, adjust=False, min_periods=n).mean().values


def wilder(x, n: int) -> np.ndarray:
    return _s(x).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().values


def rolling_max(x, n: int) -> np.ndarray:
    return _s(x).rolling(n).max().values


def rolling_min(x, n: int) -> np.ndarray:
    return _s(x).rolling(n).min().values


def shift(x, k: int) -> np.ndarray:
    return _s(x).shift(k).values


def rsi(c, n: int = 14) -> np.ndarray:
    d = np.diff(np.asarray(c, dtype=float), prepend=np.nan)
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    up[0] = dn[0] = np.nan
    au, ad = wilder(up, n), wilder(dn, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 - 100.0 / (1.0 + au / ad)
    out[(ad == 0) & np.isfinite(au)] = 100.0
    return out


def macd(c, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(c, fast) - ema(c, slow)
    sig = _s(line).ewm(span=signal, adjust=False, min_periods=signal).mean().values
    return line, sig


def stochastic(h, l, c, n: int = 14, k: int = 3, d: int = 3):
    hh, ll = rolling_max(h, n), rolling_min(l, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = 100.0 * (np.asarray(c, dtype=float) - ll) / (hh - ll)
    raw[~np.isfinite(raw)] = np.nan
    slow_k = sma(raw, k)
    return slow_k, sma(slow_k, d)


def bollinger(c, n: int = 20, k: float = 2.0):
    mid = sma(c, n)
    sd = _s(c).rolling(n).std(ddof=0).values
    return mid, mid + k * sd, mid - k * sd


def ichimoku(h, l, a: int = 9, b: int = 26, c: int = 52):
    """Tenkan, kijun, dan awan yang berlaku di candle i (sudah digeser b candle)."""
    tenkan = (rolling_max(h, a) + rolling_min(l, a)) / 2
    kijun = (rolling_max(h, b) + rolling_min(l, b)) / 2
    span_a = shift((tenkan + kijun) / 2, b)
    span_b = shift((rolling_max(h, c) + rolling_min(l, c)) / 2, b)
    return tenkan, kijun, span_a, span_b


def true_range(h, l, c) -> np.ndarray:
    h, l, c = (np.asarray(x, dtype=float) for x in (h, l, c))
    pc = np.roll(c, 1)
    pc[0] = c[0]
    return np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))


def atr(h, l, c, n: int = 14) -> np.ndarray:
    return wilder(true_range(h, l, c), n)


def adx(h, l, c, n: int = 14):
    h, l = np.asarray(h, dtype=float), np.asarray(l, dtype=float)
    up = np.diff(h, prepend=h[0])
    dn = -np.diff(l, prepend=l[0])
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = wilder(true_range(h, l, c), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100 * wilder(pdm, n) / tr
        mdi = 100 * wilder(mdm, n) / tr
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    dx[~np.isfinite(dx)] = 0.0
    dx[np.isnan(pdi)] = np.nan
    return wilder(dx, n), pdi, mdi


def obv(c, v) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    sign = np.sign(np.diff(c, prepend=c[0]))
    return np.cumsum(sign * np.nan_to_num(np.asarray(v, dtype=float)))


@njit(cache=True)
def supertrend(h, l, c, atr_arr, mult):
    """Arah Supertrend: +1 naik, -1 turun, 0 belum terdefinisi."""
    n = len(c)
    out = np.zeros(n, dtype=np.int64)
    fu = np.nan
    fl = np.nan
    d = 0
    for i in range(n):
        a = atr_arr[i]
        if np.isnan(a):
            continue
        mid = (h[i] + l[i]) / 2
        bu = mid + mult * a
        bl = mid - mult * a
        if d == 0:
            fu, fl = bu, bl
            d = 1 if c[i] >= mid else -1
        else:
            fu = bu if (bu < fu or c[i - 1] > fu) else fu
            fl = bl if (bl > fl or c[i - 1] < fl) else fl
            if d == 1 and c[i] < fl:
                d = -1
            elif d == -1 and c[i] > fu:
                d = 1
        out[i] = d
    return out


@njit(cache=True)
def psar(h, l, af0, af_max):
    """Arah Parabolic SAR setelah candle i ditutup: +1 naik, -1 turun."""
    n = len(h)
    out = np.zeros(n, dtype=np.int64)
    if n < 3:
        return out
    up = h[1] >= h[0]
    sar = l[0] if up else h[0]
    ep = h[1] if up else l[1]
    af = af0
    out[1] = 1 if up else -1
    for i in range(2, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, l[i - 1], l[i - 2])
            if l[i] < sar:
                up = False
                sar = ep
                ep = l[i]
                af = af0
            elif h[i] > ep:
                ep = h[i]
                af = min(af + af0, af_max)
        else:
            sar = max(sar, h[i - 1], h[i - 2])
            if h[i] > sar:
                up = True
                sar = ep
                ep = h[i]
                af = af0
            elif l[i] < ep:
                ep = l[i]
                af = min(af + af0, af_max)
        out[i] = 1 if up else -1
    return out


@njit(cache=True)
def latch(enter_long, exit_long, enter_short, exit_short):
    """Posisi yang bertahan: masuk saat sinyal masuk, keluar saat sinyal keluar."""
    n = len(enter_long)
    pos = np.zeros(n, dtype=np.int64)
    p = 0
    for i in range(n):
        if p == 1 and exit_long[i]:
            p = 0
        elif p == -1 and exit_short[i]:
            p = 0
        if p == 0:
            if enter_long[i]:
                p = 1
            elif enter_short[i]:
                p = -1
        pos[i] = p
    return pos


@njit(cache=True)
def pivots(h, l, k):
    """Swing high/low terkonfirmasi. piv_hi[i] = indeks swing high yang BARU
    diketahui di candle i (yaitu candle i-k), atau -1. Swing baru sah setelah k
    candle sesudahnya terbentuk, jadi tidak ada mengintip ke depan."""
    n = len(h)
    piv_hi = np.full(n, -1, dtype=np.int64)
    piv_lo = np.full(n, -1, dtype=np.int64)
    for i in range(2 * k, n):
        j = i - k
        is_hi = True
        is_lo = True
        for m in range(j - k, j + k + 1):
            if m == j:
                continue
            if h[m] >= h[j]:
                is_hi = False
            if l[m] <= l[j]:
                is_lo = False
            if not is_hi and not is_lo:
                break
        if is_hi:
            piv_hi[i] = j
        if is_lo:
            piv_lo[i] = j
    return piv_hi, piv_lo


@njit(cache=True)
def divergence(h, l, osc, k, max_gap):
    """Divergence klasik antara harga dan osilator pada dua swing berurutan.
    +1 di candle i: bullish (harga lower low, osilator higher low).
    -1 di candle i: bearish (harga higher high, osilator lower high)."""
    piv_hi, piv_lo = pivots(h, l, k)
    n = len(h)
    out = np.zeros(n, dtype=np.int64)
    last_lo = -1
    last_hi = -1
    for i in range(n):
        j = piv_lo[i]
        if j >= 0:
            if last_lo >= 0 and j - last_lo <= max_gap and not np.isnan(osc[j]) \
                    and not np.isnan(osc[last_lo]):
                if l[j] < l[last_lo] and osc[j] > osc[last_lo]:
                    out[i] = 1
            last_lo = j
        j = piv_hi[i]
        if j >= 0:
            if last_hi >= 0 and j - last_hi <= max_gap and not np.isnan(osc[j]) \
                    and not np.isnan(osc[last_hi]):
                if h[j] > h[last_hi] and osc[j] < osc[last_hi]:
                    out[i] = -1
            last_hi = j
    return out


def cross_up(a, b) -> np.ndarray:
    a, b = np.asarray(a, dtype=float), np.broadcast_to(np.asarray(b, dtype=float), np.shape(a))
    pa, pb = shift(a, 1), shift(b, 1)
    return (a > b) & (pa <= pb)


def cross_down(a, b) -> np.ndarray:
    a, b = np.asarray(a, dtype=float), np.broadcast_to(np.asarray(b, dtype=float), np.shape(a))
    pa, pb = shift(a, 1), shift(b, 1)
    return (a < b) & (pa >= pb)
