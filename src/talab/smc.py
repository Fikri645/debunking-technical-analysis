"""Supply & demand zone dan Smart Money Concepts / ICT.

Semua zona dibentuk hanya dari candle yang sudah ditutup. Order limit dipasang
setelah zona terbentuk dan terisi saat harga kembali ke zona untuk pertama kali.
Tiap fungsi mengembalikan juga geometri zona (indeks awal, batas bawah, batas atas)
untuk audit visual.
"""

from __future__ import annotations

import numpy as np
from numba import njit

BUF = 0.1  # SL diberi jarak 0,1 ATR di luar zona


@njit(cache=True)
def _out(m):
    return (np.empty(m, dtype=np.int64), np.empty(m), np.empty(m, dtype=np.int64), np.empty(m),
            np.empty(m), np.empty(m, dtype=np.int64), np.empty(m), np.empty(m))


@njit(cache=True)
def _fill(o, h, l, start, end, d, level):
    """Candle pertama yang menembus `level` (limit order). -1 jika tidak ada.
    Sentuhan persis di harga limit tidak dihitung terisi: order sungguhan yang
    antre di harga itu sering tidak terisi, dan menghitungnya memberi strategi
    harga terbaik candle itu."""
    for j in range(start, end):
        if d == 1 and l[j] < level:
            return j, min(o[j], level)
        if d == -1 and h[j] > level:
            return j, max(o[j], level)
    return -1, 0.0


@njit(cache=True)
def sd_signals(o, h, l, c, atr, dep_mult, max_base, rr, mode, max_wait):
    """Supply/demand zone seperti di materi kursus.
    - Base: 1..max_base candle kecil (rentang <= 1 ATR, badan <= separuh rentang).
    - Keberangkatan: candle berbadan >= dep_mult x ATR yang ditutup menjauhi base.
    - Zona demand: dari badan tertinggi base (proksimal) ke low terendah (distal).
      Zona supply sebaliknya.
    - Entry: limit di garis proksimal saat harga pertama kali kembali; SL di luar
      garis distal; TP = rr x risiko.
    mode 0 = semua, 1 = pembalikan saja (drop-base-rally, rally-base-drop),
    2 = lanjutan saja (rally-base-rally, drop-base-drop)."""
    n = len(c)
    e_idx, e_px, s_dir, s_sl, s_tp, z0, zlo, zhi = _out(n)
    k_out = 0
    for e in range(8, n - 1):
        a = atr[e - 1]
        if np.isnan(a) or a <= 0:
            continue
        body = c[e] - o[e]
        if abs(body) < dep_mult * a:
            continue
        d = 1 if body > 0 else -1
        nb = 0
        k = e - 1
        while nb < max_base and k >= 1:
            rng = h[k] - l[k]
            if rng <= a and abs(c[k] - o[k]) <= 0.5 * rng + 1e-12:
                nb += 1
                k -= 1
            else:
                break
        if nb == 0:
            continue
        leg_in = c[k] - o[k]
        if abs(leg_in) < 0.5 * a:
            continue
        reversal = leg_in * d < 0
        if (mode == 1 and not reversal) or (mode == 2 and reversal):
            continue
        prox = -np.inf
        dist = np.inf
        for q in range(k + 1, e):
            if d == 1:
                prox = max(prox, max(o[q], c[q]))
                dist = min(dist, l[q])
            else:
                prox = max(prox, -min(o[q], c[q]))
                dist = min(dist, -h[q])
        if d == -1:
            prox, dist = -prox, -dist
        height = d * (prox - dist)
        if height <= 0 or height > 2 * a or d * (c[e] - prox) < 0.5 * a:
            continue
        j, px = _fill(o, h, l, e + 1, min(n, e + 1 + max_wait), d, prox)
        if j < 0:
            continue
        sl = dist - d * BUF * a
        risk = d * (px - sl)
        if risk <= 0:
            continue
        e_idx[k_out] = j
        e_px[k_out] = px
        s_dir[k_out] = d
        s_sl[k_out] = sl
        s_tp[k_out] = px + d * rr * risk
        z0[k_out] = k + 1
        zlo[k_out] = min(prox, dist)
        zhi[k_out] = max(prox, dist)
        k_out += 1
    return (e_idx[:k_out], e_px[:k_out], s_dir[:k_out], s_sl[:k_out], s_tp[:k_out],
            z0[:k_out], zlo[:k_out], zhi[:k_out])


@njit(cache=True)
def structure(h, l, c, piv_hi, piv_lo):
    """Struktur pasar dari swing 3-candle.
    brk[i]   : +1 penutupan menembus swing high terakhir, -1 menembus swing low.
    kind[i]  : 1 = break of structure (searah tren), 2 = change of character (lawan tren).
    sweep[i] : +1 sumbu menembus swing low lalu ditutup kembali di atasnya
               (liquidity sweep / stop hunt bullish), -1 kebalikannya.
    trend[i] : arah struktur setelah candle i.
    leg[i]   : indeks swing lawan terakhir saat break terjadi (awal gelombang penembus).
    sh[i], sl[i] : level swing high / low terakhir yang berlaku di candle i."""
    n = len(c)
    brk = np.zeros(n, dtype=np.int64)
    kind = np.zeros(n, dtype=np.int64)
    sweep = np.zeros(n, dtype=np.int64)
    trend = np.zeros(n, dtype=np.int64)
    leg = np.full(n, -1, dtype=np.int64)
    sh_arr = np.full(n, np.nan)
    sl_arr = np.full(n, np.nan)
    sh = np.nan
    sl = np.nan
    sh_i = -1
    sl_i = -1
    sh_broken = True
    sl_broken = True
    sh_swept = True
    sl_swept = True
    t = 0
    for i in range(n):
        if piv_hi[i] >= 0:
            sh, sh_i = h[piv_hi[i]], piv_hi[i]
            sh_broken = False
            sh_swept = False
        if piv_lo[i] >= 0:
            sl, sl_i = l[piv_lo[i]], piv_lo[i]
            sl_broken = False
            sl_swept = False
        sh_arr[i] = sh
        sl_arr[i] = sl
        if not sh_broken and c[i] > sh:
            brk[i] = 1
            kind[i] = 2 if t == -1 else 1
            leg[i] = sl_i
            t = 1
            sh_broken = True
            sh_swept = True
        elif not sl_broken and c[i] < sl:
            brk[i] = -1
            kind[i] = 2 if t == 1 else 1
            leg[i] = sh_i
            t = -1
            sl_broken = True
            sl_swept = True
        else:
            if not sl_swept and l[i] < sl and c[i] > sl:
                sweep[i] = 1
                sl_swept = True
            elif not sh_swept and h[i] > sh and c[i] < sh:
                sweep[i] = -1
                sh_swept = True
        trend[i] = t
    return brk, kind, sweep, trend, leg, sh_arr, sl_arr


@njit(cache=True)
def fvg_signals(o, h, l, c, atr, trend, use_trend, rr, max_wait):
    """Fair value gap: celah antara candle i-2 dan candle i yang ditinggalkan
    candle impulsif i-1. Entry limit di tepi celah terdekat saat harga kembali,
    SL di balik candle pembentuk celah, TP = rr x risiko."""
    n = len(c)
    e_idx, e_px, s_dir, s_sl, s_tp, z0, zlo, zhi = _out(n)
    k_out = 0
    for i in range(2, n - 1):
        a = atr[i]
        if np.isnan(a):
            continue
        d = 0
        if l[i] > h[i - 2] + 0.25 * a:
            d = 1
        elif h[i] < l[i - 2] - 0.25 * a:
            d = -1
        if d == 0 or (use_trend and trend[i] != d):
            continue
        if d == 1:
            level, far = l[i], h[i - 2]
            sl = min(l[i - 1], l[i - 2]) - BUF * a
        else:
            level, far = h[i], l[i - 2]
            sl = max(h[i - 1], h[i - 2]) + BUF * a
        j, px = _fill(o, h, l, i + 1, min(n, i + 1 + max_wait), d, level)
        if j < 0:
            continue
        risk = d * (px - sl)
        if risk <= 0:
            continue
        e_idx[k_out] = j
        e_px[k_out] = px
        s_dir[k_out] = d
        s_sl[k_out] = sl
        s_tp[k_out] = px + d * rr * risk
        z0[k_out] = i - 1
        zlo[k_out] = min(level, far)
        zhi[k_out] = max(level, far)
        k_out += 1
    return (e_idx[:k_out], e_px[:k_out], s_dir[:k_out], s_sl[:k_out], s_tp[:k_out],
            z0[:k_out], zlo[:k_out], zhi[:k_out])


@njit(cache=True)
def ob_signals(o, h, l, c, atr, brk, leg, rr, max_wait):
    """Order block: candle berlawanan arah terakhir sebelum gelombang yang
    menembus struktur. Entry limit di tepi order block saat harga kembali,
    SL di balik order block."""
    n = len(c)
    e_idx, e_px, s_dir, s_sl, s_tp, z0, zlo, zhi = _out(n)
    k_out = 0
    for i in range(3, n - 1):
        d = brk[i]
        a = atr[i]
        if d == 0 or leg[i] < 0 or np.isnan(a):
            continue
        q = -1
        for k in range(i - 1, leg[i] - 1, -1):
            if d * (c[k] - o[k]) < 0:
                q = k
                break
        if q < 0 or h[q] - l[q] > 3 * a:
            continue
        level = h[q] if d == 1 else l[q]
        if d * (c[i] - level) <= 0:
            continue
        sl = (l[q] - BUF * a) if d == 1 else (h[q] + BUF * a)
        j, px = _fill(o, h, l, i + 1, min(n, i + 1 + max_wait), d, level)
        if j < 0:
            continue
        risk = d * (px - sl)
        if risk <= 0:
            continue
        e_idx[k_out] = j
        e_px[k_out] = px
        s_dir[k_out] = d
        s_sl[k_out] = sl
        s_tp[k_out] = px + d * rr * risk
        z0[k_out] = q
        zlo[k_out] = l[q]
        zhi[k_out] = h[q]
        k_out += 1
    return (e_idx[:k_out], e_px[:k_out], s_dir[:k_out], s_sl[:k_out], s_tp[:k_out],
            z0[:k_out], zlo[:k_out], zhi[:k_out])


@njit(cache=True)
def ict2022_signals(o, h, l, c, atr, sweep, sh, sl, rr, mss_window, max_wait):
    """Model ICT 2022:
    1. liquidity sweep (sumbu menembus swing lalu ditutup kembali),
    2. market structure shift: dalam `mss_window` candle, harga DITUTUP menembus
       swing di sisi berlawanan,
    3. gelombang penembus itu meninggalkan fair value gap,
    4. entry limit di tepi gap, SL di balik ujung sweep, TP = rr x risiko."""
    n = len(c)
    e_idx, e_px, s_dir, s_sl, s_tp, z0, zlo, zhi = _out(n)
    k_out = 0
    for s in range(3, n - 3):
        d = sweep[s]
        a = atr[s]
        if d == 0 or np.isnan(a):
            continue
        target = sh[s] if d == 1 else sl[s]
        if np.isnan(target):
            continue
        extreme = l[s] if d == 1 else h[s]
        m = -1
        for j in range(s + 1, min(n, s + 1 + mss_window)):
            if d == 1:
                if l[j] < extreme:
                    extreme = l[j]
                if c[j] > target:
                    m = j
                    break
            else:
                if h[j] > extreme:
                    extreme = h[j]
                if c[j] < target:
                    m = j
                    break
        if m < 0:
            continue
        g = -1
        for t in range(m, s + 1, -1):
            if d == 1 and l[t] > h[t - 2]:
                g = t
                break
            if d == -1 and h[t] < l[t - 2]:
                g = t
                break
        if g < 0:
            continue
        level = l[g] if d == 1 else h[g]
        far = h[g - 2] if d == 1 else l[g - 2]
        j, px = _fill(o, h, l, m + 1, min(n, m + 1 + max_wait), d, level)
        if j < 0:
            continue
        stop = extreme - d * BUF * a
        risk = d * (px - stop)
        if risk <= 0:
            continue
        e_idx[k_out] = j
        e_px[k_out] = px
        s_dir[k_out] = d
        s_sl[k_out] = stop
        s_tp[k_out] = px + d * rr * risk
        z0[k_out] = s
        zlo[k_out] = min(level, far)
        zhi[k_out] = max(level, far)
        k_out += 1
    return (e_idx[:k_out], e_px[:k_out], s_dir[:k_out], s_sl[:k_out], s_tp[:k_out],
            z0[:k_out], zlo[:k_out], zhi[:k_out])
