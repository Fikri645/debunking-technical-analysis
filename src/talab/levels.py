"""Support/resistance, trendline, dan Fibonacci retracement.

Semua memakai swing zigzag yang sudah terkonfirmasi (lihat patterns.zigzag),
jadi level dan garis hanya dibentuk dari informasi yang tersedia saat itu.
"""

from __future__ import annotations

import numpy as np
from numba import njit

NEAR = 0.25   # "menyentuh" level = dalam 0,25 ATR
CLUSTER = 0.5  # swing dalam 0,5 ATR dianggap satu level
SL_ATR = 1.0   # SL 1 ATR di balik level/garis
RR = 2.0


@njit(cache=True)
def sr_signals(o, h, l, c, atr, p_idx, p_px, p_conf, lookback, min_touch, mode):
    """Level horizontal = kumpulan >= min_touch swing yang berdekatan.
    mode 0: pantulan (candle menyentuh level lalu ditutup menjauh)
    mode 1: breakout (penutupan menembus level)
    mode 2: breakout lalu retest (harga kembali menyentuh level yang baru ditembus)"""
    n = len(c)
    m = len(p_idx)
    s_idx = np.empty(n, dtype=np.int64)
    s_dir = np.empty(n, dtype=np.int64)
    s_sl = np.empty(n)
    s_tp = np.empty(n)
    k_out = 0
    lev = np.empty(m)
    n_lev = 0
    first = 0   # swing aktif tertua
    nxt = 0     # swing berikutnya yang belum terkonfirmasi
    bo_level = np.nan
    bo_bar = -1
    bo_dir = 0
    for i in range(1, n):
        a = atr[i]
        if np.isnan(a):
            continue
        changed = False
        while nxt < m and p_conf[nxt] <= i:
            nxt += 1
            changed = True
        while first < nxt and p_idx[first] < i - lookback:
            first += 1
            changed = True
        if changed:
            n_lev = 0
            for q in range(first, nxt):
                cnt = 0
                tot = 0.0
                for r in range(first, nxt):
                    if abs(p_px[r] - p_px[q]) <= CLUSTER * a:
                        cnt += 1
                        tot += p_px[r]
                if cnt >= min_touch:
                    center = tot / cnt
                    dup = False
                    for z in range(n_lev):
                        if abs(lev[z] - center) <= CLUSTER * a:
                            dup = True
                            break
                    if not dup:
                        lev[n_lev] = center
                        n_lev += 1
        direction = 0
        level = 0.0
        if mode == 2:
            if bo_bar >= 0 and i > bo_bar and i - bo_bar <= 20:
                if bo_dir == 1 and l[i] <= bo_level + NEAR * a and c[i] > bo_level and c[i] > o[i]:
                    direction, level = 1, bo_level
                elif bo_dir == -1 and h[i] >= bo_level - NEAR * a and c[i] < bo_level and c[i] < o[i]:
                    direction, level = -1, bo_level
                if direction != 0:
                    bo_bar = -1
        for z in range(n_lev):
            L = lev[z]
            if mode == 0 and direction == 0:
                if l[i] <= L + NEAR * a and c[i] > L and c[i] > o[i] and c[i - 1] > L:
                    direction, level = 1, L
                elif h[i] >= L - NEAR * a and c[i] < L and c[i] < o[i] and c[i - 1] < L:
                    direction, level = -1, L
            elif mode >= 1:
                up = c[i - 1] <= L and c[i] > L + NEAR * a
                dn = c[i - 1] >= L and c[i] < L - NEAR * a
                if up or dn:
                    if mode == 1 and direction == 0:
                        direction, level = (1 if up else -1), L
                    if mode == 2:
                        bo_level, bo_bar, bo_dir = L, i, (1 if up else -1)
        if direction != 0:
            sl = level - direction * SL_ATR * a
            risk = direction * (c[i] - sl)
            if risk > 0:
                s_idx[k_out] = i
                s_dir[k_out] = direction
                s_sl[k_out] = sl
                s_tp[k_out] = c[i] + direction * RR * risk
                k_out += 1
    return s_idx[:k_out], s_dir[:k_out], s_sl[:k_out], s_tp[:k_out]


@njit(cache=True)
def trendline_signals(o, h, l, c, atr, p_idx, p_px, p_type, p_conf, mode):
    """Garis tren naik = garis lewat dua swing low terakhir yang menanjak.
    Garis tren turun = garis lewat dua swing high terakhir yang menurun.
    mode 0: pantulan di garis (searah tren). mode 1: garis tembus (lawan tren)."""
    n = len(c)
    m = len(p_idx)
    s_idx = np.empty(n, dtype=np.int64)
    s_dir = np.empty(n, dtype=np.int64)
    s_sl = np.empty(n)
    s_tp = np.empty(n)
    k_out = 0
    nxt = 0
    lo_i = np.array([-1, -1])
    lo_p = np.array([0.0, 0.0])
    hi_i = np.array([-1, -1])
    hi_p = np.array([0.0, 0.0])
    up_ok = False
    dn_ok = False
    for i in range(1, n):
        a = atr[i]
        if np.isnan(a):
            continue
        while nxt < m and p_conf[nxt] <= i:
            if p_type[nxt] == -1:
                lo_i[0], lo_p[0] = lo_i[1], lo_p[1]
                lo_i[1], lo_p[1] = p_idx[nxt], p_px[nxt]
                up_ok = lo_i[0] >= 0 and lo_p[1] > lo_p[0]
            else:
                hi_i[0], hi_p[0] = hi_i[1], hi_p[1]
                hi_i[1], hi_p[1] = p_idx[nxt], p_px[nxt]
                dn_ok = hi_i[0] >= 0 and hi_p[1] < hi_p[0]
            nxt += 1
        direction = 0
        line = 0.0
        if up_ok:
            ln = lo_p[1] + (lo_p[1] - lo_p[0]) / (lo_i[1] - lo_i[0]) * (i - lo_i[1])
            if c[i] < ln - NEAR * a:
                up_ok = False
                if mode == 1:
                    direction, line = -1, ln
            elif mode == 0 and l[i] <= ln + NEAR * a and c[i] > ln and c[i] > o[i]:
                direction, line = 1, ln
        if dn_ok and direction == 0:
            ln = hi_p[1] + (hi_p[1] - hi_p[0]) / (hi_i[1] - hi_i[0]) * (i - hi_i[1])
            if c[i] > ln + NEAR * a:
                dn_ok = False
                if mode == 1:
                    direction, line = 1, ln
            elif mode == 0 and h[i] >= ln - NEAR * a and c[i] < ln and c[i] < o[i]:
                direction, line = -1, ln
        if direction != 0:
            sl = line - direction * SL_ATR * a
            risk = direction * (c[i] - sl)
            if risk > 0:
                s_idx[k_out] = i
                s_dir[k_out] = direction
                s_sl[k_out] = sl
                s_tp[k_out] = c[i] + direction * RR * risk
                k_out += 1
    return s_idx[:k_out], s_dir[:k_out], s_sl[:k_out], s_tp[:k_out]


@njit(cache=True)
def fib_signals(o, h, l, p_idx, p_px, p_type, p_conf, level, ext):
    """Limit order di retracement `level` dari swing terakhir.
    Swing naik A->B: beli di B - level x (B - A), SL sedikit di bawah A,
    TP di B (ext = 0) atau di ekstensi B + ext x (B - A).
    Order batal jika harga membuat ekstrem baru melewati B sebelum terisi."""
    n = len(o)
    m = len(p_idx)
    e_idx = np.empty(m, dtype=np.int64)
    e_px = np.empty(m)
    s_dir = np.empty(m, dtype=np.int64)
    s_sl = np.empty(m)
    s_tp = np.empty(m)
    k_out = 0
    for k in range(1, m):
        d = p_type[k]  # +1: swing naik berakhir di high -> long
        a, b = p_px[k - 1], p_px[k]
        leg = d * (b - a)
        if leg <= 0:
            continue
        entry = b - d * level * leg
        sl = a - d * 0.05 * leg
        tp = b + d * ext * leg
        conf = p_conf[k]
        passed = False
        for j in range(p_idx[k] + 1, conf + 1):
            if (d == 1 and l[j] <= entry) or (d == -1 and h[j] >= entry):
                passed = True
                break
        if passed:
            continue
        end = min(n, conf + 1 + max(10, 3 * (p_idx[k] - p_idx[k - 1])))
        for j in range(conf + 1, end):
            if d == 1:
                if l[j] < entry:      # harus menembus, bukan sekadar menyentuh
                    e_idx[k_out] = j
                    e_px[k_out] = min(o[j], entry)
                    s_dir[k_out] = 1
                    s_sl[k_out] = sl
                    s_tp[k_out] = tp
                    k_out += 1
                    break
                if h[j] > b:
                    break
            else:
                if h[j] > entry:
                    e_idx[k_out] = j
                    e_px[k_out] = max(o[j], entry)
                    s_dir[k_out] = -1
                    s_sl[k_out] = sl
                    s_tp[k_out] = tp
                    k_out += 1
                    break
                if l[j] < b:
                    break
    return e_idx[:k_out], e_px[:k_out], s_dir[:k_out], s_sl[:k_out], s_tp[:k_out]


@njit(cache=True)
def swing_break_signals(c, p_idx, p_px, p_type, p_conf, window):
    """Kontrol tanpa pola: penutupan pertama yang menembus swing terakhir.
    Tembus di atas swing high -> long, tembus di bawah swing low -> short.
    Tidak ada syarat bentuk apa pun."""
    n = len(c)
    m = len(p_idx)
    s_idx = np.empty(m, dtype=np.int64)
    s_dir = np.empty(m, dtype=np.int64)
    k_out = 0
    for k in range(m):
        d = p_type[k]
        end = min(n, p_conf[k] + 1 + window)
        for j in range(p_conf[k] + 1, end):
            if d * (c[j] - p_px[k]) > 0:
                s_idx[k_out] = j
                s_dir[k_out] = d
                k_out += 1
                break
    return s_idx[:k_out], s_dir[:k_out]
