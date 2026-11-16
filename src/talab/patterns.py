"""Pola chart.

Definisi B ("seperti mata trader"):
- Swing high/low dicari dengan zigzag berbasis ATR pada beberapa skala. Swing baru
  dianggap ada setelah harga berbalik sejauh `mult` x ATR dari titik ekstremnya,
  jadi setiap pola hanya memakai informasi yang tersedia saat itu.
- Bentuk pola dinilai secara proporsional terhadap tinggi pola, dengan tiga
  tingkat toleransi: 0 = ketat, 1 = sedang, 2 = longgar.
- Sinyal muncul saat harga DITUTUP menembus garis leher / batas pola.
- SL dan TP mengikuti ajaran umum: SL di ujung pola, TP = tinggi pola
  diproyeksikan dari titik tembus ("measured move").

Tiap detektor ditulis untuk satu arah saja. Arah sebaliknya didapat dengan
mencerminkan harga (dikali -1), sehingga pola bullish dan bearish memakai aturan
yang persis sama.

Definisi A (Lo, Mamaysky & Wang 2000) ada di bagian bawah.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from . import indicators as ta

SCALES = (2.0, 4.0, 8.0)
NPTS = 6


@njit(cache=True)
def zigzag(h, l, c, atr, mult):
    """Swing bergantian. Mengembalikan indeks, harga, tipe (+1 high, -1 low) dan
    indeks candle saat swing itu terkonfirmasi."""
    n = len(c)
    p_idx = np.empty(n, dtype=np.int64)
    p_px = np.empty(n)
    p_type = np.empty(n, dtype=np.int64)
    p_conf = np.empty(n, dtype=np.int64)
    m = 0
    d = 0
    hi = -np.inf
    lo = np.inf
    hi_i = -1
    lo_i = -1
    for i in range(n):
        a = atr[i]
        if np.isnan(a):
            continue
        if d >= 0 and h[i] > hi:
            hi = h[i]
            hi_i = i
        if d <= 0 and l[i] < lo:
            lo = l[i]
            lo_i = i
        if d >= 0 and hi_i >= 0 and i > hi_i and c[i] < hi - mult * a:
            p_idx[m] = hi_i
            p_px[m] = hi
            p_type[m] = 1
            p_conf[m] = i
            m += 1
            d = -1
            lo = np.inf
            for j in range(hi_i + 1, i + 1):
                if l[j] < lo:
                    lo = l[j]
                    lo_i = j
        elif d <= 0 and lo_i >= 0 and i > lo_i and c[i] > lo + mult * a:
            p_idx[m] = lo_i
            p_px[m] = lo
            p_type[m] = -1
            p_conf[m] = i
            m += 1
            d = 1
            hi = -np.inf
            for j in range(lo_i + 1, i + 1):
                if h[j] > hi:
                    hi = h[j]
                    hi_i = j
    return p_idx[:m], p_px[:m], p_type[:m], p_conf[:m]


@njit(cache=True)
def _alloc(m):
    return (np.empty(m, dtype=np.int64), np.empty(m, dtype=np.int64), np.empty(m), np.empty(m),
            np.full((m, NPTS), -1, dtype=np.int64), np.full((m, NPTS), np.nan))


# ----- toleransi: [ketat, sedang, longgar] -----
HS_HEAD = np.array([0.25, 0.15, 0.05])      # kepala lebih tinggi dari bahu, x tinggi pola
HS_SYM = np.array([0.20, 0.35, 0.60])       # selisih tinggi kedua bahu
HS_NECK = np.array([0.20, 0.35, 0.60])      # kemiringan garis leher
HS_ABOVE = np.array([0.30, 0.20, 0.10])     # bahu di atas garis leher
HS_TIME = np.array([2.0, 3.0, 5.0])         # rasio lebar kiri : kanan
TREND = np.array([0.50, 0.25, 0.00])        # tren sebelum pola
EQ = np.array([0.10, 0.20, 0.35])           # dua/tiga puncak "sama tinggi"
MIN_SEP = np.array([15, 10, 5])             # jarak minimum antar puncak (candle)
FLAT = np.array([0.10, 0.15, 0.25])         # garis "datar"
RISE = np.array([0.30, 0.20, 0.15])         # garis "miring"
CONV = np.array([0.15, 0.10, 0.05])         # wedge: selisih kemiringan
FIFTH = np.array([0.12, 0.25, 9.99])        # jarak sentuhan kelima dari garisnya
FLAG_STEEP = np.array([10, 15, 25])         # durasi maksimum tiang (candle)
FLAG_RETR = np.array([0.38, 0.50, 0.62])    # koreksi maksimum terhadap tiang
CUP_RIM = np.array([0.15, 0.25, 0.40])
CUP_MINW = np.array([30, 20, 12])
CUP_ROUND = np.array([0.25, 0.15, 0.00])
CUP_CENTER = np.array([0.25, 0.20, 0.10])
CUP_HANDLE = np.array([0.35, 0.50, 0.60])
LATE = 0.5                                   # tembus sudah > 50% target: terlambat, lewati


@njit(cache=True)
def det_hs(p_idx, p_px, p_type, p_conf, c, hi, lo, t):
    """Head and shoulders (puncak) -> short saat garis leher tembus."""
    n = len(c)
    m = len(p_idx)
    s_idx, s_dir, s_sl, s_tp, pts_i, pts_p = _alloc(m)
    k_out = 0
    for k in range(5, m):
        if p_type[k] != 1:
            continue
        t0, p1, t1, p2, t2, p3 = p_px[k - 5], p_px[k - 4], p_px[k - 3], p_px[k - 2], p_px[k - 1], p_px[k]
        i1, it1, i2, it2, i3 = p_idx[k - 4], p_idx[k - 3], p_idx[k - 2], p_idx[k - 1], p_idx[k]
        height = p2 - (t1 + t2) / 2
        if height <= 0:
            continue
        if p2 - max(p1, p3) < HS_HEAD[t] * height:
            continue
        if abs(p1 - p3) > HS_SYM[t] * height:
            continue
        if abs(t1 - t2) > HS_NECK[t] * height:
            continue
        if min(p1, p3) - max(t1, t2) < HS_ABOVE[t] * height:
            continue
        left, right = i2 - i1, i3 - i2
        if left <= 0 or right <= 0 or left > HS_TIME[t] * right or right > HS_TIME[t] * left:
            continue
        if t0 > min(t1, t2) - TREND[t] * height:
            continue
        slope = (t2 - t1) / (it2 - it1)
        width = i3 - i1
        end = min(n, p_conf[k] + max(5, width // 2))
        for j in range(p_conf[k], end):
            if c[j] > p3:
                break
            neck = t2 + slope * (j - it2)
            if c[j] < neck:
                if neck - c[j] <= LATE * height:
                    s_idx[k_out] = j
                    s_dir[k_out] = -1
                    s_sl[k_out] = p3
                    s_tp[k_out] = neck - height
                    for q in range(6):
                        pts_i[k_out, q] = p_idx[k - 5 + q]
                        pts_p[k_out, q] = p_px[k - 5 + q]
                    k_out += 1
                break
    return s_idx[:k_out], s_dir[:k_out], s_sl[:k_out], s_tp[:k_out], pts_i[:k_out], pts_p[:k_out]


@njit(cache=True)
def det_double(p_idx, p_px, p_type, p_conf, c, hi, lo, t):
    """Double top -> short saat lembah di antara dua puncak tembus."""
    n = len(c)
    m = len(p_idx)
    s_idx, s_dir, s_sl, s_tp, pts_i, pts_p = _alloc(m)
    k_out = 0
    for k in range(3, m):
        if p_type[k] != 1:
            continue
        t0, p1, t1, p2 = p_px[k - 3], p_px[k - 2], p_px[k - 1], p_px[k]
        i1, i2 = p_idx[k - 2], p_idx[k]
        top = max(p1, p2)
        height = top - t1
        if height <= 0 or abs(p1 - p2) > EQ[t] * height:
            continue
        if i2 - i1 < MIN_SEP[t]:
            continue
        if t0 > t1 - TREND[t] * height:
            continue
        end = min(n, p_conf[k] + max(5, i2 - i1))
        for j in range(p_conf[k], end):
            if c[j] > top:
                break
            if c[j] < t1:
                if t1 - c[j] <= LATE * height:
                    s_idx[k_out] = j
                    s_dir[k_out] = -1
                    s_sl[k_out] = top
                    s_tp[k_out] = t1 - height
                    for q in range(4):
                        pts_i[k_out, q] = p_idx[k - 3 + q]
                        pts_p[k_out, q] = p_px[k - 3 + q]
                    k_out += 1
                break
    return s_idx[:k_out], s_dir[:k_out], s_sl[:k_out], s_tp[:k_out], pts_i[:k_out], pts_p[:k_out]


@njit(cache=True)
def det_triple(p_idx, p_px, p_type, p_conf, c, hi, lo, t):
    """Triple top -> short saat support tembus."""
    n = len(c)
    m = len(p_idx)
    s_idx, s_dir, s_sl, s_tp, pts_i, pts_p = _alloc(m)
    k_out = 0
    for k in range(5, m):
        if p_type[k] != 1:
            continue
        t0, p1, t1, p2, t2, p3 = p_px[k - 5], p_px[k - 4], p_px[k - 3], p_px[k - 2], p_px[k - 1], p_px[k]
        i1, i3 = p_idx[k - 4], p_idx[k]
        top = max(p1, max(p2, p3))
        bottom_top = min(p1, min(p2, p3))
        support = min(t1, t2)
        height = top - support
        if height <= 0 or top - bottom_top > EQ[t] * height:
            continue
        if abs(t1 - t2) > 2 * EQ[t] * height:
            continue
        if t0 > support - TREND[t] * height:
            continue
        end = min(n, p_conf[k] + max(5, (i3 - i1) // 2))
        for j in range(p_conf[k], end):
            if c[j] > top:
                break
            if c[j] < support:
                if support - c[j] <= LATE * height:
                    s_idx[k_out] = j
                    s_dir[k_out] = -1
                    s_sl[k_out] = top
                    s_tp[k_out] = support - height
                    for q in range(6):
                        pts_i[k_out, q] = p_idx[k - 5 + q]
                        pts_p[k_out, q] = p_px[k - 5 + q]
                    k_out += 1
                break
    return s_idx[:k_out], s_dir[:k_out], s_sl[:k_out], s_tp[:k_out], pts_i[:k_out], pts_p[:k_out]


@njit(cache=True)
def det_lines(p_idx, p_px, p_type, p_conf, c, hi, lo, t, mode):
    """Pola dua garis dari empat swing terakhir.
    mode 0: ascending triangle (atas datar, bawah naik) -> long saat tembus atas
    mode 1: symmetrical triangle -> ikut arah tembus
    mode 2: rectangle -> ikut arah tembus
    mode 3: rising wedge (dua garis naik, menyempit) -> short saat tembus bawah
    Tingkat ketat dan sedang mensyaratkan swing kelima menyentuh garisnya."""
    n = len(c)
    m = len(p_idx)
    s_idx, s_dir, s_sl, s_tp, pts_i, pts_p = _alloc(m)
    k_out = 0
    for k in range(4, m):
        if p_type[k] == 1:
            ih1, ih2, il1, il2 = p_idx[k - 2], p_idx[k], p_idx[k - 3], p_idx[k - 1]
            h1, h2, l1, l2 = p_px[k - 2], p_px[k], p_px[k - 3], p_px[k - 1]
        else:
            ih1, ih2, il1, il2 = p_idx[k - 3], p_idx[k - 1], p_idx[k - 2], p_idx[k]
            h1, h2, l1, l2 = p_px[k - 3], p_px[k - 1], p_px[k - 2], p_px[k]
        height = max(h1, h2) - min(l1, l2)
        if height <= 0 or ih2 == ih1 or il2 == il1:
            continue
        dh = (h2 - h1) / height
        dl = (l2 - l1) / height
        ok = False
        if mode == 0:
            ok = abs(dh) <= FLAT[t] and dl >= RISE[t]
        elif mode == 1:
            ok = dh <= -RISE[t] and dl >= RISE[t]
        elif mode == 2:
            ok = abs(dh) <= FLAT[t] and abs(dl) <= FLAT[t]
        else:
            ok = dh >= RISE[t] and dl >= RISE[t] and dl - dh >= CONV[t]
        if not ok:
            continue
        su = (h2 - h1) / (ih2 - ih1)
        sl_ = (l2 - l1) / (il2 - il1)
        first = min(ih1, il1)
        last = p_idx[k]
        if (h2 + su * (last - ih2)) - (l2 + sl_ * (last - il2)) <= 0.1 * height:
            continue
        gap_first = (h2 + su * (first - ih2)) - (l2 + sl_ * (first - il2))
        gap_last = (h2 + su * (last - ih2)) - (l2 + sl_ * (last - il2))
        if (mode == 1 or mode == 3) and gap_last > 0.8 * gap_first:
            continue  # triangle simetris dan wedge harus benar-benar menyempit
        if t < 2:
            # sentuhan kelima: swing sebelum keempat titik harus jatuh di garisnya
            q = k - 4
            line = (h2 + su * (p_idx[q] - ih2)) if p_type[q] == 1 else (l2 + sl_ * (p_idx[q] - il2))
            if abs(p_px[q] - line) > FIFTH[t] * height:
                continue
            first = p_idx[q]
        width = last - first
        end = min(n, p_conf[k] + max(5, width))
        for j in range(p_conf[k], end):
            up = h2 + su * (j - ih2)
            dn = l2 + sl_ * (j - il2)
            if up - dn <= 0:
                break
            direction = 0
            if c[j] > up:
                direction = 1
            elif c[j] < dn:
                direction = -1
            if direction == 0:
                continue
            allowed = (mode == 0 and direction == 1) or mode == 1 or mode == 2 \
                or (mode == 3 and direction == -1)
            dist = (c[j] - up) if direction == 1 else (dn - c[j])
            if allowed and dist <= LATE * height:
                s_idx[k_out] = j
                s_dir[k_out] = direction
                s_sl[k_out] = dn if direction == 1 else up
                s_tp[k_out] = c[j] + direction * height
                q0 = 4 if t < 2 else 3
                for q in range(q0 + 1):
                    pts_i[k_out, q] = p_idx[k - q0 + q]
                    pts_p[k_out, q] = p_px[k - q0 + q]
                k_out += 1
            break
    return s_idx[:k_out], s_dir[:k_out], s_sl[:k_out], s_tp[:k_out], pts_i[:k_out], pts_p[:k_out]


@njit(cache=True)
def det_flag(p_idx, p_px, p_type, p_conf, c, hi, lo, t):
    """Bull flag/pennant: tiang curam, koreksi dangkal, lalu tembus puncak tiang."""
    n = len(c)
    m = len(p_idx)
    s_idx, s_dir, s_sl, s_tp, pts_i, pts_p = _alloc(m)
    k_out = 0
    for k in range(1, m):
        if p_type[k] != 1:
            continue
        a, b = p_px[k - 1], p_px[k]
        ia, ib = p_idx[k - 1], p_idx[k]
        pole = b - a
        dur = ib - ia
        if pole <= 0 or dur <= 0 or dur > FLAG_STEEP[t]:
            continue
        cons_lo = np.inf
        for j in range(ib + 1, p_conf[k] + 1):
            if lo[j] < cons_lo:
                cons_lo = lo[j]
        end = min(n, ib + max(20, 2 * dur) + 1)
        for j in range(p_conf[k] + 1, end):
            if lo[j] < cons_lo:
                cons_lo = lo[j]
            if b - cons_lo > FLAG_RETR[t] * pole:
                break
            if c[j] > b:
                if j - ib >= 3 and c[j] - b <= LATE * pole:
                    s_idx[k_out] = j
                    s_dir[k_out] = 1
                    s_sl[k_out] = cons_lo
                    s_tp[k_out] = c[j] + pole
                    pts_i[k_out, 0] = ia
                    pts_p[k_out, 0] = a
                    pts_i[k_out, 1] = ib
                    pts_p[k_out, 1] = b
                    k_out += 1
                break
    return s_idx[:k_out], s_dir[:k_out], s_sl[:k_out], s_tp[:k_out], pts_i[:k_out], pts_p[:k_out]


@njit(cache=True)
def det_cup(p_idx, p_px, p_type, p_conf, c, hi, lo, t):
    """Cup and handle -> long saat bibir cangkir tembus."""
    n = len(c)
    m = len(p_idx)
    s_idx, s_dir, s_sl, s_tp, pts_i, pts_p = _alloc(m)
    k_out = 0
    for k in range(3, m):
        if p_type[k] != 1:
            continue
        t0, p1, t1, p2 = p_px[k - 3], p_px[k - 2], p_px[k - 1], p_px[k]
        i1, it1, i2 = p_idx[k - 2], p_idx[k - 1], p_idx[k]
        depth = min(p1, p2) - t1
        width = i2 - i1
        if depth <= 0 or abs(p1 - p2) > CUP_RIM[t] * depth or width < CUP_MINW[t]:
            continue
        pos = (it1 - i1) / width
        if pos < CUP_CENTER[t] or pos > 1 - CUP_CENTER[t]:
            continue
        if t < 2 and t0 > p1 - 0.5 * depth:
            continue
        low_zone = 0
        for j in range(i1, i2 + 1):
            if lo[j] < t1 + depth / 3:
                low_zone += 1
        if low_zone < CUP_ROUND[t] * width:
            continue
        handle_lo = np.inf
        for j in range(i2 + 1, p_conf[k] + 1):
            if lo[j] < handle_lo:
                handle_lo = lo[j]
        end = min(n, i2 + max(5, width // 2) + 1)
        for j in range(p_conf[k] + 1, end):
            if lo[j] < handle_lo:
                handle_lo = lo[j]
            if p2 - handle_lo > CUP_HANDLE[t] * depth:
                break
            if c[j] > p2:
                if j - i2 >= 3 and c[j] - p2 <= LATE * depth:
                    s_idx[k_out] = j
                    s_dir[k_out] = 1
                    s_sl[k_out] = handle_lo
                    s_tp[k_out] = c[j] + depth
                    for q in range(4):
                        pts_i[k_out, q] = p_idx[k - 3 + q]
                        pts_p[k_out, q] = p_px[k - 3 + q]
                    k_out += 1
                break
    return s_idx[:k_out], s_dir[:k_out], s_sl[:k_out], s_tp[:k_out], pts_i[:k_out], pts_p[:k_out]


def _lines(mode):
    def f(p_idx, p_px, p_type, p_conf, c, hi, lo, t):
        return det_lines(p_idx, p_px, p_type, p_conf, c, hi, lo, t, mode)
    return f


# nama -> (detektor, dicerminkan juga?)
DETECTORS = {
    "head_shoulders": (det_hs, True),
    "double_top_bottom": (det_double, True),
    "triple_top_bottom": (det_triple, True),
    "triangle_asc_desc": (_lines(0), True),
    "triangle_symmetrical": (_lines(1), False),
    "rectangle": (_lines(2), False),
    "wedge": (_lines(3), True),
    "flag_pennant": (det_flag, True),
    "cup_handle": (det_cup, True),
}

TOL_NAMES = ("ketat", "sedang", "longgar")


def detect(df: pd.DataFrame, pattern: str, tol: int, scales=SCALES) -> pd.DataFrame:
    """Semua kemunculan pola. Kolom: idx (candle sinyal), dir, sl, tp, scale, mirrored,
    serta titik-titik swing pembentuk pola (pi0..pi5, pp0..pp5) untuk audit visual."""
    o, h, l, c = (df[k].values.astype(float) for k in ("open", "high", "low", "close"))
    a = ta.atr(h, l, c)
    det, mirror = DETECTORS[pattern]
    frames = []
    for mult in scales:
        zz = zigzag(h, l, c, a, mult)
        if len(zz[0]) < 6:
            continue
        variants = [(1.0, zz[1], zz[2], c, h, l)]
        if mirror:
            variants.append((-1.0, -zz[1], -zz[2], -c, -l, -h))
        for sign, px, typ, cc, hh, ll in variants:
            s_idx, s_dir, s_sl, s_tp, pts_i, pts_p = det(zz[0], px, typ, zz[3], cc, hh, ll, tol)
            if len(s_idx) == 0:
                continue
            f = pd.DataFrame({"idx": s_idx, "dir": (s_dir * sign).astype(np.int64),
                              "sl": s_sl * sign, "tp": s_tp * sign,
                              "scale": mult, "mirrored": sign < 0})
            for q in range(NPTS):
                f[f"pi{q}"] = pts_i[:, q]
                f[f"pp{q}"] = pts_p[:, q] * sign
            frames.append(f)
    if not frames:
        return pd.DataFrame(columns=["idx", "dir", "sl", "tp", "scale", "mirrored"])
    return pd.concat(frames, ignore_index=True).sort_values("idx", kind="stable").reset_index(drop=True)


# =====================================================================
# Definisi A: Lo, Mamaysky & Wang (2000)
# =====================================================================

LMW_NAMES = {1: "HS", 2: "IHS", 3: "BTOP", 4: "BBOT", 5: "TTOP", 6: "TBOT",
             7: "RTOP", 8: "RBOT", 9: "DTOP", 10: "DBOT"}
_BW = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])


@njit(cache=True)
def _nw(y, bw, out):
    n = len(y)
    for i in range(n):
        num = 0.0
        den = 0.0
        for j in range(n):
            w = np.exp(-0.5 * ((i - j) / bw) ** 2)
            num += w * y[j]
            den += w
        out[i] = num / den


@njit(cache=True)
def _cv_bandwidth(y):
    n = len(y)
    best = _BW[0]
    best_err = np.inf
    for b in range(len(_BW)):
        bw = _BW[b]
        err = 0.0
        for i in range(n):
            num = 0.0
            den = 0.0
            for j in range(n):
                if j == i:
                    continue
                w = np.exp(-0.5 * ((i - j) / bw) ** 2)
                num += w * y[j]
                den += w
            e = y[i] - num / den
            err += e * e
        if err < best_err:
            best_err = err
            best = bw
    return best


@njit(cache=True)
def lmw(c, window=35, lag=3):
    """Kode pola per candle (0 = tidak ada). Jendela `window`+`lag` candle;
    harga dihaluskan dengan kernel regression (bandwidth 0,3 x hasil cross-validation),
    lalu lima titik ekstrem terakhir diperiksa terhadap definisi di makalah.
    Titik ekstrem terakhir harus berada di sekitar akhir jendela `window`, sehingga
    pola baru dikenali `lag` candle setelah terbentuk."""
    n = len(c)
    out = np.zeros(n, dtype=np.int64)
    w = window + lag
    sm = np.empty(w)
    e_pos = np.empty(w, dtype=np.int64)
    e_px = np.empty(w)
    e_typ = np.empty(w, dtype=np.int64)
    for i in range(w - 1, n):
        y = c[i - w + 1: i + 1]
        bw = 0.3 * _cv_bandwidth(y)
        _nw(y, bw, sm)
        ne = 0
        for x in range(1, w - 1):
            typ = 0
            if sm[x] > sm[x - 1] and sm[x] >= sm[x + 1]:
                typ = 1
            elif sm[x] < sm[x - 1] and sm[x] <= sm[x + 1]:
                typ = -1
            if typ == 0:
                continue
            best = x
            for q in (x - 1, x + 1):
                if typ == 1 and y[q] > y[best]:
                    best = q
                if typ == -1 and y[q] < y[best]:
                    best = q
            if ne > 0 and e_typ[ne - 1] == typ:
                if (typ == 1 and y[best] > e_px[ne - 1]) or (typ == -1 and y[best] < e_px[ne - 1]):
                    e_pos[ne - 1] = best
                    e_px[ne - 1] = y[best]
                continue
            e_pos[ne] = best
            e_px[ne] = y[best]
            e_typ[ne] = typ
            ne += 1
        if ne < 5:
            continue
        if abs(e_pos[ne - 1] - (window - 1)) > 1:
            continue
        e1, e2, e3, e4, e5 = e_px[ne - 5], e_px[ne - 4], e_px[ne - 3], e_px[ne - 2], e_px[ne - 1]
        top = e_typ[ne - 5] == 1
        s = 1.0 if top else -1.0  # untuk pola "bottom", bandingkan dengan tanda terbalik
        a1, a2, a3, a4, a5 = s * e1, s * e2, s * e3, s * e4, s * e5
        avg15 = (e1 + e5) / 2
        avg24 = (e2 + e4) / 2
        code = 0
        if a3 > a1 and a3 > a5 and abs(e1 - avg15) <= 0.015 * avg15 and abs(e5 - avg15) <= 0.015 * avg15 \
                and abs(e2 - avg24) <= 0.015 * avg24 and abs(e4 - avg24) <= 0.015 * avg24:
            code = 1
        elif a1 < a3 and a3 < a5 and a2 > a4:
            code = 3
        elif a1 > a3 and a3 > a5 and a2 < a4:
            code = 5
        else:
            avg_t = (e1 + e3 + e5) / 3
            avg_b = (e2 + e4) / 2
            if abs(e1 - avg_t) <= 0.0075 * avg_t and abs(e3 - avg_t) <= 0.0075 * avg_t \
                    and abs(e5 - avg_t) <= 0.0075 * avg_t and abs(e2 - avg_b) <= 0.0075 * avg_b \
                    and abs(e4 - avg_b) <= 0.0075 * avg_b \
                    and min(a1, min(a3, a5)) > max(a2, a4):
                code = 7
        if code == 0:
            # double top/bottom: dua ekstrem sejenis terpisah > 22 candle, selisih <= 1,5%
            last_typ = e_typ[ne - 1]
            for q in range(ne - 2, -1, -1):
                if e_typ[q] != last_typ or e_pos[ne - 1] - e_pos[q] <= 22:
                    continue
                avg = (e_px[q] + e_px[ne - 1]) / 2
                if abs(e_px[q] - avg) > 0.015 * avg:
                    continue
                extreme = True
                bound = max(last_typ * e_px[q], last_typ * e_px[ne - 1])
                for r in range(q + 1, ne - 1):
                    if e_typ[r] == last_typ and last_typ * e_px[r] > bound:
                        extreme = False
                if extreme:
                    out[i] = 9 if last_typ == 1 else 10
                break
            continue
        out[i] = code if top else code + 1
    return out
