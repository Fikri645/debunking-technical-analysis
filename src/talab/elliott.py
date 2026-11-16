"""Elliott Wave.

Gelombang dihitung di atas swing zigzag (patterns.zigzag). Detektor ditulis
untuk impuls naik; impuls turun didapat dengan mencerminkan harga.

Tiga aturan wajib:
  1. Wave 2 tidak turun melewati awal wave 1.
  2. Wave 3 bukan yang terpendek di antara wave 1, 3, 5 (dan melewati puncak wave 1).
  3. Wave 4 tidak masuk ke wilayah wave 1.
Pedoman Fibonacci (dipakai di varian `fib`):
  - wave 2 mengoreksi 38,2%-78,6% dari wave 1,
  - wave 3 >= 1,618 x wave 1,
  - wave 4 mengoreksi 23,6%-50% dari wave 3.

Tiga setup yang diajarkan:
  setup 0  "kejar wave 3": beli saat wave 2 selesai. SL di awal wave 1,
           TP = akhir wave 2 + 1,618 x wave 1.
  setup 1  "kejar wave 5": beli saat wave 4 selesai. SL di puncak wave 1,
           TP = akhir wave 4 + panjang wave 1.
  setup 2  "koreksi ABC": jual saat lima gelombang lengkap. SL di puncak wave 5,
           TP = akhir wave 4.
Sinyal muncul di candle saat swing terakhir terkonfirmasi, jadi hitungan hanya
memakai data sampai saat itu.
"""

from __future__ import annotations

import numpy as np
from numba import njit

NPTS = 6


@njit(cache=True)
def ew_signals(p_idx, p_px, p_type, p_conf, setup, use_fib):
    m = len(p_idx)
    s_idx = np.empty(m, dtype=np.int64)
    s_dir = np.empty(m, dtype=np.int64)
    s_sl = np.empty(m)
    s_tp = np.empty(m)
    pts_i = np.full((m, NPTS), -1, dtype=np.int64)
    pts_p = np.full((m, NPTS), np.nan)
    k_out = 0
    need = 2 if setup == 0 else (4 if setup == 1 else 5)
    for k in range(need + 2, m):
        want_type = 1 if setup == 2 else -1
        if p_type[k] != want_type:
            continue
        b = k - need  # indeks P0
        p0, p1, p2 = p_px[b], p_px[b + 1], p_px[b + 2]
        # wave 1 harus memulai tren baru: P0 adalah low yang lebih rendah dari low sebelumnya
        if p0 >= p_px[b - 2]:
            continue
        len1 = p1 - p0
        if len1 <= 0 or p2 <= p0:
            continue
        retr2 = (p1 - p2) / len1
        fib_ok = 0
        if 0.382 <= retr2 <= 0.786:
            fib_ok += 1
        direction = 1
        sl = 0.0
        tp = 0.0
        if setup == 0:
            if use_fib and fib_ok < 1:
                continue
            sl = p0
            tp = p2 + 1.618 * len1
        else:
            p3, p4 = p_px[b + 3], p_px[b + 4]
            len3 = p3 - p2
            if p3 <= p1 or p4 <= p1 or p4 >= p3:
                continue
            if len3 / len1 >= 1.618:
                fib_ok += 1
            retr4 = (p3 - p4) / len3
            if 0.236 <= retr4 <= 0.5:
                fib_ok += 1
            if setup == 1:
                if len3 < len1:
                    continue
                if use_fib and fib_ok < 2:
                    continue
                sl = p1
                tp = p4 + len1
            else:
                p5 = p_px[b + 5]
                len5 = p5 - p4
                if p5 <= p3 or (len3 < len1 and len3 < len5):
                    continue
                if use_fib and fib_ok < 2:
                    continue
                direction = -1
                sl = p5
                tp = p4
        s_idx[k_out] = p_conf[k]
        s_dir[k_out] = direction
        s_sl[k_out] = sl
        s_tp[k_out] = tp
        for q in range(need + 1):
            pts_i[k_out, q] = p_idx[b + q]
            pts_p[k_out, q] = p_px[b + q]
        k_out += 1
    return s_idx[:k_out], s_dir[:k_out], s_sl[:k_out], s_tp[:k_out], pts_i[:k_out], pts_p[:k_out]
