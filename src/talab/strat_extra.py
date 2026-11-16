"""Strategi tambahan yang ada di rencana awal tetapi belum diuji di tahap 2-5:
rounding bottom/top, channel, filter premium/discount (SMC), candlestick di area
support/resistance, dan konfluensi multi-timeframe.

Definisi dibekukan sebelum uji performa dijalankan.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from . import engine, levels, strat_candles, strat_smc
from . import indicators as ta
from .strat_indicators import _sign, arrays
from .strat_patterns import LEVEL_SCALES, _memo, _ohlc, _zz

# ---------------------------------------------------------------- rounding bottom / top

ROUND_WINDOWS = (40, 80)
ROUND_R2 = 0.80      # seberapa mulus lengkungannya
ROUND_VERTEX = 0.30  # titik terendah harus di sekitar tengah jendela
ROUND_DEPTH = 3.0    # kedalaman minimal, dalam ATR


def rounding_raw(df: pd.DataFrame):
    """Rounding bottom: harga penutupan W candle terakhir mengikuti parabola terbuka ke
    atas (R2 >= 0,8) dengan titik terendah di tengah dan kedalaman >= 3 ATR; sinyal
    beli saat penutupan menembus bibir kiri lengkungan. Rounding top = cerminannya.
    Mengembalikan idx, dir, sl, tp, awal jendela, harga bibir."""
    o, h, l, c = _ohlc(df)
    a = ta.atr(h, l, c)
    n = len(c)
    out = [[], [], [], [], [], []]
    for W in ROUND_WINDOWS:
        if n < W + 2:
            continue
        x = np.linspace(-1.0, 1.0, W)
        X = np.stack([np.ones(W), x, x * x], axis=1)
        P = np.linalg.pinv(X)                                  # 3 x W
        win = np.lib.stride_tricks.sliding_window_view(c, W)   # jendela berakhir di i = k + W - 1
        beta = win @ P.T
        xty = win @ X
        sy2 = np.einsum("ij,ij->i", win, win)
        ss_res = sy2 - np.einsum("ij,ij->i", beta, xty)
        ss_tot = sy2 - W * win.mean(axis=1) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            r2 = 1.0 - ss_res / ss_tot
            xv = -beta[:, 1] / (2.0 * beta[:, 2])
        rim = beta[:, 0] - beta[:, 1] + beta[:, 2]             # nilai parabola di ujung kiri
        bottom = beta[:, 0] + beta[:, 1] * xv + beta[:, 2] * xv * xv
        end = np.arange(W - 1, n)
        atr_e = a[end]
        lo_w, hi_w = ta.rolling_min(l, W)[end], ta.rolling_max(h, W)[end]
        prev = np.roll(c, 1)[end]
        rim_prev = np.roll(rim, 1)      # bibir menurut jendela sebelumnya: sinyal hanya di tembusan pertama
        rim_prev[0] = np.inf
        ok = (r2 >= ROUND_R2) & (np.abs(xv) <= ROUND_VERTEX) & np.isfinite(atr_e)
        for d in (1, -1):
            depth = d * (rim - bottom)
            m = ok & (d * beta[:, 2] > 0) & (depth >= ROUND_DEPTH * atr_e) & (d * (c[end] - rim) > 0) & (d * (prev - d * np.where(np.isfinite(rim_prev), d * rim_prev, np.inf)) <= 0)
            idx = end[m]
            sl = (lo_w[m] - 0.1 * atr_e[m]) if d == 1 else (hi_w[m] + 0.1 * atr_e[m])
            for store, val in zip(out, (idx, np.full(len(idx), d), sl, c[idx] + d * depth[m], idx - W + 1, rim[m])):
                store.append(val)
    if not out[0] or sum(len(v) for v in out[0]) == 0:
        return tuple(np.array([], dtype=np.int64 if k in (0, 1, 4) else float) for k in range(6))
    res = [np.concatenate(v) for v in out]
    order = np.argsort(res[0], kind="stable")
    return tuple(v[order].astype(np.int64) if k in (0, 1, 4) else v[order] for k, v in enumerate(res))


def rounding(df: pd.DataFrame, exit_rule: str) -> pd.DataFrame:
    idx, d, sl, tp, _, _ = _memo(df, "rounding", lambda: rounding_raw(df))
    if len(idx) == 0:
        return engine.empty_signals()
    if exit_rule == "h10":
        return engine.make_signals(idx + 1, d, hold=10)
    return engine.make_signals(idx + 1, d, sl=sl, tp=tp, hold=100)


# ---------------------------------------------------------------- channel

CH_PARALLEL = 0.15   # lebar channel boleh berubah paling banyak 15%
CH_MIN_WIDTH = 2.0   # lebar minimal, dalam ATR
CH_SLACK = 0.5       # penutupan boleh keluar garis paling jauh 0,5 ATR


@njit(cache=True)
def channel_signals(o, h, l, c, atr, p_idx, p_px, p_type, p_conf, mode):
    """Channel = garis lewat dua swing low terakhir dan garis lewat dua swing high
    terakhir yang hampir sejajar. mode 0: pantulan di tepi channel, target tepi
    seberang. mode 1: penutupan menembus tepi channel, target selebar channel."""
    n = len(c)
    m = len(p_idx)
    s_idx = np.empty(n, dtype=np.int64)
    s_dir = np.empty(n, dtype=np.int64)
    s_sl = np.empty(n)
    s_tp = np.empty(n)
    geo = np.empty((n, 6))
    k_out = 0
    nxt = 0
    lo_i = np.array([-1, -1])
    lo_p = np.array([0.0, 0.0])
    hi_i = np.array([-1, -1])
    hi_p = np.array([0.0, 0.0])
    alive = False
    for i in range(1, n):
        a = atr[i]
        if np.isnan(a):
            continue
        while nxt < m and p_conf[nxt] <= i:
            if p_type[nxt] == -1:
                lo_i[0], lo_p[0] = lo_i[1], lo_p[1]
                lo_i[1], lo_p[1] = p_idx[nxt], p_px[nxt]
            else:
                hi_i[0], hi_p[0] = hi_i[1], hi_p[1]
                hi_i[1], hi_p[1] = p_idx[nxt], p_px[nxt]
            alive = lo_i[0] >= 0 and hi_i[0] >= 0
            nxt += 1
        if not alive:
            continue
        s_lo = (lo_p[1] - lo_p[0]) / (lo_i[1] - lo_i[0])
        s_hi = (hi_p[1] - hi_p[0]) / (hi_i[1] - hi_i[0])
        t0 = min(lo_i[0], hi_i[0])
        w0 = (hi_p[1] + s_hi * (t0 - hi_i[1])) - (lo_p[1] + s_lo * (t0 - lo_i[1]))
        lo_ln = lo_p[1] + s_lo * (i - lo_i[1])
        hi_ln = hi_p[1] + s_hi * (i - hi_i[1])
        w = hi_ln - lo_ln
        if w0 <= 0 or w < CH_MIN_WIDTH * a or abs(w - w0) > CH_PARALLEL * w0:
            continue
        direction = 0
        sl = 0.0
        tp = 0.0
        if c[i] > hi_ln + levels.NEAR * a:
            alive = False
            if mode == 1:
                direction, sl, tp = 1, hi_ln - levels.SL_ATR * a, c[i] + w
        elif c[i] < lo_ln - levels.NEAR * a:
            alive = False
            if mode == 1:
                direction, sl, tp = -1, lo_ln + levels.SL_ATR * a, c[i] - w
        elif mode == 0:
            if l[i] <= lo_ln + levels.NEAR * a and c[i] > lo_ln and c[i] > o[i]:
                direction, sl, tp = 1, lo_ln - levels.SL_ATR * a, hi_ln
            elif h[i] >= hi_ln - levels.NEAR * a and c[i] < hi_ln and c[i] < o[i]:
                direction, sl, tp = -1, hi_ln + levels.SL_ATR * a, lo_ln
        if direction != 0:
            # channel harus benar-benar mengurung harga sejak swing pertamanya
            for j in range(t0, i):
                up_j = hi_p[1] + s_hi * (j - hi_i[1])
                dn_j = lo_p[1] + s_lo * (j - lo_i[1])
                if c[j] > up_j + CH_SLACK * a or c[j] < dn_j - CH_SLACK * a:
                    direction = 0
                    break
        if direction != 0 and direction * (c[i] - sl) > 0 and direction * (tp - c[i]) > 0:
            s_idx[k_out], s_dir[k_out], s_sl[k_out], s_tp[k_out] = i, direction, sl, tp
            geo[k_out, 0], geo[k_out, 1], geo[k_out, 2] = lo_i[0], lo_p[0], lo_ln
            geo[k_out, 3], geo[k_out, 4], geo[k_out, 5] = hi_i[0], hi_p[0], hi_ln
            k_out += 1
    return s_idx[:k_out], s_dir[:k_out], s_sl[:k_out], s_tp[:k_out], geo[:k_out]


def channel_raw(df: pd.DataFrame, mode: int):
    def f():
        o, h, l, c = _ohlc(df)
        a = ta.atr(h, l, c)
        parts = []
        for mult in LEVEL_SCALES:
            p_idx, p_px, p_type, p_conf = _zz(df, mult)
            parts.append(channel_signals(o, h, l, c, a, p_idx, p_px, p_type, p_conf, mode))
        return tuple(np.concatenate([p[k] for p in parts]) for k in range(5))
    return _memo(df, ("channel", mode), f)


def channel(df: pd.DataFrame, mode: int) -> pd.DataFrame:
    idx, d, sl, tp, _ = channel_raw(df, mode)
    if len(idx) == 0:
        return engine.empty_signals()
    return engine.make_signals(idx + 1, d, sl=sl, tp=tp, hold=50)


# ---------------------------------------------------------------- premium / discount

def premium_discount(df: pd.DataFrame, res):
    """Filter ICT: beli hanya di separuh bawah rentang swing terakhir (discount),
    jual hanya di separuh atasnya (premium)."""
    if len(res[0]) == 0:
        return res
    st = strat_smc._structure(df)
    e_idx, e_px, d = res[0], res[1], res[2]
    mid = (st[5][e_idx - 1] + st[6][e_idx - 1]) / 2.0
    keep = np.isfinite(mid) & (((d == 1) & (e_px < mid)) | ((d == -1) & (e_px > mid)))
    return tuple(x[keep] for x in res)


def fvg_pd_raw(df):
    return premium_discount(df, strat_smc.fvg_raw(df, False, 2.0))


def ob_pd_raw(df):
    return premium_discount(df, strat_smc.ob_raw(df, 2.0))


# ---------------------------------------------------------------- candlestick di area S/R

@njit(cache=True)
def sr_touch(h, l, c, atr, p_idx, p_px, p_conf, lookback, min_touch):
    """Untuk tiap candle: apakah ia menyentuh support (dan ditutup di atasnya) atau
    resistance (dan ditutup di bawahnya). Level dibentuk seperti di levels.sr_signals."""
    n = len(c)
    m = len(p_idx)
    sup = np.zeros(n, dtype=np.bool_)
    res = np.zeros(n, dtype=np.bool_)
    lev = np.empty(m)
    n_lev = 0
    first = 0
    nxt = 0
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
                    if abs(p_px[r] - p_px[q]) <= levels.CLUSTER * a:
                        cnt += 1
                        tot += p_px[r]
                if cnt >= min_touch:
                    center = tot / cnt
                    dup = False
                    for z in range(n_lev):
                        if abs(lev[z] - center) <= levels.CLUSTER * a:
                            dup = True
                            break
                    if not dup:
                        lev[n_lev] = center
                        n_lev += 1
        for z in range(n_lev):
            L = lev[z]
            if l[i] <= L + levels.NEAR * a and c[i] > L:
                sup[i] = True
            if h[i] >= L - levels.NEAR * a and c[i] < L:
                res[i] = True
    return sup, res


def _sr_context(df):
    def f():
        o, h, l, c = _ohlc(df)
        a = ta.atr(h, l, c)
        sup = np.zeros(len(c), dtype=bool)
        res = np.zeros(len(c), dtype=bool)
        for mult in LEVEL_SCALES:
            p_idx, p_px, _, p_conf = _zz(df, mult)
            s, r = sr_touch(h, l, c, a, p_idx, p_px, p_conf, 250, 2)
            sup |= s
            res |= r
        return sup, res
    return _memo(df, "sr_context", f)


CANDLES_SR = ("hammer_hanging_man", "engulfing", "pin_bar", "morning_evening_star", "doji_reversal")


def candle_sr(df: pd.DataFrame, pattern: str, exit_rule: str) -> pd.DataFrame:
    b = strat_candles.Bars(df)
    bull, bear, sl_long, sl_short = strat_candles.PATTERNS[pattern](b)
    sup, res = _sr_context(df)
    bull = np.asarray(bull, dtype=bool) & sup
    bear = np.asarray(bear, dtype=bool) & ~bull & res
    if exit_rule == "h10":
        return engine.signals_from_events(bull, bear, hold=10)
    li, si = np.flatnonzero(bull), np.flatnonzero(bear)
    idx = np.concatenate([li, si])
    d = np.concatenate([np.ones(len(li), dtype=np.int64), -np.ones(len(si), dtype=np.int64)])
    sl = np.concatenate([sl_long[li], sl_short[si]])
    risk = np.abs(b.c[idx] - sl)
    ok = np.isfinite(sl) & (risk > 0)
    return engine.make_signals(idx[ok] + 1, d[ok], sl=sl[ok], tp=(b.c[idx] + d * 2.0 * risk)[ok], hold=50)


# ---------------------------------------------------------------- multi-timeframe

def daily_trend(df: pd.DataFrame) -> np.ndarray:
    """Tren timeframe besar: EMA 20 vs EMA 50 dari penutupan HARIAN, memakai hari
    terakhir yang sudah selesai (tidak mengintip hari berjalan). +1 naik, -1 turun."""
    day = df.index.tz_convert("UTC").normalize()
    daily = df["close"].groupby(day).last()
    c = daily.values.astype(float)
    trend = pd.Series(_sign(ta.ema(c, 20) - ta.ema(c, 50)), index=daily.index).shift(1)
    if len(c) >= 50:
        trend.iloc[:50] = 0          # EMA 50 belum terbentuk
    else:
        trend[:] = 0
    return trend.reindex(day).fillna(0).values.astype(np.int64)


def mtf_state(base_pos):
    def pos(df):
        p = np.asarray(base_pos(df), dtype=np.int64)
        return np.where(p == daily_trend(df), p, 0).astype(np.int64)
    return pos


def _ema_9_21(df):
    c = arrays(df)[3]
    return _sign(ta.ema(c, 9) - ta.ema(c, 21))


def _macd_signal(df):
    line, sig = ta.macd(arrays(df)[3])
    return _sign(line - sig)


def mtf_rsi(df: pd.DataFrame, exit_rule: str) -> pd.DataFrame:
    """Beli pullback (RSI naik menembus 30) hanya saat tren harian naik; sebaliknya untuk jual."""
    r = ta.rsi(arrays(df)[3])
    t = daily_trend(df)
    lm, sm = ta.cross_up(r, 30.0) & (t == 1), ta.cross_down(r, 70.0) & (t == -1)
    if exit_rule == "h10":
        return engine.signals_from_events(lm, sm, hold=10)
    a = engine.atr(df)
    ok = np.isfinite(a)
    return engine.signals_from_events(lm & ok, sm & ok, hold=50, sl_dist=1.5 * a, tp_dist=3.0 * a)


# ---------------------------------------------------------------- registry

def build() -> dict[str, dict]:
    reg: dict[str, dict] = {}

    def add(name, fn, group, **flags):
        reg[name] = dict(fn=fn, pos=None, group=group, kind="event", **flags)

    add("rounding_mm", lambda df: rounding(df, "mm"), "pola_chart_B")
    add("rounding_h10", lambda df: rounding(df, "h10"), "pola_chart_B")
    add("channel_bounce", lambda df: channel(df, 0), "trendline")
    add("channel_break", lambda df: channel(df, 1), "trendline")
    add("smc_fvg_pd_rr2", lambda df: strat_smc._limit(fvg_pd_raw(df)), "smc_ict")
    add("smc_orderblock_pd_rr2", lambda df: strat_smc._limit(ob_pd_raw(df)), "smc_ict")
    for pat in CANDLES_SR:
        for ex in ("h10", "rr2"):
            add(f"{pat}_sr_{ex}", lambda df, p=pat, e=ex: candle_sr(df, p, e), "candlestick")
    for name, base in (("mtf_ema_9_21", _ema_9_21), ("mtf_macd_signal", _macd_signal)):
        p = mtf_state(base)
        reg[f"{name}_ls"] = dict(fn=lambda df, p=p: engine.signals_from_position(p(df)), pos=p,
                                 group="indikator", kind="state", intraday=True)
        reg[f"{name}_lo"] = dict(fn=lambda df, p=p: engine.signals_from_position(np.maximum(p(df), 0)),
                                 pos=lambda df, p=p: np.maximum(p(df), 0),
                                 group="indikator", kind="state", intraday=True)
    for ex in ("h10", "rr2"):
        add(f"mtf_rsi_cross_{ex}", lambda df, e=ex: mtf_rsi(df, e), "indikator", intraday=True)
    return reg
