"""Tahap 5: strategi lengkap ala influencer dan kontrol
"zodiak sungguhan" (4.10).

Strategi influencer disusun dari metode yang ditemukan di riset (bagian 11):
- the_grab_stoch       : kerangka flow -> pattern -> trigger. Pattern = "The Grab"
                         (false break di bawah support / di atas resistance),
                         trigger = Stochastic cross, flow = filter tren EMA200.
- pullback_ma50        : trend-following. Di atas MA200, harga menyentuh MA50 lalu
                         ditutup naik; SL 2 ATR; keluar saat ditutup di bawah MA50.
- triangle_fib1272_mtf : triangle/pennant, TP di ekstensi Fibonacci 127,2%,
                         hanya searah tren timeframe besar (EMA200).
- hs_fib1618           : head and shoulders, SL di bahu, TP di ekstensi 161,8%.
- fib_golden_candle    : pullback ke zona Fibonacci 0,5-0,786 + candle konfirmasi
                         (hammer/engulfing), SL di balik candle, RR 1:2.
Model ICT 2022, supply-demand, breakout-retest S/R, dan konfluensi EMA200+MACD+RSI
sudah diuji di tahap 2-4.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pandas as pd
from numba import njit

from . import engine, patterns, smc
from . import indicators as ta
from .strat_candles import Bars, _hammer_shape, _star_shape
from .strat_patterns import _memo, _ohlc, _zz
from .strat_smc import _atr, _structure


def _rolling_extreme(x, n, kind):
    s = pd.Series(x).rolling(n, min_periods=1)
    return (s.min() if kind == "min" else s.max()).values


# ---------- The Grab + Stochastic ----------

def the_grab_stoch(df: pd.DataFrame, use_trend: bool) -> pd.DataFrame:
    o, h, l, c = _ohlc(df)
    a = _atr(df)
    sw = _structure(df)[2]
    k, d = ta.stochastic(h, l, c)
    n = len(c)
    pos = np.arange(n)
    last_bull = np.maximum.accumulate(np.where(sw == 1, pos, -1))
    last_bear = np.maximum.accumulate(np.where(sw == -1, pos, -1))
    long_m = ta.cross_up(k, d) & (last_bull >= 0) & (pos - last_bull <= 5)
    short_m = ta.cross_down(k, d) & (last_bear >= 0) & (pos - last_bear <= 5)
    if use_trend:
        e = ta.ema(c, 200)
        long_m &= c > e
        short_m &= c < e
    ok = np.isfinite(a)
    li, si = np.flatnonzero(long_m & ok), np.flatnonzero(short_m & ok)
    idx = np.concatenate([li, si])
    dr = np.concatenate([np.ones(len(li), dtype=np.int64), -np.ones(len(si), dtype=np.int64)])
    sl = np.concatenate([_rolling_extreme(l, 6, "min")[li] - smc.BUF * a[li],
                         _rolling_extreme(h, 6, "max")[si] + smc.BUF * a[si]])
    risk = dr * (c[idx] - sl)
    keep = risk > 0
    return engine.make_signals(idx[keep] + 1, dr[keep], sl=sl[keep],
                               tp=(c[idx] + dr * 2.0 * risk)[keep], hold=50)


# ---------- pullback ke MA50 ----------

def pullback_ma50(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = _ohlc(df)
    a = _atr(df)
    m50, m200 = ta.sma(c, 50), ta.sma(c, 200)
    pc, pm = ta.shift(c, 1), ta.shift(m50, 1)
    long_m = (c > m200) & (m50 > m200) & (l <= m50) & (c > m50) & (c > o) & (pc > pm)
    short_m = (c < m200) & (m50 < m200) & (h >= m50) & (c < m50) & (c < o) & (pc < pm)
    ok = np.isfinite(a)
    n = len(c)
    parts = []
    for mask, d, exit_cond in ((long_m & ok, 1, c < m50), (short_m & ok, -1, c > m50)):
        idx = np.flatnonzero(mask)
        exits = np.flatnonzero(exit_cond)
        nxt = np.searchsorted(exits, idx, side="right")
        exit_bar = np.where(nxt < len(exits), exits[np.minimum(nxt, len(exits) - 1)] if len(exits) else n, n)
        hold = np.clip(exit_bar - idx, 1, 250)
        parts.append((idx, np.full(len(idx), d, dtype=np.int64), 2.0 * a[idx], hold))
    idx = np.concatenate([p[0] for p in parts])
    return engine.make_signals(idx + 1, np.concatenate([p[1] for p in parts]),
                               sl_dist=np.concatenate([p[2] for p in parts]),
                               hold=np.concatenate([p[3] for p in parts]))


# ---------- pola + target Fibonacci ----------

def pattern_fib_target(df: pd.DataFrame, names, ext: float, mtf: bool) -> pd.DataFrame:
    c = _ohlc(df)[3]
    frames = [_memo(df, ("det", n, 1), lambda n=n: patterns.detect(df, n, 1)) for n in names]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return engine.empty_signals()
    det = pd.concat(frames, ignore_index=True)
    idx = det["idx"].values.astype(np.int64)
    d = det["dir"].values.astype(np.int64)
    height = np.abs(det["tp"].values - c[idx])
    tp = c[idx] + d * ext * height
    keep = np.ones(len(idx), dtype=bool)
    if mtf:
        e = ta.ema(c, 200)
        keep = np.sign(c[idx] - e[idx]) == d
    return engine.make_signals(idx[keep] + 1, d[keep], sl=det["sl"].values[keep], tp=tp[keep], hold=100)


# ---------- Fibonacci golden zone + candle konfirmasi ----------

@njit(cache=True)
def _fib_candle(h, l, c, atr, p_idx, p_px, p_type, p_conf, bull, bear, lo_lv, hi_lv):
    n = len(c)
    m = len(p_idx)
    s_idx = np.empty(m, dtype=np.int64)
    s_dir = np.empty(m, dtype=np.int64)
    s_sl = np.empty(m)
    k_out = 0
    for k in range(1, m):
        d = p_type[k]
        a, b = p_px[k - 1], p_px[k]
        leg = d * (b - a)
        if leg <= 0:
            continue
        z_near = b - d * lo_lv * leg
        z_far = b - d * hi_lv * leg
        end = min(n, p_conf[k] + 1 + max(10, 3 * (p_idx[k] - p_idx[k - 1])))
        for j in range(p_conf[k], end):
            if d == 1:
                if h[j] > b or l[j] < a:
                    break
                if l[j] <= z_near and l[j] >= z_far and bull[j]:
                    s_idx[k_out], s_dir[k_out], s_sl[k_out] = j, 1, l[j] - 0.1 * atr[j]
                    k_out += 1
                    break
            else:
                if l[j] < b or h[j] > a:
                    break
                if h[j] >= z_near and h[j] <= z_far and bear[j]:
                    s_idx[k_out], s_dir[k_out], s_sl[k_out] = j, -1, h[j] + 0.1 * atr[j]
                    k_out += 1
                    break
    return s_idx[:k_out], s_dir[:k_out], s_sl[:k_out]


def fib_golden_candle(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = _ohlc(df)
    a = _atr(df)
    b = Bars(df)
    po, pc, pbody = b.p(b.o, 1), b.p(b.c, 1), b.p(b.body, 1)
    bull = _hammer_shape(b) | (b.pb(b.red, 1) & b.green & (b.o <= pc) & (b.c >= po) & (b.body > pbody))
    bear = _star_shape(b) | (b.pb(b.green, 1) & b.red & (b.o >= pc) & (b.c <= po) & (b.body > pbody))
    bull, bear = np.asarray(bull, dtype=bool), np.asarray(bear, dtype=bool)
    idx, d, sl = [], [], []
    for mult in patterns.SCALES:
        p_idx, p_px, p_type, p_conf = _zz(df, mult)
        i, dd, s = _fib_candle(h, l, c, a, p_idx, p_px, p_type, p_conf, bull, bear, 0.5, 0.786)
        idx.append(i)
        d.append(dd)
        sl.append(s)
    idx, d, sl = np.concatenate(idx), np.concatenate(d), np.concatenate(sl)
    risk = d * (c[idx] - sl)
    keep = np.isfinite(sl) & (risk > 0)
    return engine.make_signals(idx[keep] + 1, d[keep], sl=sl[keep],
                               tp=(c[idx] + d * 2.0 * risk)[keep], hold=50)


# ---------- kontrol zodiak ----------

@lru_cache(maxsize=1)
def sky() -> pd.DataFrame:
    """Posisi benda langit per hari (00:00 UTC), 2009-2027, dari pustaka ephem.
    waxing   : bulan sedang membesar (dari bulan baru menuju purnama)
    retro    : Merkurius retrograde (bujur ekliptika geosentrisnya sedang mundur)
    element  : elemen zodiak tempat Matahari berada (api, tanah, udara, air)"""
    import ephem

    days = pd.date_range("2009-06-01", "2027-06-01", freq="D", tz="UTC")
    sun = np.empty(len(days))
    moon = np.empty(len(days))
    merc = np.empty(len(days))
    for i, t in enumerate(days):
        d = ephem.Date(t.strftime("%Y/%m/%d"))
        sun[i] = math.degrees(ephem.Ecliptic(ephem.Sun(d)).lon)
        moon[i] = math.degrees(ephem.Ecliptic(ephem.Moon(d)).lon)
        merc[i] = math.degrees(ephem.Ecliptic(ephem.Mercury(d)).lon)
    step = (np.diff(merc, append=merc[-1]) + 180.0) % 360.0 - 180.0
    sign = (sun // 30).astype(int) % 12          # 0 = Aries
    element = np.array(["api", "tanah", "udara", "air"])[sign % 4]
    return pd.DataFrame({"waxing": ((moon - sun) % 360.0) < 180.0, "retro": step < 0,
                         "element": element, "sign": sign}, index=days)


def _sky_for(df: pd.DataFrame) -> pd.DataFrame:
    day = df.index.tz_convert("UTC").normalize()
    return sky().reindex(day)


def moon_phase(df: pd.DataFrame) -> np.ndarray:
    """Long saat bulan membesar, short saat bulan mengecil."""
    return np.where(_sky_for(df)["waxing"].values.astype(bool), 1, -1).astype(np.int64)


def mercury_retrograde(df: pd.DataFrame) -> np.ndarray:
    """Short saat Merkurius retrograde, long saat tidak."""
    return np.where(_sky_for(df)["retro"].values.astype(bool), -1, 1).astype(np.int64)


def zodiac_element(df: pd.DataFrame) -> np.ndarray:
    """Long saat Matahari di zodiak api/udara, short saat di zodiak tanah/air."""
    el = _sky_for(df)["element"].values
    return np.where((el == "api") | (el == "udara"), 1, -1).astype(np.int64)


def build() -> dict[str, dict]:
    reg: dict[str, dict] = {}

    def add(name, fn, group="influencer"):
        reg[name] = dict(fn=fn, pos=None, group=group, kind="event")

    add("the_grab_stoch", lambda df: the_grab_stoch(df, False))
    add("the_grab_stoch_tren", lambda df: the_grab_stoch(df, True))
    add("pullback_ma50", pullback_ma50)
    tri = ("triangle_asc_desc", "triangle_symmetrical", "flag_pennant")
    add("triangle_fib1272", lambda df: pattern_fib_target(df, tri, 1.272, False))
    add("triangle_fib1272_mtf", lambda df: pattern_fib_target(df, tri, 1.272, True))
    add("hs_fib1272", lambda df: pattern_fib_target(df, ("head_shoulders",), 1.272, False))
    add("hs_fib1618", lambda df: pattern_fib_target(df, ("head_shoulders",), 1.618, False))
    add("fib_golden_candle", fib_golden_candle)

    for name, pos_fn in (("zodiak_fase_bulan", moon_phase), ("zodiak_merkurius_retrograde", mercury_retrograde),
                         ("zodiak_elemen", zodiac_element)):
        reg[f"{name}_ls"] = dict(fn=lambda df, p=pos_fn: engine.signals_from_position(p(df)),
                                 pos=pos_fn, group="zodiak", kind="state")
        reg[f"{name}_lo"] = dict(fn=lambda df, p=pos_fn: engine.signals_from_position(np.maximum(p(df), 0)),
                                 pos=lambda df, p=pos_fn: np.maximum(p(df), 0), group="zodiak", kind="state")
    return reg
