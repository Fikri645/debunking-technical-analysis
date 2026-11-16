"""Pola candlestick.

Tiap pola mengembalikan (sinyal_long, sinyal_short, sl_long, sl_short):
sl = ujung pola (low terendah untuk long, high tertinggi untuk short).

Pola pembalikan mensyaratkan tren sebelumnya, seperti di buku: pola bullish
harus muncul setelah penurunan, pola bearish setelah kenaikan. "Tren" di sini =
harga penutupan tepat sebelum pola lebih rendah/tinggi daripada 10 candle sebelumnya.

Tiga cara keluar diuji per pola: `_h5`, `_h10` (tahan 5/10 candle) dan `_rr2`
(SL di ujung pola, TP dua kali jarak SL).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import engine
from . import indicators as ta

TREND_BARS = 10


class Bars:
    def __init__(self, df: pd.DataFrame):
        self.o, self.h, self.l, self.c = (
            df[k].values.astype(float) for k in ("open", "high", "low", "close"))
        o, h, l, c = self.o, self.h, self.l, self.c
        self.body = np.abs(c - o)
        self.rng = h - l
        self.upper = h - np.maximum(o, c)
        self.lower = np.minimum(o, c) - l
        self.green = c > o
        self.red = c < o
        self.avg_body = ta.shift(ta.sma(self.body, 10), 1)
        self.atr = ta.atr(h, l, c)

    def p(self, x, k):
        """Nilai k candle sebelumnya."""
        return ta.shift(x, k)

    def pb(self, x, k):
        return pd.Series(x).shift(k).fillna(False).values.astype(bool)

    def down(self, length):
        """Penurunan sebelum pola sepanjang `length` candle."""
        return ta.shift(self.c, length) < ta.shift(self.c, length + TREND_BARS)

    def up(self, length):
        return ta.shift(self.c, length) > ta.shift(self.c, length + TREND_BARS)

    def lowest(self, n):
        return ta.rolling_min(self.l, n)

    def highest(self, n):
        return ta.rolling_max(self.h, n)


def _hammer_shape(b: Bars):
    return (b.rng > 0) & (b.lower >= 2 * b.body) & (b.upper <= 0.25 * b.rng) & (b.body >= 0.05 * b.rng)


def _star_shape(b: Bars):
    return (b.rng > 0) & (b.upper >= 2 * b.body) & (b.lower <= 0.25 * b.rng) & (b.body >= 0.05 * b.rng)


def hammer_hanging_man(b: Bars):
    s = _hammer_shape(b)
    return s & b.down(1), s & b.up(1), b.lowest(1), b.highest(1)


def inverted_hammer_shooting_star(b: Bars):
    s = _star_shape(b)
    return s & b.down(1), s & b.up(1), b.lowest(1), b.highest(1)


def engulfing(b: Bars):
    po, pc, pbody = b.p(b.o, 1), b.p(b.c, 1), b.p(b.body, 1)
    bull = b.pb(b.red, 1) & b.green & (b.o <= pc) & (b.c >= po) & (b.body > pbody) & b.down(2)
    bear = b.pb(b.green, 1) & b.red & (b.o >= pc) & (b.c <= po) & (b.body > pbody) & b.up(2)
    return bull, bear, b.lowest(2), b.highest(2)


def harami(b: Bars):
    po, pc, pbody = b.p(b.o, 1), b.p(b.c, 1), b.p(b.body, 1)
    big_prev = pbody >= b.p(b.avg_body, 1)
    inside = (np.maximum(b.o, b.c) <= np.maximum(po, pc)) & (np.minimum(b.o, b.c) >= np.minimum(po, pc))
    small = b.body < pbody
    bull = b.pb(b.red, 1) & b.green & big_prev & inside & small & b.down(2)
    bear = b.pb(b.green, 1) & b.red & big_prev & inside & small & b.up(2)
    return bull, bear, b.lowest(2), b.highest(2)


def piercing_dark_cloud(b: Bars):
    po, pc, pbody = b.p(b.o, 1), b.p(b.c, 1), b.p(b.body, 1)
    big_prev = pbody >= b.p(b.avg_body, 1)
    mid = (po + pc) / 2
    bull = b.pb(b.red, 1) & b.green & big_prev & (b.o < pc) & (b.c > mid) & (b.c < po) & b.down(2)
    bear = b.pb(b.green, 1) & b.red & big_prev & (b.o > pc) & (b.c < mid) & (b.c > po) & b.up(2)
    return bull, bear, b.lowest(2), b.highest(2)


def morning_evening_star(b: Bars):
    o1, c1, body1 = b.p(b.o, 2), b.p(b.c, 2), b.p(b.body, 2)
    o2, c2, body2 = b.p(b.o, 1), b.p(b.c, 1), b.p(b.body, 1)
    avg1 = b.p(b.avg_body, 2)
    big1 = body1 >= avg1
    small2 = body2 <= 0.5 * avg1
    mid1 = (o1 + c1) / 2
    bull = (b.pb(b.red, 2) & big1 & small2 & (np.maximum(o2, c2) <= c1 + 0.1 * body1)
            & b.green & (b.c > mid1) & b.down(3))
    bear = (b.pb(b.green, 2) & big1 & small2 & (np.minimum(o2, c2) >= c1 - 0.1 * body1)
            & b.red & (b.c < mid1) & b.up(3))
    return bull, bear, b.lowest(3), b.highest(3)


def three_soldiers_crows(b: Bars):
    def seq(color, sign):
        ok = np.ones(len(b.c), dtype=bool)
        for k in (0, 1, 2):
            ok &= b.pb(color, k) & (b.p(b.body, k) >= 0.7 * b.p(b.avg_body, k))
        for k in (0, 1):
            c_now, c_prev = b.p(b.c, k), b.p(b.c, k + 1)
            o_now, o_prev = b.p(b.o, k), b.p(b.o, k + 1)
            ok &= sign * (c_now - c_prev) > 0
            ok &= (o_now >= np.minimum(o_prev, c_prev)) & (o_now <= np.maximum(o_prev, c_prev))
        wick = b.upper if sign > 0 else b.lower
        for k in (0, 1, 2):
            ok &= b.p(wick, k) <= 0.3 * b.p(b.body, k)
        return ok
    return seq(b.green, 1), seq(b.red, -1), b.lowest(3), b.highest(3)


def doji(b: Bars):
    """Doji sebagai tanda pembalikan: setelah turun = beli, setelah naik = jual."""
    d = (b.rng > 0) & (b.body <= 0.1 * b.rng)
    return d & b.down(1), d & b.up(1), b.lowest(1), b.highest(1)


def dragonfly_gravestone(b: Bars):
    d = (b.rng > 0) & (b.body <= 0.1 * b.rng)
    dragon = d & (b.lower >= 0.6 * b.rng) & (b.upper <= 0.1 * b.rng) & b.down(1)
    grave = d & (b.upper >= 0.6 * b.rng) & (b.lower <= 0.1 * b.rng) & b.up(1)
    return dragon, grave, b.lowest(1), b.highest(1)


def marubozu(b: Bars):
    """Candle penuh tanpa sumbu = lanjutan searah."""
    m = (b.rng > 0) & (b.body >= 0.9 * b.rng) & (b.body >= 1.5 * b.avg_body)
    return m & b.green, m & b.red, b.lowest(1), b.highest(1)


def tweezer(b: Bars):
    tol = 0.1 * b.atr
    bottom = (b.pb(b.red, 1) & b.green & (np.abs(b.l - b.p(b.l, 1)) <= tol) & b.down(2))
    top = (b.pb(b.green, 1) & b.red & (np.abs(b.h - b.p(b.h, 1)) <= tol) & b.up(2))
    return bottom, top, b.lowest(2), b.highest(2)


def inside_bar_breakout(b: Bars):
    """Inside bar lalu penutupan menembus high/low induknya."""
    mh, ml = b.p(b.h, 2), b.p(b.l, 2)
    inside = (b.p(b.h, 1) <= mh) & (b.p(b.l, 1) >= ml)
    return inside & (b.c > mh), inside & (b.c < ml), b.lowest(3), b.highest(3)


def pin_bar(b: Bars):
    """Pin bar tanpa syarat tren: sumbu panjang menolak satu arah."""
    return _hammer_shape(b), _star_shape(b), b.lowest(1), b.highest(1)


PATTERNS = {
    "hammer_hanging_man": hammer_hanging_man,
    "inverted_hammer_shooting_star": inverted_hammer_shooting_star,
    "engulfing": engulfing,
    "harami": harami,
    "piercing_dark_cloud": piercing_dark_cloud,
    "morning_evening_star": morning_evening_star,
    "three_soldiers_crows": three_soldiers_crows,
    "doji_reversal": doji,
    "dragonfly_gravestone_doji": dragonfly_gravestone,
    "marubozu": marubozu,
    "tweezer": tweezer,
    "inside_bar_breakout": inside_bar_breakout,
    "pin_bar": pin_bar,
}


def signals(pattern, exit_rule: str, df: pd.DataFrame) -> pd.DataFrame:
    b = Bars(df)
    bull, bear, sl_long, sl_short = pattern(b)
    bull = np.asarray(bull, dtype=bool)
    bear = np.asarray(bear, dtype=bool) & ~bull
    if exit_rule in ("h5", "h10"):
        return engine.signals_from_events(bull, bear, hold=int(exit_rule[1:]))
    li, si = np.flatnonzero(bull), np.flatnonzero(bear)
    idx = np.concatenate([li, si])
    d = np.concatenate([np.ones(len(li), dtype=np.int64), -np.ones(len(si), dtype=np.int64)])
    sl = np.concatenate([sl_long[li], sl_short[si]])
    risk = np.abs(b.c[idx] - sl)
    tp = b.c[idx] + d * 2.0 * risk
    ok = np.isfinite(sl) & (risk > 0)
    return engine.make_signals(idx[ok] + 1, d[ok], sl=sl[ok], tp=tp[ok], hold=50)


def build() -> dict[str, dict]:
    reg: dict[str, dict] = {}
    for name, pat in PATTERNS.items():
        for ex in ("h5", "h10", "rr2"):
            reg[f"{name}_{ex}"] = dict(
                fn=lambda df, p=pat, e=ex: signals(p, e, df),
                pos=None, group="candlestick", kind="event")
    return reg
