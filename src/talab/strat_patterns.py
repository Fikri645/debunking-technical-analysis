"""Strategi tahap 3: pola chart (definisi A dan B), S/R, trendline, Fibonacci."""

from __future__ import annotations

import weakref

import numpy as np
import pandas as pd

from . import engine, levels, patterns
from . import indicators as ta

LEVEL_SCALES = (2.0, 4.0)
FIB_REAL = {"382": 0.382, "500": 0.5, "618": 0.618, "786": 0.786}
FIB_FAKE = {"300": 0.30, "440": 0.44, "560": 0.56, "700": 0.70, "850": 0.85}

_cache: dict = {}


def _memo(df: pd.DataFrame, key, fn):
    """Simpan hasil perhitungan mahal per DataFrame (zigzag, deteksi Lo dkk.)."""
    k = (id(df), len(df), key)
    hit = _cache.get(k)
    # id() bisa dipakai ulang setelah objek lama dihapus, jadi pastikan objeknya sama
    if hit is not None and hit[0]() is df:
        return hit[1]
    if len(_cache) > 4000:
        _cache.clear()
    value = fn()
    _cache[k] = (weakref.ref(df), value)
    return value


def _ohlc(df):
    return tuple(df[k].values.astype(float) for k in ("open", "high", "low", "close"))


def _zz(df, mult):
    def f():
        o, h, l, c = _ohlc(df)
        return patterns.zigzag(h, l, c, ta.atr(h, l, c), mult)
    return _memo(df, ("zz", mult), f)


# ---------- definisi B ----------

def pattern_b(df: pd.DataFrame, name: str, tol: int, exit_rule: str) -> pd.DataFrame:
    det = _memo(df, ("det", name, tol), lambda: patterns.detect(df, name, tol))
    if det.empty:
        return engine.empty_signals()
    idx = det["idx"].values.astype(np.int64)
    d = det["dir"].values.astype(np.int64)
    if exit_rule == "h10":
        return engine.make_signals(idx + 1, d, hold=10)
    return engine.make_signals(idx + 1, d, sl=det["sl"].values, tp=det["tp"].values, hold=100)


# ---------- definisi A (Lo, Mamaysky & Wang 2000) ----------

LMW_GROUPS = {
    "lmw_head_shoulders": (1, 2),
    "lmw_broadening": (3, 4),
    "lmw_triangle": (5, 6),
    "lmw_rectangle": (7, 8),
    "lmw_double": (9, 10),
}


def pattern_a(df: pd.DataFrame, name: str, exit_rule: str) -> pd.DataFrame:
    codes = _memo(df, "lmw", lambda: patterns.lmw(df["close"].values.astype(float)))
    top, bottom = LMW_GROUPS[name]
    short_mask, long_mask = codes == top, codes == bottom
    if exit_rule == "h10":
        return engine.signals_from_events(long_mask, short_mask, hold=10)
    a = engine.atr(df)
    ok = np.isfinite(a)
    return engine.signals_from_events(long_mask & ok, short_mask & ok, hold=50,
                                      sl_dist=1.5 * a, tp_dist=3.0 * a)


# ---------- S/R, trendline ----------

def _bracket(parts) -> pd.DataFrame:
    parts = [p for p in parts if len(p[0])]
    if not parts:
        return engine.empty_signals()
    idx = np.concatenate([p[0] for p in parts])
    return engine.make_signals(idx + 1, np.concatenate([p[1] for p in parts]),
                               sl=np.concatenate([p[2] for p in parts]),
                               tp=np.concatenate([p[3] for p in parts]), hold=50)


def sr(df: pd.DataFrame, mode: int, min_touch: int) -> pd.DataFrame:
    o, h, l, c = _ohlc(df)
    a = ta.atr(h, l, c)
    parts = []
    for mult in LEVEL_SCALES:
        p_idx, p_px, _, p_conf = _zz(df, mult)
        parts.append(levels.sr_signals(o, h, l, c, a, p_idx, p_px, p_conf, 250, min_touch, mode))
    return _bracket(parts)


def trendline(df: pd.DataFrame, mode: int) -> pd.DataFrame:
    o, h, l, c = _ohlc(df)
    a = ta.atr(h, l, c)
    parts = []
    for mult in LEVEL_SCALES:
        p_idx, p_px, p_type, p_conf = _zz(df, mult)
        parts.append(levels.trendline_signals(o, h, l, c, a, p_idx, p_px, p_type, p_conf, mode))
    return _bracket(parts)


# ---------- Fibonacci ----------

def fib(df: pd.DataFrame, level: float, ext: float) -> pd.DataFrame:
    o, h, l, c = _ohlc(df)
    parts = []
    for mult in patterns.SCALES:
        p_idx, p_px, p_type, p_conf = _zz(df, mult)
        parts.append(levels.fib_signals(o, h, l, p_idx, p_px, p_type, p_conf, level, ext))
    parts = [p for p in parts if len(p[0])]
    if not parts:
        return engine.empty_signals()
    m = sum(len(p[0]) for p in parts)
    return engine.make_signals(
        np.concatenate([p[0] for p in parts]), np.concatenate([p[2] for p in parts]),
        sl=np.concatenate([p[3] for p in parts]), tp=np.concatenate([p[4] for p in parts]),
        entry_price=np.concatenate([p[1] for p in parts]), is_limit=np.ones(m, dtype=bool), hold=100)


def build() -> dict[str, dict]:
    reg: dict[str, dict] = {}

    def add(name, fn, group):
        reg[name] = dict(fn=fn, pos=None, group=group, kind="event")

    for name in patterns.DETECTORS:
        for tol, tname in enumerate(patterns.TOL_NAMES):
            for ex in ("mm", "h10"):
                add(f"{name}_{tname}_{ex}",
                    lambda df, n=name, t=tol, e=ex: pattern_b(df, n, t, e), "pola_chart_B")
    for name in LMW_GROUPS:
        for ex in ("h10", "rr2"):
            add(f"{name}_{ex}", lambda df, n=name, e=ex: pattern_a(df, n, e), "pola_chart_A")
    for mname, mode in (("bounce", 0), ("breakout", 1), ("retest", 2)):
        for touch in (2, 3):
            add(f"sr_{mname}_{touch}touch", lambda df, m=mode, t=touch: sr(df, m, t), "support_resistance")
    add("trendline_bounce", lambda df: trendline(df, 0), "trendline")
    add("trendline_break", lambda df: trendline(df, 1), "trendline")
    for tag, lv in FIB_REAL.items():
        add(f"fib_{tag}", lambda df, v=lv: fib(df, v, 0.0), "fibonacci")
        add(f"fib_{tag}_ext1272", lambda df, v=lv: fib(df, v, 0.272), "fibonacci")
    for tag, lv in FIB_FAKE.items():
        add(f"fibfake_{tag}", lambda df, v=lv: fib(df, v, 0.0), "fibonacci_palsu")
        add(f"fibfake_{tag}_ext1272", lambda df, v=lv: fib(df, v, 0.272), "fibonacci_palsu")
    return reg


def swing_break(df: pd.DataFrame, exit_rule: str) -> pd.DataFrame:
    """Kontrol: tembus swing terakhir tanpa syarat pola (lihat levels.swing_break_signals)."""
    o, h, l, c = _ohlc(df)
    idx, d = [], []
    for mult in patterns.SCALES:
        p_idx, p_px, p_type, p_conf = _zz(df, mult)
        i, dd = levels.swing_break_signals(c, p_idx, p_px, p_type, p_conf, 50)
        idx.append(i)
        d.append(dd)
    idx, d = np.concatenate(idx), np.concatenate(d)
    if len(idx) == 0:
        return engine.empty_signals()
    if exit_rule == "h10":
        return engine.make_signals(idx + 1, d, hold=10)
    a = engine.atr(df)
    ok = np.isfinite(a[idx])
    return engine.make_signals(idx[ok] + 1, d[ok], sl_dist=1.5 * a[idx[ok]], tp_dist=3.0 * a[idx[ok]], hold=50)


def build_controls() -> dict[str, dict]:
    return {
        f"kontrol_swing_break_{ex}": dict(fn=lambda df, e=ex: swing_break(df, e), pos=None,
                                          group="kontrol", kind="event")
        for ex in ("h10", "rr2")
    }
