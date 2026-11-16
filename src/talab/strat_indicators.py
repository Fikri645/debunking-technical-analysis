"""Strategi berbasis indikator dengan parameter standar.

Dua jenis:
- "state": fungsi mengembalikan posisi {-1,0,1} per candle (diputuskan di penutupan).
  Varian `_ls` = boleh long dan short, `_lo` = long saja.
- "event": fungsi mengembalikan (sinyal_long, sinyal_short). Tiap sinyal diuji dengan
  dua cara keluar: `_h10` (tahan 10 candle) dan `_rr2` (SL 1,5 ATR, TP 3 ATR, RR 1:2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import engine
from . import indicators as ta


def arrays(df: pd.DataFrame):
    return tuple(df[k].values.astype(float) for k in ("open", "high", "low", "close", "volume"))


def _sign(x) -> np.ndarray:
    return np.nan_to_num(np.sign(x)).astype(np.int64)


# ---------- state ----------

def sma_cross(f, s):
    def pos(df):
        c = arrays(df)[3]
        return _sign(ta.sma(c, f) - ta.sma(c, s))
    return pos


def ema_cross(f, s):
    def pos(df):
        c = arrays(df)[3]
        return _sign(ta.ema(c, f) - ta.ema(c, s))
    return pos


def price_vs_ema(n):
    def pos(df):
        c = arrays(df)[3]
        return _sign(c - ta.ema(c, n))
    return pos


def macd_signal(df):
    line, sig = ta.macd(arrays(df)[3])
    return _sign(line - sig)


def macd_zero(df):
    line, _ = ta.macd(arrays(df)[3])
    return _sign(line)


def rsi_obos(df):
    """Beli saat oversold (RSI < 30), jual saat overbought (RSI > 70)."""
    r = ta.rsi(arrays(df)[3])
    lo, hi = r < 30, r > 70
    return ta.latch(lo, hi, hi, lo)


def stoch_obos(df):
    _, h, l, c, _ = arrays(df)
    k, d = ta.stochastic(h, l, c)
    buy = ta.cross_up(k, d) & (k < 20)
    sell = ta.cross_down(k, d) & (k > 80)
    return ta.latch(buy, sell, sell, buy)


def bb_reversion(df):
    c = arrays(df)[3]
    mid, up, lo = ta.bollinger(c)
    return ta.latch(c < lo, c >= mid, c > up, c <= mid)


def bb_breakout(df):
    c = arrays(df)[3]
    mid, up, lo = ta.bollinger(c)
    return ta.latch(c > up, c < mid, c < lo, c > mid)


def ichimoku(df):
    _, h, l, c, _ = arrays(df)
    tenkan, kijun, sa, sb = ta.ichimoku(h, l)
    top, bot = np.fmax(sa, sb), np.fmin(sa, sb)
    pos = np.zeros(len(c), dtype=np.int64)
    pos[(c > top) & (tenkan > kijun)] = 1
    pos[(c < bot) & (tenkan < kijun)] = -1
    return pos


def supertrend(df):
    _, h, l, c, _ = arrays(df)
    return ta.supertrend(h, l, c, ta.atr(h, l, c, 10), 3.0)


def psar(df):
    _, h, l, _, _ = arrays(df)
    return ta.psar(h, l, 0.02, 0.2)


def adx_di(df):
    _, h, l, c, _ = arrays(df)
    a, pdi, mdi = ta.adx(h, l, c)
    pos = _sign(pdi - mdi)
    pos[~(a > 25)] = 0
    return pos


def donchian(df):
    _, h, l, c, _ = arrays(df)
    hi20, lo20 = ta.shift(ta.rolling_max(h, 20), 1), ta.shift(ta.rolling_min(l, 20), 1)
    hi10, lo10 = ta.shift(ta.rolling_max(h, 10), 1), ta.shift(ta.rolling_min(l, 10), 1)
    return ta.latch(c > hi20, c < lo10, c < lo20, c > hi10)


def obv_trend(df):
    _, _, _, c, v = arrays(df)
    o = ta.obv(c, v)
    return _sign(o - ta.ema(o, 20))


def vwap(df):
    """Harga di atas VWAP harian = long, di bawah = short (hanya intraday)."""
    _, h, l, c, v = arrays(df)
    day = df.index.tz_convert(None).normalize()
    tp = (h + l + c) / 3
    pv = pd.Series(tp * v).groupby(day.values).cumsum().values
    vv = pd.Series(v).groupby(day.values).cumsum().values
    with np.errstate(divide="ignore", invalid="ignore"):
        vw = pv / vv
    return _sign(c - vw)


def confluence(df):
    """Konfluensi populer: tren EMA200 + MACD + RSI searah."""
    c = arrays(df)[3]
    e = ta.ema(c, 200)
    line, sig = ta.macd(c)
    r = ta.rsi(c)
    pos = np.zeros(len(c), dtype=np.int64)
    pos[(c > e) & (line > sig) & (r > 50)] = 1
    pos[(c < e) & (line < sig) & (r < 50)] = -1
    return pos


# ---------- event ----------

def rsi_cross(df):
    r = ta.rsi(arrays(df)[3])
    return ta.cross_up(r, 30.0), ta.cross_down(r, 70.0)


def rsi_divergence(df):
    _, h, l, c, _ = arrays(df)
    d = ta.divergence(h, l, ta.rsi(c), 5, 60)
    return d == 1, d == -1


def macd_cross(df):
    line, sig = ta.macd(arrays(df)[3])
    return ta.cross_up(line, sig), ta.cross_down(line, sig)


def macd_divergence(df):
    _, h, l, c, _ = arrays(df)
    d = ta.divergence(h, l, ta.macd(c)[0], 5, 60)
    return d == 1, d == -1


def stoch_cross(df):
    _, h, l, c, _ = arrays(df)
    k, d = ta.stochastic(h, l, c)
    return ta.cross_up(k, d) & (k < 20), ta.cross_down(k, d) & (k > 80)


def bb_reentry(df):
    """Harga ditutup kembali ke dalam band setelah sempat ditutup di luarnya."""
    c = arrays(df)[3]
    _, up, lo = ta.bollinger(c)
    return ta.cross_up(c, lo), ta.cross_down(c, up)


def bb_squeeze(df):
    """Breakout dari Bollinger squeeze (band tersempit dalam 120 candle)."""
    c = arrays(df)[3]
    mid, up, lo = ta.bollinger(c)
    with np.errstate(divide="ignore", invalid="ignore"):
        bw = (up - lo) / mid
    tight = bw <= ta.rolling_min(bw, 120) * 1.1
    recent = pd.Series(tight).shift(1).rolling(5).max().fillna(0).values > 0
    return ta.cross_up(c, up) & recent, ta.cross_down(c, lo) & recent


def volume_breakout(df):
    """Tembus high/low 20 candle dengan volume lebih dari 2x rata-rata."""
    _, h, l, c, v = arrays(df)
    big = v > 2 * ta.shift(ta.sma(v, 20), 1)
    hi, lo = ta.shift(ta.rolling_max(h, 20), 1), ta.shift(ta.rolling_min(l, 20), 1)
    return (c > hi) & big, (c < lo) & big


# ---------- registry ----------

STATE = {
    "sma_20_50": (sma_cross(20, 50), {}),
    "golden_cross_50_200": (sma_cross(50, 200), {}),
    "ema_9_21": (ema_cross(9, 21), {}),
    "price_vs_ema200": (price_vs_ema(200), {}),
    "macd_signal": (macd_signal, {}),
    "macd_zero": (macd_zero, {}),
    "rsi_30_70": (rsi_obos, {}),
    "stochastic_20_80": (stoch_obos, {}),
    "bollinger_reversion": (bb_reversion, {}),
    "bollinger_breakout": (bb_breakout, {}),
    "ichimoku": (ichimoku, {}),
    "supertrend_10_3": (supertrend, {}),
    "parabolic_sar": (psar, {}),
    "adx_di_25": (adx_di, {}),
    "donchian_20_10": (donchian, {}),
    "obv_ema20": (obv_trend, {"needs_volume": True}),
    "vwap": (vwap, {"needs_volume": True, "intraday": True}),
    "confluence_ema200_macd_rsi": (confluence, {}),
}

EVENT = {
    "rsi_cross_30_70": (rsi_cross, {}),
    "rsi_divergence": (rsi_divergence, {}),
    "macd_cross": (macd_cross, {}),
    "macd_divergence": (macd_divergence, {}),
    "stochastic_cross": (stoch_cross, {}),
    "bollinger_reentry": (bb_reentry, {}),
    "bollinger_squeeze": (bb_squeeze, {}),
    "volume_breakout": (volume_breakout, {"needs_volume": True}),
}


def event_signals(masks, exit_rule: str, df: pd.DataFrame) -> pd.DataFrame:
    lm, sm = masks
    if exit_rule == "h10":
        return engine.signals_from_events(lm, sm, hold=10)
    a = engine.atr(df)
    ok = np.isfinite(a)
    return engine.signals_from_events(lm & ok, sm & ok, hold=50, sl_dist=1.5 * a, tp_dist=3.0 * a)


def build() -> dict[str, dict]:
    """Nama strategi -> dict(fn=df->sinyal, group, kind, pos=fungsi posisi atau None, flags)."""
    reg: dict[str, dict] = {}
    for name, (pos_fn, flags) in STATE.items():
        reg[f"{name}_ls"] = dict(
            fn=lambda df, p=pos_fn: engine.signals_from_position(p(df)),
            pos=pos_fn, group="indikator", kind="state", **flags)
        reg[f"{name}_lo"] = dict(
            fn=lambda df, p=pos_fn: engine.signals_from_position(np.maximum(p(df), 0)),
            pos=lambda df, p=pos_fn: np.maximum(p(df), 0),
            group="indikator", kind="state", **flags)
    for name, (mask_fn, flags) in EVENT.items():
        for ex in ("h10", "rr2"):
            reg[f"{name}_{ex}"] = dict(
                fn=lambda df, m=mask_fn, e=ex: event_signals(m(df), e, df),
                pos=None, group="indikator", kind="event", **flags)
    return reg
