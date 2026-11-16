"""Kontrol momentum polos: tanpa indikator, tanpa pola.

Aturannya satu kalimat: kalau harga penutupan lebih tinggi dari N candle lalu,
long; kalau lebih rendah, short (atau keluar untuk varian `_lo`). Dipakai untuk
menjawab: apakah indikator tren (EMA, golden cross, Ichimoku, dst.) menambah
sesuatu di atas momentum itu sendiri?

Lookback ditetapkan sebelum melihat hasil: 10, 20, 50, 100, 200.
"""

from __future__ import annotations

import numpy as np

from . import engine
from . import indicators as ta
from .strat_indicators import _sign, arrays

LOOKBACKS = (10, 20, 50, 100, 200)


def momentum(n: int):
    def pos(df):
        c = arrays(df)[3]
        return _sign(c - ta.shift(c, n))
    return pos


def build() -> dict[str, dict]:
    reg: dict[str, dict] = {}
    for n in LOOKBACKS:
        p = momentum(n)
        reg[f"momentum_polos_{n}_ls"] = dict(
            fn=lambda df, p=p: engine.signals_from_position(p(df)),
            pos=p, group="kontrol_momentum", kind="state")
        reg[f"momentum_polos_{n}_lo"] = dict(
            fn=lambda df, p=p: engine.signals_from_position(np.maximum(p(df), 0)),
            pos=lambda df, p=p: np.maximum(p(df), 0),
            group="kontrol_momentum", kind="state")
    return reg
