"""Uji anti-mengintip yang lebih rapat untuk SEMUA strategi.

Uji per-berkas hanya memotong data di satu titik dan mengabaikan sinyal di candle
terakhir, sehingga bocoran satu candle (sinyal di candle i yang memakai candle i+1)
bisa lolos. Di sini data dipotong di banyak titik, sinyal di candle terakhir
potongan ikut dibandingkan, dan semua kolom sinyal diperiksa.
"""

import numpy as np
import pandas as pd
import pytest

from talab import engine, strat_candles, strat_extra, strat_indicators, strat_influencer, strat_momentum, strat_patterns, strat_smc
from test_strategies import synth

REG = {}
for _build in (strat_indicators.build, strat_candles.build, strat_patterns.build, strat_patterns.build_controls,
               strat_smc.build, strat_influencer.build, strat_momentum.build, strat_extra.build):
    REG.update(_build())

FULL = synth(3000, seed=41)
CUTS = (1703, 1856, 2011, 2164, 2317, 2473, 2622, 2779, 2931, 2998)
PREFIX = {n: FULL.iloc[:n].copy() for n in CUTS}
COLS = ["entry_idx", "dir", "sl", "tp", "sl_dist", "tp_dist", "entry_price", "is_limit"]


def _known(sig, n):
    """Sinyal yang sudah pasti diketahui dari candle 0..n-1."""
    lim = sig["is_limit"].values
    keep = np.where(lim, sig["entry_idx"].values < n, sig["entry_idx"].values <= n)
    return sig[keep].reset_index(drop=True)


@pytest.mark.parametrize("name", sorted(REG))
def test_dense_no_look_ahead(name):
    spec = REG[name]
    if spec["pos"] is not None:
        full = np.asarray(spec["pos"](FULL))
        for n in CUTS:
            assert np.array_equal(full[:n], np.asarray(spec["pos"](PREFIX[n]))), f"posisi berubah, potongan {n}"
        return
    full = spec["fn"](FULL)
    for n in CUTS:
        part = spec["fn"](PREFIX[n])
        # sinyal market di candle n-1 punya entry_idx = n; limit order terisi paling lambat di n-1
        f = _known(full, n)
        p = _known(part, n)
        pd.testing.assert_frame_equal(f[COLS], p[COLS], obj=f"{name} potongan {n}")
        # batas lama pegang boleh bergantung pada masa depan hanya kalau keluarnya memang
        # terjadi setelah potongan
        known = (f["entry_idx"] + f["hold"] < n).values
        assert np.array_equal(f["hold"].values[known], p["hold"].values[known]), f"hold berubah, potongan {n}"


def test_dense_check_catches_one_bar_leak():
    """Strategi yang sengaja mengintip satu candle harus tertangkap oleh pemeriksaan di atas."""
    def leaky(df):
        c = df["close"].values
        up = np.zeros(len(c), dtype=bool)
        up[:-1] = c[1:] > c[:-1]          # tahu penutupan candle berikutnya
        return engine.signals_from_events(up, np.zeros(len(c), dtype=bool), hold=5)

    full = leaky(FULL)
    differs = False
    for n in CUTS:
        f = full[full["entry_idx"] <= n][COLS].reset_index(drop=True)
        p = leaky(PREFIX[n])
        p = p[p["entry_idx"] <= n][COLS].reset_index(drop=True)
        differs |= not f.equals(p)
    assert differs
