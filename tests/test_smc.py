import numpy as np
import pandas as pd
import pytest

from talab import engine, smc, strat_smc
from talab import indicators as ta
from test_patterns import path_to_df
from test_strategies import synth

REG = strat_smc.build()
COLS = ["entry_idx", "dir", "sl", "tp", "entry_price"]


@pytest.mark.parametrize("name", sorted(REG))
def test_no_look_ahead(name):
    df = synth(3000, seed=31)
    n = 2400
    full = REG[name]["fn"](df)
    part = REG[name]["fn"](df.iloc[:n].copy())
    full = full[full["entry_idx"] < n][COLS].reset_index(drop=True)
    part = part[part["entry_idx"] < n][COLS].reset_index(drop=True)
    pd.testing.assert_frame_equal(full, part)


@pytest.mark.parametrize("name", sorted(REG))
def test_fires(name):
    assert len(REG[name]["fn"](synth(20000, seed=9))) > 0


def bars(rows):
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["volume"] = 1.0
    df.index = pd.date_range("2021-01-01", periods=len(df), freq="D", tz="UTC")
    return df


def quiet(n, p=100.0):
    return [[p, p + 0.5, p - 0.5, p + (0.2 if i % 2 else -0.2)] for i in range(n)]


def test_demand_zone_drop_base_rally():
    rows = quiet(20)
    rows.append([100, 100.2, 96.8, 97])        # drop
    rows.append([97, 97.3, 96.7, 97.1])        # base
    rows.append([97.1, 97.4, 96.8, 97.0])      # base
    rows.append([97.0, 101.2, 96.9, 101])      # rally (keberangkatan)
    rows += [[101, 101.5, 100.5, 101.2], [101.2, 101.4, 99.5, 99.8], [99.8, 100, 97.05, 98.5],
             [98.5, 99.5, 98.2, 99.3]]
    df = bars(rows)
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    a = np.full(len(df), 1.0)
    e_idx, e_px, d, sl, tp, z0, zlo, zhi = smc.sd_signals(o, h, l, c, a, 1.5, 6, 2.0, 1, 200)
    assert len(e_idx) == 1 and d[0] == 1
    assert np.isclose(e_px[0], 97.1)           # proksimal = badan tertinggi base
    assert np.isclose(sl[0], 96.7 - 0.1)       # distal = low terendah base, dikurangi buffer
    assert e_idx[0] == 26                      # candle pertama yang kembali ke zona
    assert np.isclose(tp[0], 97.1 + 2 * (97.1 - 96.6))


def test_bullish_fvg_detected():
    rows = quiet(20)
    rows.append([100, 100.5, 99.8, 100.4])     # i-2: high 100.5
    rows.append([100.4, 104.2, 100.3, 104])    # i-1: impulsif
    rows.append([104, 105, 102.0, 104.5])      # i: low 102.0 > 100.5 -> gap 100.5..102.0
    rows += [[104.5, 105, 103, 103.2], [103.2, 103.5, 101.8, 102.5], [102.5, 104, 102.2, 103.8]]
    df = bars(rows)
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    a = np.full(len(df), 1.0)
    res = smc.fvg_signals(o, h, l, c, a, np.zeros(len(df), dtype=np.int64), False, 2.0, 50)
    assert len(res[0]) == 1 and res[2][0] == 1
    assert np.isclose(res[1][0], 102.0) and res[0][0] == 24
    assert np.isclose(res[6][0], 100.5) and np.isclose(res[7][0], 102.0)


def test_structure_break_and_sweep():
    # swing high 110 lalu swing low 100; sumbu ke 99 ditutup 101 = sweep; lalu tutup 111 = break
    df = path_to_df([105, 110, 100, 104], bars_per_leg=8)
    h, l, c = df["high"].values.copy(), df["low"].values.copy(), df["close"].values.copy()
    n = len(c)
    l[16] -= 0.1   # titik belok: pastikan low-nya unik
    h[8] += 0.1
    l[n - 1] = 99.0
    c[n - 1] = 101.0
    extra_c = np.array([103.0, 106.0, 109.0, 111.5])
    c2 = np.concatenate([c, extra_c])
    h2 = np.concatenate([h, extra_c + 0.2])
    l2 = np.concatenate([l, extra_c - 1.0])
    ph, pl = ta.pivots(h2, l2, 3)
    brk, kind, sweep, trend, leg, sh, sl = smc.structure(h2, l2, c2, ph, pl)
    assert sweep[n - 1] == 1
    assert brk[-1] == 1 and trend[-1] == 1


def test_elliott_impulse_counts():
    # turun dulu (supaya wave 1 memulai tren baru), lalu impuls 5 gelombang
    pts = [120, 112, 116, 100, 110, 105, 125, 118, 130, 120]
    df = path_to_df([100, 101, 100, 101, 100] + pts, bars_per_leg=10)
    w3 = strat_smc.ew_detect(df, 0, True)
    w3 = w3[(w3["dir"] == 1) & (w3["scale"] == 2.0)]
    assert len(w3) >= 1
    first = w3.iloc[0]
    assert abs(first["sl"] - 99.8) < 0.5                       # SL di awal wave 1
    assert abs(first["tp"] - (104.8 + 1.618 * 10.4)) < 1.0     # TP = wave 2 + 1,618 x wave 1
    w5 = strat_smc.ew_detect(df, 1, False)
    assert ((w5["dir"] == 1) & (abs(w5["sl"] - 110.2) < 0.5)).any()   # SL di puncak wave 1
    abc = strat_smc.ew_detect(df, 2, False)
    assert ((abc["dir"] == -1) & (abs(abc["tp"] - 117.8) < 0.5)).any()  # TP di akhir wave 4


def test_elliott_rejects_wave4_overlap():
    pts = [120, 112, 116, 100, 110, 105, 125, 108, 130, 120]   # wave 4 (108) masuk wilayah wave 1 (110)
    df = path_to_df([100, 101, 100, 101, 100] + pts, bars_per_leg=10)
    w5 = strat_smc.ew_detect(df, 1, False)
    w5 = w5[(w5["dir"] == 1) & (w5["scale"] == 2.0)]
    assert not (abs(w5["pp4"] - 107.8) < 0.5).any()
