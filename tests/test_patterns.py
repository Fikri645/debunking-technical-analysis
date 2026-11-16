import numpy as np
import pandas as pd
import pytest

from talab import indicators as ta
from talab import patterns, strat_patterns
from test_strategies import synth


REG = strat_patterns.build()
COLS = ["entry_idx", "dir", "sl", "tp", "entry_price"]


@pytest.mark.parametrize("name", sorted(REG))
def test_no_look_ahead(name):
    """Transaksi yang dibuka sebelum candle N tidak boleh berubah ketika data masa
    depan ditambahkan. (Sinyal di ujung data boleh berbeda hanya karena data habis.)"""
    df = synth(3000, seed=21)
    n = 2400
    full = REG[name]["fn"](df)
    part = REG[name]["fn"](df.iloc[:n].copy())
    full = full[full["entry_idx"] < n][COLS].reset_index(drop=True)
    part = part[part["entry_idx"] < n][COLS].reset_index(drop=True)
    pd.testing.assert_frame_equal(full, part)


def test_most_strategies_fire_on_noise():
    """Di data acak pun pola-pola ini muncul; yang tidak muncul dicatat."""
    df = synth(20000, seed=5)
    silent = [n for n, s in REG.items() if len(s["fn"](df)) == 0]
    assert len(silent) <= 6, silent


def path_to_df(points, bars_per_leg=12, noise=0.0, seed=0):
    """Bangun candle dari titik-titik belok harga."""
    rng = np.random.default_rng(seed)
    closes = []
    for a, b in zip(points[:-1], points[1:]):
        closes += list(np.linspace(a, b, bars_per_leg, endpoint=False))
    closes.append(points[-1])
    c = np.array(closes) + rng.normal(0, noise, len(closes))
    o = np.roll(c, 1)
    o[0] = c[0]
    h = np.maximum(o, c) + 0.2
    l = np.minimum(o, c) - 0.2
    idx = pd.date_range("2021-01-01", periods=len(c), freq="D", tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1.0}, index=idx)


WARMUP = [100, 101, 100, 101, 100, 101, 100]


def test_zigzag_alternates_and_confirms_late():
    df = path_to_df(WARMUP + [120, 105, 125, 100])
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    p_idx, p_px, p_type, p_conf = patterns.zigzag(h, l, c, ta.atr(h, l, c), 2.0)
    assert (np.abs(np.diff(p_type)) == 2).all()
    assert (p_conf > p_idx).all()
    assert (np.diff(p_conf) >= 0).all()


def test_head_and_shoulders_detected_with_textbook_levels():
    # naik, bahu kiri 120, leher 110, kepala 130, leher 110, bahu kanan 120, jatuh
    df = path_to_df(WARMUP + [90, 120, 110, 130, 110, 120, 95])
    det = patterns.detect(df, "head_shoulders", 1, scales=(4.0,))
    det = det[~det["mirrored"]]
    assert len(det) == 1
    row = det.iloc[0]
    assert row["dir"] == -1
    assert abs(row["sl"] - 120.2) < 0.5          # SL di bahu kanan
    assert abs(row["tp"] - (109.8 - 20.4)) < 2.5  # TP = leher - (kepala - leher)
    assert df["close"].iloc[int(row["idx"])] < 110


def test_inverse_head_and_shoulders_is_mirror():
    df = path_to_df(WARMUP + [130, 100, 110, 90, 110, 100, 125])
    det = patterns.detect(df, "head_shoulders", 1, scales=(4.0,))
    det = det[det["mirrored"]]
    assert len(det) == 1 and det.iloc[0]["dir"] == 1
    assert det.iloc[0]["sl"] < 101 and det.iloc[0]["tp"] > 125


def test_double_top_detected():
    df = path_to_df(WARMUP + [112, 100, 110, 90, 120, 108, 120.5, 95], bars_per_leg=14)
    det = patterns.detect(df, "double_top_bottom", 1, scales=(4.0,))
    det = det[~det["mirrored"]]
    assert len(det) == 1 and det.iloc[0]["dir"] == -1


def test_straight_trend_has_no_reversal_patterns():
    df = path_to_df(WARMUP + [110, 105, 120, 115, 130, 125, 140, 135, 150])
    for name in ("head_shoulders", "double_top_bottom", "triple_top_bottom"):
        assert len(patterns.detect(df, name, 0, scales=(2.0, 4.0))) == 0


def test_lmw_finds_head_and_shoulders_in_window():
    pts = [100, 110, 104, 116, 104, 110, 100]
    c = path_to_df(pts, bars_per_leg=6)["close"].values
    c = np.concatenate([np.full(10, 100.0), c, np.full(10, 100.0)])
    codes = patterns.lmw(c)
    assert (codes == 1).any()
