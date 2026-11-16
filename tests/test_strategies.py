import numpy as np
import pandas as pd
import pytest

from talab import indicators as ta
from talab import strat_candles, strat_indicators


def synth(n=1500, seed=7, freq="h"):
    rng = np.random.default_rng(seed)
    vol = 0.01 * np.exp(0.5 * np.sin(np.arange(n) / 80))
    c = 100 * np.exp(np.cumsum(rng.normal(0, 1, n) * vol))
    o = np.roll(c, 1) * np.exp(rng.normal(0, 0.001, n))
    o[0] = 100
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, n)))
    v = rng.lognormal(10, 1, n)
    idx = pd.date_range("2021-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)


REG = {**strat_indicators.build(), **strat_candles.build()}


@pytest.mark.parametrize("name", sorted(REG))
def test_no_look_ahead(name):
    """Sinyal sampai candle N tidak boleh berubah ketika data masa depan ditambahkan."""
    df = synth()
    n = 1200
    spec = REG[name]
    if spec["pos"] is not None:
        full = np.asarray(spec["pos"](df))[:n]
        part = np.asarray(spec["pos"](df.iloc[:n]))
        assert np.array_equal(full, part)
    else:
        cols = ["entry_idx", "dir", "sl", "tp", "sl_dist", "tp_dist"]
        full = spec["fn"](df)
        full = full[full["entry_idx"] <= n][cols].reset_index(drop=True)
        part = spec["fn"](df.iloc[:n])[cols].reset_index(drop=True)
        pd.testing.assert_frame_equal(full, part)


@pytest.mark.parametrize("name", sorted(REG))
def test_produces_signals(name):
    sig = REG[name]["fn"](synth(6000, seed=11))
    assert len(sig) > 0, "strategi tidak pernah memberi sinyal di data sintetis"
    assert set(np.unique(sig["dir"])) <= {-1, 1}


def test_rsi_extremes_and_reference():
    up = np.arange(1.0, 60.0)
    assert np.allclose(ta.rsi(up)[20:], 100.0)
    assert np.allclose(ta.rsi(up[::-1])[20:], 0.0)
    # Bandingkan dengan perhitungan Wilder yang ditulis manual
    rng = np.random.default_rng(0)
    c = 100 + np.cumsum(rng.normal(0, 1, 200))
    d = np.diff(c)
    g, ls = np.where(d > 0, d, 0), np.where(d < 0, -d, 0)
    ag, al = g[:14].mean(), ls[:14].mean()
    ref = []
    for i in range(14, len(d)):
        ag = (ag * 13 + g[i]) / 14
        al = (al * 13 + ls[i]) / 14
        ref.append(100 - 100 / (1 + ag / al))
    # setelah cukup lama, perbedaan nilai awal tidak lagi berpengaruh
    assert np.allclose(ta.rsi(c)[-50:], ref[-50:], atol=0.05)


def test_sma_bollinger_macd():
    c = np.arange(1.0, 101.0)
    assert np.isclose(ta.sma(c, 20)[-1], np.mean(c[-20:]))
    mid, up, lo = ta.bollinger(c)
    assert np.isclose(up[-1] - mid[-1], 2 * np.std(c[-20:]))
    line, sig = ta.macd(c)
    assert line[-1] > 0


def test_stochastic_bounds():
    df = synth()
    k, d = ta.stochastic(df["high"].values, df["low"].values, df["close"].values)
    assert np.nanmin(k) >= 0 and np.nanmax(k) <= 100


def test_pivots_are_confirmed_late():
    h = np.array([1, 2, 3, 9, 3, 2, 1, 2, 3], dtype=float)
    l = h - 0.5
    hi, lo = ta.pivots(h, l, 2)
    assert hi[5] == 3 and (hi[:5] == -1).all()


def candles(rows):
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["volume"] = 1.0
    df.index = pd.date_range("2021-01-01", periods=len(df), freq="D", tz="UTC")
    return df


def downtrend(n=14, start=200.0, step=2.0):
    rows = []
    p = start
    for _ in range(n):
        rows.append([p, p + 0.5, p - step - 0.5, p - step])
        p -= step
    return rows, p


def test_bullish_engulfing_detected():
    rows, p = downtrend()
    rows.append([p + 0.0, p + 6, p - 2.5, p + 5])  # hijau besar menelan merah sebelumnya
    b = strat_candles.Bars(candles(rows))
    bull, bear, sl, _ = strat_candles.engulfing(b)
    assert bull[-1] and not bear[-1]
    assert sl[-1] == p - 2.5


def test_hammer_needs_downtrend():
    rows, p = downtrend()
    rows.append([p, p + 0.1, p - 6, p + 0.05 + 1.0])
    rows[-1] = [p, p + 1.1, p - 6, p + 1.0]  # badan kecil di atas, sumbu bawah panjang
    b = strat_candles.Bars(candles(rows))
    bull, bear, _, _ = strat_candles.hammer_hanging_man(b)
    assert bull[-1] and not bear[-1]
    up_rows = [[100 + 2 * i, 102.5 + 2 * i, 99.5 + 2 * i, 102 + 2 * i] for i in range(14)]
    q = up_rows[-1][3]
    up_rows.append([q, q + 1.1, q - 6, q + 1.0])
    b2 = strat_candles.Bars(candles(up_rows))
    bull2, bear2, _, _ = strat_candles.hammer_hanging_man(b2)
    assert bear2[-1] and not bull2[-1]


def test_doji_shape():
    rows, p = downtrend()
    rows.append([p, p + 3, p - 3, p + 0.1])
    b = strat_candles.Bars(candles(rows))
    assert strat_candles.doji(b)[0][-1]
