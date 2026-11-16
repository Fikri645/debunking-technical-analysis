import numpy as np
import pandas as pd
import pytest

from talab import strat_extra
from test_strategies import synth

REG = strat_extra.build()
COLS = ["entry_idx", "dir", "sl", "tp", "sl_dist", "tp_dist", "entry_price"]


@pytest.mark.parametrize("name", sorted(REG))
def test_no_look_ahead(name):
    df = synth(4000, seed=23)
    n = 3100
    spec = REG[name]
    if spec["pos"] is not None:
        full = np.asarray(spec["pos"](df))[:n]
        part = np.asarray(spec["pos"](df.iloc[:n].copy()))
        assert np.array_equal(full, part)
    else:
        full = spec["fn"](df)
        part = spec["fn"](df.iloc[:n].copy())
        full = full[full["entry_idx"] < n][COLS].reset_index(drop=True)
        part = part[part["entry_idx"] < n][COLS].reset_index(drop=True)
        pd.testing.assert_frame_equal(full, part)


@pytest.mark.parametrize("name", sorted(REG))
def test_fires(name):
    assert len(REG[name]["fn"](synth(30000, seed=5))) > 0


def _bars(close):
    close = np.asarray(close, dtype=float)
    o = np.roll(close, 1)
    o[0] = close[0]
    df = pd.DataFrame({"open": o, "high": np.maximum(o, close) + 0.2, "low": np.minimum(o, close) - 0.2,
                       "close": close, "volume": 1.0})
    df.index = pd.date_range("2021-01-01", periods=len(df), freq="D", tz="UTC")
    return df


def test_rounding_bottom_detected():
    x = np.linspace(-1, 1, 40)
    cup = 100 + 12 * x * x - 12                       # dari 100 turun ke 88 lalu kembali ke 100
    close = np.concatenate([np.full(30, 100.0) + 0.3 * np.sin(np.arange(30)), cup, [100.8, 101.5, 102.0]])
    idx, d, sl, tp, start, rim = strat_extra.rounding_raw(_bars(close))
    assert (d == 1).any()
    k = np.flatnonzero(d == 1)[0]
    assert 68 <= idx[k] <= 73                          # tembus bibir tepat setelah cangkir selesai
    assert sl[k] < 89 and tp[k] > 108


def test_daily_trend_uses_completed_days_only():
    df = synth(3000, seed=3)
    t = strat_extra.daily_trend(df)
    day = df.index.normalize()
    # nilai tren konstan sepanjang satu hari, dan tidak berubah kalau hari itu dipotong di tengah
    assert all(len(set(t[day == d])) == 1 for d in day.unique()[:60])
    cut = 2990
    assert np.array_equal(t[:cut], strat_extra.daily_trend(df.iloc[:cut]))


def test_premium_discount_keeps_right_side():
    df = synth(8000, seed=12)
    from talab import strat_smc
    res = strat_extra.fvg_pd_raw(df)
    st = strat_smc._structure(df)
    mid = (st[5][res[0] - 1] + st[6][res[0] - 1]) / 2
    assert len(res[0]) > 0
    assert ((res[2] == 1) <= (res[1] < mid)).all() and ((res[2] == -1) <= (res[1] > mid)).all()
