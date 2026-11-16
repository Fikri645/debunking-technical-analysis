import numpy as np
import pandas as pd

from talab import engine, evaluate


def make_df(o, h, l, c):
    idx = pd.date_range("2020-01-01", periods=len(o), freq="D", tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1.0}, index=idx)


def random_df(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = np.roll(c, 1)
    o[0] = 100
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, n)))
    return make_df(o, h, l, c)


def run_raw(df, sig):
    a = np.ones(len(df))
    return engine.run(df, sig, 0.0, a)


def test_entry_is_next_bar_open():
    df = make_df([10, 11, 12, 13, 14], [10, 11, 12, 13, 14], [10, 11, 12, 13, 14], [10, 11, 12, 13, 14])
    mask = np.array([False, True, False, False, False])
    sig = engine.signals_from_events(mask, ~mask & False, hold=2)
    tr = run_raw(df, sig)
    assert tr["entry_idx"].tolist() == [2]
    assert tr["entry_price"].tolist() == [12]
    assert tr["exit_idx"].tolist() == [4] and tr["exit_price"].tolist() == [14]


def test_stop_checked_before_target_in_same_bar():
    df = make_df([100, 100, 100, 100], [100, 100, 110, 100], [100, 100, 90, 100], [100, 100, 100, 100])
    sig = engine.make_signals([2], [1], sl=[95.0], tp=[105.0], hold=5)
    tr = run_raw(df, sig)
    assert tr["exit_price"].tolist() == [95.0]


def test_short_target():
    df = make_df([100, 100, 100, 100], [100, 100, 101, 100], [100, 100, 94, 100], [100, 100, 96, 100])
    sig = engine.make_signals([2], [-1], sl=[105.0], tp=[95.0], hold=5)
    tr = run_raw(df, sig)
    assert tr["exit_price"].tolist() == [95.0]
    assert np.isclose(tr["gross"].iloc[0], 0.05)


def test_gap_through_stop_exits_at_open():
    df = make_df([100, 100, 100, 80, 80], [100, 100, 101, 81, 80], [100, 100, 99, 79, 80], [100, 100, 100, 80, 80])
    sig = engine.make_signals([2], [1], sl=[95.0], tp=[120.0], hold=5)
    tr = run_raw(df, sig)
    assert tr["exit_idx"].tolist() == [3] and tr["exit_price"].tolist() == [80]


def test_one_position_at_a_time():
    df = random_df(200)
    sig = engine.make_signals(np.arange(5, 150), np.ones(145, dtype=int), hold=10)
    tr = run_raw(df, sig)
    assert (tr["entry_idx"].values[1:] >= tr["exit_idx"].values[:-1]).all()
    assert len(tr) == 15


def test_limit_entry_skips_target_on_fill_bar():
    df = make_df([100, 100, 100, 100], [100, 100, 110, 110], [100, 100, 97, 100], [100, 100, 100, 100])
    sig = engine.make_signals([2], [1], sl=[90.0], tp=[105.0], hold=5,
                              entry_price=[98.0], is_limit=[True])
    tr = run_raw(df, sig)
    # TP tidak boleh dihitung di candle pengisian limit; baru kena di candle berikutnya
    assert tr["exit_idx"].tolist() == [3] and tr["exit_price"].tolist() == [105.0]


def test_position_runs():
    pos = np.array([0, 1, 1, 0, -1, -1, -1, 0])
    sig = engine.signals_from_position(pos)
    assert sig["entry_idx"].tolist() == [2, 5]
    assert sig["dir"].tolist() == [1, -1]
    assert sig["hold"].tolist() == [2, 3]


def test_null_with_zero_shift_reproduces_real_trades():
    df = random_df()
    a = engine.atr(df)
    rng = np.random.default_rng(0)
    mask = (rng.random(len(df)) < 0.03) & np.isfinite(a)
    sig = engine.signals_from_events(mask, np.zeros(len(df), bool), hold=20,
                                     sl_dist=1.5 * a, tp_dist=3.0 * a)
    tr = engine.run(df, sig, 0.001, a)
    assert len(tr) > 30
    wa = np.full(len(tr), df.index[0].value)
    wb = np.full(len(tr), df.index[-1].value)
    s_rn, s_net, cnt = engine.null_sums(df, tr, 0.001, np.array([0.0]), wa, wb, a)
    assert cnt[0] == len(tr)
    assert np.isclose(s_net[0], tr["net"].sum())
    assert np.isclose(s_rn[0], tr["rg"].sum())


def test_look_ahead_signal_is_detected_as_significant():
    """Kontrol positif: strategi yang mengintip masa depan harus terdeteksi."""
    df = random_df(4000, seed=3)
    c = df["close"].values
    future_up = np.zeros(len(c), bool)
    future_up[:-6] = c[6:] > c[:-6] * 1.01
    rows = evaluate.evaluate(
        lambda d: engine.signals_from_events(future_up, np.zeros(len(c), bool), hold=5),
        "us", "1d", K=200, cell={"X": df})
    assert rows[-1]["era"] == "ALL" and rows[-1]["p"] < 0.01


def test_random_signal_is_not_significant():
    df = random_df(4000, seed=4)
    rng = np.random.default_rng(5)
    mask = rng.random(len(df)) < 0.05
    rows = evaluate.evaluate(
        lambda d: engine.signals_from_events(mask, np.zeros(len(df), bool), hold=5),
        "us", "1d", K=200, cell={"X": df})
    assert rows[-1]["p"] > 0.01


def test_benjamini_hochberg():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.6, np.nan])
    assert evaluate.benjamini_hochberg(p, 0.05).tolist() == [True, True, False, False, False, False]


def test_permute_bars_keeps_candle_shapes():
    df = random_df(500)
    p = evaluate.permute_bars(df, 1)
    assert np.isclose(np.sort(np.log(df["high"] / df["open"])), np.sort(np.log(p["high"] / p["open"]))).all()
    assert (p["high"] >= p[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (p["low"] <= p[["open", "close"]].min(axis=1) + 1e-9).all()


def test_dead_market_bars_are_ignored():
    """Saham yang tidak bergerak (ATR ~ 0) tidak boleh menghasilkan angka raksasa."""
    df = random_df(600)
    flat = df.copy()
    flat.iloc[200:400, :4] = 100.0
    a = engine.atr(flat)
    sig = engine.make_signals(np.arange(50, 550, 7), np.ones(len(np.arange(50, 550, 7)), dtype=int), hold=5)
    tr = engine.run(flat, sig, 0.0, a)
    assert np.isfinite(tr["rg"]).all() and tr["rg"].abs().max() < 100
    wa = np.full(len(tr), flat.index[0].value)
    wb = np.full(len(tr), flat.index[-1].value)
    s_rg, _, cnt = engine.null_sums(flat, tr, 0.0, np.linspace(0.05, 0.95, 50), wa, wb, a)
    assert np.abs(s_rg / np.maximum(cnt, 1)).max() < 100
