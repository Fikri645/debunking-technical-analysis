import numpy as np
import pandas as pd
import pytest

from talab import strat_influencer
from test_strategies import synth

REG = strat_influencer.build()
COLS = ["entry_idx", "dir", "sl", "tp", "sl_dist", "entry_price"]


@pytest.mark.parametrize("name", sorted(REG))
def test_no_look_ahead(name):
    df = synth(3000, seed=41)
    n = 2400
    full = REG[name]["fn"](df)
    part = REG[name]["fn"](df.iloc[:n].copy())
    full = full[full["entry_idx"] < n][COLS].reset_index(drop=True)
    part = part[part["entry_idx"] < n][COLS].reset_index(drop=True)
    pd.testing.assert_frame_equal(full, part)


@pytest.mark.parametrize("name", sorted(REG))
def test_fires(name):
    assert len(REG[name]["fn"](synth(20000, seed=9))) > 0


def test_sky_matches_known_events():
    s = strat_influencer.sky()
    # Purnama 24 Nov 2026 dan bulan baru 9 Des 2026 (dari ephem.next_full_moon / next_new_moon)
    assert s.loc["2026-11-20", "waxing"] and not s.loc["2026-12-01", "waxing"]
    assert s.loc["2026-12-12", "waxing"]
    # Matahari di Scorpio (air) pada 16 Nov, Leo (api) pada 1 Agustus
    assert s.loc["2026-11-16", "element"] == "air" and s.loc["2026-08-01", "element"] == "api"
    # Merkurius retrograde kira-kira 3 x setahun, tiap kali sekitar 3 minggu
    frac = s["retro"].mean()
    assert 0.15 < frac < 0.25
    starts = (s["retro"] & ~s["retro"].shift(1, fill_value=False)).resample("YS").sum()
    assert starts.iloc[1:-1].between(3, 4).all()
