import numpy as np
import pytest

from talab import strat_momentum
from test_strategies import synth

REG = strat_momentum.build()


@pytest.mark.parametrize("name", sorted(REG))
def test_no_look_ahead(name):
    df = synth()
    n = 1200
    full = np.asarray(REG[name]["pos"](df))[:n]
    part = np.asarray(REG[name]["pos"](df.iloc[:n]))
    assert np.array_equal(full, part)


def test_momentum_is_sign_of_past_return():
    df = synth()
    c = df["close"].values
    pos = REG["momentum_polos_20_ls"]["pos"](df)
    assert (pos[:20] == 0).all()
    assert np.array_equal(pos[20:], np.sign(c[20:] - c[:-20]).astype(np.int64))
    lo = REG["momentum_polos_20_lo"]["pos"](df)
    assert np.array_equal(lo, np.maximum(pos, 0))
    assert len(REG["momentum_polos_20_ls"]["fn"](df)) > 0
