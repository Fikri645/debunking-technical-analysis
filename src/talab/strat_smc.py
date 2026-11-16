"""Strategi tahap 4: supply-demand, SMC/ICT, Elliott Wave."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import elliott, engine, patterns, smc
from . import indicators as ta
from .strat_patterns import _memo, _ohlc, _zz

SWING_K = 3
# Killzone ICT dalam UTC: London 07-10, New York 12-15 (tanpa penyesuaian DST)
KILLZONE_HOURS = (7, 8, 9, 12, 13, 14)


def _atr(df):
    return _memo(df, "atr", lambda: engine.atr(df))


def _structure(df):
    def f():
        o, h, l, c = _ohlc(df)
        ph, pl = ta.pivots(h, l, SWING_K)
        return smc.structure(h, l, c, ph, pl)
    return _memo(df, "structure", f)


def _limit(res, hold=100) -> pd.DataFrame:
    e_idx, e_px, d, sl, tp = res[:5]
    if len(e_idx) == 0:
        return engine.empty_signals()
    return engine.make_signals(e_idx, d, sl=sl, tp=tp, entry_price=e_px,
                               is_limit=np.ones(len(e_idx), dtype=bool), hold=hold)


# ---------- supply & demand ----------

def sd_raw(df, dep_mult, max_base, rr, mode):
    o, h, l, c = _ohlc(df)
    return smc.sd_signals(o, h, l, c, _atr(df), dep_mult, max_base, rr, mode, 200)


def sd(df, dep_mult, max_base, rr, mode) -> pd.DataFrame:
    return _limit(sd_raw(df, dep_mult, max_base, rr, mode))


# ---------- SMC / ICT ----------

def structure_break(df, kind: int, exit_rule: str) -> pd.DataFrame:
    brk, knd = _structure(df)[:2]
    lm, sm = (brk == 1) & (knd == kind), (brk == -1) & (knd == kind)
    if exit_rule == "h10":
        return engine.signals_from_events(lm, sm, hold=10)
    a = _atr(df)
    ok = np.isfinite(a)
    return engine.signals_from_events(lm & ok, sm & ok, hold=50, sl_dist=1.5 * a, tp_dist=3.0 * a)


def sweep(df, exit_rule: str) -> pd.DataFrame:
    o, h, l, c = _ohlc(df)
    sw = _structure(df)[2]
    lm, sm = sw == 1, sw == -1
    if exit_rule == "h10":
        return engine.signals_from_events(lm, sm, hold=10)
    a = _atr(df)
    idx = np.flatnonzero((lm | sm) & np.isfinite(a))
    d = sw[idx]
    sl = np.where(d == 1, l[idx], h[idx]) - d * smc.BUF * a[idx]
    risk = d * (c[idx] - sl)
    ok = risk > 0
    return engine.make_signals(idx[ok] + 1, d[ok], sl=sl[ok], tp=(c[idx] + d * 2.0 * risk)[ok], hold=50)


def fvg_raw(df, use_trend, rr):
    o, h, l, c = _ohlc(df)
    return smc.fvg_signals(o, h, l, c, _atr(df), _structure(df)[3], use_trend, rr, 50)


def ob_raw(df, rr):
    o, h, l, c = _ohlc(df)
    st = _structure(df)
    return smc.ob_signals(o, h, l, c, _atr(df), st[0], st[4], rr, 100)


def ict_raw(df, rr):
    o, h, l, c = _ohlc(df)
    st = _structure(df)
    return smc.ict2022_signals(o, h, l, c, _atr(df), st[2], st[5], st[6], rr, 20, 20)


def ict(df, rr, killzone: bool) -> pd.DataFrame:
    res = ict_raw(df, rr)
    if killzone and len(res[0]):
        hours = df.index.hour.values[res[0]]
        keep = np.isin(hours, KILLZONE_HOURS)
        res = tuple(x[keep] for x in res)
    return _limit(res)


# ---------- Elliott Wave ----------

def ew_detect(df, setup: int, use_fib: bool) -> pd.DataFrame:
    """Semua hitungan Elliott yang sah (impuls naik dan turun, semua skala)."""
    frames = []
    for mult in patterns.SCALES:
        p_idx, p_px, p_type, p_conf = _zz(df, mult)
        if len(p_idx) < 8:
            continue
        for sign in (1.0, -1.0):
            s_idx, s_dir, s_sl, s_tp, pts_i, pts_p = elliott.ew_signals(
                p_idx, p_px * sign, (p_type * sign).astype(np.int64), p_conf, setup, use_fib)
            if len(s_idx) == 0:
                continue
            f = pd.DataFrame({"idx": s_idx, "dir": (s_dir * sign).astype(np.int64),
                              "sl": s_sl * sign, "tp": s_tp * sign, "scale": mult})
            for q in range(elliott.NPTS):
                f[f"pi{q}"] = pts_i[:, q]
                f[f"pp{q}"] = pts_p[:, q] * sign
            frames.append(f)
    if not frames:
        return pd.DataFrame(columns=["idx", "dir", "sl", "tp", "scale"])
    return pd.concat(frames, ignore_index=True).sort_values("idx", kind="stable").reset_index(drop=True)


def ew(df, setup: int, use_fib: bool, exit_rule: str) -> pd.DataFrame:
    det = _memo(df, ("ew", setup, use_fib), lambda: ew_detect(df, setup, use_fib))
    if det.empty:
        return engine.empty_signals()
    idx = det["idx"].values.astype(np.int64)
    d = det["dir"].values.astype(np.int64)
    if exit_rule == "h10":
        return engine.make_signals(idx + 1, d, hold=10)
    return engine.make_signals(idx + 1, d, sl=det["sl"].values, tp=det["tp"].values, hold=200)


def build() -> dict[str, dict]:
    reg: dict[str, dict] = {}

    def add(name, fn, group, **flags):
        reg[name] = dict(fn=fn, pos=None, group=group, kind="event", **flags)

    for rr in (2, 3):
        add(f"sd_semua_rr{rr}", lambda df, r=rr: sd(df, 1.5, 6, float(r), 0), "supply_demand")
        add(f"sd_kuat_rr{rr}", lambda df, r=rr: sd(df, 2.5, 3, float(r), 0), "supply_demand")
    add("sd_pembalikan_rr2", lambda df: sd(df, 1.5, 6, 2.0, 1), "supply_demand")
    add("sd_lanjutan_rr2", lambda df: sd(df, 1.5, 6, 2.0, 2), "supply_demand")

    add("smc_bos_h10", lambda df: structure_break(df, 1, "h10"), "smc_ict")
    add("smc_bos_rr2", lambda df: structure_break(df, 1, "rr2"), "smc_ict")
    add("smc_choch_h10", lambda df: structure_break(df, 2, "h10"), "smc_ict")
    add("smc_choch_rr2", lambda df: structure_break(df, 2, "rr2"), "smc_ict")
    add("smc_sweep_h10", lambda df: sweep(df, "h10"), "smc_ict")
    add("smc_sweep_rr2", lambda df: sweep(df, "rr2"), "smc_ict")
    add("smc_fvg_rr2", lambda df: _limit(fvg_raw(df, False, 2.0)), "smc_ict")
    add("smc_fvg_tren_rr2", lambda df: _limit(fvg_raw(df, True, 2.0)), "smc_ict")
    add("smc_orderblock_rr2", lambda df: _limit(ob_raw(df, 2.0)), "smc_ict")
    add("smc_orderblock_rr3", lambda df: _limit(ob_raw(df, 3.0)), "smc_ict")
    add("ict2022_rr2", lambda df: ict(df, 2.0, False), "smc_ict")
    add("ict2022_rr3", lambda df: ict(df, 3.0, False), "smc_ict")
    add("ict2022_killzone_rr3", lambda df: ict(df, 3.0, True), "smc_ict", intraday=True)

    for sname, setup in (("wave3", 0), ("wave5", 1), ("abc", 2)):
        for fname, fib in (("aturan", False), ("fib", True)):
            add(f"elliott_{sname}_{fname}", lambda df, s=setup, f=fib: ew(df, s, f, "native"), "elliott")
            add(f"elliott_{sname}_{fname}_h10", lambda df, s=setup, f=fib: ew(df, s, f, "h10"), "elliott")
    return reg
