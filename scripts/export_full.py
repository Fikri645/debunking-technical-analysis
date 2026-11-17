"""Ekspor SELURUH riwayat uji untuk chart interaktif di website.

- docs/data/c/<pasar>_<tf>/<simbol>.dat : semua candle instrumen itu.
- docs/data/t/<strategi>/<pasar>_<tf>.dat : semua transaksi strategi itu di semua instrumen sel
  tersebut (candle masuk, candle keluar, arah, cara keluar, SL/TP), plus titik pola atau zona
  yang dipakai bila ada.

Format berkas: gzip dari [panjang header (uint32)] [header JSON] [larik biner berurutan].
Crypto 15 menit hanya diekspor 2 tahun terakhir (LIMIT_BARS) supaya ukuran website wajar;
angka hasil uji tetap dari seluruh data.

Jalankan: uv run python scripts/export_full.py [pasar_tf ...] [--strategi nama ...]
"""

from __future__ import annotations

import gzip
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_site as es  # noqa: E402
from run_tests import CELLS, applicable  # noqa: E402
from talab import engine, evaluate, patterns, strat_extra, strat_patterns, strat_smc  # noqa: E402

SITE = Path(__file__).resolve().parents[1] / "docs" / "data"
LIMIT_BARS = {"crypto_15m": 70080}
PAT_CODES = list(patterns.DETECTORS)


def write_dat(path: Path, header: dict, arrays: list[tuple[str, np.ndarray]]) -> None:
    blobs, meta = [], []
    for name, arr in arrays:
        a = np.ascontiguousarray(arr)
        raw = a.tobytes()
        meta.append([name, a.dtype.str.lstrip("<|"), int(a.size)])
        blobs.append(raw + b"\0" * (-len(raw) % 4))
    header = dict(header, arrays=meta)
    hj = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode()
    hj += b" " * (-len(hj) % 4)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(struct.pack("<I", len(hj)) + hj + b"".join(blobs), 6, mtime=0))


def fname(sym: str) -> str:
    return sym.replace("^", "_").replace("=", "_")


def export_candles(key: str, cell: dict) -> dict[str, int]:
    offs = {}
    for sym, df in cell.items():
        off = max(0, len(df) - LIMIT_BARS.get(key, 10 ** 9))
        offs[sym] = off
        d = df.iloc[off:]
        t = (engine.ts_ns(d.index) // 60_000_000_000).astype(np.uint32)
        vol = d["volume"].values if "volume" in d else np.zeros(len(d))
        write_dat(SITE / "c" / key / f"{fname(sym)}.dat", dict(sym=sym, n=len(d)),
                  [("t", t)] + [(k[0], d[k].values.astype(np.float32)) for k in ("open", "high", "low", "close")]
                  + [("v", np.nan_to_num(vol).astype(np.float32))])
    return offs


def base_of(name: str) -> str:
    parts = name.split("_")
    base = "_".join(parts[:-1]) if parts[-1] in es.EXIT_TEXT else name
    if base.rsplit("_", 1)[-1] in es.TOL_TEXT:
        base = base.rsplit("_", 1)[0]
    return base


def _merge(tr: pd.DataFrame, det: pd.DataFrame, on_det: list[str], cols: list[str]) -> np.ndarray:
    """Cocokkan tiap transaksi dengan baris deteksi pertama yang kuncinya sama."""
    out = np.full((len(tr), len(cols)), np.nan, dtype=np.float32)
    if len(tr) == 0 or len(det) == 0:
        return out
    d = det.drop_duplicates(on_det)[on_det + cols]
    m = tr[["_k0", "_k1"]].merge(d, how="left", left_on=["_k0", "_k1"], right_on=on_det)
    return m[cols].to_numpy(dtype=np.float32)


PT_COLS = [f"pi{q}" for q in range(6)] + [f"pp{q}" for q in range(6)]


def _pts(tr, det):
    for c in PT_COLS:
        if c not in det:
            det = det.assign(**{c: np.nan})
    g = _merge(tr, det, ["idx", "dir"], PT_COLS)
    g[:, :6][g[:, :6] < 0] = np.nan
    return g


def geo_for(name: str, base: str, group: str, df: pd.DataFrame, tr: pd.DataFrame):
    """(deskripsi untuk header, matriks float per transaksi) atau (None, None)."""
    tr = tr.assign(_k0=tr.entry_idx - 1, _k1=tr.dir)
    if group == "pola_chart_B" and name.count("_") >= 2 and name.rsplit("_", 2)[0] in patterns.DETECTORS:
        pat, tol, _ = name.rsplit("_", 2)
        t = patterns.TOL_NAMES.index(tol)
        det = strat_patterns._memo(df, ("det", pat, t), lambda: patterns.detect(df, pat, t))
        return dict(kind="pts", pat=pat), _pts(tr, det)
    if base in ("triangle_fib1272", "triangle_fib1272_mtf", "hs_fib1272", "hs_fib1618"):
        names = ("head_shoulders",) if base.startswith("hs") else ("triangle_asc_desc", "triangle_symmetrical", "flag_pennant")
        g = np.full((len(tr), 13), np.nan, dtype=np.float32)
        for pat in names:
            det = strat_patterns._memo(df, ("det", pat, 1), lambda p=pat: patterns.detect(df, p, 1))
            cand = _pts(tr, det)
            use = np.isnan(g[:, 0]) & ~np.isnan(cand[:, 0])
            g[use, :12] = cand[use]
            g[use, 12] = PAT_CODES.index(pat)
        return dict(kind="pts", pats=PAT_CODES), g
    if group == "elliott":
        p = name.split("_")
        setup, fib = es.EW_SETUP[p[1]], p[2] == "fib"
        det = strat_patterns._memo(df, ("ew", setup, fib), lambda: strat_smc.ew_detect(df, setup, fib))
        return dict(kind="pts", pat="elliott"), _pts(tr, det)
    res = label = None
    if base in es.SD_ARGS:
        dep, mb, mode = es.SD_ARGS[base]
        res, label = strat_smc.sd_raw(df, dep, mb, 3.0 if name.endswith("rr3") else 2.0, mode), "zona supply/demand"
    elif base in ("smc_fvg", "smc_fvg_tren"):
        res, label = strat_smc.fvg_raw(df, base.endswith("tren"), 2.0), "fair value gap"
    elif base == "smc_orderblock":
        res, label = strat_smc.ob_raw(df, 3.0 if name.endswith("rr3") else 2.0), "order block"
    elif base in ("ict2022", "ict2022_killzone"):
        res, label = strat_smc.ict_raw(df, 3.0 if name.endswith("rr3") else 2.0), "FVG setelah sweep"
    elif base == "smc_fvg_pd":
        res, label = strat_extra.fvg_pd_raw(df), "fair value gap"
    elif base == "smc_orderblock_pd":
        res, label = strat_extra.ob_pd_raw(df), "order block"
    if res is not None:
        e_idx, _, dirs, _, _, z0, zlo, zhi = res
        det = pd.DataFrame({"idx": np.asarray(e_idx) - 1, "dir": dirs, "z0": z0, "zlo": zlo, "zhi": zhi})
        return dict(kind="zone", label=label), _merge(tr, det, ["idx", "dir"], ["z0", "zlo", "zhi"])
    if group == "candlestick" or (base.endswith("_sr") and base[:-3] in es.CANDLE_LEN):
        return dict(kind="span", len=es.CANDLE_LEN.get(base[:-3] if base.endswith("_sr") else base, 1)), None
    if base.startswith("momentum_polos_"):
        return dict(kind="mom", n=int(base.split("_")[2])), None
    return None, None


def export_strategy_cell(name: str, spec: dict, market: str, tf: str, cell: dict, offs: dict) -> int:
    key = f"{market}_{tf}"
    base = base_of(name)
    cols = {k: [] for k in ("e", "x", "k", "sl", "tp", "ep", "xp")}
    geos, syms, desc, pos = [], {}, None, 0
    for sym, df in cell.items():
        cost = evaluate.cost_for(market, sym)
        tr = es.trades_of(df, spec["fn"](df), cost)
        off = offs[sym]
        total = len(tr)
        tr = tr[tr.entry_idx >= off].reset_index(drop=True)
        o, c = df["open"].values.astype(float), df["close"].values.astype(float)
        xi = tr.exit_idx.values
        xp = tr.xp.values
        kind = np.full(len(tr), 4, dtype=np.uint8)
        kind[xp == c[xi]] = 1
        kind[xp == tr.tp.values] = 3
        kind[xp == tr.sl.values] = 2
        kind[xp == o[xi]] = 0
        cols["e"].append((tr.entry_idx.values - off).astype(np.uint32))
        cols["x"].append((xi - off).astype(np.uint32))
        cols["k"].append((kind | np.where(tr.dir.values > 0, 8, 0)).astype(np.uint8))
        cols["sl"].append(tr.sl.values.astype(np.float32))
        cols["tp"].append(tr.tp.values.astype(np.float32))
        cols["ep"].append(np.where(tr.limit.values, tr.ep.values, np.nan).astype(np.float32))
        cols["xp"].append(np.where(kind == 4, xp, np.nan).astype(np.float32))
        desc, g = geo_for(name, base, spec["group"], df, tr)
        if g is not None:
            w = 6 if desc["kind"] == "pts" else 1
            g = g.copy()
            g[:, :w] -= off          # indeks candle ikut digeser
            geos.append(g)
        syms[sym] = [pos, len(tr), cost, total]
        pos += len(tr)
    arr = {k: np.concatenate(v) if v else np.array([]) for k, v in cols.items()}
    arrays = [("e", arr["e"]), ("x", arr["x"]), ("k", arr["k"])]
    for k in ("sl", "tp", "ep", "xp"):
        if len(arr[k]) and np.isfinite(arr[k]).any():
            arrays.append((k, arr[k]))
    if geos:
        g = np.concatenate(geos)
        desc["w"] = int(g.shape[1])
        arrays.append(("g", g.ravel()))
    write_dat(SITE / "t" / name / f"{key}.dat", dict(strategy=name, cell=key, base=base, geo=desc, syms=syms), arrays)
    return pos


def main() -> None:
    args = sys.argv[1:]
    only_s = args[args.index("--strategi") + 1:] if "--strategi" in args else []
    only_c = args[:args.index("--strategi")] if "--strategi" in args else args
    reg = es.registry()
    for market, tf in CELLS:
        key = f"{market}_{tf}"
        if only_c and key not in only_c:
            continue
        cell = evaluate.load_cell(market, tf)
        offs = export_candles(key, cell)
        t0, total = time.time(), 0
        for i, (name, spec) in enumerate(reg.items(), 1):
            if (only_s and name not in only_s) or not applicable(spec, market, tf):
                continue
            total += export_strategy_cell(name, spec, market, tf, cell, offs)
            if i % 25 == 0:
                print(f"  {key}: {i}/{len(reg)} strategi, {total:,} transaksi, {time.time() - t0:.0f} dtk", flush=True)
        print(f"selesai {key}: {total:,} transaksi, {time.time() - t0:.0f} dtk", flush=True)
    cells = {f"{m}_{tf}": dict(syms=list(evaluate.load_cell(m, tf))) for m, tf in CELLS} if not (only_c or only_s) else None
    if cells:
        (SITE / "full.json").write_text(json.dumps(dict(cells=cells, limit=LIMIT_BARS), separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
