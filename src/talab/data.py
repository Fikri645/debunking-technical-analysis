"""Unduh dan muat data OHLCV. Semua file disimpan sebagai parquet dengan kolom
open, high, low, close, volume dan indeks waktu UTC."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"

COLS = ["open", "high", "low", "close", "volume"]
BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"


def path_for(base: Path, market: str, symbol: str, tf: str) -> Path:
    safe = symbol.replace("^", "_").replace("=", "_")
    return base / market / tf / f"{safe}.parquet"


def fetch_yf(symbol: str, tf: str, start: str) -> pd.DataFrame:
    import yfinance as yf

    kwargs = dict(interval=tf, auto_adjust=True, progress=False, threads=False)
    if tf == "1d":
        df = yf.download(symbol, start=start, **kwargs)
    else:
        df = yf.download(symbol, period="730d", **kwargs)
    if df is None or df.empty:
        return pd.DataFrame(columns=COLS)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[COLS]
    idx = pd.DatetimeIndex(df.index)
    df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    df.index.name = "time"
    return df


def fetch_binance(symbol: str, tf: str, start: str) -> pd.DataFrame:
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    rows: list[list] = []
    while True:
        r = requests.get(
            BINANCE_URL,
            params={"symbol": symbol, "interval": tf, "startTime": start_ms, "limit": 1000},
            timeout=30,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        start_ms = batch[-1][6] + 1  # close time + 1 ms
        if len(batch) < 1000:
            break
        time.sleep(0.05)
    if not rows:
        return pd.DataFrame(columns=COLS)
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["time"] + COLS
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.set_index("time").astype(float)
    # candle terakhir mungkin belum selesai
    return df.iloc[:-1]


def download(market: str, symbol: str, tf: str, start: str, force: bool = False) -> Path:
    out = path_for(RAW, market, symbol, tf)
    if out.exists() and not force:
        return out
    df = fetch_binance(symbol, tf, start) if market.startswith("crypto") else fetch_yf(symbol, tf, start)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    return out


def clean(df: pd.DataFrame, allow_extreme: bool = False,
          wick_ratio: float = 1.5) -> tuple[pd.DataFrame, dict]:
    """Buang baris rusak. Mengembalikan data bersih dan catatan kualitas."""
    n0 = len(df)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    dup = n0 - len(df)
    bad = (
        df[["open", "high", "low", "close"]].isna().any(axis=1)
        | (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (df["high"] < df[["open", "close"]].max(axis=1) - 1e-12)
        | (df["low"] > df[["open", "close"]].min(axis=1) + 1e-12)
    )
    # candle tanpa pergerakan sama sekali (hari libur yang terisi harga lama)
    flat = (df["high"] == df["low"]) & (df["volume"].fillna(0) == 0)
    out = df[~bad & ~flat]

    glitch_n = wick_n = 0
    dropped_before = ""
    if not allow_extreme and len(out) > 20:
        # Harga yang melompat jauh dari median 5 candle sebelum dan sesudahnya lalu
        # kembali lagi adalah data rusak (mis. penyesuaian stock split yang salah),
        # bukan pergerakan pasar. Crypto dikecualikan karena lonjakan seperti itu nyata.
        c = out["close"]
        before = c.shift(1).rolling(5, min_periods=3).median()
        after = c[::-1].shift(1).rolling(5, min_periods=3).median()[::-1]
        ref = pd.concat([before, after], axis=1).mean(axis=1)
        ratio = c / ref
        glitch = (ratio > 1.6) | (ratio < 1 / 1.6)
        glitch_n = int(glitch.sum())
        out = out[~glitch]
        # Sumbu yang mustahil panjangnya
        body_hi = out[["open", "close"]].max(axis=1)
        body_lo = out[["open", "close"]].min(axis=1)
        wick = (out["high"] > body_hi * wick_ratio) | (out["low"] < body_lo / wick_ratio)
        wick_n = int(wick.sum())
        out = out[~wick]
        # Lubang data panjang: pakai hanya bagian setelah lubang terakhir
        gap_days = out.index.to_series().diff().dt.total_seconds() / 86400
        big = np.flatnonzero(gap_days.values > 60)
        if len(big):
            dropped_before = str(out.index[big[-1]].date())
            out = out.iloc[big[-1]:]

    ret = out["close"].pct_change().abs()
    report = {
        "rows_raw": n0,
        "rows_clean": len(out),
        "duplicates": int(dup),
        "bad_ohlc": int(bad.sum()),
        "flat_no_volume": int((flat & ~bad).sum()),
        "price_glitches": glitch_n,
        "impossible_wicks": wick_n,
        "dropped_before_gap": dropped_before,
        "max_abs_return": float(ret.max()) if len(out) > 1 else float("nan"),
        "start": str(out.index[0].date()) if len(out) else "",
        "end": str(out.index[-1].date()) if len(out) else "",
    }
    return out, report


def load(market: str, symbol: str, tf: str) -> pd.DataFrame:
    return pd.read_parquet(path_for(CLEAN, market, symbol, tf))
