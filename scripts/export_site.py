"""Ekspor data untuk website interaktif (docs/data/).

Untuk tiap strategi diambil sampel ACAK transaksi (bukan yang terbaik) dari empat
sel pasar, lengkap dengan candle di sekitarnya dan geometri yang dipakai strategi
(titik pola, zona, level, garis indikator), supaya pembaca bisa melihat sendiri
apa yang diuji dan di mana posisi dibuka/ditutup.

Jalankan setelah final_summary.py:
    uv run python scripts/export_site.py [nama_strategi ...]
"""

from __future__ import annotations

import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from talab import (engine, evaluate, levels, patterns, strat_candles, strat_indicators,  # noqa: E402
                   strat_influencer, strat_momentum, strat_patterns, strat_smc)
from talab import indicators as ta  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final"
SITE = ROOT / "docs" / "data"
CELLS = [("idx", "1d"), ("us", "1d"), ("forex", "1d"), ("crypto", "4h")]
CELLS_INTRADAY = [("idx", "1h"), ("us", "1h"), ("forex", "1h"), ("crypto", "1h")]
N_TRADES = 6
BEFORE, AFTER, MAX_AFTER = 70, 18, 220

GROUP_LABEL = {
    "indikator": "Indikator", "candlestick": "Pola candlestick",
    "pola_chart_B": "Pola chart", "pola_chart_A": "Pola chart versi riset 2000 (Lo dkk.)",
    "support_resistance": "Support / resistance", "trendline": "Trendline", "fibonacci": "Fibonacci",
    "supply_demand": "Supply & demand", "smc_ict": "SMC / ICT", "elliott": "Elliott Wave",
    "influencer": "Strategi lengkap ala kursus",
    "kontrol": "Kontrol: tembus swing tanpa pola", "fibonacci_palsu": "Kontrol: Fibonacci palsu",
    "zodiak": "Kontrol: zodiak sungguhan", "kontrol_momentum": "Kontrol: momentum polos",
}
MARKET_LABEL = {"idx": "Saham Indonesia", "us": "Saham AS", "forex": "Forex & emas", "crypto": "Crypto"}

# ---------------------------------------------------------------- nama & aturan

EXIT_TEXT = {
    "ls": "Posisi long dan short; keluar saat sinyal berbalik.",
    "lo": "Long saja; keluar saat sinyal hilang.",
    "h5": "Keluar setelah 5 candle.", "h10": "Keluar setelah 10 candle.",
    "rr2": "SL di ujung pola atau 1,5 ATR; TP dua kali jarak SL.",
    "rr3": "TP tiga kali jarak SL.",
    "mm": "SL dan TP ala buku: SL di balik pola, TP setinggi pola (measured move).",
}
BASE = {
    "sma_20_50": ("SMA 20/50 cross", "Long saat SMA 20 di atas SMA 50, short saat di bawah."),
    "golden_cross_50_200": ("Golden cross / death cross", "Long saat SMA 50 di atas SMA 200, short saat di bawah."),
    "ema_9_21": ("EMA 9/21 cross", "Long saat EMA 9 di atas EMA 21, short saat di bawah."),
    "price_vs_ema200": ("Harga vs EMA 200", "Long saat harga ditutup di atas EMA 200, short saat di bawah."),
    "macd_signal": ("MACD vs garis sinyal", "Long saat MACD di atas garis sinyalnya, short saat di bawah."),
    "macd_zero": ("MACD vs garis nol", "Long saat MACD di atas nol, short saat di bawah."),
    "rsi_30_70": ("RSI 30/70", "Beli saat RSI di bawah 30 (oversold), jual saat di atas 70 (overbought)."),
    "stochastic_20_80": ("Stochastic 20/80", "Beli saat %K memotong %D ke atas di bawah 20, jual saat memotong ke bawah di atas 80."),
    "bollinger_reversion": ("Bollinger: pantulan", "Beli saat ditutup di bawah band bawah, keluar di garis tengah (dan sebaliknya)."),
    "bollinger_breakout": ("Bollinger: breakout", "Beli saat ditutup di atas band atas, keluar di garis tengah (dan sebaliknya)."),
    "ichimoku": ("Ichimoku", "Long saat harga di atas awan dan tenkan di atas kijun; short saat sebaliknya."),
    "supertrend_10_3": ("Supertrend (10, 3)", "Ikuti arah Supertrend."),
    "parabolic_sar": ("Parabolic SAR", "Ikuti arah Parabolic SAR."),
    "adx_di_25": ("ADX + DI", "Ikuti +DI/−DI, hanya saat ADX di atas 25."),
    "donchian_20_10": ("Donchian / Turtle 20-10", "Beli saat tembus high 20 candle, keluar saat tembus low 10 candle (dan sebaliknya)."),
    "obv_ema20": ("OBV vs EMA 20", "Long saat OBV di atas EMA 20-nya, short saat di bawah."),
    "vwap": ("VWAP harian", "Long saat harga di atas VWAP hari itu, short saat di bawah."),
    "confluence_ema200_macd_rsi": ("Konfluensi EMA200 + MACD + RSI", "Long hanya saat harga di atas EMA 200, MACD di atas sinyal, dan RSI di atas 50 (dan sebaliknya)."),
    "rsi_cross_30_70": ("RSI keluar dari oversold/overbought", "Beli saat RSI naik menembus 30, jual saat turun menembus 70."),
    "rsi_divergence": ("Divergensi RSI", "Harga membuat low lebih rendah tetapi RSI low lebih tinggi: beli (dan sebaliknya)."),
    "macd_cross": ("MACD cross", "Beli saat MACD memotong garis sinyal ke atas, jual saat ke bawah."),
    "macd_divergence": ("Divergensi MACD", "Harga membuat low lebih rendah tetapi MACD low lebih tinggi: beli (dan sebaliknya)."),
    "stochastic_cross": ("Stochastic cross di area ekstrem", "Beli saat %K memotong %D ke atas di bawah 20, jual saat ke bawah di atas 80."),
    "bollinger_reentry": ("Bollinger: kembali ke dalam band", "Beli saat harga ditutup kembali di atas band bawah, jual saat kembali di bawah band atas."),
    "bollinger_squeeze": ("Bollinger squeeze breakout", "Setelah band menyempit, beli saat tembus band atas, jual saat tembus band bawah."),
    "volume_breakout": ("Breakout dengan volume", "Tembus high/low 20 candle dengan volume lebih dari 2x rata-rata."),
    # candlestick
    "hammer_hanging_man": ("Hammer / hanging man", "Hammer setelah penurunan: beli. Hanging man setelah kenaikan: jual."),
    "inverted_hammer_shooting_star": ("Inverted hammer / shooting star", "Inverted hammer setelah penurunan: beli. Shooting star setelah kenaikan: jual."),
    "engulfing": ("Engulfing", "Candle besar menelan badan candle sebelumnya, berlawanan arah tren sebelumnya."),
    "harami": ("Harami", "Candle kecil di dalam badan candle besar sebelumnya, berlawanan arah tren sebelumnya."),
    "piercing_dark_cloud": ("Piercing line / dark cloud cover", "Candle kedua menembus lebih dari separuh badan candle pertama."),
    "morning_evening_star": ("Morning star / evening star", "Tiga candle: besar, kecil, lalu besar berlawanan arah."),
    "three_soldiers_crows": ("Three white soldiers / three black crows", "Tiga candle besar searah berturut-turut."),
    "doji_reversal": ("Doji", "Doji setelah tren: sinyal pembalikan."),
    "dragonfly_gravestone_doji": ("Dragonfly / gravestone doji", "Doji dengan sumbu panjang satu sisi setelah tren."),
    "marubozu": ("Marubozu", "Candle penuh tanpa sumbu: ikuti arahnya."),
    "tweezer": ("Tweezer top / bottom", "Dua candle dengan high/low yang sama setelah tren."),
    "inside_bar_breakout": ("Inside bar breakout", "Candle di dalam rentang candle sebelumnya, lalu ditembus."),
    "pin_bar": ("Pin bar", "Sumbu panjang menolak harga: masuk berlawanan arah sumbu."),
    # pola chart
    "head_shoulders": ("Head and shoulders", "Tiga puncak, yang tengah tertinggi; jual saat neckline tembus (kebalikannya untuk inverse)."),
    "double_top_bottom": ("Double top / bottom", "Dua puncak (atau lembah) sejajar; masuk saat neckline tembus."),
    "triple_top_bottom": ("Triple top / bottom", "Tiga puncak (atau lembah) sejajar; masuk saat neckline tembus."),
    "triangle_asc_desc": ("Ascending / descending triangle", "Satu sisi datar, satu sisi miring; masuk saat sisi datar tembus."),
    "triangle_symmetrical": ("Symmetrical triangle", "Dua garis mengerucut; masuk searah tembusnya."),
    "rectangle": ("Rectangle", "Harga bolak-balik di antara dua level; masuk searah tembusnya."),
    "wedge": ("Rising / falling wedge", "Dua garis mengerucut searah; masuk saat tembus berlawanan arah."),
    "flag_pennant": ("Flag / pennant", "Konsolidasi kecil setelah gerakan tajam; masuk saat lanjut searah."),
    "cup_handle": ("Cup and handle", "Cekungan bulat lalu koreksi kecil; beli saat tembus bibir cangkir."),
    "lmw_head_shoulders": ("Head and shoulders (Lo dkk.)", "Definisi kernel-regression dari riset tahun 2000."),
    "lmw_broadening": ("Broadening top/bottom (Lo dkk.)", "Definisi kernel-regression dari riset tahun 2000."),
    "lmw_triangle": ("Triangle (Lo dkk.)", "Definisi kernel-regression dari riset tahun 2000."),
    "lmw_rectangle": ("Rectangle (Lo dkk.)", "Definisi kernel-regression dari riset tahun 2000."),
    "lmw_double": ("Double top/bottom (Lo dkk.)", "Definisi kernel-regression dari riset tahun 2000."),
    "trendline_bounce": ("Trendline: pantulan", "Masuk searah tren saat harga memantul di garis tren. SL 1 ATR di balik garis, TP 1:2."),
    "trendline_break": ("Trendline: tembus", "Masuk berlawanan tren saat garis tren tembus. SL 1 ATR di balik garis, TP 1:2."),
    # smc
    "smc_bos": ("Break of structure (BOS)", "Masuk searah saat harga menembus swing terakhir searah tren."),
    "smc_choch": ("Change of character (CHoCH)", "Masuk saat harga menembus swing terakhir melawan tren sebelumnya."),
    "smc_sweep": ("Liquidity sweep", "Sumbu menembus swing terakhir lalu ditutup kembali: masuk berlawanan arah sumbu."),
    "smc_fvg": ("Fair value gap", "Limit order saat harga kembali ke celah (FVG). SL di balik celah."),
    "smc_fvg_tren": ("Fair value gap searah tren", "FVG, hanya yang searah struktur pasar."),
    "smc_orderblock": ("Order block", "Limit order di candle berlawanan arah terakhir sebelum gelombang yang menembus struktur."),
    "ict2022": ("Model ICT 2022", "Sweep likuiditas, lalu market structure shift, lalu entry di FVG."),
    "ict2022_killzone": ("Model ICT 2022 + killzone", "Model ICT 2022, hanya di jam London dan New York."),
    "sd_semua": ("Supply & demand: semua zona", "Limit order saat harga kembali ke zona base sebelum gerakan tajam."),
    "sd_kuat": ("Supply & demand: zona kuat", "Hanya zona dengan keberangkatan sangat tajam dan base pendek."),
    "sd_pembalikan": ("Supply & demand: pola pembalikan", "Hanya zona drop-base-rally dan rally-base-drop."),
    "sd_lanjutan": ("Supply & demand: pola lanjutan", "Hanya zona rally-base-rally dan drop-base-drop."),
    # influencer
    "the_grab_stoch": ("Sweep + Stochastic", "Setelah liquidity sweep, masuk saat Stochastic berpotongan. SL di ujung sweep, TP 1:2."),
    "the_grab_stoch_tren": ("Sweep + Stochastic + filter tren", "Sama, hanya searah EMA 200."),
    "pullback_ma50": ("Pullback ke MA 50", "Dalam tren, masuk saat harga menyentuh MA 50 lalu ditutup menjauh. SL 2 ATR, keluar saat tutup di balik MA 50."),
    "triangle_fib1272": ("Triangle/pennant + target Fibonacci 127,2%", "Breakout triangle atau pennant dengan TP di ekstensi 127,2%."),
    "triangle_fib1272_mtf": ("Triangle/pennant + Fib 127,2% + filter tren", "Sama, hanya searah EMA 200."),
    "hs_fib1272": ("Head and shoulders + target Fibonacci 127,2%", "SL di atas bahu, TP 127,2% dari tinggi kepala."),
    "hs_fib1618": ("Head and shoulders + target Fibonacci 161,8%", "SL di atas bahu, TP 161,8% dari tinggi kepala."),
    "fib_golden_candle": ("Golden zone Fibonacci + candle konfirmasi", "Harga masuk zona 50–78,6% lalu muncul hammer/engulfing. TP 1:2."),
    "zodiak_fase_bulan": ("Fase bulan", "Long saat bulan membesar, short saat bulan mengecil."),
    "zodiak_merkurius_retrograde": ("Merkurius retrograde", "Short saat Merkurius retrograde, long saat tidak."),
    "zodiak_elemen": ("Elemen zodiak", "Long saat Matahari di zodiak api/udara, short saat di zodiak tanah/air."),
    "kontrol_swing_break": ("Tembus swing terakhir (tanpa pola)", "Masuk saat penutupan menembus swing high/low terakhir. Tidak ada syarat bentuk."),
}
TOL_TEXT = {"ketat": "toleransi ketat", "sedang": "toleransi sedang", "longgar": "toleransi longgar"}
EW_TEXT = {"wave3": ("Elliott: kejar wave 3", "Setelah wave 1 dan koreksi wave 2, masuk searah wave 1. SL di awal wave 1."),
           "wave5": ("Elliott: kejar wave 5", "Setelah wave 4, masuk searah tren. SL di puncak wave 1."),
           "abc": ("Elliott: koreksi ABC", "Setelah lima gelombang lengkap, masuk melawan tren. TP di akhir wave 4.")}


def describe(name: str) -> tuple[str, str]:
    """Nama tampilan dan aturan satu-dua kalimat."""
    parts = name.split("_")
    suffix = parts[-1] if parts[-1] in EXIT_TEXT else None
    base = "_".join(parts[:-1]) if suffix else name
    exit_text = EXIT_TEXT.get(suffix, "")
    tag = {"ls": "long & short", "lo": "long saja", "h5": "tahan 5 candle", "h10": "tahan 10 candle",
           "rr2": "RR 1:2", "rr3": "RR 1:3", "mm": "SL/TP ala buku"}.get(suffix, "")
    if base in BASE:
        label, rule = BASE[base]
    elif base.rsplit("_", 1)[-1] in TOL_TEXT and base.rsplit("_", 1)[0] in BASE:
        b, tol = base.rsplit("_", 1)
        label, rule = BASE[b]
        tag = f"{TOL_TEXT[tol]}, {tag}"
    elif base.startswith("sr_"):
        _, mode, touch = base.split("_")
        mode_t = {"bounce": "pantulan", "breakout": "breakout", "retest": "breakout lalu retest"}[mode]
        label = f"Support/resistance: {mode_t}"
        rule = f"Level = minimal {touch[0]} swing di harga yang sama. SL 1 ATR di balik level, TP 1:2."
        tag = f"{touch[0]} sentuhan"
        exit_text = ""
    elif base.startswith("fib_") or base.startswith("fibfake_") or name.startswith(("fib_", "fibfake_")):
        full = name
        lv = int(full.split("_")[1]) / 10
        ext = "ext1272" in full
        fake = full.startswith("fibfake")
        label = f"{'Level karangan' if fake else 'Fibonacci'} {str(lv).replace('.', ',')}%"
        rule = (f"Limit order di retracement {str(lv).replace('.', ',')}% dari swing terakhir. SL sedikit di balik "
                f"awal swing, TP di {'ekstensi 127,2%' if ext else 'ujung swing'}.")
        tag = "TP ekstensi 127,2%" if ext else "TP ujung swing"
        exit_text = ""
    elif base.startswith("elliott_") or name.startswith("elliott_"):
        p = name.split("_")
        label, rule = EW_TEXT[p[1]]
        tag = ("dengan pedoman Fibonacci" if p[2] == "fib" else "aturan wajib saja") + \
              (", tahan 10 candle" if p[-1] == "h10" else ", SL/TP ala Elliott")
        exit_text = ""
    elif base.startswith("momentum_polos_"):
        n = base.split("_")[2]
        label = f"Momentum polos {n} candle"
        rule = f"Long kalau harga penutupan lebih tinggi dari {n} candle lalu; kalau tidak, {'keluar' if suffix == 'lo' else 'short'}."
        exit_text = ""
    else:
        label, rule = name, ""
    full_label = f"{label} ({tag})" if tag else label
    return full_label, (rule + " " + exit_text).strip()


# ---------------------------------------------------------------- registry

def registry() -> dict[str, dict]:
    reg = {}
    for build in (strat_indicators.build, strat_candles.build, strat_patterns.build, strat_patterns.build_controls,
                  strat_smc.build, strat_influencer.build, strat_momentum.build):
        reg.update(build())
    return reg


def trades_of(df: pd.DataFrame, sig: pd.DataFrame, cost: float) -> pd.DataFrame:
    """Sama dengan engine.run, tetapi menyimpan harga SL/TP mutlak."""
    o, h, l, c = (df[k].values.astype(float) for k in ("open", "high", "low", "close"))
    taken, e_px, x_idx, x_px, sl, tp = engine._simulate(
        o, h, l, c, sig["entry_idx"].values, sig["dir"].values, sig["sl"].values, sig["tp"].values,
        sig["sl_dist"].values, sig["tp_dist"].values, sig["hold"].values,
        sig["entry_price"].values, sig["is_limit"].values)
    a = engine.atr(df)
    ei = sig["entry_idx"].values[taken]
    d = sig["dir"].values[taken]
    out = pd.DataFrame({"entry_idx": ei, "exit_idx": x_idx[taken], "dir": d, "ep": e_px[taken], "xp": x_px[taken],
                        "sl": sl[taken], "tp": tp[taken], "limit": sig["is_limit"].values[taken]})
    out["net"] = d * (out["xp"] / out["ep"] - 1.0) - cost
    ap = a[ei - 1] / out["ep"].values
    return out[np.isfinite(ap) & (ap >= engine.MIN_ATR_PCT)].reset_index(drop=True)


# ---------------------------------------------------------------- geometri

def r6(x):
    return None if x is None or not np.isfinite(x) else float(f"{float(x):.6g}")


def series(x, a, b):
    return [r6(v) for v in np.asarray(x, dtype=float)[a:b]]


def ov(name, values, a, b, pane="price", ref=None):
    d = {"name": name, "pane": pane, "v": series(values, a, b)}
    if ref:
        d["ref"] = ref
    return d


def overlays(base: str, df: pd.DataFrame, a: int, b: int) -> list[dict]:
    o, h, l, c, v = strat_indicators.arrays(df)
    if base == "sma_20_50":
        return [ov("SMA 20", ta.sma(c, 20), a, b), ov("SMA 50", ta.sma(c, 50), a, b)]
    if base == "golden_cross_50_200":
        return [ov("SMA 50", ta.sma(c, 50), a, b), ov("SMA 200", ta.sma(c, 200), a, b)]
    if base == "pullback_ma50":
        return [ov("SMA 50", ta.sma(c, 50), a, b), ov("SMA 200", ta.sma(c, 200), a, b)]
    if base == "ema_9_21":
        return [ov("EMA 9", ta.ema(c, 9), a, b), ov("EMA 21", ta.ema(c, 21), a, b)]
    if base == "price_vs_ema200":
        return [ov("EMA 200", ta.ema(c, 200), a, b)]
    if base in ("macd_signal", "macd_zero", "macd_cross", "macd_divergence"):
        line, sig = ta.macd(c)
        return [ov("MACD", line, a, b, "sub", [0]), ov("Sinyal", sig, a, b, "sub")]
    if base == "confluence_ema200_macd_rsi":
        return [ov("EMA 200", ta.ema(c, 200), a, b), ov("RSI 14", ta.rsi(c), a, b, "sub", [50])]
    if base in ("rsi_30_70", "rsi_cross_30_70", "rsi_divergence"):
        return [ov("RSI 14", ta.rsi(c), a, b, "sub", [30, 70])]
    if base in ("stochastic_20_80", "stochastic_cross", "the_grab_stoch", "the_grab_stoch_tren"):
        k, d = ta.stochastic(h, l, c)
        out = [ov("%K", k, a, b, "sub", [20, 80]), ov("%D", d, a, b, "sub")]
        if base.endswith("tren"):
            out.append(ov("EMA 200", ta.ema(c, 200), a, b))
        return out
    if base.startswith("bollinger"):
        mid, up, lo = ta.bollinger(c)
        return [ov("Band atas", up, a, b), ov("Tengah", mid, a, b), ov("Band bawah", lo, a, b)]
    if base == "ichimoku":
        tk, kj, sa, sb = ta.ichimoku(h, l)
        return [ov("Tenkan", tk, a, b), ov("Kijun", kj, a, b), ov("Span A", sa, a, b), ov("Span B", sb, a, b)]
    if base == "adx_di_25":
        ad, pdi, mdi = ta.adx(h, l, c)
        return [ov("ADX", ad, a, b, "sub", [25]), ov("+DI", pdi, a, b, "sub"), ov("−DI", mdi, a, b, "sub")]
    if base in ("donchian_20_10", "volume_breakout"):
        return [ov("High 20", ta.shift(ta.rolling_max(h, 20), 1), a, b),
                ov("Low 20", ta.shift(ta.rolling_min(l, 20), 1), a, b)]
    if base == "obv_ema20":
        ob = ta.obv(c, v)
        return [ov("OBV", ob, a, b, "sub"), ov("EMA 20 OBV", ta.ema(ob, 20), a, b, "sub")]
    if base in ("triangle_fib1272_mtf",):
        return [ov("EMA 200", ta.ema(c, 200), a, b)]
    return []


CANDLE_LEN = {"hammer_hanging_man": 1, "inverted_hammer_shooting_star": 1, "engulfing": 2, "harami": 2,
              "piercing_dark_cloud": 2, "morning_evening_star": 3, "three_soldiers_crows": 3, "doji_reversal": 1,
              "dragonfly_gravestone_doji": 1, "marubozu": 1, "tweezer": 2, "inside_bar_breakout": 3, "pin_bar": 1}
SD_ARGS = {"sd_semua": (1.5, 6, 0), "sd_kuat": (2.5, 3, 0), "sd_pembalikan": (1.5, 6, 1), "sd_lanjutan": (1.5, 6, 2)}
EW_SETUP = {"wave3": 0, "wave5": 1, "abc": 2}


def _match_points(det: pd.DataFrame, sig: int, d: int):
    if det.empty:
        return None
    hit = det[(det["idx"] == sig) & (det["dir"] == d)]
    if hit.empty:
        return None
    row = hit.iloc[0]
    return [(int(row[f"pi{q}"]), float(row[f"pp{q}"])) for q in range(6)
            if f"pi{q}" in row and row[f"pi{q}"] >= 0]


def _zone(res, entry: int, d: int, label: str):
    e_idx, _, dirs, _, _, z0, zlo, zhi = res
    hit = np.flatnonzero((e_idx == entry) & (dirs == d))
    if len(hit) == 0:
        return []
    k = hit[0]
    return [dict(k="zone", i0=int(z0[k]), i1=int(entry), lo=float(zlo[k]), hi=float(zhi[k]), label=label)]


def geometry(name: str, base: str, group: str, df: pd.DataFrame, tr) -> list[dict]:
    """Bentuk yang dipakai strategi, dalam indeks candle mutlak (digeser belakangan)."""
    entry, d = int(tr.entry_idx), int(tr.dir)
    sig = entry - 1
    o, h, l, c = strat_patterns._ohlc(df)
    if group == "pola_chart_B":
        pat, tol, _ = name.rsplit("_", 2)
        t = patterns.TOL_NAMES.index(tol)
        det = strat_patterns._memo(df, ("det", pat, t), lambda: patterns.detect(df, pat, t))
        pts = _match_points(det, sig, d)
        return [dict(k="poly", pts=pts, label="pola")] if pts else []
    if base in ("triangle_fib1272", "triangle_fib1272_mtf", "hs_fib1272", "hs_fib1618"):
        names = ("head_shoulders",) if base.startswith("hs") else ("triangle_asc_desc", "triangle_symmetrical", "flag_pennant")
        for pat in names:
            det = strat_patterns._memo(df, ("det", pat, 1), lambda p=pat: patterns.detect(df, p, 1))
            pts = _match_points(det, sig, d)
            if pts:
                return [dict(k="poly", pts=pts, label=BASE[pat][0])]
        return []
    if group == "elliott":
        p = name.split("_")
        setup, fib = EW_SETUP[p[1]], p[2] == "fib"
        det = strat_patterns._memo(df, ("ew", setup, fib), lambda: strat_smc.ew_detect(df, setup, fib))
        pts = _match_points(det, sig, d)
        return [dict(k="poly", pts=pts, label="gelombang", numbered=True)] if pts else []
    if base in SD_ARGS:
        dep, mb, mode = SD_ARGS[base]
        rr = 3.0 if name.endswith("rr3") else 2.0
        return _zone(strat_smc.sd_raw(df, dep, mb, rr, mode), entry, d, "zona demand" if d == 1 else "zona supply")
    if base in ("smc_fvg", "smc_fvg_tren"):
        return _zone(strat_smc.fvg_raw(df, base.endswith("tren"), 2.0), entry, d, "fair value gap")
    if base == "smc_orderblock":
        return _zone(strat_smc.ob_raw(df, 3.0 if name.endswith("rr3") else 2.0), entry, d, "order block")
    if base in ("ict2022", "ict2022_killzone"):
        return _zone(strat_smc.ict_raw(df, 3.0 if name.endswith("rr3") else 2.0), entry, d, "FVG setelah sweep")
    if base in ("smc_bos", "smc_choch", "smc_sweep"):
        st = strat_smc._structure(df)
        if base == "smc_sweep":
            lvl = st[6][sig - 1] if d == 1 else st[5][sig - 1]
            label = "swing low yang disapu" if d == 1 else "swing high yang disapu"
        else:
            lvl = st[5][sig - 1] if d == 1 else st[6][sig - 1]
            label = "swing high yang ditembus" if d == 1 else "swing low yang ditembus"
        if np.isfinite(lvl) and lvl > 0:
            return [dict(k="hline", p=float(lvl), i0=sig - 30, i1=sig, label=label)]
        return []
    if group == "support_resistance":
        a = ta.atr(h, l, c)
        lvl = tr.sl + d * levels.SL_ATR * a[sig]
        return [dict(k="hline", p=float(lvl), i0=sig - 120, i1=int(tr.exit_idx), label="level S/R")]
    if group == "trendline":
        a = ta.atr(h, l, c)
        line = tr.sl + d * levels.SL_ATR * a[sig]
        want = -d if base == "trendline_bounce" else d
        for mult in strat_patterns.LEVEL_SCALES:
            p_idx, p_px, p_type, p_conf = strat_patterns._zz(df, mult)
            ok = np.flatnonzero((p_conf <= sig) & (p_type == want))
            if len(ok) < 2:
                continue
            i0, i1 = ok[-2], ok[-1]
            slope = (p_px[i1] - p_px[i0]) / (p_idx[i1] - p_idx[i0])
            ln = p_px[i1] + slope * (sig - p_idx[i1])
            if abs(ln - line) <= 1e-6 * abs(line):
                return [dict(k="line", i0=int(p_idx[i0]), p0=float(p_px[i0]), i1=sig, p1=float(ln), label="garis tren")]
        return [dict(k="hline", p=float(line), i0=sig - 10, i1=sig, label="garis tren (nilai saat sinyal)")]
    if group in ("fibonacci", "fibonacci_palsu"):
        lv = int(name.split("_")[1]) / 1000
        ext = 0.272 if "ext1272" in name else 0.0
        leg = d * (tr.tp - tr.sl) / (1.05 + ext)
        a_px = tr.sl + d * 0.05 * leg
        b_px = a_px + d * leg
        ratios = sorted({0.0, 0.382, 0.5, 0.618, 0.786, 1.0, lv})
        lo_i = max(0, entry - 400)
        hb = h if d == 1 else l
        ha = l if d == 1 else h
        ib = [i for i in range(entry - 1, lo_i, -1) if abs(hb[i] - b_px) <= 1e-6 * abs(b_px)]
        ib = ib[0] if ib else entry - 5
        ia = [i for i in range(ib - 1, lo_i, -1) if abs(ha[i] - a_px) <= 1e-6 * abs(a_px)]
        ia = ia[0] if ia else max(lo_i, ib - 20)
        return [dict(k="fib", i0=int(ia), i1=int(tr.exit_idx), ia=int(ia), ib=int(ib), a=float(a_px), b=float(b_px),
                     levels=[[r, float(b_px - d * r * leg)] for r in ratios], used=lv)]
    if group == "candlestick":
        n = CANDLE_LEN.get(base, 1)
        return [dict(k="span", i0=sig - n + 1, i1=sig, label="pola")]
    if group == "pola_chart_A":
        return [dict(k="span", i0=sig - 37, i1=sig, label="jendela 38 candle")]
    if base == "kontrol_swing_break":
        for mult in patterns.SCALES:
            p_idx, p_px, p_type, p_conf = strat_patterns._zz(df, mult)
            for k in np.flatnonzero((p_type == d) & (p_conf < sig) & (p_conf >= sig - 51)):
                seg = c[p_conf[k] + 1: sig]
                if d * (c[sig] - p_px[k]) > 0 and not (d * (seg - p_px[k]) > 0).any():
                    return [dict(k="hline", p=float(p_px[k]), i0=int(p_idx[k]), i1=sig, label="swing terakhir")]
        return []
    if base.startswith("momentum_polos_"):
        n = int(base.split("_")[2])
        if sig - n >= 0:
            return [dict(k="hline", p=float(c[sig - n]), i0=sig - n, i1=sig, label=f"harga {n} candle lalu")]
    return []


def shift_shape(s: dict, a: int, n: int) -> dict:
    s = dict(s)
    for key in ("i0", "i1", "ia", "ib"):
        if key in s:
            s[key] = int(np.clip(s[key] - a, 0, n - 1))
    if "pts" in s:
        s["pts"] = [[int(i - a), r6(p)] for i, p in s["pts"]]
    for key in ("p", "p0", "p1", "lo", "hi", "a", "b"):
        if key in s:
            s[key] = r6(s[key])
    if "levels" in s:
        s["levels"] = [[r, r6(p)] for r, p in s["levels"]]
    return s


def pack_trade(name, base, group, market, sym, df, tr) -> dict:
    entry, exit_ = int(tr.entry_idx), int(tr.exit_idx)
    shapes = geometry(name, base, group, df, tr)
    first = entry
    for s in shapes:
        first = min([first] + [s[k] for k in ("i0", "ia") if k in s] + [p[0] for p in s.get("pts", [])])
    a = max(0, min(entry - BEFORE, first - 12))
    a = max(a, entry - 320)
    b = min(len(df), min(exit_, entry + MAX_AFTER) + AFTER + 1)
    n = b - a
    t = (engine.ts_ns(df.index[a:b]) // 60_000_000_000).tolist()
    out = dict(sym=sym, market=market, t=t,
               o=series(df["open"].values, a, b), h=series(df["high"].values, a, b),
               l=series(df["low"].values, a, b), c=series(df["close"].values, a, b),
               e=entry - a, x=(exit_ - a) if exit_ < b else None, dir=int(tr.dir),
               ep=r6(tr.ep), xp=r6(tr.xp), sl=r6(tr.sl), tp=r6(tr.tp), net=round(float(tr.net) * 100, 2),
               bars=exit_ - entry, limit=bool(tr.limit),
               shapes=[shift_shape(s, a, n) for s in shapes if not s.get("pts") or min(p[0] for p in s["pts"]) >= a],
               ov=overlays(base, df, a, b))
    return out


def export_strategy(name: str, spec: dict) -> dict:
    parts = name.split("_")
    base = "_".join(parts[:-1]) if parts[-1] in EXIT_TEXT else name
    if base.rsplit("_", 1)[-1] in TOL_TEXT:
        base = base.rsplit("_", 1)[0]
    cells = CELLS_INTRADAY if spec.get("intraday") else CELLS
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    out = {}
    for market, tf in cells:
        if spec.get("needs_volume") and market == "forex":
            continue
        cell = evaluate.load_cell(market, tf)
        pool = []
        for sym, df in cell.items():
            tr = trades_of(df, spec["fn"](df), evaluate.cost_for(market, sym))
            pool += [(sym, row) for row in tr.itertuples(index=False)]
        if not pool:
            continue
        pick = rng.choice(len(pool), size=min(N_TRADES, len(pool)), replace=False)
        out[f"{market}_{tf}"] = [pack_trade(name, base, spec["group"], market, pool[i][0], cell[pool[i][0]], pool[i][1])
                                 for i in sorted(pick)]
    return out


def write_index(reg: dict) -> None:
    v = pd.read_csv(FINAL / "verdicts.csv")
    strategies = []
    for name, spec in reg.items():
        label, rule = describe(name)
        rows = v[v.strategy == name]
        cells = []
        for r in rows.itertuples(index=False):
            cells.append(dict(market=r.market, tf=r.tf, n=int(r.n) if np.isfinite(r.n) else 0,
                              win=r6(r.win_rate), net=r6(r.mean_net), rnd=r6(r.null_mean_net), p=r6(r.p), z=r6(r.z),
                              netC=r6(r.mean_net_C), pC=r6(r.p_C), beats=bool(r.beats_random),
                              profit=bool(r.profitable), passes=bool(r.passes)))
        strategies.append(dict(id=name, group=spec["group"], label=label, rule=rule, cells=cells,
                               passes=int(sum(c["passes"] for c in cells))))
    ta_rows = v[~v.control]
    matrix = []
    for g in GROUP_LABEL:
        d = v[v.group == g]
        if d.empty:
            continue
        matrix.append(dict(group=g, label=GROUP_LABEL[g], control=bool(d.control.iloc[0]),
                           cells={m: [int(d[d.market == m].passes.sum()), int((d.market == m).sum())]
                                  for m in MARKET_LABEL}))
    index = dict(groups=GROUP_LABEL, markets=MARKET_LABEL, strategies=strategies, matrix=matrix,
                 findings=findings(v),
                 totals=dict(strategies=int(ta_rows.strategy.nunique()), combos=int(len(ta_rows)),
                             passes=int(ta_rows.passes.sum()),
                             non_crypto=[int(ta_rows[ta_rows.market != "crypto"].passes.sum()),
                                         int((ta_rows.market != "crypto").sum())],
                             crypto=[int(ta_rows[ta_rows.market == "crypto"].passes.sum()),
                                     int((ta_rows.market == "crypto").sum())]))
    (SITE / "index.json").write_text(json.dumps(index, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def findings(v: pd.DataFrame) -> dict:
    """Angka untuk grafik ringkas di halaman (sama dengan scripts/make_figures.py)."""
    proof = []
    for g in ("zodiak", "candlestick", "fibonacci", "smc_ict", "supply_demand", "support_resistance"):
        d = v[v.group == g]
        proof.append(dict(label=GROUP_LABEL[g].replace("Kontrol: ", ""), untung=round(100 * d.profitable.mean(), 1),
                          bukti=round(100 * d.beats_random.mean(), 1)))
    f = v[v.group.isin(["fibonacci", "fibonacci_palsu"])].copy()
    f["level"] = f.strategy.str.extract(r"_(\d{3})")[0].astype(int) / 10
    fib = [dict(level=float(lv), fake=grp == "fibonacci_palsu", z=round(float(d.z.mean()), 2))
           for (lv, grp), d in f.groupby(["level", "group"])]
    allt = pd.read_csv(FINAL / "all_tests.csv")
    trend = ["sma_20_50", "golden_cross_50_200", "ema_9_21", "price_vs_ema200", "macd_signal", "macd_zero",
             "ichimoku", "supertrend_10_3", "parabolic_sar", "adx_di_25", "donchian_20_10", "bollinger_breakout"]
    e = allt[allt.era.isin(["B_2018_2021", "C_2022_2026"]) & (allt.market == "crypto") & (allt.tf == "4h")
             & (allt.group == "indikator") & (allt.kind == "state") & allt.strategy.str[:-3].isin(trend)]
    era = [dict(label=k[2:].replace("_", "–"), edge=round(float((d.mean_rg - d.null_mean_rg).mean()), 2))
           for k, d in e.groupby("era")]
    s = pd.read_csv(FINAL / "momentum_vs_indikator_sharpe.csv")
    s = s[(s.market == "crypto") & (s.sisi == "lo") & s.tf.isin(["1d", "4h"])]
    mom = [dict(tf=r.tf, buyhold=round(r.sharpe_buyhold, 2), indikator=round(r.sharpe_indikator, 2),
                momentum=round(r.sharpe_momentum, 2)) for r in s.itertuples()]
    return dict(proof=proof, fib=fib, era=era, momentum=mom)


QUIZ = [("crypto", "1d", "BTCUSDT"), ("us", "1d", "AAPL"), ("idx", "1d", "BBCA.JK"), ("forex", "1d", "EURUSD=X"),
        ("crypto", "4h", "ETHUSDT"), ("us", "1d", "TSLA"), ("idx", "1d", "TLKM.JK"), ("forex", "1d", "GC=F"),
        ("crypto", "1d", "SOLUSDT"), ("us", "1d", "NVDA")]


def write_quiz() -> None:
    """Kuis: satu chart asli, tiga hasil mengacak urutan candle dari jendela yang sama."""
    rng = np.random.default_rng(2026)
    rounds = []
    for k, (market, tf, sym) in enumerate(QUIZ):
        df = evaluate.load_cell(market, tf).get(sym)
        if df is None or len(df) < 600:
            continue
        start = int(rng.integers(200, len(df) - 200))
        real = df.iloc[start: start + 160]
        answer = int(rng.integers(0, 4))
        charts = []
        for slot in range(4):
            d = real if slot == answer else evaluate.permute_bars(real, seed=100 * k + slot)
            base = float(d["open"].values[0])   # dinormalkan supaya tidak bisa ditebak dari angka harganya
            charts.append({c: [round(float(x) / base * 100, 3) for x in d[col].values]
                           for c, col in (("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"))})
        rounds.append(dict(sym=sym.replace("=X", "").replace(".JK", ""), tf=tf,
                           period=f"{real.index[0]:%b %Y} – {real.index[-1]:%b %Y}", answer=answer, charts=charts))
    (SITE / "quiz.json").write_text(json.dumps(rounds, separators=(",", ":")), encoding="utf-8")
    print("kuis:", len(rounds), "ronde")


def main() -> None:
    (SITE / "trades").mkdir(parents=True, exist_ok=True)
    reg = registry()
    write_index(reg)
    write_quiz()
    if sys.argv[1:] == ["--index"]:
        return
    only = sys.argv[1:]
    t0 = time.time()
    for i, (name, spec) in enumerate(reg.items(), 1):
        if only and name not in only:
            continue
        out = SITE / "trades" / f"{name}.json"
        if out.exists() and not only:
            continue
        data = export_strategy(name, spec)
        out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        if i % 10 == 0 or only:
            print(f"{i}/{len(reg)} {name} {out.stat().st_size // 1024} KB, {time.time() - t0:.0f} dtk", flush=True)


if __name__ == "__main__":
    main()
