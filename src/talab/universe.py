"""Daftar instrumen yang diuji. Dibekukan sebelum uji performa."""

IDX = [
    "^JKSE",
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "BBTN.JK", "BRIS.JK", "ARTO.JK",
    "TLKM.JK", "EXCL.JK", "ISAT.JK", "TOWR.JK",
    "ASII.JK", "UNTR.JK", "AKRA.JK",
    "UNVR.JK", "ICBP.JK", "INDF.JK", "KLBF.JK", "SIDO.JK", "CPIN.JK", "JPFA.JK",
    "AMRT.JK", "MAPI.JK", "ACES.JK", "ERAA.JK",
    "ADRO.JK", "PTBA.JK", "ITMG.JK", "HRUM.JK", "MEDC.JK", "PGAS.JK", "ESSA.JK",
    "ANTM.JK", "INCO.JK", "MDKA.JK",
    "SMGR.JK", "INTP.JK", "INKP.JK", "TKIM.JK", "BRPT.JK", "TPIA.JK", "GOTO.JK",
]

US = [
    "SPY", "QQQ",
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "JPM", "V",
    "UNH", "XOM", "JNJ", "WMT", "PG", "MA", "HD", "CVX", "LLY", "ABBV",
    "MRK", "PEP", "KO", "COST", "AVGO", "ADBE", "CRM", "NFLX", "AMD", "INTC",
    "CSCO", "ORCL", "DIS", "BA", "CAT", "GE", "GS", "MS", "BAC", "WFC",
    "C", "T", "VZ", "PFE", "NKE", "MCD", "IBM", "QCOM", "TXN",
]

FOREX = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCHF=X", "USDCAD=X",
    "NZDUSD=X", "USDIDR=X",
    "GC=F",  # emas (futures) sebagai proxy XAU/USD
]

CRYPTO = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT", "TRXUSDT", "DOTUSDT",
]

YF_MARKETS = {"idx": IDX, "us": US, "forex": FOREX}
YF_TIMEFRAMES = ["1d", "1h"]
CRYPTO_TIMEFRAMES = ["1d", "4h", "1h", "15m"]

START = "2010-01-01"

# Pembagian era
ERAS = {
    "A_2010_2017": ("2010-01-01", "2017-12-31"),
    "B_2018_2021": ("2018-01-01", "2021-12-31"),
    "C_2022_2026": ("2022-01-01", "2026-12-31"),
}
