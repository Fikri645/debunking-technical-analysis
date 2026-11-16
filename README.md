# Debunking Technical Analysis

210 strategi technical analysis yang populer di kelas trading diuji melawan entry acak, di 115 saham Indonesia, saham AS, pasangan forex/emas, dan koin crypto, data 2010–2026.

**Website interaktif:** https://fikri645.github.io/debunking-technical-analysis/

Di website Anda bisa melihat tiap pola, zona, dan level yang diuji langsung di chart, lengkap dengan posisi entry, SL, TP, dan exit.

## Hasil singkat

| | Lolos / diuji |
|---|---|
| Saham Indonesia, saham AS, forex, emas | 1 / 1.233 |
| Crypto | 30 / 833 |

- Tidak ada yang lolos di pasar mana pun: pola candlestick, support/resistance, Fibonacci, supply & demand, SMC/ICT, strategi lengkap ala kursus.
- Level Fibonacci tidak bisa dibedakan dari level karangan (30 / 44 / 56 / 70 / 85%).
- Yang lolos di crypto semuanya ikut-tren atau breakout. Aturan satu kalimat tanpa indikator ("long kalau harga lebih tinggi dari N candle lalu") hasilnya setara dengan 14 indikator tren, dan "tembus swing terakhir tanpa syarat bentuk" setara dengan pola chart.
- Keunggulan ikut-tren di crypto 4 jam turun sekitar 65% dari 2018–2021 ke 2022–2026.
- Kontrol zodiak sungguhan (fase bulan, Merkurius retrograde, elemen zodiak): lebih dari separuh kombinasinya "untung", tidak satu pun mengalahkan entry acak.

Satu-satunya yang lolos di luar crypto hanya punya 57 transaksi dan kemungkinan besar kebetulan.

## Cara uji

- **Aturan backtest:** sinyal dihitung di penutupan candle, entry di pembukaan candle berikutnya. Jika SL dan TP kena di candle yang sama, SL dianggap kena dulu. Satu posisi per instrumen.
- **Pembanding acak:** semua transaksi strategi digeser ke waktu acak (1.000 kali; 20.000 kali untuk yang menjanjikan) dengan arah, jarak SL/TP dalam ATR, dan lama pegang yang sama. Statistik uji: rata-rata return kotor per satuan volatilitas.
- **Biaya pulang-pergi:** saham Indonesia 0,40%, saham AS 0,10%, crypto 0,20%, forex 0,015%.
- **Syarat lolos:** mengalahkan entry acak (false discovery rate 5%, Benjamini–Hochberg, dihitung global untuk 6.045 uji) dan untung setelah biaya, di seluruh periode dan di 2022–2026.
- **Pemeriksaan mesin uji:** strategi sinyal acak (harus gagal), strategi yang mengintip masa depan (harus lolos), data dengan urutan candle diacak, dan uji tanpa-mengintip untuk setiap strategi. 406 uji otomatis.

## Menjalankan sendiri

Butuh [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python scripts/download_data.py
uv run python scripts/run_tests.py stage2
uv run python scripts/final_summary.py
uv run pytest
```

Tahap lain: `stage3`, `stage3_kontrol`, `stage4`, `stage5`, `stage6_momentum`. Hasil per sel disimpan di `results/`, ringkasan di `results/final/`.

Website: `uv run python scripts/export_site.py`, lalu `python -m http.server 8765 --directory docs`.

## Isi repo

- `src/talab/` — mesin backtest, pembanding acak, indikator, detektor pola, strategi
- `scripts/` — unduh data, jalankan uji, ringkasan, grafik, ekspor website
- `tests/` — uji otomatis
- `results/` — semua hasil uji (CSV)
- `figures/audit/` — audit visual deteksi pola, dipotong di candle sinyal
- `docs/` — website
- `post/` — grafik ringkasan

## Batasan

- Yang diuji adalah aturan yang bisa ditulis jadi kode dengan parameter standar, bukan penilaian diskresioner seorang trader.
- Data dari Yahoo Finance dan Binance; data intraday saham hanya mencakup sekitar dua tahun terakhir.
- Instrumen yang diuji adalah yang masih ada sekarang, jadi ada bias terhadap yang bertahan.
- Bukan saran investasi.
