# Debunking Technical Analysis

232 strategi technical analysis yang populer di kelas trading diuji melawan entry acak dan melawan beli-lalu-diamkan, di 115 saham Indonesia, saham AS, pasangan forex/emas, dan koin crypto, data 2010–2026.

**Website interaktif:** https://fikri645.github.io/debunking-technical-analysis/

Di website Anda bisa melihat tiap pola, zona, dan level yang diuji langsung di chart, lengkap dengan posisi entry, SL, TP, dan exit.

## Hasil singkat

| | Lolos / diuji | Mengalahkan beli-lalu-diamkan |
|---|---|---|
| Saham Indonesia | 1 / 453 | 125 / 418, hanya 7 yang untung |
| Saham AS | 0 / 451 | 0 / 409 |
| Forex dan emas | 0 / 435 | 89 / 397, hanya 34 yang untung |
| Crypto | 21 / 913 | 29 / 867 |

"Diuji" adalah kombinasi strategi × pasar × timeframe. Kolom beli-lalu-diamkan hanya menghitung kombinasi yang punya cukup transaksi.

- Satu-satunya yang lolos di luar crypto: pola rectangle toleransi longgar di data 1 jam saham Indonesia (288 transaksi dalam sekitar 3 tahun, untung di 21 dari 34 saham). Ia baru lolos setelah definisi pola diperketat sesudah hasil pertama terbit (lihat METODE.md), jadi perlakukan sebagai temuan yang belum teruji ulang.
- Tidak ada yang lolos di pasar mana pun: pola candlestick, support/resistance, Fibonacci, supply & demand, SMC/ICT, pola chart versi Lo, Mamaysky & Wang (2000), strategi lengkap ala kursus.
- Level Fibonacci tidak bisa dibedakan dari level karangan (30 / 44 / 56 / 70 / 85%). Satu level karangan malah lolos di crypto 1 jam; level Fibonacci asli tidak ada yang lolos.
- Yang lolos di crypto hampir semuanya ikut-tren atau breakout. Aturan satu kalimat tanpa indikator ("long kalau harga lebih tinggi dari N candle lalu") hasilnya setara dengan 14 indikator tren, dan "tembus swing terakhir tanpa syarat bentuk" ikut lolos seperti pola chart.
- Beli-lalu-diamkan: di saham AS dan di data harian saham Indonesia tidak ada satu pun strategi yang menang. Di data 1 jam saham Indonesia dan di forex, sebagian strategi "menang" karena harga turun pada periode itu dan strategi rugi lebih sedikit. Di crypto, 29 kombinasi menang, hampir semuanya indikator ikut-tren di timeframe harian dan 4 jam.
- Keunggulan ikut-tren di crypto 4 jam turun sekitar 60% dari 2018–2021 ke 2022–2026.
- Strategi yang lolos di crypto tetap mengalami penurunan modal yang dalam: median 79% dari puncak per koin (beli-lalu-diamkan 93%); lihat `scripts/drawdown.py`.
- Kontrol zodiak sungguhan (fase bulan, Merkurius retrograde, elemen zodiak): 57% kombinasinya "untung", tidak satu pun mengalahkan entry acak.

Metode lengkap, kriteria, dan semua koreksi selama pengerjaan: [METODE.md](METODE.md).

## Cara uji

- **Aturan backtest:** sinyal dihitung di penutupan candle, entry di pembukaan candle berikutnya. Jika SL dan TP kena di candle yang sama, SL dianggap kena dulu. Satu posisi per instrumen.
- **Pembanding acak:** semua transaksi strategi digeser sekian candle secara acak (1.000 kali; 20.000 kali untuk yang menjanjikan) dengan arah, jarak SL/TP dalam ATR, dan lama pegang yang sama. Statistik uji: rata-rata return kotor per satuan volatilitas.
- **Biaya pulang-pergi:** saham Indonesia 0,40%, saham AS 0,10%, crypto 0,20%, forex 0,015%.
- **Syarat lolos:** mengalahkan entry acak (false discovery rate 5%, Benjamini–Hochberg, dihitung global untuk 6.619 uji), untung setelah biaya, dan untung di minimal 60% instrumen; ketiganya di seluruh periode dan di 2022–2026.
- **Beli-lalu-diamkan:** per instrumen, hasil majemuk strategi dibanding membeli di awal data dan mendiamkannya, setelah biaya.
- **Pemeriksaan mesin uji:** strategi sinyal acak (harus gagal), strategi yang mengintip masa depan (harus lolos), data dengan urutan candle diacak, dan uji tanpa-mengintip untuk setiap strategi. 714 uji otomatis.

## Menjalankan sendiri

Butuh [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python scripts/download_data.py
uv run python scripts/run_tests.py stage2
uv run python scripts/final_summary.py
uv run pytest
```

Tahap lain: `stage3`, `stage3_kontrol`, `stage4`, `stage5`, `stage6_momentum`, `stage7_tambahan`; lalu `scripts/per_instrument.py` sebelum ringkasan. Hasil per sel disimpan di `results/`, ringkasan di `results/final/`.

Website: `uv run python scripts/export_site.py` (ringkasan dan contoh transaksi) dan `uv run python scripts/export_full.py` (seluruh candle dan semua posisi tiap strategi untuk chart interaktif), lalu `python -m http.server 8765 --directory docs`.

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
