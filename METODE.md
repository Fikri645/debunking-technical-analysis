# Metode

Dokumen ini mencatat apa yang diuji, bagaimana, kriteria yang ditetapkan sebelum hasil keluar, dan semua koreksi yang terjadi selama pengerjaan. Angka hasil ada di [README.md](README.md) dan `results/final/`.

## 1. Pertanyaan

Apakah indikator, pola, dan strategi technical analysis yang populer di materi belajar trading lebih baik daripada entry acak setelah biaya transaksi, dan apakah lebih baik daripada beli lalu diamkan?

Bukan tujuan: membuat strategi yang menguntungkan, memberi saran investasi, atau menguji analisis fundamental dan trading kuantitatif.

## 2. Aturan yang ditetapkan sebelum melihat hasil

1. Definisi dan parameter dibekukan sebelum uji performa dijalankan. Untuk pola dan zona, definisi diperiksa lewat audit visual (bagian 5) sebelum hasil performanya dihitung.
2. Semua hasil dilaporkan, termasuk yang lolos.
3. Kriteria lolos (bagian 7) tidak diubah setelah hasil keluar.
4. Kode dan hasil terbuka.

## 3. Data

| Pasar | Instrumen | Sumber | Timeframe |
|---|---|---|---|
| Saham Indonesia | IHSG dan 42 saham besar | Yahoo Finance | harian 2010–2026, 1 jam sekitar 3 tahun terakhir |
| Saham AS | SPY, QQQ, dan 49 saham besar | Yahoo Finance | sama |
| Forex dan emas | 8 pasangan mata uang dan emas berjangka | Yahoo Finance | sama |
| Crypto | 12 koin berpasangan USDT | Binance | harian, 4 jam, 1 jam, 15 menit sejak listing |

Pembersihan data: candle dengan OHLC tidak konsisten, candle datar tanpa volume, lonjakan harga satu candle yang langsung kembali (salah data), sumbu yang mustahil, dan data sebelum jeda panjang (lebih dari 60 hari) dibuang.

Tiga era dilaporkan terpisah: 2010–2017, 2018–2021, 2022–2026.

## 4. Yang diuji

- **Indikator** dengan parameter standar: SMA/EMA cross, golden cross, harga vs EMA 200, MACD, RSI, Stochastic, Bollinger Bands, Ichimoku, Supertrend, Parabolic SAR, ADX, Donchian, OBV, VWAP, divergensi, breakout volume, konfluensi EMA 200 + MACD + RSI, dan konfluensi dengan tren harian (multi-timeframe).
- **13 pola candlestick**, masing-masing dengan tiga cara keluar; lima yang paling populer juga diuji khusus di area support/resistance.
- **Pola chart, dua definisi:**
  - definisi sendiri: titik swing dari zigzag berbasis ATR di tiga skala, aturan geometri dari buku teks, tiga tingkat toleransi (ketat, sedang, longgar). Head and shoulders, double/triple top-bottom, triangle, rectangle, wedge, flag/pennant, cup and handle, rounding bottom/top.
  - definisi Lo, Mamaysky & Wang (2000): kernel regression.
- **Support/resistance** (pantulan, breakout, retest), **trendline** dan **channel**, **Fibonacci retracement** dengan level asli dan level karangan sebagai kontrol.
- **Supply & demand**, **SMC/ICT** (BOS, CHoCH, liquidity sweep, fair value gap, order block, filter premium/discount, model ICT 2022 termasuk versi killzone), **Elliott Wave** (kejar wave 3, kejar wave 5, koreksi ABC).
- **Strategi lengkap** seperti yang diajarkan di kursus: entry, SL, dan TP.
- **Kontrol:** tembus swing terakhir tanpa syarat bentuk, Fibonacci palsu, momentum polos tanpa indikator, dan zodiak sungguhan (fase bulan, Merkurius retrograde, elemen zodiak).

Setiap sinyal diuji dengan cara keluar yang diajarkan (SL/TP dari pola) dan cara keluar sederhana (tahan sejumlah candle). Varian "tahan N candle" berfungsi sebagai studi kejadian: return setelah sinyal dibandingkan dengan return setelah waktu acak.

## 5. Audit visual

Untuk pola chart, zona supply-demand, order block, FVG, model ICT, hitungan Elliott, rounding, dan channel, sampel acak hasil deteksi digambar dengan chart yang dipotong tepat di candle sinyal, sehingga kelanjutan harga tidak terlihat saat menilai bentuknya. Definisi yang tidak terlihat seperti yang digambar trader diperbaiki sebelum uji performa. Galerinya ada di `figures/audit/`.

Perbaikan yang terjadi di tahap ini: wedge dan symmetrical triangle diwajibkan punya sentuhan kelima dan benar-benar mengerucut; channel diwajibkan mengurung harga sejak swing pertamanya dan hampir sejajar.

## 6. Mesin uji

- Sinyal dihitung di penutupan candle; posisi dibuka di pembukaan candle berikutnya. Limit order terisi di harga limit hanya jika harga menembus level itu; sentuhan persis di harga limit tidak dihitung terisi.
- Jika SL dan TP kena di candle yang sama, SL dianggap kena lebih dulu. Jika harga pembukaan melompati SL/TP, keluar di harga pembukaan.
- Satu posisi dalam satu waktu per instrumen.
- Biaya pulang-pergi: saham Indonesia 0,40%, saham AS 0,10%, crypto 0,20%, forex 0,015% (USD/IDR 0,10%, emas 0,03%).

Pemeriksaan mesin:
- strategi dengan sinyal acak harus gagal (tingkat lolos palsu sesuai harapan);
- strategi yang sengaja mengintip masa depan harus lolos;
- data dengan urutan candle diacak tidak boleh menghasilkan strategi yang "berhasil";
- setiap strategi punya uji otomatis: sinyal sampai candle N tidak boleh berubah ketika data setelah N ditambahkan. Data dipotong di sepuluh titik, sinyal di candle terakhir potongan ikut dibandingkan, dan uji itu sendiri diuji dengan strategi yang sengaja mengintip satu candle.

## 7. Statistik

**Pembanding acak.** Semua transaksi sebuah strategi di satu pasar digeser sekian candle secara melingkar di dalam eranya, dihitung pada candle milik instrumen itu sendiri, dengan pecahan geseran yang sama untuk semua instrumen (supaya transaksi yang bergerombol tetap bergerombol), lalu dijalankan ulang dengan arah, jarak SL/TP dalam satuan ATR, dan batas lama pegang yang sama. Ini diulang 1.000 kali; strategi yang tampak menjanjikan (p < 0,01) diulang 20.000 kali. Statistik ujinya rata-rata return kotor per transaksi dalam satuan volatilitas. Keuntungan dinilai dari return bersih setelah biaya.

**Koreksi uji berganda.** False discovery rate 5% (Benjamini–Hochberg), dihitung sekaligus untuk semua uji strategi.

**Kriteria lolos.** Ketiganya harus terpenuhi:
1. mengalahkan pembanding acak pada FDR 5%, di seluruh periode dan di 2022–2026;
2. rata-rata return bersih positif, di seluruh periode dan di 2022–2026;
3. untung di minimal 60% instrumen yang punya sedikitnya 5 transaksi (dan minimal 2 instrumen), di seluruh periode dan di 2022–2026.

**Beli lalu diamkan.** Untuk tiap instrumen, return majemuk dari menjalankan strategi terus-menerus dibandingkan dengan membeli instrumen itu di awal data dan mendiamkannya, keduanya setelah biaya. Sebuah kombinasi disebut mengalahkan beli-lalu-diamkan jika menang di lebih dari separuh instrumennya.

## 8. Yang berbeda dari rencana awal

- Rencana menyebut uji bootstrap ala White's Reality Check. Yang dipakai hanya kontrol false discovery rate; keduanya menjawab masalah yang sama (banyaknya uji), tetapi Reality Check tidak dijalankan.
- Rencana menyebut 5.000–10.000 simulasi acak per strategi; yang dijalankan 1.000, dan 20.000 untuk yang menjanjikan.
- Biaya saham Indonesia dan saham AS dibulatkan menjadi satu angka pulang-pergi.

## 9. Koreksi selama pengerjaan

Semua ini ditemukan dan diperbaiki sebelum hasil akhir; dicatat supaya bisa diperiksa.

- **Statistik uji.** Versi awal membagi return bersih (sudah dipotong biaya) dengan volatilitas, sehingga strategi yang kebetulan aktif saat pasar ramai terlihat lebih baik. Diganti menjadi return kotor per volatilitas; semua uji diulang.
- **Volatilitas mendekati nol.** Pada saham yang lama tidak bergerak, pembagian dengan ATR yang nyaris nol menghasilkan angka tak masuk akal. Transaksi saat ATR di bawah 0,01% harga kini diabaikan, di strategi maupun pembanding acak.
- **Salah data harga.** Lonjakan satu candle dan sumbu palsu dari sumber data tertangkap saat audit visual dan dibersihkan.
- **Bug cache.** Cache hasil perhitungan bisa tertukar antar-data; ditemukan oleh uji otomatis, diperbaiki, dan tiga sel yang diulang menghasilkan angka identik.
- **Tren per era.** Laporan awal memakai return mentah dan menyertakan 2017 yang datanya terlalu tipis. Dikoreksi menjadi keunggulan atas pembanding acak, 2018–2021 dibanding 2022–2026.
- **Kriteria ketiga.** Laporan awal menghitung "lolos" hanya dari dua kriteria pertama. Kriteria ketiga (bekerja di banyak instrumen) kini diterapkan di semua angka.

### Temuan tinjauan kode independen

Setelah hasil pertama dipublikasikan, kode ditinjau oleh peninjau terpisah yang ditugasi mencari bug. Tidak ditemukan strategi yang mengintip masa depan. Empat kelemahan ditemukan; semuanya membuat strategi terlihat lebih baik dari seharusnya, dan semuanya diperbaiki sebelum seluruh uji diulang dari nol:

- **Pembanding acak di data intraday saham.** Geseran dihitung dalam waktu kalender, sehingga transaksi pembanding yang jatuh di luar jam bursa menumpuk di candle pembuka sesi (lebih dari 80% entry pembanding), yang perilakunya tidak tipikal. Kini geseran dihitung dalam jumlah candle.
- **Jendela era per pasar.** Transaksi pembanding yang jatuh sebelum sebuah koin mulai diperdagangkan terbuang, sehingga pembanding untuk strategi beli di crypto terlalu rendah. Dengan geseran per candle instrumen, ini tidak terjadi lagi.
- **Limit order terisi saat disentuh.** Level supply-demand, order block, dan FVG adalah harga persis dari candle sebelumnya, sehingga candle berikutnya sering menyentuhnya tepat di titik terendah. Menghitung sentuhan sebagai terisi memberi strategi harga terbaik candle itu. Kini order baru dihitung terisi jika harga menembus level. Aturan ini konservatif: sebagian order yang hanya tersentuh sebenarnya akan terisi.
- **Uji anti-mengintip terlalu longgar** untuk menangkap bocoran satu candle. Diperketat (lihat bagian 6).

Selain itu, pasar yang datanya hanya mencakup satu era tidak lagi dihitung dua kali dalam koreksi uji berganda.

### Definisi pola diperketat setelah hasil terbit

Ini satu-satunya perubahan definisi yang dilakukan setelah hasil dipublikasikan, jadi dicatat terpisah. Saat contoh deteksi ditampilkan di website, terlihat tiga kelemahan bentuk:

- **Cup and handle** tidak punya syarat kebulatan yang berarti. Bentuk V lurus menghabiskan 33% waktunya di sepertiga terbawah cangkir, sedangkan ambang lama 25% / 15% / 0%, sehingga bentuk V lolos sebagai "cangkir". Ambang baru 42% / 38% / 35% untuk sepertiga terbawah, ditambah syarat 62% / 60% / 56% candle ditutup di bawah tengah kedalaman (bentuk V: 50%, bentuk U: 87%), dan gagang minimal 4 candle.
- **Garis "datar"** pada ascending/descending triangle dan rectangle boleh miring sampai 10% / 15% / 25% tinggi pola di antara dua sentuhan. Ambang baru 4% / 7% / 10%.
- **Double top/bottom longgar** menerima dua puncak yang hanya berjarak 5 candle. Kini minimal 8.
- **Garis leher head and shoulders** hanya dibatasi selisih kedua lembahnya, sehingga dua lembah yang berdekatan menghasilkan garis leher curam yang "ditembus" harga hanya karena bergerak mendatar. Kini kemiringan diukur sepanjang lebar pola, dan kedua bahu harus tetap jelas di sisi yang benar dari garis leher, termasuk saat garis itu ditembus.

Akibatnya di data harian: deteksi cup and handle turun dari 19 / 246 / 969 menjadi 1 / 57 / 195 (ketat / sedang / longgar), ascending/descending triangle dari 66 / 411 / 4.104 menjadi 19 / 182 / 1.585, rectangle dari 166 / 727 / 6.097 menjadi 25 / 156 / 816, head and shoulders dari 96 / 534 / 1.969 menjadi 73 / 416 / 1.609. Semua uji pola chart dan strategi kursus yang memakainya diulang, dan koreksi uji berganda dihitung ulang untuk semua uji. Perubahan hanya ke arah bentuk yang lebih rapi; tidak ada ambang yang dipilih dengan melihat hasil performanya.

Hasil sesudahnya: yang lolos berubah dari 23 menjadi 22. Dua strategi crypto 4 jam yang bukan pola (`sma_20_50_lo`, `trendline_break`) tidak lagi lolos karena ambang koreksi uji berganda untuk 2022–2026 bergeser saat jumlah uji berubah; keduanya memang berada tepat di ambang. Satu strategi baru lolos, dan ini satu-satunya di luar crypto: `rectangle_longgar_mm` di data 1 jam saham Indonesia (288 transaksi dalam sekitar 3 tahun, untung di 21 dari 34 saham). Karena ia muncul dari definisi yang diubah setelah hasil terbit, dari data yang hanya mencakup satu era, dan dari sekitar 1.300 kombinasi di luar crypto, ia dilaporkan apa adanya tetapi belum bisa dianggap temuan yang kokoh.

Yang diketahui tetapi tidak diubah: transaksi yang harga pembukaannya sudah melewati SL atau TP dilewati (itu bisa diketahui saat entry), sedangkan pembanding acak tidak punya saringan serupa; dan limit order tidak pernah mengambil TP di candle pengisiannya, yang merugikan strategi.

## 10. Batasan

- Definisi algoritmik tidak identik dengan cara tiap trader menggambar. Jawabannya di sini: beberapa tingkat toleransi, dua definisi pola, audit visual, dan kode terbuka.
- Daftar instrumen diambil dari yang masih ada sekarang (survivorship bias). Ini menguntungkan posisi beli, termasuk beli-lalu-diamkan; pembanding acak memakai data yang sama, jadi perbandingan itu tetap adil.
- Data intraday saham dan forex hanya sekitar dua tahun.
- Parameter yang diuji adalah parameter standar; tidak ada optimasi parameter.
- Diskresi manusia (memilih setup, membaca berita) tidak diuji.
- Slippage dan dampak pasar di luar biaya tetap tidak dimodelkan.
