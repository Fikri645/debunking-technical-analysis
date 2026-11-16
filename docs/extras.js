// Grafik temuan dan tabel semua hasil.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const nf = (x, d = 2) => x.toLocaleString('id-ID', { minimumFractionDigits: d, maximumFractionDigits: d });
  const pct = (x, d = 2) => (x > 0 ? '+' : x < 0 ? '−' : '') + nf(Math.abs(x), d) + '%';
  const TF = { '1d': 'harian', '4h': '4 jam', '1h': '1 jam', '15m': '15 menit' };
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  // Catatan temuan per kelompok, ditampilkan di bawah aturan strategi
  window.TA_GROUP_NOTES = {
    indikator: 'Temuan kelompok ini: yang lolos hanya indikator ikut-tren di crypto. "Beli saat oversold" (RSI, Stochastic, Bollinger bawah) di crypto malah lebih buruk dari entry acak, dan di saham AS maupun data harian saham Indonesia tidak satu pun mengalahkan beli-lalu-diamkan.',
    candlestick: 'Temuan kelompok ini: dari 13 pola candlestick dengan tiga cara keluar, tidak satu pun lolos di pasar mana pun.',
    pola_chart_B: 'Temuan kelompok ini: yang lolos hanya sinyal breakout di crypto. Kontrol "tembus swing terakhir tanpa syarat bentuk" juga lolos di sana, jadi bentuk polanya tidak menambah apa-apa.',
    pola_chart_A: 'Temuan kelompok ini: definisi pola dari riset Lo, Mamaysky & Wang (2000) tidak lolos di pasar mana pun.',
    support_resistance: 'Temuan kelompok ini: pantulan, breakout, maupun retest di level horizontal tidak lolos di pasar mana pun.',
    trendline: 'Temuan kelompok ini: pantulan di garis tren tidak lolos. Tembus garis tren lolos di crypto saja, dan itu sinyal ikut-arah, sama seperti momentum.',
    fibonacci: 'Temuan kelompok ini: tidak ada level Fibonacci yang lolos, dan hasilnya tidak bisa dibedakan dari level karangan 30 / 44 / 56 / 70 / 85%.',
    fibonacci_palsu: 'Ini kontrol: level retracement karangan yang tidak ada hubungannya dengan deret Fibonacci. Hasilnya setara dengan level Fibonacci asli; satu level karangan (30%) malah lolos di crypto 1 jam, sedangkan level asli tidak ada yang lolos.',
    supply_demand: 'Temuan kelompok ini: zona supply dan demand tidak lolos di pasar mana pun, termasuk zona "kuat".',
    smc_ict: 'Temuan kelompok ini: BOS, CHoCH, liquidity sweep, fair value gap, order block, dan model ICT 2022 tidak lolos di pasar mana pun.',
    elliott: 'Temuan kelompok ini: hanya "kejar wave 3" di crypto 4 jam yang lolos, dan itu entry searah gerakan terakhir. Sepertiga sinyal Elliott punya hitungan lain yang sama sahnya tetapi berlawanan arah dalam 10 candle.',
    influencer: 'Temuan kelompok ini: delapan strategi lengkap ala kursus, tidak satu pun lolos di pasar mana pun.',
    kontrol: 'Ini kontrol: masuk saat harga menembus swing terakhir, tanpa syarat bentuk pola. Kalau ini lolos, bentuk polanya tidak menambah apa-apa.',
    zodiak: 'Ini kontrol: ramalan zodiak sungguhan. Lebih dari separuhnya "untung" karena pasar cenderung naik, tetapi tidak satu pun mengalahkan entry acak.',
    kontrol_momentum: 'Ini kontrol: aturan satu kalimat tanpa indikator. Di crypto harian dan 4 jam hasilnya setara dengan 14 indikator tren.',
  };

  // ---------------------------------------------------------------- temuan
  function bars(rows, max, unit) {
    return '<div class="bars">' + rows.map((r) =>
      `<div class="bar-row"><span class="bar-label">${esc(r.label)}</span>` +
      `<span class="bar-track"><span class="bar" style="width:${Math.max(0.5, 100 * r.value / max)}%;background:${r.color}"></span></span>` +
      `<span class="bar-value">${r.text || nf(r.value, unit === '%' ? 0 : 2) + (unit || '')}</span></div>`).join('') + '</div>';
  }
  function diverging(rows, max) {
    return '<div class="bars">' + rows.map((r) => {
      const w = 50 * Math.abs(r.value) / max;
      const style = r.value >= 0 ? `left:50%;width:${w}%` : `left:${50 - w}%;width:${w}%`;
      return `<div class="bar-row"><span class="bar-label">${esc(r.label)}</span>` +
        `<span class="bar-track mid"><span class="bar" style="${style};background:${r.color}"></span></span>` +
        `<span class="bar-value">${(r.value > 0 ? '+' : r.value < 0 ? '−' : '') + nf(Math.abs(r.value), 2)}</span></div>`;
    }).join('') + '</div>';
  }

  function renderFindings(f) {
    const GREY = '#aab3c5', BLUE = '#2f6fb0', AMBER = '#d98a12', RED = '#cf4034';
    const proofRows = [];
    for (const p of f.proof) {
      proofRows.push({ label: p.label + ': "untung"', value: p.untung, color: GREY, text: nf(p.untung, 0) + '%' });
      proofRows.push({ label: p.label + ': kalahkan acak', value: p.bukti, color: BLUE, text: nf(p.bukti, 0) + '%' });
    }
    const drop = Math.round(100 * (1 - f.era[1].edge / f.era[0].edge));
    const mom = [];
    for (const m of f.momentum) {
      const t = 'Crypto ' + TF[m.tf];
      mom.push({ label: t + ': beli lalu diamkan', value: m.buyhold, color: GREY });
      mom.push({ label: t + ': 14 indikator tren', value: m.indikator, color: AMBER });
      mom.push({ label: t + ': aturan satu kalimat', value: m.momentum, color: BLUE });
    }
    const cards = [
      ['"Backtest saya untung" tidak membuktikan apa-apa',
        'Persen kombinasi yang "untung" dibanding yang mengalahkan entry acak. Ramalan zodiak sungguhan paling sering "untung", karena pasar memang cenderung naik.',
        bars(proofRows, Math.max(...f.proof.map((p) => p.untung)), '%')],
      ['Level Fibonacci setara dengan angka karangan',
        'Keunggulan atas entry acak (skor z, rata-rata semua pasar). Di atas +1,96 baru berarti; tidak ada yang sampai. Kuning: level Fibonacci. Abu-abu: level karangan.',
        diverging(f.fib.map((x) => ({ label: nf(x.level, 1) + '%' + (x.fake ? ' (karangan)' : ' (Fibonacci)'), value: x.z, color: x.fake ? GREY : AMBER })), 2)],
      ['Yang bekerja momentumnya, bukan indikatornya',
        'Sharpe ratio setelah biaya, median 12 koin. Aturan "beli kalau harga lebih tinggi dari N candle lalu, selain itu keluar" setara dengan indikator tren.',
        bars(mom, 1, '')],
      [`Keunggulan itu pun sudah menyusut ${drop}%`,
        'Keunggulan strategi ikut-tren atas entry acak di crypto 4 jam, per transaksi, dalam satuan volatilitas.',
        bars(f.era.map((e, k) => ({ label: e.label, value: e.edge, color: k ? RED : BLUE })), f.era[0].edge, '')],
    ];
    if (f.buyhold) {
      const tot = f.buyhold.reduce((a, b) => a + b.total, 0), win = f.buyhold.reduce((a, b) => a + b.menang, 0);
      cards.unshift([`Beli lalu diamkan mengalahkan ${nf(100 * (1 - win / tot), 0)}% strategi`,
        'Persen kombinasi strategi yang hasil akhirnya lebih tinggi dari membeli instrumen yang sama di awal data lalu mendiamkannya, di lebih dari separuh instrumennya. Keduanya setelah biaya. Di saham Indonesia dan forex sebagian besar yang menang tetap rugi: harganya turun pada periode itu dan strateginya hanya rugi lebih sedikit.',
        bars(f.buyhold.map((b) => ({ label: `${b.label}: ${b.menang} dari ${nf(b.total, 0)}, ${b.untung} untung`, value: 100 * b.menang / b.total,
          color: BLUE, text: nf(100 * b.menang / b.total, 0) + '%' })), 100, '%')]);
    }
    $('findings').innerHTML = cards.map(([h, p, body]) => `<article><h3>${esc(h)}</h3><p>${esc(p)}</p>${body}</article>`).join('');
  }

  // ---------------------------------------------------------------- tabel semua hasil
  const table = { rows: [], sort: 'p', dir: 1, limit: 50, api: null };
  const COLS = [['label', 'Strategi'], ['market', 'Pasar'], ['n', 'Transaksi'], ['win', 'Menang'], ['net', 'Rata-rata per transaksi'],
    ['rnd', 'Entry acak'], ['p', 'Nilai p'], ['cagr', 'Strategi per tahun'], ['bh', 'Beli-diamkan per tahun'], ['verdict', 'Kesimpulan']];

  function renderTable() {
    const m = $('a-market').value, g = $('a-group').value, v = $('a-verdict').value;
    let rows = table.rows.filter((r) => (!m || r.market === m) && (!g || r.group === g) && (!v || r.verdict === v));
    const k = table.sort;
    rows.sort((a, b) => {
      const x = a[k], y = b[k];
      if (x === null) return 1;
      if (y === null) return -1;
      return (typeof x === 'string' ? x.localeCompare(y, 'id') : x - y) * table.dir;
    });
    $('a-count').textContent = `${nf(rows.length, 0)} baris`;
    $('a-more').hidden = rows.length <= table.limit;
    const head = '<thead><tr>' + COLS.map(([key, name]) =>
      `<th scope="col" aria-sort="${key === k ? (table.dir > 0 ? 'ascending' : 'descending') : 'none'}"><button type="button" data-k="${key}">${name}${key === k ? (table.dir > 0 ? ' ▲' : ' ▼') : ''}</button></th>`).join('') + '</tr></thead>';
    $('a-table').innerHTML = head + '<tbody>' + rows.slice(0, table.limit).map((r) =>
      `<tr data-id="${r.id}" data-m="${r.market}" tabindex="0"><td>${esc(r.label)}</td><td>${esc(r.marketLabel)}</td><td>${nf(r.n, 0)}</td>` +
      `<td>${r.win === null ? '–' : nf(r.win * 100, 1) + '%'}</td>` +
      `<td class="${r.net > 0 ? 'pos' : 'neg'}">${r.net === null ? '–' : pct(r.net * 100, 3)}</td>` +
      `<td>${r.rnd === null ? '–' : pct(r.rnd * 100, 3)}</td>` +
      `<td>${r.p === null ? '–' : r.p < 0.001 ? '&lt; 0,001' : nf(r.p, 3)}</td>` +
      `<td class="${r.cagr > r.bh ? 'pos' : ''}">${r.cagr === null ? '–' : pct(r.cagr * 100, 1)}</td>` +
      `<td>${r.bh === null ? '–' : pct(r.bh * 100, 1)}</td>` +
      `<td class="${r.verdict === 'Lolos' ? 'tag-pass' : ''}">${r.verdict}</td></tr>`).join('') + '</tbody>';
  }

  window.addEventListener('ta-ready', (ev) => {
    const { index, verdict, open } = ev.detail;
    table.api = open;
    renderFindings(index.findings);
    for (const s of index.strategies) for (const c of s.cells) {
      table.rows.push({ id: s.id, label: s.label, group: s.group, market: c.market, marketLabel: `${index.markets[c.market]}, ${TF[c.tf]}`,
        n: c.n, win: c.win, net: c.net, rnd: c.rnd, p: c.p, cagr: c.inst ? c.cagr : null, bh: c.inst ? c.bh : null, verdict: verdict(c) });
    }
    $('a-market').innerHTML = '<option value="">Semua pasar</option>' + Object.entries(index.markets).map(([k, v]) => `<option value="${k}">${v}</option>`).join('');
    $('a-group').innerHTML = '<option value="">Semua kelompok</option>' + Object.entries(index.groups).map(([k, v]) => `<option value="${k}">${v}</option>`).join('');
    renderTable();
  });
  for (const id of ['a-market', 'a-group', 'a-verdict']) $(id).addEventListener('change', () => { table.limit = 50; renderTable(); });
  $('a-more').addEventListener('click', () => { table.limit += 50; renderTable(); });
  $('a-table').addEventListener('click', (ev) => {
    const h = ev.target.closest('th button');
    if (h) {
      table.dir = table.sort === h.dataset.k ? -table.dir : 1;
      table.sort = h.dataset.k;
      renderTable();
      return;
    }
    const tr = ev.target.closest('tbody tr');
    if (tr && table.api) table.api(tr.dataset.id, tr.dataset.m);
  });
  $('a-table').addEventListener('keydown', (ev) => {
    const tr = ev.target.closest('tbody tr');
    if (tr && ev.key === 'Enter' && table.api) table.api(tr.dataset.id, tr.dataset.m);
  });
})();
