(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const TF = { '1d': 'harian', '4h': '4 jam', '1h': '1 jam', '15m': '15 menit' };
  const C = { ink: '#141b33', muted: '#5b6480', grid: '#e3e8f2', up: '#12806c', down: '#cf4034', mark: '#6a45d9', amber: '#b97a06' };
  const OV_COLORS = ['#2f6fb0', '#d98a12', '#7d8496', '#b0559a'];
  const nf = (x, d = 2) => x.toLocaleString('id-ID', { minimumFractionDigits: d, maximumFractionDigits: d });
  const pct = (x, d = 2) => (x > 0 ? '+' : x < 0 ? '−' : '') + nf(Math.abs(x), d) + '%';
  const price = (x) => {
    const a = Math.abs(x);
    const d = a >= 1000 ? 0 : a >= 100 ? 1 : a >= 1 ? 2 : a >= 0.01 ? 4 : 6;
    return nf(x, d);
  };

  let INDEX = null;
  const cache = new Map();
  const state = { strategy: null, trades: null, cell: null, i: 0, revealed: false, hover: -1, shown: 0, guess: null, timer: 0 };
  const score = { right: 0, total: 0 };
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ---------------------------------------------------------------- data
  async function loadTrades(id) {
    if (!cache.has(id)) cache.set(id, fetch(`data/trades/${id}.json`).then((r) => (r.ok ? r.json() : { _error: true })).catch(() => ({ _error: true })));
    return cache.get(id);
  }

  function verdictOf(c) {
    if (!c || !c.n) return ['fail', 'Terlalu sedikit transaksi untuk diuji.'];
    if (c.passes) return ['pass', `Lolos: mengalahkan entry acak, untung setelah biaya termasuk di 2022–2026, dan untung di ${c.pos} dari ${c.inst} instrumen.`];
    if (c.beats && c.profit) return ['mixed', `Mengalahkan entry acak dan untung secara rata-rata, tetapi tidak merata: hanya untung di ${c.pos} dari ${c.inst} instrumen (atau tidak bertahan merata di 2022–2026), jadi tidak lolos.`];
    if (c.beats) return ['mixed', 'Lebih baik dari entry acak, tetapi tidak untung setelah biaya di semua periode.'];
    if (c.p !== null && c.p < 0.05) return ['mixed', 'Sekilas lebih baik dari entry acak, tetapi tidak bertahan setelah koreksi untuk banyaknya uji atau di 2022–2026.'];
    if (c.profit) return ['fail', 'Untung, tetapi tidak lebih baik dari entry acak. Untungnya datang dari arah pasar, bukan dari sinyal.'];
    return ['fail', 'Tidak lebih baik dari entry acak.'];
  }
  const shortVerdict = (c) => (!c.n ? 'data kurang' : c.passes ? 'Lolos' : c.beats && c.profit ? 'tidak merata' : c.beats ? 'kalah oleh biaya' : c.profit ? 'untung, setara acak' : 'setara acak');
  const bhText = (c) => (!c.inst || c.cagr === null ? '–' : `${pct(c.cagr * 100, 1)} vs ${pct(c.bh * 100, 1)}`);

  // ---------------------------------------------------------------- matriks
  function renderMatrix() {
    const m = INDEX.markets;
    const t = $('matrix');
    let html = '<thead><tr><th></th>' + Object.values(m).map((x) => `<th scope="col">${x}</th>`).join('') + '</tr></thead><tbody>';
    let seenControl = false;
    for (const row of INDEX.matrix) {
      const sep = row.control && !seenControl ? ' sep' : '';
      seenControl = seenControl || row.control;
      html += `<tr class="${row.control ? 'control' : ''}${sep}"><th scope="row">${row.label}</th>`;
      for (const k of Object.keys(m)) {
        const [p, n] = row.cells[k];
        html += n ? `<td><button type="button" class="${p ? 'pass' : ''}" data-g="${row.group}" data-m="${k}" aria-label="${row.label}, ${m[k]}: ${p} dari ${n} lolos">${p} dari ${n}</button></td>` : '<td></td>';
      }
      html += '</tr>';
    }
    t.innerHTML = html + '</tbody>';
    t.addEventListener('click', (ev) => {
      const b = ev.target.closest('button');
      if (!b) return;
      $('f-group').value = b.dataset.g;
      $('f-search').value = '';
      $('f-pass').checked = false;
      renderList();
      const list = INDEX.strategies.filter((s) => s.group === b.dataset.g);
      const best = list.find((s) => s.cells.some((c) => c.market === b.dataset.m && c.passes)) || list[0];
      selectStrategy(best.id, b.dataset.m);
      $('explorer').scrollIntoView();
    });
    const tt = INDEX.totals;
    $('headline-result').textContent =
      `Hasilnya: di saham dan forex ${tt.non_crypto[0]} dari ${nf(tt.non_crypto[1], 0)} kombinasi lolos. ` +
      `Di crypto ${tt.crypto[0]} dari ${nf(tt.crypto[1], 0)}, hampir semuanya ikut-tren atau breakout, dan aturan satu kalimat tanpa indikator hasilnya setara.`;
    $('lede-count').textContent = nf(tt.strategies, 0);
  }

  // ---------------------------------------------------------------- daftar
  function renderGroups() {
    const sel = $('f-group');
    sel.innerHTML = '<option value="">Semua kelompok</option>' +
      Object.entries(INDEX.groups).map(([k, v]) => `<option value="${k}">${v}</option>`).join('');
  }

  function renderList() {
    const g = $('f-group').value, q = $('f-search').value.trim().toLowerCase(), onlyPass = $('f-pass').checked;
    const items = INDEX.strategies.filter((s) => (!g || s.group === g) && (!onlyPass || s.passes > 0) &&
      (!q || s.label.toLowerCase().includes(q) || s.id.includes(q)));
    const ul = $('strategy-list');
    ul.innerHTML = items.length ? items.map((s) =>
      `<li><button type="button" data-id="${s.id}" aria-current="${state.strategy && state.strategy.id === s.id}">` +
      `<span>${s.label}</span>${s.passes ? `<span class="dot">lolos ${s.passes}</span>` : ''}</button></li>`).join('')
      : '<li class="empty">Tidak ada strategi yang cocok. Hapus kata pencarian atau pilih kelompok lain.</li>';
  }

  // ---------------------------------------------------------------- strategi
  async function selectStrategy(id, market) {
    const s = INDEX.strategies.find((x) => x.id === id);
    if (!s) return;
    state.strategy = s;
    $('s-title').textContent = s.label;
    $('s-rule').textContent = s.rule;
    $('s-group').textContent = (window.TA_GROUP_NOTES || {})[s.group] || '';
    renderList();
    const cur = $('strategy-list').querySelector('[aria-current=true]');
    const ul = $('strategy-list');
    if (cur) {   // gulir daftarnya saja, bukan halamannya
      const top = cur.getBoundingClientRect().top - ul.getBoundingClientRect().top;
      if (top < 0 || top > ul.clientHeight - 40) ul.scrollTop += top - 60;
    }
    renderStats();
    state.trades = await loadTrades(id);
    if (state.strategy !== s) return;
    const failed = state.trades._error;
    if (failed) { cache.delete(id); state.trades = {}; }
    const keys = Object.keys(state.trades);
    const want = keys.find((k) => market && k.startsWith(market + '_'));
    const passed = keys.find((k) => s.cells.some((c) => c.passes && `${c.market}_${c.tf}` === k));
    renderCells(keys, failed);
    selectCell(want || passed || keys[0]);
  }

  function renderCells(keys, failed) {
    $('s-cells').innerHTML = keys.map((k) => {
      const [m, tf] = k.split('_');
      return `<button type="button" role="tab" data-k="${k}">${INDEX.markets[m]}, ${TF[tf]}</button>`;
    }).join('') || `<span class="note">${failed ? 'Data contoh transaksi gagal dimuat. Muat ulang halaman; kalau membuka secara lokal, pastikan server menayangkan folder docs.' : 'Strategi ini tidak menghasilkan transaksi di pasar yang ditampilkan.'}</span>`;
  }

  function selectCell(k) {
    state.cell = k || null;
    state.i = 0;
    for (const b of $('s-cells').querySelectorAll('button')) b.setAttribute('aria-selected', b.dataset.k === k);
    const v = $('s-verdict');
    if (k) {
      const [m, tf] = k.split('_');
      const c = state.strategy.cells.find((x) => x.market === m && x.tf === tf);
      const [cls, text] = verdictOf(c);
      v.className = 'verdict ' + cls;
      v.textContent = text + (c && c.n ? ` (${nf(c.n, 0)} transaksi, rata-rata ${pct(c.net * 100, 3)} per transaksi setelah biaya; entry acak ${pct(c.rnd * 100, 3)}.)` : '');
      renderBell(c);
      $('s-bh').textContent = c && c.inst && c.cagr !== null
        ? `Dibanding beli lalu diamkan: strategi ini menghasilkan ${pct(c.cagr * 100, 1)} per tahun (median antar-instrumen), ` +
          `beli lalu diamkan ${pct(c.bh * 100, 1)} per tahun. Strategi menang di ${c.bhWin} dari ${c.inst} instrumen, ` +
          `dengan modal berada di pasar ${nf(Math.min(100, c.expo * 100), 0)}% waktu.`
        : '';
    } else { v.className = 'verdict'; v.textContent = ''; $('bell').innerHTML = ''; $('s-bh').textContent = ''; }
    showTrade();
    syncHash();
  }

  // Posisi strategi di antara versi acaknya, digambar dari skor z uji tersebut
  function renderBell(c) {
    const el = $('bell');
    if (!c || c.z === null || c.p === null) { el.innerHTML = ''; return; }
    const W = 460, H = 118, x0 = 14, x1 = W - 14, base = 86, top = 14;
    const X = (z) => x0 + (z + 4.5) / 9 * (x1 - x0);
    const Y = (z) => base - Math.exp(-z * z / 2) * (base - top);
    let d = '', tail = `M${X(1.645)},${base}`;
    for (let z = -4.5; z <= 4.5001; z += 0.1) d += `${z === -4.5 ? 'M' : 'L'}${X(z).toFixed(1)},${Y(z).toFixed(1)}`;
    for (let z = 1.645; z <= 4.5001; z += 0.1) tail += `L${X(z).toFixed(1)},${Y(z).toFixed(1)}`;
    tail += `L${X(4.5)},${base}Z`;
    const zc = Math.max(-4.3, Math.min(4.3, c.z));
    const col = c.z > 1.645 ? '#12806c' : '#cf4034';
    const off = Math.abs(c.z) > 4.3 ? (c.z > 0 ? ' (jauh di kanan)' : ' (jauh di kiri)') : '';
    el.innerHTML =
      `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Skor z ${nf(c.z, 2)}: posisi strategi dibanding sebaran entry acak">` +
      `<path d="${tail}" fill="#d3efe4"/><path d="${d}" fill="none" stroke="#5b6480" stroke-width="1.5"/>` +
      `<line x1="${x0}" y1="${base}" x2="${x1}" y2="${base}" stroke="#cfd6e6"/>` +
      `<line x1="${X(zc)}" y1="${top - 6}" x2="${X(zc)}" y2="${base}" stroke="${col}" stroke-width="2.5"/>` +
      `<circle cx="${X(zc)}" cy="${top - 6}" r="4.5" fill="${col}"/>` +
      `<text x="${x0}" y="${H - 8}" font-size="12" fill="#5b6480">lebih buruk dari acak</text>` +
      `<text x="${X(0)}" y="${H - 8}" font-size="12" fill="#5b6480" text-anchor="middle">rata-rata entry acak</text>` +
      `<text x="${x1}" y="${H - 8}" font-size="12" fill="#0b5a4b" text-anchor="end">5% teratas</text></svg>` +
      `<p>Garis tegak: strategi ini${off}. Kurva: sebaran hasil ribuan versi entry acak. ` +
      `Peluang entry acak menyamai strategi ini: ${c.p < 0.001 ? 'kurang dari 0,1%' : nf(c.p * 100, 1) + '%'}.</p>`;
  }

  function renderStats() {
    const rows = state.strategy.cells.map((c) => {
      const cls = c.net > 0 ? 'pos' : 'neg';
      return `<tr><td>${INDEX.markets[c.market]}, ${TF[c.tf]}</td><td>${nf(c.n, 0)}</td>` +
        `<td>${c.win === null ? '–' : nf(c.win * 100, 1) + '%'}</td>` +
        `<td class="${cls}">${c.net === null ? '–' : pct(c.net * 100, 3)}</td>` +
        `<td>${c.rnd === null ? '–' : pct(c.rnd * 100, 3)}</td>` +
        `<td>${c.p === null ? '–' : c.p < 0.001 ? '&lt; 0,001' : nf(c.p, 3)}</td>` +
        `<td>${c.inst ? c.pos + ' dari ' + c.inst : '–'}</td><td>${bhText(c)}</td>` +
        `<td class="${c.passes ? 'tag-pass' : ''}">${shortVerdict(c)}</td></tr>`;
    }).join('');
    $('s-stats').innerHTML = '<thead><tr><th>Pasar</th><th>Transaksi</th><th>Menang</th><th>Rata-rata per transaksi</th>' +
      '<th>Entry acak</th><th>Nilai p</th><th>Instrumen untung</th><th>Per tahun: strategi vs beli-diamkan</th><th>Kesimpulan</th></tr></thead><tbody>' + rows + '</tbody>';
  }

  // ---------------------------------------------------------------- transaksi
  const current = () => (state.cell && state.trades[state.cell] ? state.trades[state.cell][state.i] : null);

  function showTrade() {
    clearInterval(state.timer);
    state.revealed = false;
    state.guess = null;
    state.shown = 0;
    state.hover = -1;
    const t = current();
    const list = state.cell ? state.trades[state.cell] : [];
    $('btn-prev').disabled = $('btn-next').disabled = list.length < 2;
    $('t-chips').innerHTML = list.map((_, k) =>
      `<button type="button" data-i="${k}" aria-current="${k === state.i}" aria-label="Contoh ${k + 1}">${k + 1}</button>`).join('');
    $('t-count').textContent = t ? `Contoh ${state.i + 1} dari ${list.length}` : '';
    if (t) {
      const tf = state.cell.split('_')[1];
      $('t-title').textContent = `${t.sym.replace('=X', '').replace('.JK', '')}, ${TF[tf]}, ${fmtDate(t.t[t.e], tf)}`;
    } else $('t-title').textContent = '';
    renderLegend(t);
    updateResult();
    draw();
  }

  function updateResult() {
    const t = current(), el = $('t-result');
    const done = t && state.revealed && state.shown >= t.c.length;
    for (const id of ['btn-up', 'btn-down', 'btn-reveal']) $(id).disabled = !t || state.revealed;
    $('btn-replay').disabled = !done;
    $('t-score').textContent = score.total ? `Tebakan benar: ${score.right} dari ${score.total}` : '';
    if (!t) { el.textContent = ''; return; }
    const side = t.dir === 1 ? 'long' : 'short';
    if (!state.revealed) {
      el.className = 'result';
      el.textContent = `Strategi membuka posisi ${side} di ${price(t.ep)}. Posisi ini akan untung atau rugi?`;
      return;
    }
    if (!done) { el.className = 'result'; el.textContent = 'Memutar candle setelah entry…'; return; }
    el.className = 'result ' + (t.net > 0 ? 'win' : 'loss');
    const where = t.x === null ? ' (keluar di luar tampilan chart)' : '';
    const g = state.guess === null ? '' : ((t.net > 0) === (state.guess === 'up') ? ' Tebakan Anda benar.' : ' Tebakan Anda meleset.');
    el.textContent = `Posisi ${side} dari ${price(t.ep)}, keluar di ${price(t.xp)} setelah ${t.bars} candle${where}: ${pct(t.net)} setelah biaya.${g}`;
  }

  function reveal(guess) {
    const t = current();
    if (!t || state.revealed) return;
    state.revealed = true;
    state.guess = guess;
    if (guess !== null) { score.total += 1; if ((t.net > 0) === (guess === 'up')) score.right += 1; }
    play();
  }

  function play() {
    const t = current();
    if (!t) return;
    clearInterval(state.timer);
    const n = t.c.length;
    state.shown = reduced ? n : t.e;
    const stepSize = Math.max(1, Math.ceil((n - t.e) / 60));
    const tick = () => {
      state.shown = Math.min(n, state.shown + stepSize);
      if (state.shown >= n) clearInterval(state.timer);
      updateResult();
      draw();
    };
    if (reduced) { updateResult(); draw(); } else state.timer = setInterval(tick, 33);
  }

  function renderLegend(t) {
    const items = [];
    if (t) {
      if (t.shapes.length) items.push([C.mark, 'pola / zona / level yang dipakai strategi']);
      t.ov.forEach((o, k) => items.push([OV_COLORS[k % OV_COLORS.length], o.name]));
      if (t.sl !== null) items.push([C.down, 'stop loss']);
      if (t.tp !== null) items.push([C.up, 'take profit']);
    }
    $('legend').innerHTML = items.map(([c, n]) => `<span><i style="background:${c}"></i>${n}</span>`).join('');
  }

  function fmtDate(min, tf) {
    const d = new Date(min * 60000);
    const o = { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' };
    if (tf !== '1d') Object.assign(o, { hour: '2-digit', minute: '2-digit' });
    return d.toLocaleString('id-ID', o) + (tf !== '1d' ? ' UTC' : '');
  }

  // ---------------------------------------------------------------- chart
  const canvas = $('chart');
  const ctx = canvas.getContext('2d');
  let geo = null;

  function draw() {
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth, H = canvas.clientHeight;
    if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
      canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const t = current();
    geo = null;
    if (!t) {
      ctx.fillStyle = C.muted; ctx.font = '16px "Atkinson Hyperlegible Next", sans-serif'; ctx.textAlign = 'center';
      ctx.fillText('Pilih strategi di daftar untuk melihat contoh transaksinya.', W / 2, H / 2);
      return;
    }
    const n = t.c.length;
    const nVis = state.revealed ? Math.min(n, Math.max(state.shown, t.e)) : t.e;   // candle yang digambar
    const done = state.revealed && state.shown >= n;
    const nScale = state.revealed ? n : Math.min(n, t.e + Math.max(10, Math.round(t.e * 0.18)));
    const subs = t.ov.filter((o) => o.pane === 'sub');
    const L = 10, R = 68, T = 30, B = 26, subH = subs.length ? Math.min(130, H * 0.24) : 0, gap = subs.length ? 14 : 0;
    const pH = H - T - B - subH - gap, pW = W - L - R;
    const bw = pW / nScale;
    const X = (i) => L + (i + 0.5) * bw;

    // rentang harga
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < Math.max(nVis, 1); i++) { lo = Math.min(lo, t.l[i]); hi = Math.max(hi, t.h[i]); }
    const span0 = hi - lo || hi * 0.01;
    const extra = [t.ep];
    if (t.sl !== null) extra.push(t.sl);
    if (t.tp !== null) extra.push(t.tp);
    for (const s of t.shapes) {
      if (s.pts) s.pts.forEach((p) => extra.push(p[1]));
      for (const k of ['p', 'p0', 'p1', 'lo', 'hi']) if (s[k] !== undefined && s[k] !== null) extra.push(s[k]);
      if (s.levels) s.levels.forEach((p) => extra.push(p[1]));
    }
    const soft = [];   // garis indikator hanya boleh sedikit melebarkan skala
    for (const o of t.ov) if (o.pane === 'price') for (let i = 0; i < nVis; i++) if (o.v[i] !== null) soft.push(o.v[i]);
    const lo0 = lo, hi0 = hi;
    for (const v of extra) if (v > lo0 - 1.2 * span0 && v < hi0 + 1.2 * span0) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    for (const v of soft) if (v > lo0 - 0.25 * span0 && v < hi0 + 0.25 * span0) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const pad = (hi - lo) * 0.06; lo -= pad; hi += pad;
    const Y = (p) => T + (hi - p) / (hi - lo) * pH;
    geo = { L, bw, nScale, nVis };

    ctx.font = '12.5px "Atkinson Hyperlegible Next", sans-serif';
    ctx.textBaseline = 'middle';

    // grid + sumbu harga
    const step = niceStep((hi - lo) / 5);
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1; ctx.fillStyle = C.muted; ctx.textAlign = 'left';
    for (let p = Math.ceil(lo / step) * step; p < hi; p += step) {
      const y = Math.round(Y(p)) + 0.5;
      ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(L + pW, y); ctx.stroke();
      ctx.fillText(price(p), L + pW + 6, y);
    }
    // sumbu waktu
    const tf = state.cell.split('_')[1];
    ctx.textAlign = 'center';
    const every = Math.max(1, Math.round(nScale / Math.max(2, Math.floor(pW / 150))));
    for (let i = Math.floor(every / 2); i < nScale; i += every) {
      ctx.fillText(fmtDate(t.t[Math.min(i, n - 1)], '1d'), X(i), H - B / 2 + 2);
    }

    ctx.save();
    ctx.beginPath(); ctx.rect(L, T - 8, pW, pH + 16); ctx.clip();
    const lastX = (i) => Math.min(i, state.revealed ? Math.max(nVis - 1, t.e) : t.e);

    // latar: span & zona
    for (const s of t.shapes) {
      if (s.k === 'span') {
        ctx.fillStyle = 'rgba(106,69,217,.13)';
        ctx.fillRect(X(s.i0) - bw / 2, T - 8, (s.i1 - s.i0 + 1) * bw, pH + 16);
      } else if (s.k === 'fill') {
        ctx.fillStyle = 'rgba(106,69,217,.13)';
        ctx.beginPath();
        s.pts.forEach((p, q) => (q ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1]))));
        ctx.closePath(); ctx.fill();
      } else if (s.k === 'zone') {
        ctx.fillStyle = 'rgba(106,69,217,.16)';
        ctx.fillRect(X(s.i0) - bw / 2, Y(s.hi), (lastX(s.i1) - s.i0 + 1) * bw, Y(s.lo) - Y(s.hi));
        ctx.strokeStyle = C.mark; ctx.lineWidth = 1;
        ctx.strokeRect(X(s.i0) - bw / 2 + 0.5, Y(s.hi), (lastX(s.i1) - s.i0 + 1) * bw, Y(s.lo) - Y(s.hi));
      }
    }
    // area untung/rugi setelah dibuka
    if (done) {
      const x1 = t.x === null ? n - 1 : t.x;
      ctx.fillStyle = t.net > 0 ? 'rgba(18,128,108,.12)' : 'rgba(207,64,52,.12)';
      ctx.fillRect(X(t.e) - bw / 2, Math.min(Y(t.ep), Y(t.xp)), (x1 - t.e + 1) * bw, Math.abs(Y(t.ep) - Y(t.xp)));
    }
    // candle
    const body = Math.max(1, Math.min(bw * 0.7, 14));
    for (let i = 0; i < nVis; i++) {
      const up = t.c[i] >= t.o[i];
      ctx.strokeStyle = ctx.fillStyle = up ? C.up : C.down;
      const x = Math.round(X(i)) + 0.5;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, Y(t.h[i])); ctx.lineTo(x, Y(t.l[i])); ctx.stroke();
      const y0 = Y(Math.max(t.o[i], t.c[i])), hh = Math.max(1, Math.abs(Y(t.o[i]) - Y(t.c[i])));
      ctx.fillRect(x - body / 2, y0, body, hh);
    }
    // garis indikator di panel harga
    t.ov.forEach((o, k) => { if (o.pane === 'price') line(o.v, nVis, X, Y, OV_COLORS[k % OV_COLORS.length]); });

    // bentuk
    ctx.lineWidth = 2; ctx.strokeStyle = C.mark; ctx.fillStyle = C.mark;
    for (const s of t.shapes) {
      if (s.k === 'poly' || s.k === 'dots') {
        const lead = s.lead || 0;   // ruas awal = tren sebelum pola, digambar tipis
        if (s.k === 'poly') {
          ctx.lineWidth = s.thin ? 1.2 : 2; ctx.globalAlpha = s.thin ? 0.55 : 1;
          if (lead) {
            ctx.save(); ctx.globalAlpha = 0.45; ctx.setLineDash([4, 4]); ctx.lineWidth = 1.4; ctx.beginPath();
            for (let q = 0; q <= lead; q++) (q ? ctx.lineTo(X(s.pts[q][0]), Y(s.pts[q][1])) : ctx.moveTo(X(s.pts[q][0]), Y(s.pts[q][1])));
            ctx.stroke(); ctx.restore();
          }
          ctx.beginPath();
          for (let q = lead; q < s.pts.length; q++) (q > lead ? ctx.lineTo(X(s.pts[q][0]), Y(s.pts[q][1])) : ctx.moveTo(X(s.pts[q][0]), Y(s.pts[q][1])));
          ctx.stroke();
        }
        ctx.lineWidth = 2; ctx.globalAlpha = 1;
        s.pts.forEach((p, q) => {
          if (s.dots !== false && q >= lead) { ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 3.5, 0, 6.3); ctx.fill(); }
          if (s.numbered) { ctx.textAlign = 'center'; ctx.fillText(String(q), X(p[0]), Y(p[1]) + (q % 2 ? -12 : 13)); }
          const lb = s.labels && s.labels[q];
          if (lb) tag(lb[0], X(p[0]), Y(p[1]) - lb[1] * 17);
        });
      } else if (s.k === 'zone' && s.tag) {
        label(s.tag, X(s.i0) + 4, Y(s.lo) + 12, C.mark, 'left', true);
      } else if (s.k === 'hline') {
        dash(X(s.i0) - bw / 2, Y(s.p), X(lastX(s.i1)) + bw / 2, Y(s.p), C.mark, [6, 4], 1.6);
        label(s.label, X(s.i0), Y(s.p) - 9, C.mark, 'left');
      } else if (s.k === 'line') {
        ctx.save(); if (s.dash) ctx.setLineDash([6, 4]);
        ctx.beginPath(); ctx.moveTo(X(s.i0), Y(s.p0)); ctx.lineTo(X(s.i1), Y(s.p1)); ctx.stroke(); ctx.restore();
        if (s.label) label(s.label, X(s.i0), Y(s.p0) + (s.p1 > s.p0 ? 12 : -12), C.mark, 'left');
      } else if (s.k === 'fib') {
        ctx.beginPath(); ctx.moveTo(X(s.ia), Y(s.a)); ctx.lineTo(X(s.ib), Y(s.b)); ctx.stroke();
        for (const [r, p] of s.levels) {
          const used = Math.abs(r - s.used) < 1e-9;
          dash(X(s.ia), Y(p), X(lastX(s.i1)) + bw / 2, Y(p), used ? C.mark : 'rgba(106,69,217,.45)', used ? [] : [3, 4], used ? 2 : 1);
          label(nf(r * 100, 1) + '%', X(s.ia) - 4, Y(p), used ? C.mark : C.muted, 'right');
        }
      }
    }
    // SL / TP
    const xe = X(t.e) - bw / 2, xr = state.revealed ? X(t.x === null ? n - 1 : t.x) + bw / 2 : L + pW;
    if (t.sl !== null) { dash(xe, Y(t.sl), xr, Y(t.sl), C.down, [5, 4], 1.4); label('SL', xr - 4, Y(t.sl) - 9, C.down, 'right'); }
    if (t.tp !== null) { dash(xe, Y(t.tp), xr, Y(t.tp), C.up, [5, 4], 1.4); label('TP', xr - 4, Y(t.tp) - 9, C.up, 'right'); }
    // entry
    marker(X(t.e), Y(t.ep), t.dir, t.dir === 1 ? 'Long' : 'Short');
    if (done && t.x !== null) {
      ctx.strokeStyle = C.ink; ctx.lineWidth = 2.2;
      const x = X(t.x), y = Y(t.xp);
      ctx.beginPath(); ctx.moveTo(x - 5, y - 5); ctx.lineTo(x + 5, y + 5); ctx.moveTo(x + 5, y - 5); ctx.lineTo(x - 5, y + 5); ctx.stroke();
      const far = x > L + pW - 110;
      label('Keluar ' + pct(t.net), far ? x - 9 : x + 9, y, t.net > 0 ? C.up : C.down, far ? 'right' : 'left', true);
    }
    // crosshair
    if (state.hover >= 0 && state.hover < nVis) {
      const x = Math.round(X(state.hover)) + 0.5;
      ctx.strokeStyle = 'rgba(20,27,51,.35)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, T - 8); ctx.lineTo(x, T + pH + 8); ctx.stroke();
    }
    ctx.restore();

    // panel bawah (oscillator)
    if (subs.length) {
      const y0 = T + pH + gap;
      let slo = Infinity, shi = -Infinity;
      for (const o of subs) {
        for (let i = 0; i < nVis; i++) if (o.v[i] !== null) { slo = Math.min(slo, o.v[i]); shi = Math.max(shi, o.v[i]); }
        (o.ref || []).forEach((r) => { slo = Math.min(slo, r); shi = Math.max(shi, r); });
      }
      const sp = (shi - slo || 1) * 0.08; slo -= sp; shi += sp;
      const YS = (v) => y0 + (shi - v) / (shi - slo) * subH;
      ctx.strokeStyle = C.grid; ctx.strokeRect(L + 0.5, y0 + 0.5, pW, subH);
      ctx.save(); ctx.beginPath(); ctx.rect(L, y0, pW, subH); ctx.clip();
      for (const o of subs) for (const r of o.ref || []) {
        dash(L, YS(r), L + pW, YS(r), 'rgba(20,27,51,.35)', [3, 4], 1);
      }
      t.ov.forEach((o, k) => { if (o.pane === 'sub') line(o.v, nVis, X, YS, OV_COLORS[k % OV_COLORS.length]); });
      ctx.restore();
      ctx.fillStyle = C.muted; ctx.textAlign = 'left';
      for (const o of subs) for (const r of o.ref || []) ctx.fillText(nf(r, 0), L + pW + 6, YS(r));
    }
  }

  function niceStep(raw) {
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    const f = raw / p;
    return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p;
  }
  function line(v, nVis, X, Y, color) {
    ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.beginPath();
    let pen = false;
    for (let i = 0; i < nVis; i++) {
      if (v[i] === null) { pen = false; continue; }
      if (pen) ctx.lineTo(X(i), Y(v[i])); else ctx.moveTo(X(i), Y(v[i]));
      pen = true;
    }
    ctx.stroke();
  }
  function dash(x0, y0, x1, y1, color, pattern, w) {
    ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = w; ctx.setLineDash(pattern);
    ctx.beginPath(); ctx.moveTo(x0, Math.round(y0) + 0.5); ctx.lineTo(x1, Math.round(y1) + 0.5); ctx.stroke(); ctx.restore();
  }
  function label(text, x, y, color, align, bold) {
    ctx.save();
    ctx.font = (bold ? '700 ' : '600 ') + '12.5px "Atkinson Hyperlegible Next", sans-serif';
    ctx.textAlign = align; ctx.textBaseline = 'middle';
    const w = ctx.measureText(text).width;
    const bx = align === 'left' ? x - 3 : align === 'right' ? x - w - 3 : x - w / 2 - 3;
    ctx.fillStyle = 'rgba(251,252,254,.85)'; ctx.fillRect(bx, y - 9, w + 6, 18);
    ctx.fillStyle = color; ctx.fillText(text, x, y); ctx.restore();
  }
  function tag(text, x, y) {   // label bagian pola, gaya kotak seperti di platform charting
    ctx.save();
    ctx.font = '700 12px "Atkinson Hyperlegible Next", sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    const w = ctx.measureText(text).width + 12;
    ctx.fillStyle = C.mark; ctx.fillRect(x - w / 2, y - 10, w, 20);
    ctx.fillStyle = '#fff'; ctx.fillText(text, x, y + 0.5); ctx.restore();
  }
  function marker(x, y, dir, text) {
    const col = dir === 1 ? C.up : C.down, s = 9;
    ctx.fillStyle = col; ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5;
    ctx.beginPath();
    if (dir === 1) { ctx.moveTo(x, y); ctx.lineTo(x - s, y + s * 1.5); ctx.lineTo(x + s, y + s * 1.5); }
    else { ctx.moveTo(x, y); ctx.lineTo(x - s, y - s * 1.5); ctx.lineTo(x + s, y - s * 1.5); }
    ctx.closePath(); ctx.fill(); ctx.stroke();
    label(text, x, y + dir * (s * 1.5 + 12), col, 'center', true);
  }

  canvas.addEventListener('mousemove', (ev) => {
    const t = current();
    if (!t || !geo) return;
    const r = canvas.getBoundingClientRect();
    const i = Math.floor((ev.clientX - r.left - geo.L) / geo.bw);
    const ok = i >= 0 && i < geo.nVis;
    state.hover = ok ? i : -1;
    const tf = state.cell.split('_')[1];
    $('readout').textContent = ok
      ? `${fmtDate(t.t[i], tf)}   O ${price(t.o[i])}  H ${price(t.h[i])}  L ${price(t.l[i])}  C ${price(t.c[i])}` +
        t.ov.map((o) => (o.v[i] === null ? '' : `  ${o.name} ${price(o.v[i])}`)).join('')
      : '';
    draw();
  });
  canvas.addEventListener('mouseleave', () => { state.hover = -1; $('readout').textContent = ''; draw(); });

  // ---------------------------------------------------------------- kontrol
  function step(d) {
    const list = state.cell ? state.trades[state.cell] : [];
    if (!list.length) return;
    state.i = (state.i + d + list.length) % list.length;
    showTrade();
  }
  $('btn-up').addEventListener('click', () => reveal('up'));
  $('btn-down').addEventListener('click', () => reveal('down'));
  $('btn-reveal').addEventListener('click', () => reveal(null));
  $('btn-replay').addEventListener('click', play);
  $('t-chips').addEventListener('click', (ev) => {
    const b = ev.target.closest('button');
    if (b) { state.i = +b.dataset.i; showTrade(); }
  });
  $('btn-prev').addEventListener('click', () => step(-1));
  $('btn-next').addEventListener('click', () => step(1));
  $('s-cells').addEventListener('click', (ev) => { const b = ev.target.closest('button'); if (b) selectCell(b.dataset.k); });
  $('strategy-list').addEventListener('click', (ev) => { const b = ev.target.closest('button'); if (b) selectStrategy(b.dataset.id); });
  $('f-group').addEventListener('change', renderList);
  $('f-search').addEventListener('input', renderList);
  $('f-pass').addEventListener('change', renderList);
  document.addEventListener('keydown', (ev) => {
    if (/INPUT|SELECT|TEXTAREA/.test(ev.target.tagName)) return;
    if (ev.key === 'ArrowRight') step(1);
    else if (ev.key === 'ArrowLeft') step(-1);
  });
  window.addEventListener('resize', draw);

  function syncHash() {
    if (state.strategy) history.replaceState(null, '', `#${state.strategy.id}${state.cell ? '/' + state.cell : ''}`);
  }

  window.addEventListener('hashchange', () => {
    const [id, cell] = decodeURIComponent(location.hash.slice(1)).split('/');
    if (INDEX && id && (!state.strategy || state.strategy.id !== id)) selectStrategy(id, cell ? cell.split('_')[0] : null);
  });

  // ---------------------------------------------------------------- mulai
  fetch('data/index.json').then((r) => r.json()).then((idx) => {
    INDEX = idx;
    renderMatrix();
    renderGroups();
    renderList();
    const [id, cell] = decodeURIComponent(location.hash.slice(1)).split('/');
    const start = INDEX.strategies.find((s) => s.id === id) ? id : 'head_shoulders_sedang_mm';
    selectStrategy(start, cell ? cell.split('_')[0] : null);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(draw);
    window.dispatchEvent(new CustomEvent('ta-ready', { detail: {
      index: idx, verdict: shortVerdict,
      open: (sid, market) => { selectStrategy(sid, market); $('explorer').scrollIntoView(); },
    } }));
  }).catch(() => {
    $('headline-result').textContent = 'Data tidak bisa dimuat. Buka halaman ini lewat server web, bukan langsung dari berkas.';
  });
})();
