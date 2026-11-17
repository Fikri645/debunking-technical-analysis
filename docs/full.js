// Chart seluruh riwayat uji: semua candle satu instrumen, semua posisi strategi, bisa digeser dan di-zoom.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const TF = { '1d': 'harian', '4h': '4 jam', '1h': '1 jam', '15m': '15 menit' };
  const C = { ink: '#141b33', muted: '#5b6480', grid: '#e3e8f2', up: '#12806c', down: '#cf4034', mark: '#6a45d9' };
  const OV_COLORS = ['#2f6fb0', '#d98a12', '#7d8496', '#b0559a'];
  const FONT = '"Atkinson Hyperlegible Next", sans-serif';
  const nf = (x, d = 2) => x.toLocaleString('id-ID', { minimumFractionDigits: d, maximumFractionDigits: d });
  const pct = (x, d = 1) => (x > 0 ? '+' : x < 0 ? '−' : '') + nf(Math.abs(x), d) + '%';
  const price = (x) => { const a = Math.abs(x); return nf(x, a >= 1000 ? 0 : a >= 100 ? 1 : a >= 1 ? 2 : a >= 0.01 ? 4 : 6); };
  const symName = (s) => s.replace('=X', '').replace('.JK', '').replace('^', '');
  const fileName = (s) => s.replace('^', '_').replace('=', '_');
  const fmtDate = (min, tf) => {
    const o = { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' };
    if (tf !== '1d') Object.assign(o, { hour: '2-digit', minute: '2-digit' });
    return new Date(min * 60000).toLocaleString('id-ID', o);
  };

  // ---------------------------------------------------------------- berkas biner
  const TYPES = { u4: Uint32Array, f4: Float32Array, u1: Uint8Array };
  const datCache = new Map();
  function loadDat(url) {
    if (!datCache.has(url)) {
      datCache.set(url, (async () => {
        const r = await fetch(url);
        if (!r.ok) throw new Error(url);
        const buf = await new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
        const hl = new DataView(buf).getUint32(0, true);
        const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)));
        let off = 4 + hl;
        const A = {};
        for (const [name, type, len] of header.arrays) {
          const T = TYPES[type];
          A[name] = new T(buf, off, len);
          off = (off + len * T.BYTES_PER_ELEMENT + 3) & ~3;
        }
        return { header, A };
      })().catch((e) => { datCache.delete(url); throw e; }));
    }
    return datCache.get(url);
  }

  // ---------------------------------------------------------------- indikator (sama dengan src/talab/indicators.py)
  const nanArr = (n) => new Float64Array(n).fill(NaN);
  function sma(x, w) {
    const n = x.length, out = nanArr(n);
    let s = 0, bad = 0;
    for (let i = 0; i < n; i++) {
      if (Number.isNaN(x[i])) bad++; else s += x[i];
      if (i >= w) { if (Number.isNaN(x[i - w])) bad--; else s -= x[i - w]; }
      if (i >= w - 1 && !bad) out[i] = s / w;
    }
    return out;
  }
  function ewm(x, alpha, minp) {
    const n = x.length, out = nanArr(n);
    let y = NaN, cnt = 0;
    for (let i = 0; i < n; i++) {
      if (Number.isNaN(x[i])) continue;
      y = Number.isNaN(y) ? x[i] : alpha * x[i] + (1 - alpha) * y;
      if (++cnt >= minp) out[i] = y;
    }
    return out;
  }
  const ema = (x, w) => ewm(x, 2 / (w + 1), w);
  const wilder = (x, w) => ewm(x, 1 / w, w);
  function roll(x, w, max) {
    const n = x.length, out = nanArr(n);
    for (let i = w - 1; i < n; i++) {
      let m = x[i];
      for (let j = i - w + 1; j < i; j++) if (max ? x[j] > m : x[j] < m) m = x[j];
      out[i] = m;
    }
    return out;
  }
  const shiftArr = (x, k) => { const out = nanArr(x.length); for (let i = k; i < x.length; i++) out[i] = x[i - k]; return out; };
  const zip = (a, b, f) => { const out = nanArr(a.length); for (let i = 0; i < a.length; i++) out[i] = f(a[i], b[i]); return out; };
  function atr(K, w = 14) {
    const n = K.n, tr = nanArr(n);
    for (let i = 0; i < n; i++) {
      const pc = i ? K.c[i - 1] : K.c[0];
      tr[i] = Math.max(K.h[i] - K.l[i], Math.abs(K.h[i] - pc), Math.abs(K.l[i] - pc));
    }
    return wilder(tr, w);
  }
  function rsi(c, w = 14) {
    const n = c.length, up = nanArr(n), dn = nanArr(n);
    for (let i = 1; i < n; i++) { const d = c[i] - c[i - 1]; up[i] = d > 0 ? d : 0; dn[i] = d < 0 ? -d : 0; }
    const au = wilder(up, w), ad = wilder(dn, w);
    return zip(au, ad, (a, b) => (Number.isNaN(a) ? NaN : b === 0 ? 100 : 100 - 100 / (1 + a / b)));
  }
  function macd(c) {
    const line = zip(ema(c, 12), ema(c, 26), (a, b) => a - b);
    return [line, ewm(line, 2 / 10, 9)];
  }
  function stochastic(K) {
    const hh = roll(K.h, 14, true), ll = roll(K.l, 14, false), raw = nanArr(K.n);
    for (let i = 0; i < K.n; i++) { const v = 100 * (K.c[i] - ll[i]) / (hh[i] - ll[i]); raw[i] = Number.isFinite(v) ? v : NaN; }
    const k = sma(raw, 3);
    return [k, sma(k, 3)];
  }
  function bollinger(c, w = 20, k = 2) {
    const mid = sma(c, w), up = nanArr(c.length), lo = nanArr(c.length);
    for (let i = w - 1; i < c.length; i++) {
      let s = 0;
      for (let j = i - w + 1; j <= i; j++) s += (c[j] - mid[i]) ** 2;
      const sd = Math.sqrt(s / w);
      up[i] = mid[i] + k * sd; lo[i] = mid[i] - k * sd;
    }
    return [up, mid, lo];
  }
  function supertrendLine(K, a, mult) {
    const out = nanArr(K.n);
    let fu = NaN, fl = NaN, d = 0;
    for (let i = 0; i < K.n; i++) {
      if (Number.isNaN(a[i])) continue;
      const mid = (K.h[i] + K.l[i]) / 2, bu = mid + mult * a[i], bl = mid - mult * a[i];
      if (d === 0) { fu = bu; fl = bl; d = K.c[i] >= mid ? 1 : -1; } else {
        fu = (bu < fu || K.c[i - 1] > fu) ? bu : fu;
        fl = (bl > fl || K.c[i - 1] < fl) ? bl : fl;
        if (d === 1 && K.c[i] < fl) d = -1; else if (d === -1 && K.c[i] > fu) d = 1;
      }
      out[i] = d === 1 ? fl : fu;
    }
    return out;
  }
  function psarLine(K, af0 = 0.02, afMax = 0.2) {
    const n = K.n, out = nanArr(n), h = K.h, l = K.l;
    if (n < 3) return out;
    let up = h[1] >= h[0], sar = up ? l[0] : h[0], ep = up ? h[1] : l[1], af = af0;
    for (let i = 2; i < n; i++) {
      sar += af * (ep - sar);
      if (up) {
        sar = Math.min(sar, l[i - 1], l[i - 2]);
        if (l[i] < sar) { up = false; sar = ep; ep = l[i]; af = af0; } else if (h[i] > ep) { ep = h[i]; af = Math.min(af + af0, afMax); }
      } else {
        sar = Math.max(sar, h[i - 1], h[i - 2]);
        if (h[i] > sar) { up = true; sar = ep; ep = h[i]; af = af0; } else if (l[i] < ep) { ep = l[i]; af = Math.min(af + af0, afMax); }
      }
      out[i] = sar;
    }
    return out;
  }
  function adx(K, w = 14) {
    const n = K.n, pdm = nanArr(n), mdm = nanArr(n);
    for (let i = 0; i < n; i++) {
      const u = i ? K.h[i] - K.h[i - 1] : 0, d = i ? K.l[i - 1] - K.l[i] : 0;
      pdm[i] = u > d && u > 0 ? u : 0; mdm[i] = d > u && d > 0 ? d : 0;
    }
    const tr = atr(K, w), pdi = zip(wilder(pdm, w), tr, (a, b) => 100 * a / b), mdi = zip(wilder(mdm, w), tr, (a, b) => 100 * a / b);
    const dx = zip(pdi, mdi, (a, b) => { if (Number.isNaN(a)) return NaN; const v = 100 * Math.abs(a - b) / (a + b); return Number.isFinite(v) ? v : 0; });
    return [wilder(dx, w), pdi, mdi];
  }
  function obv(K) {
    const out = nanArr(K.n);
    let s = 0;
    for (let i = 0; i < K.n; i++) { s += (i ? Math.sign(K.c[i] - K.c[i - 1]) : 0) * K.v[i]; out[i] = s; }
    return out;
  }
  function vwap(K) {
    const out = nanArr(K.n);
    let day = -1, pv = 0, vv = 0;
    for (let i = 0; i < K.n; i++) {
      const dd = Math.floor(K.t[i] / 1440);
      if (dd !== day) { day = dd; pv = 0; vv = 0; }
      pv += (K.h[i] + K.l[i] + K.c[i]) / 3 * K.v[i]; vv += K.v[i];
      out[i] = vv ? pv / vv : NaN;
    }
    return out;
  }
  const O = (name, v, pane = 'price', ref) => ({ name, v, pane, ref });
  function overlays(base, K) {
    const c = K.c;
    const M = () => { const [a, b] = macd(c); return [O('MACD', a, 'sub', [0]), O('Sinyal', b, 'sub')]; };
    const ST = () => { const [k, d] = stochastic(K); return [O('%K', k, 'sub', [20, 80]), O('%D', d, 'sub')]; };
    if (base === 'supertrend_10_3') return [O('Supertrend', supertrendLine(K, atr(K, 10), 3))];
    if (base === 'parabolic_sar') return [O('Parabolic SAR', psarLine(K))];
    if (base === 'vwap') return [O('VWAP', vwap(K))];
    if (base === 'sma_20_50') return [O('SMA 20', sma(c, 20)), O('SMA 50', sma(c, 50))];
    if (base === 'golden_cross_50_200' || base === 'pullback_ma50') return [O('SMA 50', sma(c, 50)), O('SMA 200', sma(c, 200))];
    if (base === 'ema_9_21' || base === 'mtf_ema_9_21') return [O('EMA 9', ema(c, 9)), O('EMA 21', ema(c, 21))];
    if (base === 'price_vs_ema200' || base === 'triangle_fib1272_mtf') return [O('EMA 200', ema(c, 200))];
    if (base.startsWith('macd') || base === 'mtf_macd_signal') return M();
    if (base === 'confluence_ema200_macd_rsi') return [O('EMA 200', ema(c, 200)), O('RSI 14', rsi(c), 'sub', [50])];
    if (base.startsWith('rsi') || base === 'mtf_rsi_cross') return [O('RSI 14', rsi(c), 'sub', [30, 70])];
    if (base.startsWith('stochastic') || base.startsWith('the_grab_stoch')) return base.endsWith('tren') ? [...ST(), O('EMA 200', ema(c, 200))] : ST();
    if (base.startsWith('bollinger')) { const [u, m, l] = bollinger(c); return [O('Band atas', u), O('Tengah', m), O('Band bawah', l)]; }
    if (base === 'ichimoku') {
      const mid = (w) => zip(roll(K.h, w, true), roll(K.l, w, false), (a, b) => (a + b) / 2);
      const tk = mid(9), kj = mid(26);
      return [O('Tenkan', tk), O('Kijun', kj), O('Span A', shiftArr(zip(tk, kj, (a, b) => (a + b) / 2), 26)), O('Span B', shiftArr(mid(52), 26))];
    }
    if (base === 'adx_di_25') { const [a, p, m] = adx(K); return [O('ADX', a, 'sub', [25]), O('+DI', p, 'sub'), O('−DI', m, 'sub')]; }
    if (base === 'donchian_20_10' || base === 'volume_breakout') return [O('High 20', shiftArr(roll(K.h, 20, true), 1)), O('Low 20', shiftArr(roll(K.l, 20, false), 1))];
    if (base === 'obv_ema20') { const ob = obv(K); return [O('OBV', ob, 'sub'), O('EMA 20 OBV', ema(ob, 20), 'sub')]; }
    return [];
  }

  // ---------------------------------------------------------------- bentuk pola dari titik swing detektor
  function arc(i0, p0, ib, pb, i1, p1, n = 20) {
    const out = [];
    for (let q = 0; q <= n; q++) { const v = 1 - q / n; out.push([i0 + (ib - i0) * q / n, pb + (p0 - pb) * (1 - Math.sqrt(Math.max(0, 1 - v * v)))]); }
    for (let q = 1; q <= n; q++) { const v = q / n; out.push([ib + (i1 - ib) * q / n, pb + (p1 - pb) * (1 - Math.sqrt(Math.max(0, 1 - v * v)))]); }
    return out;
  }
  function patternShapes(pat, P, d, sig, K) {
    const side = P.map((p, q) => (p[1] > P[q ? q - 1 : 1][1] ? 1 : -1));
    const lab = (q, text) => [text, side[q]];
    const body = (neck, names, neckLabel) => {
      const last = names.length;
      const ga = P[0][1] - neck(P[0][0]), gb = P[1][1] - neck(P[1][0]);
      let left = null;
      if (ga * gb < 0) { const t = ga / (ga - gb), i = P[0][0] + t * (P[1][0] - P[0][0]); left = [i, neck(i)]; }
      const pts = [...(left ? [left] : []), ...P.slice(1, last + 1), [sig, neck(sig)]];
      const labels = [...(left ? [null] : []), ...names.map((nm, q) => (nm ? lab(q + 1, nm) : null)), null];
      return [{ k: 'fill', pts }, { k: 'poly', pts, labels }, { k: 'line', i0: pts[0][0], p0: neck(pts[0][0]), i1: sig, p1: neck(sig), label: neckLabel, dash: true }];
    };
    if (pat === 'head_shoulders' && P.length >= 6) {
      const [i1, t1] = P[2], [i2, t2] = P[4], s = (t2 - t1) / (i2 - i1);
      return body((i) => t2 + s * (i - i2), ['bahu kiri', null, 'kepala', null, 'bahu kanan'], 'garis leher');
    }
    if (pat === 'double_top_bottom' && P.length >= 4) {
      const w = d === -1 ? 'puncak' : 'dasar', lvl = P[2][1];
      return body(() => lvl, [w + ' 1', null, w + ' 2'], 'garis leher');
    }
    if (pat === 'triple_top_bottom' && P.length >= 6) {
      const w = d === -1 ? 'puncak' : 'dasar', lvl = d === -1 ? Math.min(P[2][1], P[4][1]) : Math.max(P[2][1], P[4][1]);
      return body(() => lvl, [w + ' 1', null, w + ' 2', null, w + ' 3'], d === -1 ? 'support' : 'resistance');
    }
    if (['triangle_asc_desc', 'triangle_symmetrical', 'rectangle', 'wedge'].includes(pat) && P.length >= 4) {
      const i0 = P[0][0], lines = {};
      for (const sd of [1, -1]) {
        const two = [];
        for (let q = P.length - 4; q < P.length; q++) if (side[q] === sd) two.push(P[q]);
        if (two.length < 2) return [{ k: 'poly', pts: P }];
        const s = (two[1][1] - two[0][1]) / (two[1][0] - two[0][0]);
        lines[sd] = { k: 'line', i0, p0: two[1][1] + s * (i0 - two[1][0]), i1: sig, p1: two[1][1] + s * (sig - two[1][0]), label: sd === 1 ? 'garis atas' : 'garis bawah' };
      }
      const u = lines[1], lo = lines[-1];
      return [{ k: 'fill', pts: [[i0, u.p0], [sig, u.p1], [sig, lo.p1], [i0, lo.p0]] }, u, lo, { k: 'dots', pts: P }];
    }
    if (pat === 'flag_pennant' && P.length >= 2) {
      const [ia, a] = P[0], [ib, b] = P[1];
      let x0 = ib + 1, x1 = sig - 1;
      if (x1 - x0 < 1) { x0 = ib; x1 = sig; }
      let m = 0;
      const cnt = x1 - x0 + 1;
      if (cnt > 3) {
        let sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (let i = x0; i <= x1; i++) { sx += i; sy += K.c[i]; sxx += i * i; sxy += i * K.c[i]; }
        m = (cnt * sxy - sx * sy) / (cnt * sxx - sx * sx);
      }
      let up = -Infinity, dn = Infinity;
      for (let i = x0; i <= x1; i++) { up = Math.max(up, K.h[i] - m * i); dn = Math.min(dn, K.l[i] - m * i); }
      const fu = (i) => up + m * i, fd = (i) => dn + m * i;
      return [{ k: 'fill', pts: [[ib, fu(ib)], [sig, fu(sig)], [sig, fd(sig)], [ib, fd(ib)]] },
        { k: 'line', i0: ia, p0: a, i1: ib, p1: b, label: 'tiang' },
        { k: 'line', i0: ib, p0: fu(ib), i1: sig, p1: fu(sig), label: d === -1 ? 'bendera' : '' },
        { k: 'line', i0: ib, p0: fd(ib), i1: sig, p1: fd(sig), label: d === 1 ? 'bendera' : '' }];
    }
    if (pat === 'cup_handle' && P.length >= 4) {
      const [i1, p1] = P[1], [it1, t1] = P[2], [i2, p2] = P[3];
      let j = i2 + 1, ext = d === 1 ? Infinity : -Infinity;
      for (let i = i2 + 1; i <= sig; i++) { const v = d === 1 ? K.l[i] : K.h[i]; if (d === 1 ? v < ext : v > ext) { ext = v; j = i; } }
      const cup = arc(i1, p1, it1, t1, i2, p2), handle = arc(i2, p2, j, ext, sig, p2, 8);
      const mid = (a, text) => a.map((_, q) => (q === (a.length >> 1) ? [text, -d] : null));
      return [{ k: 'poly', pts: cup, dots: false, labels: mid(cup, 'cangkir') }, { k: 'poly', pts: handle, dots: false, labels: mid(handle, 'gagang') },
        { k: 'hline', p: p2, i0: i1, i1: sig, label: 'bibir cangkir' }];
    }
    return [{ k: 'poly', pts: P, numbered: pat === 'elliott' }];
  }

  // ---------------------------------------------------------------- keadaan
  const S = { id: null, cell: null, tf: '1d', strategy: null, markets: {}, T: null, K: null, sym: null, trades: [], ov: [], eq: null, bh: null,
    v0: 0, bars: 300, hover: -1, sel: -1, drag: null, token: 0 };
  const canvas = $('fchart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  window.addEventListener('ta-cell', async (ev) => {
    const { id, cell, strategy, markets } = ev.detail;
    Object.assign(S, { id, cell, strategy, markets, tf: cell ? cell.split('_')[1] : '1d', T: null, K: null, trades: [], sel: -1 });
    const token = ++S.token;
    $('fc-sym').innerHTML = '';
    if (!cell) { message('Pilih strategi dan pasar untuk melihat seluruh riwayat ujinya.'); return; }
    message('Memuat semua transaksi…');
    try {
      const T = await loadDat(`data/t/${id}/${cell}.dat`);
      if (token !== S.token) return;
      S.T = T;
      const syms = Object.entries(T.header.syms);
      $('fc-sym').innerHTML = syms.map(([s, v]) => `<option value="${s}">${symName(s)} (${nf(v[3], 0)} posisi)</option>`).join('');
      const keep = S.sym && T.header.syms[S.sym] && T.header.syms[S.sym][1] ? S.sym : (syms.find(([, v]) => v[1] > 0) || syms[0])[0];
      $('fc-sym').value = keep;
      await loadSymbol(keep, token);
    } catch (e) {
      if (token === S.token) message('Data riwayat penuh untuk strategi ini gagal dimuat. Muat ulang halaman.');
    }
  });

  async function loadSymbol(sym, token = ++S.token) {
    S.sym = sym;
    message('Memuat candle…');
    const { header, A } = await loadDat(`data/c/${S.cell}/${fileName(sym)}.dat`);
    if (token !== S.token) return;
    const K = { n: header.n, t: A.t, o: A.o, h: A.h, l: A.l, c: A.c, v: A.v };
    const T = S.T, [start, count, cost, total] = T.header.syms[sym], a = T.A, geo = T.header.geo;
    const trades = [];
    const eq = new Float64Array(K.n);
    for (let q = start; q < start + count; q++) {
      const e = a.e[q], x = a.x[q], kind = a.k[q] & 7, dir = a.k[q] & 8 ? 1 : -1;
      const sl = a.sl ? a.sl[q] : NaN, tp = a.tp ? a.tp[q] : NaN;
      const ep = a.ep && !Number.isNaN(a.ep[q]) ? a.ep[q] : K.o[e];
      const xp = kind === 0 ? K.o[x] : kind === 1 ? K.c[x] : kind === 2 ? sl : kind === 3 ? tp : a.xp[q];
      const net = dir * (xp / ep - 1) - cost;
      const t = { e, x, dir, ep, xp, sl, tp, net, g: null };
      if (geo && geo.w) t.g = a.g.subarray(q * geo.w, (q + 1) * geo.w);
      trades.push(t);
      eq[x] += Math.log1p(Math.max(net, -0.99));
    }
    for (let i = 1; i < K.n; i++) eq[i] += eq[i - 1];
    const bh = new Float64Array(K.n);
    for (let i = 0; i < K.n; i++) bh[i] = Math.log(K.c[i] / K.o[0]) + Math.log1p(-cost);
    Object.assign(S, { K, trades, eq, bh, ov: overlays(T.header.base, K), sel: -1, hover: -1, shown: total === count ? null : [count, total] });
    S.bars = Math.min(K.n, 300);
    S.v0 = K.n - S.bars;
    if (trades.length) { const last = trades[trades.length - 1]; if (last.x < S.v0) S.v0 = Math.max(0, Math.min(K.n - S.bars, last.e - S.bars * 0.6)); }
    summary();
    legend();
    describe(-1);
    draw();
  }

  function message(text) { S.K = null; $('fc-summary').textContent = text; $('fc-trade').textContent = ''; $('fc-legend').innerHTML = ''; draw(); }

  function summary() {
    const { trades, K, eq, bh } = S, n = trades.length;
    const tfTxt = TF[S.tf], range = `${fmtDate(K.t[0], '1d')} sampai ${fmtDate(K.t[K.n - 1], '1d')}`;
    const cut = S.shown ? ` Untuk timeframe ini chart hanya memuat 2 tahun terakhir (${nf(S.shown[0], 0)} dari ${nf(S.shown[1], 0)} posisi); angka hasil uji tetap dari seluruh data.` : '';
    if (!n) { $('fc-summary').textContent = `${symName(S.sym)}, ${tfTxt}, ${range}: strategi ini tidak membuka posisi di instrumen ini.${cut}`; return; }
    const win = trades.filter((t) => t.net > 0).length;
    $('fc-summary').textContent = `${symName(S.sym)}, ${tfTxt}, ${range}: ${nf(n, 0)} posisi, ${nf(100 * win / n, 0)}% untung. ` +
      `Hasil majemuk strategi ${pct(100 * Math.expm1(eq[K.n - 1]))}, beli lalu diamkan ${pct(100 * Math.expm1(bh[K.n - 1]))}, keduanya setelah biaya.${cut}`;
  }

  function legend() {
    const items = [[C.up, 'posisi untung'], [C.down, 'posisi rugi']];
    if (S.T.header.geo) items.push([C.mark, 'pola / zona yang dipakai strategi']);
    S.ov.forEach((o, k) => items.push([OV_COLORS[k % OV_COLORS.length], o.name]));
    items.push([C.ink, 'modal strategi'], ['#9aa3b8', 'modal beli lalu diamkan']);
    $('fc-legend').innerHTML = items.map(([c, n]) => `<span><i style="background:${c}"></i>${n}</span>`).join('');
  }

  // ---------------------------------------------------------------- gambar
  let G = null;
  const firstTrade = (i) => {   // transaksi pertama yang keluarnya >= i
    let lo = 0, hi = S.trades.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (S.trades[m].x < i) lo = m + 1; else hi = m; }
    return lo;
  };
  function tag(text, x, y) {
    ctx.save(); ctx.font = `700 11.5px ${FONT}`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    const w = ctx.measureText(text).width + 10;
    ctx.fillStyle = C.mark; ctx.fillRect(x - w / 2, y - 9, w, 18); ctx.fillStyle = '#fff'; ctx.fillText(text, x, y + 0.5); ctx.restore();
  }
  function text(t, x, y, color, align = 'left', bold = false) {
    ctx.save(); ctx.font = `${bold ? 700 : 600} 12px ${FONT}`; ctx.textAlign = align; ctx.textBaseline = 'middle';
    const w = ctx.measureText(t).width, bx = align === 'left' ? x - 3 : align === 'right' ? x - w - 3 : x - w / 2 - 3;
    ctx.fillStyle = 'rgba(251,252,254,.85)'; ctx.fillRect(bx, y - 9, w + 6, 18); ctx.fillStyle = color; ctx.fillText(t, x, y); ctx.restore();
  }
  function seriesLine(v, i0, i1, step, X, Y, color, width = 1.5) {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
    let pen = false;
    for (let i = i0; i <= i1; i += step) {
      if (Number.isNaN(v[i])) { pen = false; continue; }
      if (pen) ctx.lineTo(X(i), Y(v[i])); else ctx.moveTo(X(i), Y(v[i]));
      pen = true;
    }
    ctx.stroke();
  }
  function drawShapes(shapes, X, Y, labels) {
    ctx.strokeStyle = C.mark; ctx.fillStyle = C.mark; ctx.lineWidth = 1.8;
    for (const s of shapes) {
      if (s.k === 'fill') {
        ctx.save(); ctx.fillStyle = 'rgba(106,69,217,.13)'; ctx.beginPath();
        s.pts.forEach((p, q) => (q ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1]))));
        ctx.closePath(); ctx.fill(); ctx.restore();
      } else if (s.k === 'poly' || s.k === 'dots') {
        if (s.k === 'poly') { ctx.beginPath(); s.pts.forEach((p, q) => (q ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1])))); ctx.stroke(); }
        s.pts.forEach((p, q) => {
          if (s.k === 'dots') { ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 3, 0, 6.3); ctx.fill(); }
          if (!labels) return;
          if (s.numbered) text(String(q), X(p[0]), Y(p[1]) + (q % 2 ? -12 : 13), C.mark, 'center', true);
          const lb = s.labels && s.labels[q];
          if (lb) tag(lb[0], X(p[0]), Y(p[1]) - lb[1] * 16);
        });
      } else if (s.k === 'line' || s.k === 'hline') {
        const p0 = s.k === 'line' ? s.p0 : s.p, p1 = s.k === 'line' ? s.p1 : s.p;
        ctx.save(); if (s.dash || s.k === 'hline') ctx.setLineDash([6, 4]);
        ctx.beginPath(); ctx.moveTo(X(s.i0), Y(p0)); ctx.lineTo(X(s.i1), Y(p1)); ctx.stroke(); ctx.restore();
        if (labels && s.label) text(s.label, X(s.i0), Y(p0) + (p1 > p0 ? 12 : -12), C.mark);
      }
    }
  }

  function draw() {
    const dpr = window.devicePixelRatio || 1, W = canvas.clientWidth, H = canvas.clientHeight;
    if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) { canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr); }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    G = null;
    const K = S.K;
    if (!K) return;
    const subs = S.ov.filter((o) => o.pane === 'sub');
    const L = 8, R = 70, T = 12, B = 24, eqH = Math.round(H * 0.2), subH = subs.length ? Math.round(H * 0.16) : 0, gap = 12;
    const pH = H - T - B - eqH - gap - (subH ? subH + gap : 0), pW = W - L - R;
    S.bars = Math.max(20, Math.min(K.n + 40, S.bars));
    S.v0 = Math.max(-S.bars * 0.5, Math.min(K.n - S.bars * 0.5, S.v0));
    const bw = pW / S.bars, X = (i) => L + (i - S.v0 + 0.5) * bw;
    const i0 = Math.max(0, Math.floor(S.v0)), i1 = Math.min(K.n - 1, Math.ceil(S.v0 + S.bars));
    if (i1 < i0) return;
    let lo = Infinity, hi = -Infinity;
    for (let i = i0; i <= i1; i++) { if (K.l[i] < lo) lo = K.l[i]; if (K.h[i] > hi) hi = K.h[i]; }
    const t0 = firstTrade(i0);
    let t1 = t0;
    while (t1 < S.trades.length && S.trades[t1].e <= i1) t1++;
    const nVis = t1 - t0, detail = bw >= 2.5 && nVis <= 400, span0 = hi - lo || hi * 0.01;
    if (detail) for (let q = t0; q < t1; q++) for (const v of [S.trades[q].sl, S.trades[q].tp]) if (v > lo - span0 && v < hi + span0) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const pad = (hi - lo) * 0.07 || 1; lo -= pad; hi += pad;
    const Y = (p) => T + (hi - p) / (hi - lo) * pH;
    G = { L, bw, pW, T, pH, i0, i1, t0, t1 };

    ctx.font = `12px ${FONT}`; ctx.textBaseline = 'middle';
    // grid harga
    const raw = (hi - lo) / 5, p10 = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p10, step = (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p10;
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1; ctx.fillStyle = C.muted; ctx.textAlign = 'left';
    for (let p = Math.ceil(lo / step) * step; p < hi; p += step) {
      const y = Math.round(Y(p)) + 0.5;
      ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(L + pW, y); ctx.stroke(); ctx.fillText(price(p), L + pW + 6, y);
    }
    // sumbu waktu
    ctx.textAlign = 'center';
    const every = Math.max(1, Math.round(S.bars / Math.max(2, Math.floor(pW / 140))));
    for (let i = Math.ceil(i0 / every) * every; i <= i1; i += every) ctx.fillText(fmtDate(K.t[i], S.bars < 400 ? S.tf : '1d'), X(i), H - B / 2 + 1);

    ctx.save(); ctx.beginPath(); ctx.rect(L, T, pW, pH); ctx.clip();
    // latar: rentang posisi dan zona
    const geo = S.T.header.geo;
    for (let q = t0; q < t1; q++) {
      const t = S.trades[q], xa = X(t.e) - bw / 2, w = Math.max(1, (t.x - t.e + 1) * bw);
      ctx.fillStyle = t.net > 0 ? 'rgba(18,128,108,.09)' : 'rgba(207,64,52,.09)';
      ctx.fillRect(xa, T, w, pH);
      if (detail) { ctx.fillStyle = t.net > 0 ? 'rgba(18,128,108,.2)' : 'rgba(207,64,52,.2)'; ctx.fillRect(xa, Math.min(Y(t.ep), Y(t.xp)), w, Math.max(1, Math.abs(Y(t.ep) - Y(t.xp)))); }
      if (q === S.sel) { ctx.strokeStyle = C.ink; ctx.lineWidth = 1; ctx.strokeRect(xa + 0.5, T + 1.5, w - 1, pH - 3); }
      if (!geo || bw < 1) continue;
      if (geo.kind === 'span') { ctx.fillStyle = 'rgba(106,69,217,.16)'; ctx.fillRect(X(t.e - geo.len) - bw / 2, T, geo.len * bw, pH); }
      if (geo.kind === 'zone' && t.g && !Number.isNaN(t.g[0])) {
        ctx.fillStyle = 'rgba(106,69,217,.16)'; ctx.strokeStyle = C.mark; ctx.lineWidth = 1;
        const zx = X(t.g[0]) - bw / 2, zw = (t.x - t.g[0] + 1) * bw, zy = Y(t.g[2]), zh = Y(t.g[1]) - Y(t.g[2]);
        ctx.fillRect(zx, zy, zw, zh); ctx.strokeRect(zx + 0.5, zy, zw, zh);
      }
    }
    // candle
    if (bw >= 3) {
      const body = Math.max(1, Math.min(bw * 0.7, 13));
      for (let i = i0; i <= i1; i++) {
        ctx.strokeStyle = ctx.fillStyle = K.c[i] >= K.o[i] ? C.up : C.down;
        const x = Math.round(X(i)) + 0.5;
        ctx.beginPath(); ctx.moveTo(x, Y(K.h[i])); ctx.lineTo(x, Y(K.l[i])); ctx.stroke();
        ctx.fillRect(x - body / 2, Y(Math.max(K.o[i], K.c[i])), body, Math.max(1, Math.abs(Y(K.o[i]) - Y(K.c[i]))));
      }
    } else {   // banyak candle per piksel: satu garis tinggi-rendah per kolom piksel
      const per = Math.max(1, Math.ceil(1 / bw));
      for (let i = i0; i <= i1; i += per) {
        let h = -Infinity, l = Infinity;
        const e = Math.min(i1, i + per - 1);
        for (let j = i; j <= e; j++) { if (K.h[j] > h) h = K.h[j]; if (K.l[j] < l) l = K.l[j]; }
        ctx.strokeStyle = K.c[e] >= K.o[i] ? C.up : C.down;
        const x = Math.round(X(i)) + 0.5;
        ctx.beginPath(); ctx.moveTo(x, Y(h)); ctx.lineTo(x, Math.max(Y(l), Y(h) + 1)); ctx.stroke();
      }
    }
    const stepLine = Math.max(1, Math.floor(1 / bw));
    S.ov.forEach((o, k) => { if (o.pane === 'price') seriesLine(o.v, i0, i1, stepLine, X, Y, OV_COLORS[k % OV_COLORS.length]); });
    // pola
    if (geo && bw >= 0.8 && nVis <= 250) {
      for (let q = t0; q < t1; q++) {
        const t = S.trades[q], sig = t.e - 1;
        if (geo.kind === 'pts' && t.g && !Number.isNaN(t.g[0])) {
          const P = [];
          for (let j = 0; j < 6; j++) if (!Number.isNaN(t.g[j]) && t.g[j] >= 0) P.push([t.g[j], t.g[6 + j]]);
          const pat = geo.pats ? geo.pats[t.g[12]] : geo.pat;
          if (P.length >= 2) drawShapes(patternShapes(pat, P, t.dir, sig, K), X, Y, bw >= 2.5 && nVis <= 40);
        } else if (geo.kind === 'mom' && sig - geo.n >= 0) {
          drawShapes([{ k: 'hline', p: K.c[sig - geo.n], i0: sig - geo.n, i1: sig, label: `harga ${geo.n} candle lalu` }], X, Y, bw >= 4 && nVis <= 20);
        }
      }
    }
    // tanda masuk / keluar, SL / TP
    if (detail) {
      for (let q = t0; q < t1; q++) {
        const t = S.trades[q], xe = X(t.e), xx = X(t.x), col = t.dir === 1 ? C.up : C.down, s = Math.max(4, Math.min(8, bw * 0.9));
        ctx.save(); ctx.setLineDash([4, 3]); ctx.lineWidth = 1.2;
        if (!Number.isNaN(t.sl)) { ctx.strokeStyle = C.down; ctx.beginPath(); ctx.moveTo(xe, Y(t.sl)); ctx.lineTo(xx, Y(t.sl)); ctx.stroke(); }
        if (!Number.isNaN(t.tp)) { ctx.strokeStyle = C.up; ctx.beginPath(); ctx.moveTo(xe, Y(t.tp)); ctx.lineTo(xx, Y(t.tp)); ctx.stroke(); }
        ctx.restore();
        ctx.fillStyle = col; ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.2; ctx.beginPath();
        const y = Y(t.ep);
        if (t.dir === 1) { ctx.moveTo(xe, y); ctx.lineTo(xe - s, y + s * 1.5); ctx.lineTo(xe + s, y + s * 1.5); } else { ctx.moveTo(xe, y); ctx.lineTo(xe - s, y - s * 1.5); ctx.lineTo(xe + s, y - s * 1.5); }
        ctx.closePath(); ctx.fill(); ctx.stroke();
        ctx.strokeStyle = C.ink; ctx.lineWidth = 1.8; const yx = Y(t.xp), r = Math.max(3, s * 0.6);
        ctx.beginPath(); ctx.moveTo(xx - r, yx - r); ctx.lineTo(xx + r, yx + r); ctx.moveTo(xx + r, yx - r); ctx.lineTo(xx - r, yx + r); ctx.stroke();
        if (bw >= 5 && nVis <= 30) text(pct(t.net * 100, 2), xx + 8, yx, t.net > 0 ? C.up : C.down, 'left', true);
      }
    }
    if (S.hover >= i0 && S.hover <= i1) {
      const x = Math.round(X(S.hover)) + 0.5;
      ctx.strokeStyle = 'rgba(20,27,51,.35)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, T + pH); ctx.stroke();
    }
    ctx.restore();

    // panel osilator
    let y0 = T + pH + gap;
    if (subs.length) {
      let slo = Infinity, shi = -Infinity;
      for (const o of subs) { for (let i = i0; i <= i1; i++) if (!Number.isNaN(o.v[i])) { slo = Math.min(slo, o.v[i]); shi = Math.max(shi, o.v[i]); } (o.ref || []).forEach((r) => { slo = Math.min(slo, r); shi = Math.max(shi, r); }); }
      const sp = (shi - slo || 1) * 0.08; slo -= sp; shi += sp;
      const YS = (v) => y0 + (shi - v) / (shi - slo) * subH;
      ctx.strokeStyle = C.grid; ctx.strokeRect(L + 0.5, y0 + 0.5, pW, subH);
      ctx.save(); ctx.beginPath(); ctx.rect(L, y0, pW, subH); ctx.clip();
      ctx.setLineDash([3, 4]); ctx.strokeStyle = 'rgba(20,27,51,.35)';
      for (const o of subs) for (const r of o.ref || []) { ctx.beginPath(); ctx.moveTo(L, YS(r)); ctx.lineTo(L + pW, YS(r)); ctx.stroke(); }
      ctx.setLineDash([]);
      S.ov.forEach((o, k) => { if (o.pane === 'sub') seriesLine(o.v, i0, i1, stepLine, X, YS, OV_COLORS[k % OV_COLORS.length]); });
      ctx.restore();
      ctx.fillStyle = C.muted; ctx.textAlign = 'left';
      for (const o of subs) for (const r of o.ref || []) ctx.fillText(nf(r, 0), L + pW + 6, YS(r));
      y0 += subH + gap;
    }
    // panel modal: strategi vs beli lalu diamkan (skala log, dihitung dari awal data)
    let elo = Infinity, ehi = -Infinity;
    for (let i = i0; i <= i1; i++) { elo = Math.min(elo, S.eq[i], S.bh[i]); ehi = Math.max(ehi, S.eq[i], S.bh[i]); }
    const ep = (ehi - elo || 0.02) * 0.12; elo -= ep; ehi += ep;
    const YE = (v) => y0 + (ehi - v) / (ehi - elo) * eqH;
    ctx.strokeStyle = C.grid; ctx.strokeRect(L + 0.5, y0 + 0.5, pW, eqH);
    ctx.save(); ctx.beginPath(); ctx.rect(L, y0, pW, eqH); ctx.clip();
    if (elo < 0 && ehi > 0) { ctx.setLineDash([3, 4]); ctx.strokeStyle = 'rgba(20,27,51,.35)'; ctx.beginPath(); ctx.moveTo(L, YE(0)); ctx.lineTo(L + pW, YE(0)); ctx.stroke(); ctx.setLineDash([]); }
    seriesLine(S.bh, i0, i1, stepLine, X, YE, '#9aa3b8', 1.5);
    seriesLine(S.eq, i0, i1, stepLine, X, YE, C.ink, 2);
    ctx.restore();
    ctx.textAlign = 'left'; ctx.font = `600 11.5px ${FONT}`;
    const ye = YE(S.eq[i1]), yb = YE(S.bh[i1]), sep = Math.abs(ye - yb) < 13 ? (ye <= yb ? -7 : 7) : 0;
    ctx.fillStyle = C.ink; ctx.fillText(pct(100 * Math.expm1(S.eq[i1]), 0), L + pW + 6, ye + sep);
    ctx.fillStyle = '#7a849b'; ctx.fillText(pct(100 * Math.expm1(S.bh[i1]), 0), L + pW + 6, yb - sep);
    ctx.fillStyle = C.muted; ctx.font = `11.5px ${FONT}`; ctx.fillText('Modal sejak awal data', L + 6, y0 + 10);
  }

  // ---------------------------------------------------------------- interaksi
  const idxAt = (clientX) => (G ? Math.floor(S.v0 + (clientX - canvas.getBoundingClientRect().left - G.L) / G.bw) : -1);
  const tradeAt = (i) => { const q = firstTrade(i); return q < S.trades.length && S.trades[q].e <= i ? q : -1; };
  function zoom(factor, center) {
    if (!S.K) return;
    const c = center === undefined ? S.v0 + S.bars / 2 : center, nb = Math.max(20, Math.min(S.K.n + 40, S.bars * factor));
    S.v0 = c - (c - S.v0) * nb / S.bars; S.bars = nb; draw();
  }
  function describe(q) {
    const el = $('fc-trade');
    if (q < 0 || !S.K) { el.className = 'result'; el.textContent = S.trades.length ? 'Geser chart ke kiri atau kanan, perbesar dengan roda mouse atau tombol, dan pilih sebuah posisi untuk melihat rinciannya.' : ''; return; }
    const t = S.trades[q], K = S.K;
    el.className = 'result ' + (t.net > 0 ? 'win' : 'loss');
    el.textContent = `Posisi ${q + 1} dari ${nf(S.trades.length, 0)}: ${t.dir === 1 ? 'long' : 'short'} di ${price(t.ep)} (${fmtDate(K.t[t.e], S.tf)}), ` +
      `keluar di ${price(t.xp)} (${fmtDate(K.t[t.x], S.tf)}) setelah ${nf(t.x - t.e, 0)} candle: ${pct(t.net * 100, 2)} setelah biaya.`;
  }
  function goTrade(q) {
    if (!S.trades.length) return;
    q = Math.max(0, Math.min(S.trades.length - 1, q));
    const t = S.trades[q];
    S.sel = q;
    S.bars = Math.max(120, Math.min(S.K.n, (t.x - t.e) * 3 + 120));
    S.v0 = t.e - S.bars * 0.65;   // sisakan ruang di kiri untuk pola yang mendahului posisi
    describe(q); draw();
  }
  canvas.addEventListener('pointerdown', (ev) => { if (!S.K) return; canvas.setPointerCapture(ev.pointerId); S.drag = { x: ev.clientX, v0: S.v0, moved: false }; });
  canvas.addEventListener('pointermove', (ev) => {
    if (!S.K || !G) return;
    if (S.drag) {
      const dx = ev.clientX - S.drag.x;
      if (Math.abs(dx) > 3) S.drag.moved = true;
      S.v0 = S.drag.v0 - dx / G.bw; S.hover = -1; draw(); return;
    }
    const i = idxAt(ev.clientX), ok = i >= 0 && i < S.K.n;
    S.hover = ok ? i : -1;
    const K = S.K, q = ok ? tradeAt(i) : -1;
    $('fc-readout').textContent = ok ? `${fmtDate(K.t[i], S.tf)}   O ${price(K.o[i])}  H ${price(K.h[i])}  L ${price(K.l[i])}  C ${price(K.c[i])}` +
      (q >= 0 ? `   posisi ${S.trades[q].dir === 1 ? 'long' : 'short'} ${pct(S.trades[q].net * 100, 2)}` : '') : '';
    draw();
  });
  const endDrag = (ev) => {
    if (!S.drag) return;
    const moved = S.drag.moved; S.drag = null;
    if (!moved) { const q = tradeAt(idxAt(ev.clientX)); S.sel = q; describe(q); draw(); }
  };
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', () => { S.drag = null; });
  canvas.addEventListener('pointerleave', () => { if (!S.drag) { S.hover = -1; $('fc-readout').textContent = ''; draw(); } });
  canvas.addEventListener('wheel', (ev) => {
    if (!S.K || !G) return;
    ev.preventDefault();
    if (Math.abs(ev.deltaX) > Math.abs(ev.deltaY)) { S.v0 += ev.deltaX / G.bw; draw(); } else zoom(ev.deltaY > 0 ? 1.2 : 1 / 1.2, S.v0 + (ev.clientX - canvas.getBoundingClientRect().left - G.L) / G.bw);
  }, { passive: false });
  canvas.addEventListener('keydown', (ev) => {
    if (!S.K) return;
    const k = ev.key;
    if (k === 'ArrowLeft') S.v0 -= S.bars * 0.15; else if (k === 'ArrowRight') S.v0 += S.bars * 0.15;
    else if (k === '+' || k === '=') return zoom(1 / 1.4); else if (k === '-') return zoom(1.4); else return;
    ev.preventDefault(); draw();
  });
  $('fc-sym').addEventListener('change', (ev) => loadSymbol(ev.target.value).catch(() => message('Data candle gagal dimuat. Muat ulang halaman.')));
  $('fc-in').addEventListener('click', () => zoom(1 / 1.6));
  $('fc-out').addEventListener('click', () => zoom(1.6));
  $('fc-all').addEventListener('click', () => { if (S.K) { S.bars = S.K.n + 10; S.v0 = -5; draw(); } });
  $('fc-prev').addEventListener('click', () => goTrade(S.sel < 0 ? (G ? Math.max(0, G.t0 - 1) : 0) : S.sel - 1));
  $('fc-next').addEventListener('click', () => goTrade(S.sel < 0 ? (G ? G.t0 : 0) : S.sel + 1));
  window.addEventListener('resize', draw);
})();
