/* PrivateLearn demo player — replays a trace.json. Vanilla JS, no framework.
   It never computes results: every number on screen comes from the trace.
   Serve from privatelearn/ with `python -m http.server 8000` and open /web/index.html.
   Loads ../trace.json (T9 output) and falls back to ../data/sample_trace.json;
   force a file with index.html?trace=<path>.   */

const TRACE_PARAM = new URLSearchParams(location.search).get('trace');
const TRACE_CANDIDATES = TRACE_PARAM ? [TRACE_PARAM] : ['../trace.json', '../data/sample_trace.json'];

const $ = (id) => document.getElementById(id);
const graph = $('graph');
const chart = $('chart');
const lossChart = $('lossChart');
const logEl = $('log');
const attackEl = $('attack');
const playBtn = $('play');
const scrub = $('scrub');
const roundLabel = $('roundLabel');
const showMask = $('showMask');
const showDp = $('showDp');

const HOSPITALS = ['h0', 'h1', 'h2'];
const DEFAULT_COLORS = { h0: '#4c6ef5', h1: '#40c057', h2: '#fa5252', coord: '#fab005' };
const C = { line: '#2a2e3a', muted: '#8b93a7', text: '#e7e9f0', bad: '#fa5252', good: '#4ade80', accent: '#fab005', cen: '#4c6ef5' };

let trace = null;
let round = 0; // committed state: number of completed rounds shown (0 = before round 1)
let timer = null;
let phase = 0; // 0..1 animation of round+1 while playing: 0–0.5 updates fly in, 0.5–1 broadcast out
let attackProtected = false;

// ---------- canvas helpers (crisp on HiDPI; logical size = width/height attributes) ----------

function setupCanvas(canvas) {
  const w = +canvas.getAttribute('width');
  const h = +canvas.getAttribute('height');
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

const G = setupCanvas(graph);
const A = setupCanvas(chart);
const L = setupCanvas(lossChart);

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

const fmt = (v, d = 3) => (typeof v === 'number' ? v.toFixed(d) : '—');

// ---------- graph ----------

function nodePos(id) {
  const i = HOSPITALS.indexOf(id);
  if (i >= 0) return { x: 160 + i * 320, y: 72 };
  return { x: 480, y: 222 };
}

function node(id) {
  return trace.nodes.find((n) => n.id === id) || { id, label: id };
}

function drawNode(id, { active = false, dim = false, sub = '', sub2 = '' } = {}) {
  const ctx = G.ctx;
  const p = nodePos(id);
  const n = node(id);
  ctx.globalAlpha = dim ? 0.35 : 1;
  ctx.beginPath();
  ctx.arc(p.x, p.y, active ? 31 : 27, 0, Math.PI * 2);
  ctx.fillStyle = n.color || DEFAULT_COLORS[id];
  ctx.fill();
  if (active) {
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#fff';
    ctx.stroke();
  }
  ctx.fillStyle = '#fff';
  ctx.font = '600 17px ui-sans-serif, system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(n.label, p.x, id === 'coord' ? p.y + 54 : p.y - 40);
  ctx.globalAlpha = 1;
  ctx.font = '15px ui-sans-serif, system-ui, sans-serif';
  const below = id === 'coord' ? p.y + 76 : p.y + 52;
  if (sub) { ctx.fillStyle = C.muted; ctx.fillText(sub, p.x, below); }
  if (sub2) { ctx.fillStyle = C.accent; ctx.fillText(sub2, p.x, below + 20); }
  ctx.textAlign = 'left';
}

function drawLock(x, y) {
  const ctx = G.ctx;
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(x, y - 3, 3, Math.PI, 0);
  ctx.stroke();
  ctx.fillStyle = '#fff';
  ctx.fillRect(x - 4.5, y - 3, 9, 7);
}

function drawPacket(from, to, t, { color = '#fff', locked = false, rejected = false } = {}) {
  const ctx = G.ctx;
  const a = nodePos(from);
  const b = nodePos(to);
  const tt = rejected ? Math.min(t, 0.8) : t; // a rejected update is stopped before it is aggregated
  const x = a.x + (b.x - a.x) * tt;
  const y = a.y + (b.y - a.y) * tt;
  ctx.beginPath();
  ctx.arc(x, y, locked ? 9 : 6, 0, Math.PI * 2);
  ctx.fillStyle = rejected && t > 0.8 ? C.bad : color;
  ctx.fill();
  if (locked) drawLock(x, y + 1);
  if (rejected && t > 0.8) {
    ctx.strokeStyle = C.bad;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x + 10, y - 10); ctx.lineTo(x + 20, y);
    ctx.moveTo(x + 20, y - 10); ctx.lineTo(x + 10, y);
    ctx.stroke();
  }
}

function updateFor(r, h) {
  return r ? r.updates.find((u) => u.hospital === h) : null;
}

function renderGraph() {
  const ctx = G.ctx;
  ctx.clearRect(0, 0, G.w, G.h);
  const playing = timer !== null && phase > 0;
  const shown = playing ? trace.rounds[round] : trace.rounds[round - 1]; // round being animated or last completed
  const coord = nodePos('coord');

  ctx.strokeStyle = C.line;
  ctx.lineWidth = 1;
  HOSPITALS.forEach((h) => {
    const p = nodePos(h);
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
    ctx.lineTo(coord.x, coord.y);
    ctx.stroke();
  });

  HOSPITALS.forEach((h) => {
    const u = updateFor(shown, h);
    const offline = u && !u.sent;
    const rejected = u && u.rejected;
    let sub = `${u ? u.n : trace.meta.rows_per_hospital} rows · never leave`;
    if (offline) sub = 'offline this round';
    else if (rejected) sub = `rejected: ${u.reason || 'outlier'}`;
    const sub2 = showDp.checked && u && u.sent ? `ε ${fmt(u.dp_eps, 1)} this round` : '';
    drawNode(h, { active: !!(u && u.sent && !rejected), dim: offline, sub, sub2 });
  });

  const eps = shown ? shown.eps_total : 0;
  drawNode('coord', {
    sub: showMask.checked ? 'sees only the masked sum Σ' : 'aggregates updates',
    sub2: showDp.checked ? `ε spent: ${fmt(eps, 1)}` : '',
  });

  if (playing) {
    const r = trace.rounds[round];
    if (phase < 0.5) {
      r.updates.forEach((u) => {
        if (!u.sent) return;
        drawPacket(u.hospital, 'coord', phase * 2, {
          color: node(u.hospital).color || DEFAULT_COLORS[u.hospital],
          locked: showMask.checked && u.masked,
          rejected: !!u.rejected,
        });
      });
    } else {
      HOSPITALS.forEach((h) => {
        // everyone online gets the new model, including a hospital whose update was rejected
        if (updateFor(r, h)?.sent !== false) drawPacket('coord', h, (phase - 0.5) * 2, { color: C.accent });
      });
    }
  }

  ctx.fillStyle = C.muted;
  ctx.font = '14px ui-sans-serif, system-ui, sans-serif';
  ctx.fillText('only 7 weight deltas + a loss cross each line — never a patient row', 16, G.h - 14);
}

// ---------- charts ----------

function drawSeries(S, series, upto, { refs = [], color = C.accent, pad = 26, decimals = 2 } = {}) {
  const { ctx, w, h } = S;
  ctx.clearRect(0, 0, w, h);
  const vals = series.concat(refs.map((r) => r.v)).filter((v) => typeof v === 'number');
  let min = Math.min(...vals);
  let max = Math.max(...vals);
  const span = Math.max(max - min, 0.02);
  min -= span * 0.12;
  max += span * 0.12;
  const n = series.length;
  const X = (i) => pad + (n > 1 ? i / (n - 1) : 0) * (w - pad - 10);
  const Y = (v) => h - 18 - ((v - min) / (max - min)) * (h - 28);

  ctx.font = '10px ui-monospace, monospace';
  ctx.fillStyle = C.muted;
  ctx.strokeStyle = C.line;
  ctx.lineWidth = 1;
  for (let i = 0; i < n; i++) {
    ctx.textAlign = 'center';
    ctx.fillText(`r${i}`, X(i), h - 4);
  }
  ctx.textAlign = 'right';
  ctx.fillText(max.toFixed(decimals), pad - 4, 12);
  ctx.fillText(min.toFixed(decimals), pad - 4, h - 18);
  ctx.textAlign = 'left';

  refs.forEach((r) => {
    if (typeof r.v !== 'number') return;
    ctx.strokeStyle = r.color;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(pad, Y(r.v));
    ctx.lineTo(w - 10, Y(r.v));
    ctx.stroke();
    ctx.setLineDash([]);
  });

  const last = Math.min(upto, n - 1);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i <= last; i++) {
    i === 0 ? ctx.moveTo(X(i), Y(series[i])) : ctx.lineTo(X(i), Y(series[i]));
  }
  ctx.stroke();
  for (let i = 0; i <= last; i++) {
    ctx.beginPath();
    ctx.arc(X(i), Y(series[i]), i === last ? 4 : 2.5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }
}

function renderCharts() {
  const m = trace.metrics;
  drawSeries(A, m.auc_series, round, {
    refs: [
      { v: trace.meta.auc_solo, color: C.muted },
      { v: trace.meta.auc_centralized, color: C.cen },
    ],
  });
  $('aucNow').textContent = fmt(m.auc_series[Math.min(round, m.auc_series.length - 1)]);
  if (Array.isArray(m.loss_series) && m.loss_series.length) {
    drawSeries(L, m.loss_series, round, { color: '#40c057' });
    $('lossNow').textContent = fmt(m.loss_series[Math.min(round, m.loss_series.length - 1)]);
  }
}

// ---------- decision log ----------

function renderLog() {
  logEl.replaceChildren();
  if (round === 0) {
    logEl.appendChild(el('li', 'empty', 'press Play — each round the Coordinator picks a strategy'));
    return;
  }
  trace.rounds.slice(0, round).forEach((r, i) => {
    const li = el('li', i === round - 1 ? 'current' : '');
    const head = el('div');
    head.append(el('b', '', `r${r.round} `), el('span', 'badge', r.strategy), document.createTextNode(r.decision || ''));
    li.appendChild(head);
    const meta = el('span', 'meta',
      `AUC ${fmt(r.auc_before)} → ${fmt(r.auc_after)} · ε total ${fmt(r.eps_total, 1)}`);
    li.appendChild(meta);
    const out = (r.updates || []).filter((u) => !u.sent || u.rejected);
    if (out.length) {
      const warn = el('span', 'meta warn',
        out.map((u) => `${node(u.hospital).label}: ${u.sent ? 'rejected' : 'offline'}${u.reason ? ` (${u.reason})` : ''}`).join(' · '));
      li.appendChild(warn);
    }
    logEl.appendChild(li);
  });
}

// ---------- attack ----------

function renderAttack() {
  const a = trace.attack;
  const s = attackProtected ? a.with_privacy : a.without_privacy;
  $('atkOff').classList.toggle('on', !attackProtected);
  $('atkOn').classList.toggle('on', attackProtected);
  attackEl.replaceChildren();
  attackEl.appendChild(el('div', `stamp ${s.recovered ? 'recovered' : 'fail'}`, s.recovered ? 'ROW RECOVERED' : 'ATTACK FAILS'));
  const line = el('div');
  line.append(document.createTextNode(`target ${node(a.target).label} · feature error `), el('b', '', fmt(s.feature_error, 2)));
  attackEl.appendChild(line);
  // bar on a shared scale so the two scenarios are comparable
  const top = Math.max(a.without_privacy.feature_error || 0, a.with_privacy.feature_error || 0, 1e-9);
  const bar = el('div', 'bar');
  const fill = el('i');
  fill.style.width = `${Math.max(2, (100 * (s.feature_error || 0)) / top)}%`;
  fill.style.background = s.recovered ? C.bad : C.good;
  bar.appendChild(fill);
  attackEl.appendChild(bar);
  attackEl.appendChild(el('div', 'muted', s.recovered
    ? 'from one unmasked update the attacker reconstructs a real-looking patient row'
    : 'masked + noised updates: the reconstruction is noise'));
}

// ---------- playback ----------

function render() {
  roundLabel.textContent = `round ${round} / ${trace.rounds.length}`;
  scrub.value = round;
  renderGraph();
  renderCharts();
  renderLog();
  renderAttack();
}

function stop() {
  if (timer) clearInterval(timer);
  timer = null;
  phase = 0;
  playBtn.textContent = 'Play';
}

function seek(r) {
  stop();
  round = Math.max(0, Math.min(trace.rounds.length, r));
  render();
}

function play() {
  if (timer) { stop(); render(); return; }
  if (round >= trace.rounds.length) round = 0;
  playBtn.textContent = 'Pause';
  phase = 0;
  const msPerRound = Math.max(1200, Math.min(6000, (trace.timeline_ms || 16000) / trace.rounds.length / 2));
  const tick = 40;
  timer = setInterval(() => {
    phase += tick / msPerRound;
    if (phase >= 1) {
      phase = 0;
      round += 1; // the animated round is now complete
      if (round >= trace.rounds.length) { stop(); }
    }
    render();
  }, tick);
}

$('reset').onclick = () => seek(0);
playBtn.onclick = play;
scrub.oninput = () => seek(+scrub.value);
showMask.onchange = render;
showDp.onchange = render;
$('atkOff').onclick = () => { attackProtected = false; renderAttack(); };
$('atkOn').onclick = () => { attackProtected = true; renderAttack(); };
document.addEventListener('keydown', (e) => {
  if (!trace || (e.target.tagName === 'INPUT' && e.target.type !== 'checkbox')) return;
  if (e.key === ' ') { e.preventDefault(); play(); }
  if (e.key === 'ArrowRight') seek(round + 1);
  if (e.key === 'ArrowLeft') seek(round - 1);
});

async function loadTrace() {
  const errors = [];
  for (const url of TRACE_CANDIDATES) {
    try {
      const res = await fetch(url, { cache: 'no-store' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return { url, data: await res.json() };
    } catch (e) {
      errors.push(`${url}: ${e.message}`);
    }
  }
  throw new Error(errors.join('\n'));
}

loadTrace()
  .then(({ url, data }) => {
    trace = data;
    scrub.max = trace.rounds.length;
    $('bytes').textContent = '0';
    $('source').textContent = `replaying ${url.replace(/^\.\.\//, '')}`;
    render();
  })
  .catch((e) => {
    const pre = el('pre', '', `failed to load a trace\n${e.message}\n\nserve privatelearn/ with: python -m http.server 8000\nthen open http://localhost:8000/web/index.html  (or ?trace=<path>)`);
    pre.style.cssText = 'color:#fa5252;padding:2em;white-space:pre-wrap';
    document.body.replaceChildren(pre);
  });
