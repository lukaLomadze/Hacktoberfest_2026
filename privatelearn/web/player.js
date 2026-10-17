/* PrivateLearn demo player — replays a trace.json per docs/DESIGN.md. Vanilla JS, no framework.
   It never computes results: every number on screen comes from the trace.
   Colors come from the CSS tokens in player.css (read at load), never raw hex.
   Serve privatelearn/ with `python -m http.server 8000`, open /web/index.html.
   Loads ../trace.json (T9 output), falls back to ../data/sample_trace.json; ?trace=<path> overrides. */

const TRACE_PARAM = new URLSearchParams(location.search).get('trace');
const TRACE_CANDIDATES = TRACE_PARAM ? [TRACE_PARAM] : ['../trace.json', '../data/sample_trace.json'];
const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const $ = (id) => document.getElementById(id);
const graph = $('graph');
const playBtn = $('play');
const scrub = $('scrub');
const showMask = $('showMask');
const showDp = $('showDp');

const css = getComputedStyle(document.documentElement);
const T = (name) => css.getPropertyValue(`--${name}`).trim();
const COL = {
  bgRaise: T('bg-raise'), panel: T('panel'), panel2: T('panel-2'), border: T('border'),
  text: T('text'), dim: T('text-dim'), faint: T('text-faint'),
  h0: T('hosp-a'), h1: T('hosp-b'), h2: T('hosp-c'), coord: T('coord'), ok: T('ok'), danger: T('danger'),
};
const FONT_DATA = T('font-data');
const FONT_UI = T('font-ui');

const HOSPITALS = ['h0', 'h1', 'h2'];
const AGENTS = [['schema', 'S', 'Schema'], ['planner', 'P', 'Planner'], ['trainer', 'T', 'Trainer'],
  ['challenger', 'C', 'Challenger']];
const HOP_MS = 600; // DESIGN §8: ~600 ms per hop

let trace = null;
let round = 0; // completed rounds shown (0 = before round 1)
let timer = null;
let phase = 0; // 0..1 while animating round+1: 0–0.5 updates in, 0.5–1 broadcast out

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
const A = setupCanvas($('chart'));
const L = setupCanvas($('lossChart'));

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

const fmt = (v, d = 3) => (typeof v === 'number' && isFinite(v) ? v.toFixed(d) : '—');
const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2); // ease-in-out

// ---------- graph ----------

function pos(id) {
  const i = HOSPITALS.indexOf(id);
  if (i >= 0) return { x: 160 + i * 320, y: 78 };
  return { x: 480, y: 232 };
}

function node(id) {
  return trace.nodes.find((n) => n.id === id) || { id, label: id };
}

function roundedRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawGrid() {
  const ctx = G.ctx;
  ctx.fillStyle = COL.bgRaise;
  for (let x = 16; x < G.w; x += 32) {
    for (let y = 16; y < G.h; y += 32) ctx.fillRect(x - 1, y - 1, 2, 2);
  }
}

function text(str, x, y, { size = 13, color = COL.dim, font = FONT_UI, weight = '', align = 'center' } = {}) {
  const ctx = G.ctx;
  ctx.font = `${weight} ${size}px ${font}`.trim();
  ctx.fillStyle = color;
  ctx.textAlign = align;
  ctx.fillText(str, x, y);
  ctx.textAlign = 'left';
}

function drawLock(x, y, color) {
  const ctx = G.ctx;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(x, y - 3, 3, Math.PI, 0);
  ctx.stroke();
  ctx.fillStyle = color;
  ctx.fillRect(x - 4.5, y - 3, 9, 7);
}

function drawHospital(id, u) {
  const ctx = G.ctx;
  const p = pos(id);
  const offline = u && !u.sent;
  const rejected = u && u.rejected;
  const active = u && u.sent && !rejected;
  ctx.globalAlpha = offline ? 0.3 : 1;
  ctx.beginPath();
  ctx.arc(p.x, p.y, 26, 0, Math.PI * 2);
  ctx.fillStyle = COL[id];
  ctx.fill();
  if (active) {
    ctx.lineWidth = 2;
    ctx.strokeStyle = COL.text;
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  text(node(id).label, p.x, p.y - 38, { size: 16, color: COL.text, weight: 600 });
  let sub = `${u ? u.n : trace.meta.rows_per_hospital} rows · never leave`;
  let subColor = COL.faint;
  if (offline) { sub = 'offline this round'; subColor = COL.danger; }
  else if (rejected) { sub = 'excluded this round'; subColor = COL.danger; } // why: ticker + decision log
  text(sub, p.x, p.y + 46, { size: 13, color: subColor });
  // the hospital's real agent team — visible, not just a count
  text('Schema · Planner · Trainer · Challenger', p.x, p.y + 64, { size: 10, color: COL.faint });
  if (showDp.checked && u && u.sent && !rejected) {
    text(`ε ${fmt(u.dp_eps, 1)} this round`, p.x, p.y + 80, { size: 12, color: COL.dim, font: FONT_DATA });
  }
  drawAgentChips(id, p);
}

// The four agents inside a hospital: a chip lights up when that agent acted in the shown round
// (round 0 = schema negotiation). Read from trace.agents; hidden for traces without it.
function agentActivity(id, r) {
  return ((trace.agents || {})[id] || []).filter((a) => a.round === r);
}

function drawAgentChips(id, p) {
  if (!trace.agents) return;
  const ctx = G.ctx;
  const acted = new Set(agentActivity(id, shownRound()).map((a) => a.agent));
  const w = 20, gap = 4, x0 = p.x - (AGENTS.length * w + (AGENTS.length - 1) * gap) / 2, y = p.y + 76;
  AGENTS.forEach(([key, letter], i) => {
    const x = x0 + i * (w + gap);
    roundedRect(ctx, x, y, w, 16, 4);
    if (acted.has(key)) {
      ctx.fillStyle = COL[id];
      ctx.fill();
    } else {
      ctx.strokeStyle = COL.border;
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    text(letter, x + w / 2, y + 12, { size: 11, weight: 700, font: FONT_DATA, color: acted.has(key) ? COL.panel : COL.faint });
  });
}

// the round whose state the graph shows: the one animating, else the last completed (0 = setup)
function shownRound() {
  return timer !== null && phase > 0 ? round + 1 : round;
}

function drawCoordinator(shown) {
  const ctx = G.ctx;
  const p = pos('coord');
  roundedRect(ctx, p.x - 28, p.y - 28, 56, 56, 8);
  ctx.fillStyle = COL.coord;
  ctx.fill();
  text('Coordinator', p.x, p.y + 50, { size: 16, color: COL.text, weight: 600 });
  text(showMask.checked ? 'sees only the masked sum Σ' : 'aggregates the updates', p.x, p.y + 70, { size: 13 });
  if (showDp.checked) {
    text(`ε spent ${fmt(shown ? shown.eps_total : 0, 1)}`, p.x, p.y + 88, { size: 12, color: COL.dim, font: FONT_DATA });
  }
}

// DESIGN §7.1: a packet is 7 numbers, drawn as a small stack of bars.
function drawPacket(from, to, t, { color, locked = false, rejected = false } = {}) {
  const ctx = G.ctx;
  const a = pos(from);
  const b = pos(to);
  const stop = rejected ? 0.78 : 1; // a rejected update never reaches aggregation
  const k = ease(Math.min(t, 1)) * stop;
  const x = a.x + (b.x - a.x) * k;
  const y = a.y + (b.y - a.y) * k;
  const halted = rejected && t >= 1;
  ctx.fillStyle = halted ? COL.danger : color;
  [10, 14, 8].forEach((h, i) => ctx.fillRect(x - 7 + i * 5, y + 7 - h, 4, h));
  if (locked) drawLock(x + 14, y + 2, COL.text);
  if (halted) {
    ctx.strokeStyle = COL.danger;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x - 10, y - 14); ctx.lineTo(x + 10, y + 6);
    ctx.moveTo(x + 10, y - 14); ctx.lineTo(x - 10, y + 6);
    ctx.stroke();
  }
}

const updateFor = (r, h) => (r ? r.updates.find((u) => u.hospital === h) : null);

function renderGraph() {
  const ctx = G.ctx;
  ctx.clearRect(0, 0, G.w, G.h);
  drawGrid();
  const playing = timer !== null && phase > 0;
  const shown = playing ? trace.rounds[round] : trace.rounds[round - 1];

  const c = pos('coord');
  ctx.strokeStyle = COL.border;
  ctx.lineWidth = 1;
  HOSPITALS.forEach((h) => {
    const p = pos(h);
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
    ctx.lineTo(c.x, c.y);
    ctx.stroke();
  });

  HOSPITALS.forEach((h) => drawHospital(h, updateFor(shown, h)));
  drawCoordinator(shown);

  if (playing && !REDUCED) {
    const r = trace.rounds[round];
    if (phase < 0.5) {
      r.updates.forEach((u) => {
        if (!u.sent) return;
        drawPacket(u.hospital, 'coord', phase * 2, {
          color: COL[u.hospital], locked: showMask.checked && u.masked, rejected: !!u.rejected,
        });
      });
    } else {
      HOSPITALS.forEach((h) => {
        // every hospital that is online gets the new model, including one whose update was excluded
        if (updateFor(r, h)?.sent !== false) drawPacket('coord', h, (phase - 0.5) * 2, { color: COL.coord });
      });
    }
  }
}

function renderLegend() {
  const lg = $('legend');
  lg.replaceChildren();
  HOSPITALS.concat('coord').forEach((id) => {
    const s = el('span');
    const sw = el('i', id === 'coord' ? 'sq' : '');
    sw.style.background = COL[id];
    s.append(sw, document.createTextNode(node(id).label));
    lg.appendChild(s);
  });
}

// ---------- charts ----------

function drawSeries(S, series, upto, { refs = [], color, pad = 30, decimals = 1, fixedMax, step } = {}) {
  const { ctx, w, h } = S;
  ctx.clearRect(0, 0, w, h);
  const vals = series.concat(refs.map((r) => r.v)).filter((v) => typeof v === 'number');
  let min, max;
  if (step) {
    // DESIGN §7.2: fit the axis to the data (labels show the true extent), so the
    // federated climb is visible instead of a flat smear in the top of the panel.
    const lo = Math.min(...vals), hi = Math.max(...vals);
    min = Math.floor((lo - 0.01) / step) * step;
    max = Math.ceil((hi + 0.01) / step) * step;
    if (max - min < step * 3) max = min + step * 3;
  } else {
    const lo = Math.min(...vals), hi = Math.max(...vals), span = Math.max(hi - lo, 1e-3);
    min = lo - span * 0.15; max = hi + span * 0.15;
  }
  const n = series.length;
  const top = 6, bottom = step ? 18 : 6;
  const ticks = [];
  if (step) for (let v = min; v <= max + 1e-9; v += step) ticks.push(v);

  ctx.font = `12px ${FONT_DATA}`;
  // the y labels are right-aligned at pad - 4: make room for the widest one (#31)
  const widest = Math.max(0, ...ticks.map((v) => ctx.measureText(v.toFixed(decimals)).width));
  pad = Math.max(pad, Math.ceil(widest) + 8);
  const X = (i) => pad + (n > 1 ? i / (n - 1) : 0) * (w - pad - 8);
  const Y = (v) => h - bottom - ((v - min) / (max - min)) * (h - top - bottom);

  if (step) {
    ctx.strokeStyle = COL.border;
    ctx.lineWidth = 1;
    ctx.fillStyle = COL.faint;
    ctx.textAlign = 'right';
    ticks.forEach((v) => {
      ctx.beginPath(); ctx.moveTo(pad, Y(v)); ctx.lineTo(w - 8, Y(v)); ctx.stroke();
      ctx.fillText(v.toFixed(decimals), pad - 4, Y(v) + 4);
    });
    ctx.textAlign = 'center';
    for (let i = 0; i < n; i++) ctx.fillText(`r${i}`, X(i), h - 4);
    ctx.textAlign = 'left';
  }

  refs.forEach((r) => {
    if (typeof r.v !== 'number') return;
    ctx.strokeStyle = r.color;
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, Y(r.v)); ctx.lineTo(w - 8, Y(r.v)); ctx.stroke();
    ctx.setLineDash([]);
  });

  const last = Math.min(upto, n - 1);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i <= last; i++) (i === 0 ? ctx.moveTo : ctx.lineTo).call(ctx, X(i), Y(series[i]));
  ctx.stroke();
  ctx.fillStyle = color;
  for (let i = 0; i <= last; i++) {
    ctx.beginPath();
    ctx.arc(X(i), Y(series[i]), i === last ? 4 : 2.5, 0, Math.PI * 2);
    ctx.fill();
  }
}

function renderCharts() {
  const m = trace.metrics;
  const meta = trace.meta;
  // The red line is the hospital that is worst off alone — the one federation is for.
  // The best single hospital is still stated, so nothing is hidden.
  let solo = meta.auc_solo;
  let soloLabel = 'solo';
  const byH = meta.auc_solo_by_hospital;
  if (byH && Object.keys(byH).length) {
    const [hid, v] = Object.entries(byH).sort((a, b) => a[1] - b[1])[0];
    solo = v;
    soloLabel = `${node(hid).label} alone`;
    $('soloNote').textContent = `best single hospital alone: ${fmt(Math.max(...Object.values(byH)))}`;
  }
  drawSeries(A, m.auc_series, round, {
    color: COL.coord, step: 0.05, decimals: 2,
    refs: [{ v: solo, color: COL.danger }, { v: meta.auc_centralized, color: COL.ok }],
  });
  $('aucNow').textContent = fmt(m.auc_series[Math.min(round, m.auc_series.length - 1)]);
  $('soloLabel').textContent = soloLabel;
  $('aucSolo').textContent = fmt(solo);
  $('aucFed').textContent = fmt(meta.auc_federated);
  $('aucCen').textContent = fmt(meta.auc_centralized);
  if (Array.isArray(m.loss_series) && m.loss_series.length) {
    drawSeries(L, m.loss_series, round, { color: COL.dim, pad: 8 });
    $('lossNow').textContent = fmt(m.loss_series[Math.min(round, m.loss_series.length - 1)]);
  }
}

// ---------- decision log ----------

function renderLog() {
  const log = $('log');
  log.replaceChildren();
  if (round === 0) {
    log.appendChild(el('li', 'empty', 'Press Play — each round the Coordinator picks a strategy.'));
    return;
  }
  trace.rounds.slice(0, round).forEach((r, i) => {
    const excluded = r.excluded || [];
    const li = el('li', [excluded.length ? 'excluded' : '', i < round - 1 ? 'past' : ''].join(' ').trim());
    const row = el('div', 'row');
    row.append(el('span', `badge ${r.strategy}`, r.strategy), el('span', 'num', `round ${r.round}`),
               el('span', 'tag coord', 'coordinator'));
    excluded.forEach((h) => row.appendChild(el('span', 'tag', `excluded ${node(h).label}`)));
    li.append(row, el('div', 'text', r.decision || ''));
    // the hospital agents that actually acted this round, from trace.agents
    const acts = HOSPITALS.map((h) => {
      const names = (trace.agents?.[h] || []).filter((a) => a.round === r.round).map((a) => a.agent);
      return names.length ? `${node(h).label}: ${names.join(' → ')}` : null;
    }).filter(Boolean);
    if (acts.length) li.appendChild(el('div', 'agents', acts.join(' · ')));
    li.appendChild(el('div', 'meta', `AUC ${fmt(r.auc_before)} → ${fmt(r.auc_after)} · ε ${fmt(r.eps_total, 1)}`));
    log.appendChild(li);
  });
}

// ---------- attack (DESIGN §7.4: two cards, before / after) ----------

function attackCard(label, s) {
  const card = el('div', 'attack-card');
  card.appendChild(el('div', 'label', label));
  card.appendChild(el('span', `stamp ${s.recovered ? 'recovered' : 'failed'}`, s.recovered ? 'RECOVERED' : 'FAILED'));
  const err = el('div', 'err', 'feature error ');
  err.appendChild(el('b', '', fmt(s.feature_error, 2)));
  card.appendChild(err);
  return card;
}

function renderAttack() {
  const a = trace.attack;
  $('attack').replaceChildren(
    attackCard('Without protection', a.without_privacy),
    attackCard('With masking + DP', a.with_privacy),
  );
}

// ---------- agent ticker (stage caption) ----------

let STATIC_CAPTION = '';

function setTicker(parts) {
  const t = $('ticker');
  t.replaceChildren(...parts.map((p) => (typeof p === 'string' ? document.createTextNode(p) : p)));
}

function renderTicker() {
  if (!trace.agents) return;
  const name = (id) => el('b', '', node(id).label);
  const r = shownRound();
  if (timer !== null && phase > 0) {
    // narrate one hospital's agents at a time while the round animates
    const id = HOSPITALS[Math.min(HOSPITALS.length - 1, Math.floor(phase * HOSPITALS.length))];
    const acts = agentActivity(id, r);
    const label = (k) => (AGENTS.find((a) => a[0] === k) || [k, k, k])[2];
    setTicker(acts.length
      ? [name(id), ' · ', acts.map((a) => `${label(a.agent)}: ${a.action}`).join(' → ')]
      : [name(id), ' · no agent activity this round']);
    return;
  }
  if (r === 0) {
    setTicker([STATIC_CAPTION]);
    return;
  }
  // paused on a completed round: what each hospital's Challenger measured on its own data
  const parts = [`Round ${r} · each Challenger scored the new model on its own patients: `];
  HOSPITALS.forEach((id, i) => {
    const last = agentActivity(id, r).filter((a) => a.agent === 'challenger').pop();
    const m = last && /local AUC ([0-9.]+)/.exec(last.action);
    const offline = updateFor(trace.rounds[r - 1], id)?.sent === false; // from the trace, not inferred
    parts.push(name(id), m ? ` ${m[1]}` : offline ? ' offline this round' : ' —');
    if (i < HOSPITALS.length - 1) parts.push(' · ');
  });
  setTicker(parts);
}

// ---------- playback ----------

function render() {
  $('roundLabel').textContent = `Round ${round} / ${trace.rounds.length}`;
  scrub.value = round;
  renderGraph();
  renderCharts();
  renderLog();
  renderTicker();
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
  if (REDUCED) { seek(trace.rounds.length); return; } // DESIGN §8: jump to the final state
  playBtn.textContent = 'Pause';
  phase = 0;
  const tick = 30;
  const roundMs = 4 * HOP_MS; // update hop + broadcast hop, each with a short hold
  timer = setInterval(() => {
    phase += tick / roundMs;
    if (phase >= 1) {
      phase = 0;
      round += 1;
      if (round >= trace.rounds.length) stop();
    }
    render();
  }, tick);
}

$('reset').onclick = () => seek(0);
playBtn.onclick = play;
scrub.oninput = () => seek(+scrub.value);
showMask.onchange = render;
showDp.onchange = render;
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

// DESIGN §9 loading state
G.ctx.clearRect(0, 0, G.w, G.h);
text('loading trace…', G.w / 2, G.h / 2, { size: 13, color: COL.faint, font: FONT_DATA });

loadTrace()
  .then(({ url, data }) => {
    trace = data;
    STATIC_CAPTION = $('ticker').textContent.replace(/\s+/g, ' ').trim();
    scrub.max = trace.rounds.length;
    $('bytes').textContent = '0';
    $('source').textContent = `replaying ${url.replace(/^\.\.\//, '')}`;
    renderLegend();
    renderAttack();
    render();
  })
  .catch((e) => {
    const pre = el('pre', 'state error',
      `failed to load a trace\n${e.message}\n\nserve privatelearn/ with: python -m http.server 8000\nthen open http://localhost:8000/web/index.html  (or ?trace=<path>)`);
    document.body.replaceChildren(pre);
  });
