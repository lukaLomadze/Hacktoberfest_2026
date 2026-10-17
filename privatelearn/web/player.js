/* PrivateLearn demo player — replays a trace.json per docs/DESIGN.md.
   Vanilla JS, no framework. Colors and type come from player.css tokens. */

const TRACE_URL = new URLSearchParams(location.search).get('trace') || '../data/sample_trace.json';
const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const $ = (id) => document.getElementById(id);
const graph = $('graph'), g = graph.getContext('2d');
const chart = $('chart'), c = chart.getContext('2d');
const logEl = $('log'), attackEl = $('attack'), legendEl = $('legend'), chartLegendEl = $('chartLegend');
const playBtn = $('play'), resetBtn = $('reset'), scrub = $('scrub'), roundLabel = $('roundLabel'), bytesEl = $('bytes');

const CSS = getComputedStyle(document.documentElement);
const T = (n) => CSS.getPropertyValue(n).trim();
const COL = {
  hosp: [T('--hosp-a'), T('--hosp-b'), T('--hosp-c')],
  coord: T('--coord'), ok: T('--ok'), danger: T('--danger'),
  border: T('--border'), bgRaise: T('--bg-raise'),
  text: T('--text'), dim: T('--text-dim'), faint: T('--text-faint'),
};

const HOSP = ['h0', 'h1', 'h2'];
const nodeColor = (id) => (id === 'coord' ? COL.coord : COL.hosp[HOSP.indexOf(id)]);

let trace = null;
let round = 0;       // rounds completed so far
let phase = 0;       // packet animation progress 0..1
let timer = null;

/* ---------------- graph ---------------- */
function pos(id) {
  if (id === 'coord') return { x: 480, y: 320 };
  return { x: 160 + HOSP.indexOf(id) * 320, y: 90 };
}

function roundedRect(x, y, w, h, r) {
  g.beginPath();
  g.moveTo(x + r, y);
  g.arcTo(x + w, y, x + w, y + h, r);
  g.arcTo(x + w, y + h, x, y + h, r);
  g.arcTo(x, y + h, x, y, r);
  g.arcTo(x, y, x + w, y, r);
  g.closePath();
}

function drawGrid() {
  g.fillStyle = COL.bgRaise;
  for (let x = 0; x <= graph.width; x += 32) {
    for (let y = 0; y <= graph.height; y += 32) {
      g.fillRect(x, y, 1.5, 1.5);
    }
  }
}

function drawEdges() {
  const coord = pos('coord');
  g.strokeStyle = COL.border;
  g.lineWidth = 1;
  HOSP.forEach((h) => {
    const p = pos(h);
    g.beginPath();
    g.moveTo(p.x, p.y + 30);
    g.lineTo(coord.x, coord.y - 26);
    g.stroke();
  });
}

function drawNode(id, label) {
  const p = pos(id);
  const color = nodeColor(id);
  if (id === 'coord') {
    g.fillStyle = color;
    roundedRect(p.x - 60, p.y - 18, 120, 36, 10);
    g.fill();
    g.fillStyle = '#14100a';
    g.font = '600 12px ' + 'ui-monospace, monospace';
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.fillText(label, p.x, p.y);
  } else {
    g.beginPath();
    g.arc(p.x, p.y, 26, 0, Math.PI * 2);
    g.fillStyle = color;
    g.fill();
    // short letter inside the disc
    g.fillStyle = '#fff';
    g.font = '700 16px ' + 'ui-monospace, monospace';
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.fillText(label.slice(-1), p.x, p.y);
    // full name + role count below
    g.font = '11px ' + 'ui-monospace, monospace';
    g.fillStyle = COL.dim;
    g.fillText(label, p.x, p.y + 40);
    g.fillStyle = COL.faint;
    g.fillText('4 agents', p.x, p.y + 52);
  }
  g.textBaseline = 'alphabetic';
  g.textAlign = 'left';
}

function drawPacket(from, to, t, color) {
  const a = pos(from), b = pos(to);
  const x = a.x + (b.x - a.x) * t;
  const y = a.y + (b.y - a.y) * t;
  // a small stack of three bars = "numbers", not a dot
  g.fillStyle = color;
  const w = [10, 6, 8];
  w.forEach((len, i) => {
    g.fillRect(x - len / 2, y - 6 + i * 5, len, 3);
  });
}

function renderGraph() {
  g.clearRect(0, 0, graph.width, graph.height);
  drawGrid();
  drawEdges();

  const current = trace.rounds[round - 1];

  HOSP.forEach((h) => {
    const node = trace.nodes.find((n) => n.id === h);
    drawNode(h, (node && node.label) || h.toUpperCase());
  });
  drawNode('coord', 'Coordinator');

  if (current) {
    // updates: hospital -> coordinator
    current.updates.forEach((u) => {
      if (u.sent && !u.rejected) drawPacket(u.hospital, 'coord', phase, nodeColor(u.hospital));
    });
    // broadcast: coordinator -> hospitals (not excluded)
    if (phase > 0.5) {
      HOSP.forEach((h) => {
        if (!current.excluded.includes(h)) drawPacket('coord', h, (phase - 0.5) * 2, COL.coord);
      });
    }
  }
}

/* ---------------- legend ---------------- */
function renderLegend() {
  legendEl.innerHTML = '';
  const items = [
    ...trace.nodes.filter((n) => n.id !== 'coord').map((n) => [n.label, nodeColor(n.id)]),
    ['Coordinator', COL.coord],
  ];
  items.forEach(([label, color]) => {
    const span = document.createElement('span');
    span.className = 'legend-item';
    const dot = document.createElement('span');
    dot.className = 'legend-dot';
    dot.style.background = color;
    span.appendChild(dot);
    span.appendChild(document.createTextNode(label));
    legendEl.appendChild(span);
  });
}

/* ---------------- chart ---------------- */
function renderChart() {
  c.clearRect(0, 0, chart.width, chart.height);
  const W = chart.width, H = chart.height;
  const L = 34, R = 8, top = 10, bottom = 20;
  const y0 = 0.6, y1 = 1.0;

  const X = (i) => L + (i / (trace.rounds.length)) * (W - L - R);
  const Y = (v) => top + (1 - (v - y0) / (y1 - y0)) * (H - top - bottom);

  // gridlines + y labels
  c.font = '9px ' + 'ui-monospace, monospace';
  c.fillStyle = COL.faint;
  for (let v = 0.6; v <= 1.0001; v += 0.1) {
    const y = Y(v);
    c.strokeStyle = COL.border;
    c.lineWidth = 1;
    c.beginPath();
    c.moveTo(L, y);
    c.lineTo(W - R, y);
    c.stroke();
    c.textAlign = 'right';
    c.textBaseline = 'middle';
    c.fillText(v.toFixed(1), L - 6, y);
  }
  // x labels
  c.textAlign = 'center';
  c.textBaseline = 'top';
  for (let i = 0; i <= trace.rounds.length; i++) {
    c.fillText(String(i), X(i), H - bottom + 4);
  }

  // reference lines: solo (danger), centralized (ok)
  const refs = [
    [trace.meta.auc_solo, COL.danger, 'solo'],
    [trace.meta.auc_centralized, COL.ok, 'centralized'],
  ];
  refs.forEach(([v, color, label]) => {
    const y = Y(v);
    c.strokeStyle = color;
    c.setLineDash([4, 3]);
    c.lineWidth = 1;
    c.beginPath();
    c.moveTo(L, y);
    c.lineTo(W - R, y);
    c.stroke();
    c.setLineDash([]);
    c.textAlign = 'left';
    c.textBaseline = 'bottom';
    c.fillStyle = color;
    c.fillText(label, L + 4, y - 1);
  });

  // federated line
  const s = trace.metrics.auc_series;
  c.strokeStyle = COL.coord;
  c.lineWidth = 2;
  c.beginPath();
  s.forEach((v, i) => {
    i === 0 ? c.moveTo(X(i), Y(v)) : c.lineTo(X(i), Y(v));
  });
  c.stroke();
  s.forEach((v, i) => {
    c.fillStyle = COL.coord;
    c.beginPath();
    c.arc(X(i), Y(v), 3, 0, Math.PI * 2);
    c.fill();
  });
  c.textAlign = 'left';
  c.textBaseline = 'alphabetic';

  // legend
  chartLegendEl.innerHTML = '';
  [['solo', COL.danger], ['federated', COL.coord], ['centralized', COL.ok]].forEach(([lbl, colr]) => {
    const span = document.createElement('span');
    const sw = document.createElement('span');
    sw.className = 'swatch';
    sw.style.background = colr;
    span.appendChild(sw);
    span.appendChild(document.createTextNode(lbl));
    chartLegendEl.appendChild(span);
  });
}

/* ---------------- decision log ---------------- */
function renderLog() {
  logEl.innerHTML = '';
  trace.rounds.slice(0, round).forEach((r) => {
    const li = document.createElement('li');
    if (r.excluded.length) li.classList.add('warn');

    const badge = document.createElement('span');
    badge.className = 'badge ' + r.strategy;
    badge.textContent = r.strategy.replace('_', '-');

    const num = document.createElement('span');
    num.className = 'round-num';
    num.textContent = 'r' + r.round;

    const text = document.createElement('span');
    text.textContent = r.decision;

    li.appendChild(badge);
    li.appendChild(num);
    li.appendChild(text);

    if (r.excluded.length) {
      const tag = document.createElement('span');
      tag.className = 'excluded-tag';
      tag.textContent = 'excluded ' + r.excluded.join(',');
      li.appendChild(tag);
    }
    logEl.appendChild(li);
  });
}

/* ---------------- attack ---------------- */
function attackCard(label, recovered, error) {
  const card = document.createElement('div');
  card.className = 'attack-card';
  const l = document.createElement('span');
  l.className = 'label';
  l.textContent = label;
  const stamp = document.createElement('span');
  stamp.className = 'stamp ' + (recovered ? 'recovered' : 'failed');
  stamp.textContent = recovered ? 'RECOVERED' : 'FAILED';
  const err = document.createElement('span');
  err.className = 'attack-err';
  err.textContent = 'error ' + error;
  card.appendChild(l);
  card.appendChild(stamp);
  card.appendChild(err);
  return card;
}

function renderAttack() {
  attackEl.innerHTML = '';
  const a = trace.attack;
  attackEl.appendChild(attackCard('Without protection', a.without_privacy.recovered, a.without_privacy.feature_error));
  attackEl.appendChild(attackCard('With masking + DP', a.with_privacy.recovered, a.with_privacy.feature_error));
}

/* ---------------- orchestration ---------------- */
function render() {
  renderGraph();
  renderChart();
  renderLog();
  renderAttack();
  roundLabel.textContent = 'Round ' + round + ' / ' + trace.rounds.length;
}

function step(direction) {
  round = Math.max(0, Math.min(trace.rounds.length, round + direction));
  scrub.value = round;
  phase = 0;
  render();
}

function play() {
  if (timer) {
    clearInterval(timer);
    timer = null;
    playBtn.textContent = 'Play';
    return;
  }
  playBtn.textContent = 'Pause';
  if (round >= trace.rounds.length) step(-trace.rounds.length);
  if (REDUCED) {
    round = trace.rounds.length;
    phase = 1;
    render();
    playBtn.textContent = 'Play';
    return;
  }
  timer = setInterval(() => {
    phase += 0.04;
    if (phase >= 1) {
      phase = 0;
      step(1);
      if (round >= trace.rounds.length) {
        clearInterval(timer);
        timer = null;
        playBtn.textContent = 'Play';
      }
    }
    render();
  }, 40);
}

resetBtn.onclick = () => step(-trace.rounds.length);
playBtn.onclick = play;
scrub.oninput = () => { round = +scrub.value; phase = 0; render(); };

/* ---------------- load ---------------- */
fetch(TRACE_URL)
  .then((r) => r.json())
  .then((t) => {
    trace = t;
    scrub.max = trace.rounds.length;
    bytesEl.textContent = '0';
    renderLegend();
    render();
  })
  .catch((e) => {
    document.body.innerHTML =
      '<pre style="color:#ff5c5c;padding:2em;font:12px monospace">' +
      'failed to load ' + TRACE_URL + '\n' + e + '\n\n' +
      'serve the repo with: python -m http.server 8000\n' +
      'then open /web/index.html</pre>';
  });
