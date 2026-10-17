/* PrivateLearn demo player — replays a trace.json. Vanilla JS, no framework.
   Skeleton for T10: correct data plumbing; visual polish is Person C's lane.
   Serve from the repo root with `python -m http.server`, or open with a trace URL:
     index.html?trace=data/sample_trace.json   */

const TRACE_URL = new URLSearchParams(location.search).get('trace') || '../data/sample_trace.json';

const graph = document.getElementById('graph');
const gctx = graph.getContext('2d');
const chart = document.getElementById('chart');
const cctx = chart.getContext('2d');
const logEl = document.getElementById('log');
const attackEl = document.getElementById('attack');
const playBtn = document.getElementById('play');
const resetBtn = document.getElementById('reset');
const scrub = document.getElementById('scrub');
const roundLabel = document.getElementById('roundLabel');
const bytesEl = document.getElementById('bytes');

let trace = null;
let round = 0; // 0 = before round 1
let timer = null;
let packetPhase = 0; // animation progress 0..1

const COLORS = {
  hospital: ['#4c6ef5', '#40c057', '#fa5252'],
  coordinator: '#fab005',
};

// fixed layout: three hospitals along the top, coordinator below centre
function nodePos(id) {
  const hospitals = ['h0', 'h1', 'h2'];
  const i = hospitals.indexOf(id);
  if (i >= 0) {
    const x = 160 + i * 320;
    return { x, y: 60 };
  }
  return { x: 480, y: 250 }; // coordinator
}

function drawNode(id, label, color, active) {
  const p = nodePos(id);
  gctx.beginPath();
  gctx.arc(p.x, p.y, active ? 30 : 26, 0, Math.PI * 2);
  gctx.fillStyle = color;
  gctx.fill();
  gctx.fillStyle = '#fff';
  gctx.font = '13px monospace';
  gctx.textAlign = 'center';
  gctx.fillText(label, p.x, p.y + 4);
  gctx.textAlign = 'left';
}

function drawPacket(from, to, t) {
  const a = nodePos(from);
  const b = nodePos(to);
  const x = a.x + (b.x - a.x) * t;
  const y = a.y + (b.y - a.y) * t;
  gctx.beginPath();
  gctx.arc(x, y, 4, 0, Math.PI * 2);
  gctx.fillStyle = '#fff';
  gctx.fill();
}

function render() {
  gctx.clearRect(0, 0, graph.width, graph.height);
  // edges (static)
  const hospitals = ['h0', 'h1', 'h2'];
  const coord = nodePos('coord');
  gctx.strokeStyle = '#2a2e3a';
  hospitals.forEach((h) => {
    const p = nodePos(h);
    gctx.beginPath();
    gctx.moveTo(p.x, p.y);
    gctx.lineTo(coord.x, coord.y);
    gctx.stroke();
  });

  // nodes
  hospitals.forEach((h, i) => {
    const active = trace.rounds[round - 1]?.updates.some((u) => u.hospital === h && u.sent);
    drawNode(h, trace.nodes.find((n) => n.id === h)?.label || h, COLORS.hospital[i], active);
  });
  drawNode('coord', 'Coordinator', COLORS.coordinator, false);

  // packets for the current round
  if (round >= 1) {
    const r = trace.rounds[round - 1];
    r.updates.forEach((u) => {
      if (u.sent && !u.rejected) drawPacket(u.hospital, 'coord', packetPhase);
    });
    // broadcast back
    if (packetPhase > 0.5) {
      hospitals.forEach((h) => {
        if (!r.excluded.includes(h)) drawPacket('coord', h, (packetPhase - 0.5) * 2);
      });
    }
  }

  roundLabel.textContent = `round ${round} / ${trace.rounds.length}`;
  renderChart();
  renderLog();
  renderAttack();
}

function renderChart() {
  cctx.clearRect(0, 0, chart.width, chart.height);
  const s = trace.metrics.auc_series;
  const points = trace.rounds.length + 1;
  const pad = 20;
  const min = Math.min(...s, trace.meta.auc_solo) - 0.05;
  const max = Math.max(...s, trace.meta.auc_centralized) + 0.02;
  const X = (i) => pad + (i / (points - 1)) * (chart.width - 2 * pad);
  const Y = (v) => chart.height - pad - ((v - min) / (max - min)) * (chart.height - 2 * pad);

  // solo / centralized reference lines
  [trace.meta.auc_solo, trace.meta.auc_centralized].forEach((v, k) => {
    cctx.strokeStyle = k === 0 ? '#5c6474' : '#5c6474';
    cctx.setLineDash([4, 4]);
    cctx.beginPath();
    cctx.moveTo(pad, Y(v));
    cctx.lineTo(chart.width - pad, Y(v));
    cctx.stroke();
    cctx.setLineDash([]);
  });

  cctx.strokeStyle = '#fab005';
  cctx.beginPath();
  s.forEach((v, i) => {
    i === 0 ? cctx.moveTo(X(i), Y(v)) : cctx.lineTo(X(i), Y(v));
  });
  cctx.stroke();
}

function renderLog() {
  logEl.innerHTML = '';
  trace.rounds.slice(0, round).forEach((r) => {
    const li = document.createElement('li');
    li.innerHTML = `<b>r${r.round}</b> ${r.strategy} — ${r.decision}`;
    if (r.excluded.length) li.classList.add('warn');
    logEl.appendChild(li);
  });
}

function renderAttack() {
  const a = trace.attack;
  attackEl.innerHTML =
    `target: ${a.target}<br>` +
    `no privacy → recovered <b>${a.without_privacy.recovered ? 'yes' : 'no'}</b> (err ${a.without_privacy.feature_error})<br>` +
    `masking + DP → recovered <b>${a.with_privacy.recovered ? 'yes' : 'no'}</b> (err ${a.with_privacy.feature_error})`;
}

function step(direction) {
  round = Math.max(0, Math.min(trace.rounds.length, round + direction));
  scrub.max = trace.rounds.length;
  scrub.value = round;
  packetPhase = 0;
  render();
}

async function play() {
  if (timer) { clearInterval(timer); timer = null; playBtn.textContent = 'Play'; return; }
  playBtn.textContent = 'Pause';
  if (round >= trace.rounds.length) step(-trace.rounds.length);
  timer = setInterval(() => {
    packetPhase += 0.04;
    if (packetPhase >= 1) {
      packetPhase = 0;
      step(1);
      if (round >= trace.rounds.length) { clearInterval(timer); timer = null; playBtn.textContent = 'Play'; }
    }
    render();
  }, 40);
}

resetBtn.onclick = () => step(-trace.rounds.length);
playBtn.onclick = play;
scrub.oninput = () => { round = +scrub.value; packetPhase = 0; render(); };

fetch(TRACE_URL)
  .then((r) => r.json())
  .then((t) => {
    trace = t;
    scrub.max = trace.rounds.length;
    bytesEl.textContent = '0';
    render();
  })
  .catch((e) => {
    document.body.innerHTML = `<pre style="color:#fa5252;padding:2em">failed to load ${TRACE_URL}\n${e}\n\nserve the repo with: python -m http.server 8000\nor open index.html?trace=&lt;path&gt;</pre>`;
  });
