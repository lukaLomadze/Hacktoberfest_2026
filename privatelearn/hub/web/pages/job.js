// #/jobs/{id} — the live process view (PLATFORM-API §5.5, §6). It is driven only by §6
// events: GET /jobs/{id}/events to catch up, then the SSE stream while the job runs. A
// finished job can be replayed from its event history with a player bar (play/pause, step one
// event, jump a round, stop, skip to the end, speed 0.25–8×, a position slider; keys: Space,
// →, Shift+→). ?replay=1 starts the replay at once; this is also how the mock event list plays.

import { api } from "../api.js";
import { errorBox, h, meta, session, statusPill } from "../app.js";
import { createCanvas } from "../live/canvas.js";
import { createChart } from "../live/chart.js";
import { createDecisions } from "../live/decisions.js";
import { createFeed } from "../live/feed.js";
import { applyEvent, initialState } from "../live/state.js";
import { createStepper } from "../live/stepper.js";

const LIVE = new Set(["queued", "running"]);
// replay pacing per event type, in ms at 1×: packets are quick, agent messages are read
const PACE = { packet: 160, "agent.message": 420, "round.decision": 700, "round.completed": 700, default: 220 };
const SPEEDS = [0.25, 0.5, 1, 2, 4, 8];

const STYLE_ID = "page-job-style";
const STYLE = `
.page[data-page="job"] .job-head { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.page[data-page="job"] .job-head h1 { overflow-wrap: anywhere; }
.page[data-page="job"] .job-facts { display: flex; gap: 16px; flex-wrap: wrap; color: var(--text-dim); }
.page[data-page="job"] .job-facts b { color: var(--text); font-family: var(--font-data); font-weight: 600; }
.page[data-page="job"] .zero { color: var(--ok); font-family: var(--font-data); font-weight: 600; }
.page[data-page="job"] .replay-bar { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; padding: 8px 12px; }
.page[data-page="job"] .replay-bar .btn { min-width: 0; }
.page[data-page="job"] .replay-bar select { width: auto; min-width: 0; }
.page[data-page="job"] .replay-bar input[type="range"] { flex: 1 1 160px; min-width: 120px; accent-color: var(--coord); }
.page[data-page="job"] .replay-pos { font: 12px var(--font-data); color: var(--text-dim); white-space: nowrap; }
.page[data-page="job"] .live-dot { font: 12px var(--font-data); color: var(--text-dim); border: 1px dashed var(--border-strong); border-radius: 999px; padding: 0 8px; }
.page[data-page="job"] .published { border-color: var(--ok); color: var(--text); display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
.page[data-page="job"] .published b { color: var(--ok); font-family: var(--font-data); }
.page[data-page="job"] .job-grid { display: grid; grid-template-columns: minmax(0, 1fr) 360px; gap: 16px; align-items: start; }
.page[data-page="job"] .job-main, .page[data-page="job"] .job-rail { display: flex; flex-direction: column; gap: 16px; min-width: 0; }
.page[data-page="job"] .live-head { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }
.page[data-page="job"] .live-head b { margin-left: auto; }
.page[data-page="job"] .stage { padding: 0; overflow: hidden; }
.page[data-page="job"] .stage-box { width: 100%; }
.page[data-page="job"] .live-canvas { display: block; }
.page[data-page="job"] .stage-caption { padding: 8px 16px; border-top: 1px solid var(--border); color: var(--text-dim); }
/* stepper */
.page[data-page="job"] .live-stepper { list-style: none; margin: 0; padding: 0; display: grid; gap: 8px; counter-reset: step;
  grid-template-columns: repeat(auto-fit, minmax(min(120px, 100%), 1fr)); }
.page[data-page="job"] .live-step { display: flex; align-items: center; gap: 8px; padding: 8px; border: 1px solid var(--border); border-radius: 8px; color: var(--text-faint); font-size: 12px; font-weight: 600; counter-increment: step; }
.page[data-page="job"] .step-dot { flex: none; width: 22px; height: 22px; border-radius: 50%; display: grid; place-items: center; border: 1px solid var(--border-strong); font: 600 12px var(--font-data); }
.page[data-page="job"] .step-dot::before { content: counter(step); }
.page[data-page="job"] .step-done { color: var(--text-dim); }
.page[data-page="job"] .step-done .step-dot { background: var(--ok); border-color: var(--ok); color: var(--bg); }
.page[data-page="job"] .step-done .step-dot::before { content: "✓"; }
.page[data-page="job"] .step-now { color: var(--text); border-color: var(--coord); background: var(--panel); }
.page[data-page="job"] .step-now .step-dot { background: var(--coord); border-color: var(--coord); color: var(--bg); }
.page[data-page="job"] .step-failed { color: var(--danger); border-color: var(--danger); }
.page[data-page="job"] .step-failed .step-dot { background: var(--danger); border-color: var(--danger); color: var(--bg); }
/* feed */
.page[data-page="job"] .live-feed { display: flex; flex-direction: column; }
.page[data-page="job"] .live-feed-list { list-style: none; margin: 0; padding: 0 4px 0 0; display: flex; flex-direction: column; gap: 8px; max-height: 640px; overflow-y: auto; }
.page[data-page="job"] .feed-item { border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; background: var(--panel-2); }
.page[data-page="job"] .feed-item.from-coord { border-color: var(--border-strong); }
.page[data-page="job"] .feed-meta { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; font-size: 12px; }
.page[data-page="job"] .feed-dot { width: 8px; height: 8px; border-radius: 999px; background: var(--text-faint); flex: none; }
.page[data-page="job"] .feed-dot.coord { background: var(--coord); border-radius: 2px; }
.page[data-page="job"] .feed-agent { font-family: var(--font-data); color: var(--text-dim); }
.page[data-page="job"] .feed-round { margin-left: auto; font-family: var(--font-data); }
.page[data-page="job"] .feed-text { margin-top: 4px; overflow-wrap: anywhere; }
.page[data-page="job"] .feed-tag { font: 12px/1.4 var(--font-data); border: 1px solid var(--border-strong); border-radius: 999px; padding: 0 6px; color: var(--text-dim); }
.page[data-page="job"] .feed-tag.made-llm { color: var(--text); border-style: dashed; }
.page[data-page="job"] .feed-unknown code { font-size: 12px; color: var(--text-faint); overflow-wrap: anywhere; }
/* decisions */
.page[data-page="job"] .live-decisions { list-style: none; margin: 0; padding: 0; display: grid; grid-template-columns: repeat(auto-fill, minmax(min(240px, 100%), 1fr)); gap: 8px; }
.page[data-page="job"] .decision { border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; background: var(--panel-2); display: flex; flex-direction: column; gap: 4px; }
.page[data-page="job"] .decision.has-excluded { border-left: 2px solid var(--danger); }
.page[data-page="job"] .decision-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.page[data-page="job"] .decision-text { color: var(--text-dim); overflow-wrap: anywhere; }
.page[data-page="job"] .decision-meta { font-size: 12px; }
.page[data-page="job"] .decision-excluded { font-size: 12px; color: var(--danger); }
.page[data-page="job"] .badge { font: 600 12px/1.6 var(--font-data); padding: 0 8px; border-radius: 999px; color: var(--bg); background: var(--text-dim); }
/* DESIGN §7.3 badge colors */
.page[data-page="job"] .badge-fedavg { background: var(--coord); }
.page[data-page="job"] .badge-fedprox { background: var(--hosp-a); }
.page[data-page="job"] .badge-trimmed_mean { background: var(--hosp-c); }
/* chart */
.page[data-page="job"] .chart-box { width: 100%; }
.page[data-page="job"] .chart-box canvas { display: block; }
.page[data-page="job"] .chart-legend { display: flex; gap: 12px; flex-wrap: wrap; font: 12px var(--font-data); color: var(--text-dim); margin: 8px 0 16px; }
.page[data-page="job"] .sw { display: inline-block; width: 12px; height: 2px; vertical-align: middle; margin-right: 4px; background: var(--coord); }
.page[data-page="job"] .sw.dash { background: none; border-top: 2px dashed; height: 0; }
.page[data-page="job"] .sw-ok { color: var(--ok); }
.page[data-page="job"] .sw-bad { color: var(--danger); }
/* run switch (mock mode): the scripted fixture vs real agent runs */
.page[data-page="job"] .run-switch { display: flex; gap: 8px; flex-wrap: wrap; }
.page[data-page="job"] .run-tab { display: flex; flex-direction: column; gap: 2px; padding: 8px 12px; border: 1px solid var(--border); border-radius: 8px; background: var(--panel-2); color: var(--text-dim); text-decoration: none; min-width: 0; }
.page[data-page="job"] .run-tab b { color: var(--text); }
.page[data-page="job"] .run-tab span { font-size: 12px; overflow-wrap: anywhere; }
.page[data-page="job"] .run-tab.active { border-color: var(--coord); }
@media (max-width: 900px) {
  .page[data-page="job"] .job-grid { grid-template-columns: minmax(0, 1fr); }
  .page[data-page="job"] .live-feed-list { max-height: 420px; }
}
`;

function ensureStyle() {
  if (!document.getElementById(STYLE_ID)) document.head.append(h("style", { id: STYLE_ID }, STYLE));
}

const fmt = (v, d = 1) => (typeof v === "number" && Number.isFinite(v) ? v.toFixed(d) : "—");

export async function render({ el, params, query }) {
  ensureStyle();
  el.replaceChildren(h("p.loading", "loading job…"));
  const id = params.id;

  let job;
  let history;
  try {
    [job, history] = await Promise.all([api.get(`/jobs/${encodeURIComponent(id)}`),
                                        api.get(`/jobs/${encodeURIComponent(id)}/events`, { since: 0 })]);
  } catch (err) {
    el.replaceChildren(errorBox(err));
    return null;
  }
  const events = (history && history.items) || [];
  const [group, metaInfo] = await Promise.all([
    api.get(`/groups/${encodeURIComponent(job.group_id)}`).catch(() => null),
    meta().catch(() => null),
  ]);
  const spec = group && group.task_spec;
  const secure = !spec || !spec.privacy || spec.privacy.secure_aggregation !== false;

  // ---------------------------------------------------------------- the pieces
  const stepper = createStepper(metaInfo && metaInfo.enums && metaInfo.enums.job_step);
  const canvas = createCanvas({ secure });
  const feed = createFeed();
  const decisions = createDecisions();
  const chart = createChart();

  const statusSlot = h("span");
  const roundSlot = h("b", "0");
  const epsSlot = h("b", "0.0");
  const liveSlot = h("span.live-dot", { hidden: true });
  const btn = (label, title) => h("button.btn.ghost.small", { type: "button", title }, label);
  const playBtn = h("button.btn.primary.small", { type: "button", title: "Play / pause (Space)" }, "▶ Play");
  const stepBtn = btn("Step ›", "Show the next event (→)");
  const roundBtn = btn("Next round »", "Play up to the end of the next round (Shift+→)");
  const stopBtn = btn("■ Stop", "Stop and go back to the start");
  const endBtn = btn("Skip to end ⇥", "Show the finished job");
  const speedSel = h("select", { "aria-label": "Replay speed", title: "Replay speed" },
    SPEEDS.map((v) => h("option", { value: String(v), selected: v === 1 }, `${v}×`)));
  const scrub = h("input", { type: "range", min: "0", max: "0", value: "0", step: "1", "aria-label": "Replay position" });
  const posSlot = h("span.replay-pos");
  const replayBar = h("div.panel.replay-bar", { hidden: true, role: "toolbar", "aria-label": "Replay controls" },
    playBtn, stepBtn, roundBtn, stopBtn, endBtn, h("span.faint", "speed"), speedSel, scrub, posSlot);
  const publishedSlot = h("div.panel.published", { hidden: true, role: "status" });
  const runsSlot = h("nav.run-switch", { hidden: true, "aria-label": "Which run to show" });
  const errorSlot = h("div");

  const title = group ? `${group.name}: ${job.kind === "finetune" ? "fine-tune" : "training"}` : `Job ${job.id}`;
  el.replaceChildren(
    runsSlot,
    h("div.job-head",
      h("h1", title), statusSlot, liveSlot),
    h("div.job-facts",
      h("span", "round ", roundSlot, ` / ${job.rounds ?? "agent decides"}`),
      h("span", "ε spent ", epsSlot),
      h("span", "participants ", h("b", String((job.participants || []).length))),
      h("span", h("span.zero", "0"), " data rows left any company"),
      h("span.faint.mono", job.id)),
    replayBar, stepper.el, errorSlot, publishedSlot,
    h("div.job-grid",
      h("div.job-main",
        h("section.panel.stage", h("div.stage-box", canvas.el),
          h("p.stage-caption", "Each company runs its own agents (Schema · Planner · Trainer · Challenger); a chip lights " +
            "when that agent acts this round. Packets are masked updates in, the new model out.")),
        decisions.el, chart.el),
      h("div.job-rail", feed.el)));
  canvas.mount();
  chart.mount();

  // mock mode: switch between the scripted demo (job_1) and real agent runs (heart_diseases_simulation)
  if (api.isMock) {
    api.get("/demo/runs").then((r) => {
      const runs = (r && r.items) || [];
      if (runs.length < 2) return;
      runsSlot.replaceChildren(...runs.map((run) => h("a.run-tab", {
        href: `#/jobs/${encodeURIComponent(run.job_id)}`, class: run.job_id === job.id ? "active" : null,
        "aria-current": run.job_id === job.id ? "page" : null,
      }, h("b", run.label), h("span", `${run.detail} · ${run.status}`))));
      runsSlot.hidden = false;
    }).catch(() => { /* an older mock without /demo/runs: no switch */ });
  }

  // ---------------------------------------------------------------- drawing
  let state = initialState(job);

  function draw(changed) {
    const all = !changed;
    if (all || changed.has("header")) {
      statusSlot.replaceChildren(statusPill(state.status));
      roundSlot.textContent = String(state.round);
      epsSlot.textContent = fmt(state.eps_total);
      errorSlot.replaceChildren(...(state.error ? [errorBox(new Error(state.error))] : []));
    }
    if (all || changed.has("stepper") || changed.has("header")) stepper.render(state);
    if (all || changed.has("canvas")) canvas.render(state);
    if (all || changed.has("feed")) feed.render(state);
    if (all || changed.has("decisions")) decisions.render(state);
    if (all || changed.has("chart")) chart.render(state);
    if (all || changed.has("published")) {
      const p = state.published;
      publishedSlot.hidden = !p;
      if (p) {
        const auc = p.metrics && p.metrics.auc;
        const link = session.user && session.user.role === "admin" ? h("a", { href: `#/groups/${job.group_id}` }, "Open the group") : null;
        // replaceChildren() prints null as the text "null", so drop the missing parts first
        publishedSlot.replaceChildren(...[h("span", "Published ", h("b", `v${p.version ?? "?"}`)),
          typeof auc === "number" ? h("span", "AUC ", h("b", auc.toFixed(3))) : null, link].filter(Boolean));
      }
    }
  }

  function apply(e, animate) {
    const changed = applyEvent(state, e);
    if (changed.has("packet")) canvas.packet(e, animate);
    if (changed.size) draw(changed);
  }

  for (const e of events) applyEvent(state, e); // catch up silently, then draw once
  if (!events.length) state.status = job.status; // nothing happened yet: trust the Job object
  draw();

  // ---------------------------------------------------------------- live stream
  let stopStream = null;
  function goLive() {
    // the event history is the fresher answer when there is one: a job whose history already
    // ends in a final job.status gets no stream (nothing would ever close it)
    if (!LIVE.has(events.length ? state.status : job.status)) return;
    liveSlot.hidden = false;
    liveSlot.textContent = "live";
    stopStream = api.stream(job.id, (e) => {
      if (typeof e.seq === "number" && e.seq <= state.last_seq) return; // resent after a reconnect
      events.push(e); // so a replay after the job finishes includes what streamed live
      apply(e, true);
      if (!LIVE.has(state.status)) {
        liveSlot.textContent = "finished";
        cursor = events.length;
        replayBar.hidden = false;
        sync();
        // the server ends the stream after the last event; EventSource would reconnect every
        // few seconds for as long as the page is open unless it is closed here
        if (stopStream) stopStream();
        stopStream = null;
      }
    }, { since: state.last_seq });
  }

  // ---------------------------------------------------------------- replay (finished jobs, mocks)
  // cursor = how many events the picture shows; it starts at the end (the finished job)
  let timer = null;
  let cursor = events.length;
  let playing = false;
  let speed = 1;
  const atEnd = () => cursor >= events.length;

  function sync() {
    playBtn.textContent = playing ? "⏸ Pause" : atEnd() ? "↻ Replay" : cursor === 0 ? "▶ Play" : "▶ Resume";
    scrub.max = String(events.length);
    scrub.value = String(cursor);
    posSlot.textContent = `event ${cursor} / ${events.length} · round ${state.round}`;
  }
  function restart() { // an empty picture, before the first event
    state = initialState(job);
    cursor = 0;
    draw();
  }
  function pause() {
    if (timer) clearTimeout(timer);
    timer = null;
    playing = false;
    sync();
  }
  function tick() {
    timer = null;
    if (atEnd()) { pause(); return; }
    const e = events[cursor++];
    apply(e, true);
    sync();
    if (atEnd()) { pause(); return; }
    timer = setTimeout(tick, (PACE[e.type] ?? PACE.default) / speed);
  }
  function play() {
    if (atEnd()) restart();
    playing = true;
    sync();
    tick();
  }
  function stepOnce() {
    pause();
    if (atEnd()) restart();
    apply(events[cursor++], true);
    sync();
  }
  function nextRound() { // up to and including the next round.completed (or the job's end)
    pause();
    if (atEnd()) restart();
    while (!atEnd()) {
      const e = events[cursor++];
      applyEvent(state, e);
      if (e.type === "round.completed" || (e.type === "job.status" && !LIVE.has(e.status))) break;
    }
    draw();
    sync();
  }
  function seek(k) {
    pause();
    state = initialState(job);
    cursor = 0;
    while (cursor < Math.min(k, events.length)) applyEvent(state, events[cursor++]);
    draw();
    sync();
  }

  playBtn.addEventListener("click", () => (playing ? pause() : play()));
  stepBtn.addEventListener("click", stepOnce);
  roundBtn.addEventListener("click", nextRound);
  stopBtn.addEventListener("click", () => { pause(); restart(); sync(); });
  endBtn.addEventListener("click", () => seek(events.length));
  scrub.addEventListener("input", () => seek(Number(scrub.value)));
  speedSel.addEventListener("change", () => {
    speed = Number(speedSel.value) || 1;
    if (playing && timer) { clearTimeout(timer); timer = setTimeout(tick, 60); } // apply the new speed now
  });
  const onKey = (ev) => {
    const t = ev.target && ev.target.closest ? ev.target : null;
    if (replayBar.hidden || ev.ctrlKey || ev.metaKey || ev.altKey) return;
    if (t && t.closest("input, select, textarea, [contenteditable]")) return; // typing, or the slider's own keys
    if (ev.key === " ") {
      if (t && t.closest("button, a")) return; // Space already presses a focused button
      ev.preventDefault();
      if (playing) pause(); else play();
    }
    else if (ev.key === "ArrowRight" && ev.shiftKey) { ev.preventDefault(); nextRound(); }
    else if (ev.key === "ArrowRight") { ev.preventDefault(); stepOnce(); }
  };
  document.addEventListener("keydown", onKey);

  replayBar.hidden = LIVE.has(state.status) || !events.length;
  sync();
  goLive();
  if (query.replay === "1" && !replayBar.hidden) { restart(); play(); }

  return () => {
    if (stopStream) stopStream();
    if (timer) clearTimeout(timer);
    document.removeEventListener("keydown", onKey);
    canvas.destroy();
    chart.destroy();
  };
}
