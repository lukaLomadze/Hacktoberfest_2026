// #/jobs/{id} — the live process view (PLATFORM-API §5.5, §6). It is driven only by §6
// events: GET /jobs/{id}/events to catch up, then the SSE stream while the job runs. A
// finished job can be replayed from its event history (?replay=1 starts the replay at once),
// which is also how the view works on the mock event list.

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

const STYLE_ID = "page-job-style";
const STYLE = `
.page[data-page="job"] .job-head { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.page[data-page="job"] .job-head h1 { overflow-wrap: anywhere; }
.page[data-page="job"] .job-facts { display: flex; gap: 16px; flex-wrap: wrap; color: var(--text-dim); }
.page[data-page="job"] .job-facts b { color: var(--text); font-family: var(--font-data); font-weight: 600; }
.page[data-page="job"] .zero { color: var(--ok); font-family: var(--font-data); font-weight: 600; }
.page[data-page="job"] .job-controls { display: flex; align-items: center; gap: 8px; margin-left: auto; flex-wrap: wrap; }
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
.page[data-page="job"] .live-stepper { list-style: none; margin: 0; padding: 0; display: flex; gap: 4px; flex-wrap: wrap; }
.page[data-page="job"] .live-step { display: flex; align-items: center; gap: 8px; padding: 4px 12px; border: 1px solid var(--border); border-radius: 999px; color: var(--text-faint); font-size: 12px; }
.page[data-page="job"] .step-dot { width: 8px; height: 8px; border-radius: 999px; border: 1px solid var(--border-strong); }
.page[data-page="job"] .step-done { color: var(--text-dim); }
.page[data-page="job"] .step-done .step-dot { background: var(--ok); border-color: var(--ok); }
.page[data-page="job"] .step-now { color: var(--text); border-color: var(--border-strong); background: var(--panel); }
.page[data-page="job"] .step-now .step-dot { background: var(--text); border-color: var(--text); }
.page[data-page="job"] .step-failed { color: var(--danger); border-color: var(--danger); }
.page[data-page="job"] .step-failed .step-dot { background: var(--danger); border-color: var(--danger); }
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
  const replayBtn = h("button.btn.ghost", { type: "button" }, "Replay");
  const endBtn = h("button.btn.ghost", { type: "button", hidden: true }, "Skip to end");
  const publishedSlot = h("div.panel.published", { hidden: true, role: "status" });
  const errorSlot = h("div");

  const title = group ? `${group.name}: ${job.kind === "finetune" ? "fine-tune" : "training"}` : `Job ${job.id}`;
  el.replaceChildren(
    h("div.job-head",
      h("h1", title), statusSlot, liveSlot,
      h("div.job-controls", replayBtn, endBtn)),
    h("div.job-facts",
      h("span", "round ", roundSlot, ` / ${job.rounds ?? "—"}`),
      h("span", "ε spent ", epsSlot),
      h("span", "participants ", h("b", String((job.participants || []).length))),
      h("span", h("span.zero", "0"), " data rows left any company"),
      h("span.faint.mono", job.id)),
    stepper.el, errorSlot, publishedSlot,
    h("div.job-grid",
      h("div.job-main",
        h("section.panel.stage", h("div.stage-box", canvas.el),
          h("p.stage-caption", "Each company runs its own agents (Schema · Planner · Trainer · Challenger); a chip lights " +
            "when that agent acts this round. Packets are masked updates in, the new model out.")),
        decisions.el, chart.el),
      h("div.job-rail", feed.el)));
  canvas.mount();
  chart.mount();

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
        publishedSlot.replaceChildren(h("span", "Published ", h("b", `v${p.version ?? "?"}`)),
          typeof auc === "number" ? h("span", "AUC ", h("b", auc.toFixed(3))) : null, link);
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
    if (!LIVE.has(job.status) && !LIVE.has(state.status)) return;
    liveSlot.hidden = false;
    liveSlot.textContent = "live";
    stopStream = api.stream(job.id, (e) => {
      if (typeof e.seq === "number" && e.seq <= state.last_seq) return; // resent after a reconnect
      events.push(e); // so a replay after the job finishes includes what streamed live
      apply(e, true);
      if (!LIVE.has(state.status)) {
        liveSlot.textContent = "finished";
        replayBtn.hidden = false;
      }
    }, { since: state.last_seq });
  }

  // ---------------------------------------------------------------- replay (finished jobs, mocks)
  let timer = null;
  let cursor = 0;
  let replaying = false;

  function stopReplay() {
    if (timer) clearTimeout(timer);
    timer = null;
    replaying = false;
    replayBtn.textContent = cursor && cursor < events.length ? "Resume" : "Replay";
    endBtn.hidden = !(cursor && cursor < events.length);
  }
  function step() {
    if (cursor >= events.length) {
      cursor = 0;
      stopReplay();
      return;
    }
    const e = events[cursor++];
    apply(e, true);
    timer = setTimeout(step, PACE[e.type] ?? PACE.default);
  }
  replayBtn.addEventListener("click", () => {
    if (replaying) { stopReplay(); return; }
    if (cursor === 0) { // start over from an empty picture
      state = initialState(job);
      draw();
    }
    replaying = true;
    replayBtn.textContent = "Pause";
    endBtn.hidden = false;
    step();
  });
  endBtn.addEventListener("click", () => {
    if (timer) clearTimeout(timer);
    while (cursor < events.length) applyEvent(state, events[cursor++]);
    draw();
    cursor = 0;
    stopReplay();
  });

  replayBtn.hidden = LIVE.has(state.status) || !events.length;
  goLive();
  if (query.replay === "1" && !replayBtn.hidden) replayBtn.click();

  return () => {
    if (stopStream) stopStream();
    if (timer) clearTimeout(timer);
    canvas.destroy();
    chart.destroy();
  };
}
