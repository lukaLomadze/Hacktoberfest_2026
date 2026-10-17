// The live view's model: a pure reducer over PLATFORM-API §6 events.
//
// Everything on #/jobs/{id} is drawn from this state, and this state comes only from events
// (plus the Job object for the starting roster). Catch-up (/events), the live SSE stream and
// a replay all go through applyEvent, so they always draw the same picture.

export const AGENT_LETTERS = [["schema", "S"], ["planner", "P"], ["trainer", "T"], ["challenger", "C"]];

/** A fresh state for this job. `job` is GET /jobs/{id}; only its roster and header are used. */
export function initialState(job) {
  const participants = new Map();
  for (const p of job.participants || []) {
    participants.set(p.node_id || p.membership_id, {
      node_id: p.node_id || p.membership_id,
      membership_id: p.membership_id,
      company_name: p.company_name || p.node_id || p.membership_id,
      color: p.color || null,
      state: "waiting",
      detail: "",
      acted: new Map(), // round -> Set(agent) — lights the S/P/T/C chips
    });
  }
  return {
    job_id: job.id,
    kind: job.kind,
    rounds_total: job.rounds || null,
    status: "queued",
    error: null,
    step: null,
    round: 0,
    participants,
    messages: [], // agent.message events, in order
    decisions: new Map(), // round -> {decision fields, completed fields}
    auc: [], // [{round, value}] — round 0 is the starting model
    loss: [],
    eps_total: 0,
    published: null,
    unknown: [], // events of a type this page does not know, shown as text
    last_seq: 0,
  };
}

function participant(state, nodeId, e) {
  if (!nodeId || nodeId === "coord") return null;
  let p = state.participants.get(nodeId);
  if (!p) {
    // an event names a participant the Job did not list (a late joiner): add it, don't drop it
    p = { node_id: nodeId, membership_id: e.membership_id || null,
          company_name: (e.from && e.from.company_name) || nodeId, color: null, state: "waiting",
          detail: "", acted: new Map() };
    state.participants.set(nodeId, p);
  }
  return p;
}

function decision(state, round) {
  if (!state.decisions.has(round)) state.decisions.set(round, { round });
  return state.decisions.get(round);
}

/**
 * Apply one event in place and return what changed, so the page redraws only that part:
 * a set of "header" | "stepper" | "canvas" | "feed" | "decisions" | "chart" | "published".
 * Events at or below last_seq are ignored (a reconnect may resend them).
 */
export function applyEvent(state, e) {
  const changed = new Set();
  if (!e || typeof e !== "object") return changed;
  if (typeof e.seq === "number") {
    if (e.seq <= state.last_seq) return changed;
    state.last_seq = e.seq;
  }
  if (typeof e.round === "number" && e.round > state.round) {
    state.round = e.round;
    changed.add("header").add("canvas");
  }
  switch (e.type) {
    case "job.status":
      state.status = e.status;
      state.error = e.error || null;
      changed.add("header").add("stepper");
      break;
    case "job.step":
      state.step = e.step;
      if (state.status === "queued") state.status = "running";
      changed.add("stepper").add("header");
      break;
    case "participant.state": {
      const p = participant(state, e.node_id, e);
      if (p) {
        p.state = e.state;
        p.detail = e.detail || "";
        changed.add("canvas");
      }
      break;
    }
    case "agent.message": {
      state.messages.push(e);
      const from = e.from || {};
      const p = participant(state, from.node_id, e);
      if (p && from.agent) {
        if (!p.acted.has(e.round)) p.acted.set(e.round, new Set());
        p.acted.get(e.round).add(from.agent);
        changed.add("canvas");
      }
      changed.add("feed");
      break;
    }
    case "packet":
      changed.add("packet"); // drawn as an animation; nothing to keep
      break;
    case "round.decision":
      Object.assign(decision(state, e.round), {
        strategy: e.strategy, participants: e.participants || [], excluded: e.excluded || [],
        reasoning: e.reasoning || "", decided_by: e.decided_by || "rule",
      });
      changed.add("decisions");
      break;
    case "round.completed": {
      Object.assign(decision(state, e.round), {
        auc_before: e.auc_before, auc_after: e.auc_after, loss: e.loss, eps_total: e.eps_total,
        applied: e.applied, completed: true,
      });
      if (!state.auc.length && typeof e.auc_before === "number") state.auc.push({ round: e.round - 1, value: e.auc_before });
      if (typeof e.auc_after === "number") state.auc.push({ round: e.round, value: e.auc_after });
      if (typeof e.loss === "number") state.loss.push({ round: e.round, value: e.loss });
      if (typeof e.eps_total === "number") state.eps_total = e.eps_total;
      changed.add("decisions").add("chart").add("header");
      break;
    }
    case "model.published":
      state.published = { version_id: e.version_id, version: e.version, metrics: e.metrics || {} };
      changed.add("published").add("chart");
      break;
    default:
      // a newer server may send event types this page does not know: show them as text
      state.unknown.push(e);
      changed.add("feed");
  }
  return changed;
}

/** Display name for a node id ("coord" → "Coordinator"). */
export function nameOf(state, nodeId) {
  if (nodeId === "coord") return "Coordinator";
  if (nodeId === "*") return "everyone";
  const p = state.participants.get(nodeId);
  return p ? p.company_name : String(nodeId);
}
