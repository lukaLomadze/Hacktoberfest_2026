// Decision cards (DESIGN.md §7.3, for the platform): one card per round, built from that
// round's round.decision (strategy, who took part, why) and round.completed (AUC, ε).

import { h } from "../app.js";
import { nameOf } from "./state.js";

const fmt = (v, d = 3) => (typeof v === "number" && Number.isFinite(v) ? v.toFixed(d) : "—");

export function createDecisions() {
  const list = h("ol.live-decisions");
  const el = h("section.panel", { "aria-label": "Coordinator decisions" },
    h("div.live-head", h("h2", "Decisions"), h("span.faint", "each round the coordinator picks a strategy")), list);

  function card(state, d) {
    const excluded = d.excluded || [];
    return h("li.decision", { class: excluded.length ? "has-excluded" : null },
      h("div.decision-row",
        d.strategy ? h("span.badge", { class: `badge-${d.strategy}` }, d.strategy) : null,
        h("span.mono", `round ${d.round}`),
        d.decided_by ? h("span.feed-tag", { class: `made-${d.decided_by}` }, d.decided_by) : null,
        excluded.map((id) => h("span.decision-excluded", `excluded ${nameOf(state, id)}`)),
        d.applied === false ? h("span.decision-excluded", "round discarded") : null),
      d.reasoning ? h("p.decision-text", d.reasoning) : null,
      d.completed ? h("p.mono.decision-meta",
        `AUC ${fmt(d.auc_before)} → ${fmt(d.auc_after)} · ε ${fmt(d.eps_total, 1)}`) : h("p.faint", "training…"));
  }

  function render(state) {
    const rounds = [...state.decisions.values()].sort((a, b) => a.round - b.round);
    list.replaceChildren(...(rounds.length ? rounds.map((d) => card(state, d))
      : [h("li.faint", "No round decided yet.")]));
  }

  return { el, render };
}
