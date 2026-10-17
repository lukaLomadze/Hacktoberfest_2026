// The agent conversation feed: one bubble per agent.message (§6), newest at the bottom.
// Each bubble says who spoke (company · agent), to whom, in which round, and whether the
// decision was the LLM's or the rule fallback's (made_by, §9). Event types this page does not
// know are shown as a plain line instead of being dropped.

import { h } from "../app.js";
import { nameOf } from "./state.js";

const MAX_BUBBLES = 400; // keep the DOM small on long jobs; older messages stay in the state

export function createFeed() {
  const list = h("ol.live-feed-list");
  const el = h("section.panel.live-feed", { "aria-label": "Agent conversation" },
    h("div.live-head", h("h2", "Agent conversation"), h("span.faint", "what each agent decided and why")),
    list);
  let shown = 0;
  let unknownShown = 0;

  function bubble(state, e) {
    const from = e.from || {};
    const who = from.node_id === "coord" || from.agent === "coordinator" ? "Coordinator" : nameOf(state, from.node_id);
    const p = state.participants.get(from.node_id);
    const dot = h("span.feed-dot", { "aria-hidden": "true" });
    if (p && p.color) dot.style.background = p.color; // the server's per-company color (§4.5)
    else if (who === "Coordinator") dot.classList.add("coord");
    return h("li.feed-item", { class: who === "Coordinator" ? "from-coord" : null },
      h("div.feed-meta", dot, h("b", who), from.agent && from.agent !== "coordinator" ? h("span.feed-agent", from.agent) : null,
        h("span.faint", `→ ${nameOf(state, e.to)}`), h("span.feed-round.faint", `r${e.round ?? 0}`),
        e.made_by ? h("span.feed-tag", { class: `made-${e.made_by}` }, e.made_by) : null),
      h("p.feed-text", e.text || ""));
  }

  function render(state) {
    const stick = list.scrollHeight - list.scrollTop - list.clientHeight < 24; // follow only if at the bottom
    if (state.messages.length < shown) { // a replay restarted
      list.replaceChildren();
      shown = 0;
      unknownShown = 0;
    }
    for (; shown < state.messages.length; shown++) list.append(bubble(state, state.messages[shown]));
    for (; unknownShown < state.unknown.length; unknownShown++) {
      const e = state.unknown[unknownShown];
      list.append(h("li.feed-item.feed-unknown", h("span.faint", `r${e.round ?? 0} · ${e.type || "event"}`),
        " ", h("code", JSON.stringify(e).slice(0, 200))));
    }
    while (list.childElementCount > MAX_BUBBLES) list.firstElementChild.remove();
    if (!list.childElementCount) list.append(h("li.faint.feed-empty", "No agent messages yet."));
    else list.querySelector(".feed-empty")?.remove();
    if (stick) list.scrollTop = list.scrollHeight;
  }

  return { el, render };
}
