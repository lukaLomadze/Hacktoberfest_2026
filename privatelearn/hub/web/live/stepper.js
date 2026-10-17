// The job stepper: one chip per job.step, in the order GET /meta gives them (§2), so a new
// step added on the server shows up without a frontend change. A step the meta list does not
// know (an older cached meta) is appended and shown as text.

import { h } from "../app.js";

const FALLBACK = ["waiting_for_nodes", "key_exchange", "schema_negotiation", "normalization", "training",
                  "evaluating", "published"];

export function createStepper(steps) {
  const order = Array.isArray(steps) && steps.length ? [...steps] : FALLBACK;
  const el = h("ol.live-stepper", { "aria-label": "Job progress" });

  function render(state) {
    const list = order.includes(state.step) || !state.step ? order : [...order, state.step];
    const current = state.step ? list.indexOf(state.step) : -1;
    const finished = state.status === "succeeded";
    const failed = state.status === "failed" || state.status === "cancelled";
    el.replaceChildren(...list.map((step, i) => {
      let tone = "todo";
      if (i < current || (finished && i <= current)) tone = "done";
      else if (i === current) tone = failed ? "failed" : "now";
      const label = step === "training" && state.rounds_total
        ? `training ${Math.min(state.round, state.rounds_total)}/${state.rounds_total}`
        : step.replaceAll("_", " ");
      return h("li.live-step", { class: `step-${tone}`, "aria-current": tone === "now" ? "step" : null, title: step },
        h("span.step-dot", { "aria-hidden": "true" }), h("span.step-label", label));
    }));
  }

  return { el, render };
}
