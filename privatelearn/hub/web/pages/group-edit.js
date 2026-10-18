// #/groups/new and #/groups/{id}/edit — the admin writes a group's TaskSpec (PLATFORM-API
// §4.3, §5.3). "Draft with agent" asks the architect agent (POST /groups/draft-spec) to
// fill the whole spec from a plain question; the admin edits the question, basics and task,
// saves a draft, publishes. Inputs, model, training and privacy are not edited here: they
// keep the agent's draft (or the defaults below), and the agents settle them during the job.
// The spec is editable only while the group is a draft.

import { api } from "../api.js";
import { errorBox, h } from "../app.js";

const TASK_TYPES = ["binary_classification", "multiclass_classification", "regression", "image_classification", "text_generation"];

const blankSpec = () => ({
  question: "", task_type: "binary_classification",
  label: { name: "", classes: ["no", "yes"], positive: "yes" },
  input_schema: [],
  model: { type: "logistic", hyperparams: { epochs: 4, lr: 0.1 } },
  training: { rounds: 4, min_participants: 2, aggregation: "adaptive" },
  privacy: { secure_aggregation: true, dp: { enabled: true, epsilon_per_round: 1.5, delta: 1e-5 } },
  metric: "auc",
});

export async function render({ el, params, navigate }) {
  const id = params.id || null;
  let name = "";
  let description = "";
  let spec = blankSpec();
  let status = "draft";
  let notes = "";
  let draft = null;

  el.replaceChildren(h("p.loading", "Loading…"));
  try {
    const g = id ? await api.get(`/groups/${encodeURIComponent(id)}`) : null;
    if (g) { name = g.name; description = g.description || ""; spec = g.task_spec || blankSpec(); status = g.status; }
  } catch (err) { el.replaceChildren(errorBox(err)); return; }

  const ro = status !== "draft";
  const err = h("div");
  // replaceChildren() would print a skipped (null) part as the text "null"
  const show = (...parts) => el.replaceChildren(...parts.filter((p) => p !== null && p !== undefined && p !== false));

  // input/select bound to a getter/setter on the spec; text edits do not redraw the page
  const bind = (tag, get, set, attrs = {}) => {
    const input = h(tag, { ...attrs, disabled: ro || attrs.disabled });
    if (input.type === "checkbox") { input.checked = !!get(); input.onchange = () => { set(input.checked); draw(); }; }
    else { input.value = get() ?? ""; input.oninput = () => set(input.type === "number" ? Number(input.value) : input.value); }
    return input;
  };
  const select = (options, get, set, attrs = {}, disabledOf = () => false, labelOf = (o) => o) => {
    const s = h("select", { ...attrs, disabled: ro },
      options.map((o) => h("option", { value: o, disabled: disabledOf(o), selected: o === get() }, labelOf(o))));
    s.onchange = () => { set(s.value); draw(); };
    return s;
  };
  const field = (label, input, hint) => h("div.field", h("label", label), input, hint ? h("span.hint", hint) : null);
  const panel = (title, ...children) => h("section.panel.stack", h("h2", title), ...children);

  function draw() {
    const draftBtn = h("button.btn.primary", { type: "button", disabled: ro, onclick: async () => {
      draftBtn.disabled = true; draftBtn.textContent = "…"; err.replaceChildren();
      try {
        draft = await api.post("/groups/draft-spec", { question: spec.question, notes });
        spec = draft.task_spec;
        if (!name && spec.question) name = spec.question.replace(/\?\s*$/, "").slice(0, 60);
        draw();
      } catch (e) { err.replaceChildren(errorBox(e)); draftBtn.disabled = false; draftBtn.textContent = "Draft with agent"; }
    } }, "Draft with agent");

    show(
      h("h1", id ? `Edit ${name || id}` : "New group"),
      ro ? h("p", { style: "color:var(--coord)" }, "Published groups can't change their spec.") : null,
      err,
      h("form.stack", { onsubmit: (e) => e.preventDefault() },
        panel("Describe what the model should answer",
          field("Question", bind("input", () => spec.question, (v) => { spec.question = v; },
            { name: "question", placeholder: "Does this patient have diabetes?", style: "font-size:16px" })),
          field("Notes (optional)", bind("textarea", () => notes, (v) => { notes = v; }, { rows: 3 })),
          h("div", draftBtn),
          draft ? h("div.stack", { style: "gap:8px" },
            h("span.made-by", { class: draft.made_by === "llm" ? "llm" : "" }, draft.made_by === "llm" ? "LLM" : "rule"),
            h("blockquote.quote", `“${draft.reasoning}”`)) : null),
        panel("Basics", h("div.fields",
          field("Name", bind("input", () => name, (v) => { name = v; }, { name: "name" })),
          field("Description", bind("input", () => description, (v) => { description = v; }, { name: "description" })))),
        panel("Task", h("div.fields",
          field("Task type", select(TASK_TYPES, () => spec.task_type, (v) => { spec.task_type = v; })),
          field("Label name", bind("input.mono", () => spec.label.name, (v) => { spec.label.name = v; }, { name: "task_spec.label.name" })),
          field("Classes (comma-separated)", (() => {
            const i = bind("input.mono", () => (spec.label.classes || []).join(","),
              (v) => { spec.label.classes = v.split(",").map((x) => x.trim()); });
            i.onchange = () => draw();
            return i;
          })()),
          field("Positive class", select((spec.label.classes || []).filter(Boolean), () => spec.label.positive,
            (v) => { spec.label.positive = v; }))))),
      h("div.footer-bar",
        actionButton("Save draft", "btn.ghost", false),
        actionButton("Publish", "btn.primary", true)));
  }

  function actionButton(label, cls, publish) {
    const b = h(`button.${cls}`, { type: "button", disabled: ro,
                                   title: ro ? "Published groups can't change their spec." : null, onclick: async () => {
      b.disabled = true; b.textContent = "…"; err.replaceChildren();
      el.querySelectorAll(".field-error").forEach((n) => n.remove());
      try {
        const body = { name, description, task_spec: spec };
        const gid = id ? (await api.patch(`/groups/${encodeURIComponent(id)}`, body), id) : (await api.post("/groups", body)).id;
        if (publish) { await api.post(`/groups/${encodeURIComponent(gid)}/publish`); navigate(`/groups/${gid}`); }
        else if (!id) navigate(`/groups/${gid}/edit`);
        else { b.textContent = "Saved"; setTimeout(() => { b.disabled = false; b.textContent = label; }, 1200); return; }
      } catch (e) {
        const input = e && e.field ? el.querySelector(`[name="${CSS.escape(e.field)}"]`) : null;
        if (input) { input.insertAdjacentElement("afterend", h("p.field-error", e.message)); input.focus(); }
        else err.replaceChildren(errorBox(e));
        b.disabled = false; b.textContent = label;
      }
    } }, label);
    return b;
  }

  draw();
}
