// #/memberships/{id} — one company in one group (PLATFORM-API §4.4, §5.4, §5.6). Two tabs:
// "Connect data" (node, column statistics, the agent's schema mapping, validation) and
// "Use model" (a form generated from the TaskSpec's input_schema → POST /predict).
// Only names and statistics are shown here: the node never sends a row.

import { api } from "../api.js";
import { errorBox, fieldErrors, h, statusPill } from "../app.js";

const STEPS = ["requested", "approved", "node_connected", "schema_proposed", "schema_confirmed", "ready", "training", "active"];
// one entry per STEPS status: what that step means for the company
const JOURNEY = [
  { title: "Join request", desc: "You asked to join this group." },
  { title: "Approved", desc: "The admin let you in." },
  { title: "Data connected", desc: "Your CSV was read on your computer." },
  { title: "Columns mapped", desc: "Your agent matched your columns to the model's inputs." },
  { title: "You confirmed", desc: "You checked the mapping." },
  { title: "Validated", desc: "Your data passed the checks." },
  { title: "Training", desc: "Your data trains the shared model. Rows stay with you." },
  { title: "Model ready", desc: "Use the shared model." },
];
const NEXT = {
  requested: "waiting for the admin to approve your request.",
  approved: "upload your data as a CSV file below. Only column statistics leave this computer.",
  node_connected: "your agent is reading your columns…",
  schema_proposed: "check the mapping your agent proposed, then click Confirm mapping.",
  schema_confirmed: "your agent is validating the data against the mapping…",
  ready: "all set. The admin starts training once enough companies are ready.",
  training: "the shared model is training with your data. No row leaves your company.",
  active: "Open the Use model tab to get predictions from the shared model.",
};
// "Fill example": the first patient of the Pima dataset, in the group's canonical units
const EXAMPLE = { pregnancies: 6, glucose: 148, blood_pressure: 72, bmi: 33.6, insulin: 0, age: 50 };

export async function render({ el, params }) {
  let m = null;
  let g = null;
  let tab = null;
  let liveJob = null;
  const body = h("div.stack");

  async function load() {
    m = await api.get(`/memberships/${encodeURIComponent(params.id)}`);
    g = await api.get(`/groups/${encodeURIComponent(m.group_id)}`);
    if (!tab) tab = m.status === "active" ? "use" : "connect";
    liveJob = null;
    if (m.status === "training") {
      const jobs = await api.get(`/groups/${encodeURIComponent(m.group_id)}/jobs`).catch(() => ({ items: [] }));
      liveJob = (jobs.items || []).find((j) => j.status === "running" || j.status === "queued") || null;
    }
    draw();
    // after a confirm the node validates the data and reports back: follow it until ready
    if (m.status === "schema_confirmed" || m.status === "training") setTimeout(() => { if (el.isConnected) load().catch(showError); }, 2300);
  }
  function showError(err) { body.replaceChildren(errorBox(err)); }

  function draw() {
    const cur = STEPS.indexOf(m.status);
    const finished = cur === STEPS.length - 1;
    const tracker = h("ol.journey", JOURNEY.map((s, i) => {
      const done = i < cur || (finished && i === cur);
      const now = i === cur && !done;
      return h("li", { class: done ? "done" : now ? "now" : "", "aria-current": now ? "step" : null },
        h("span.num", done ? "✓" : String(i + 1)),
        h("div", h("div.title", s.title), h("div.desc", s.desc)));
    }));
    const next = NEXT[m.status];
    const callout = next ? h("div.next", h("strong", finished ? "Done. " : "Now: "), next,
      liveJob ? [" ", h("a", { href: `#/jobs/${liveJob.id}` }, "Watch it live")] : null) : null;
    const tabs = h("div.tabs", { role: "tablist" }, [["connect", "Connect data"], ["use", "Use model"]].map(([k, label]) =>
      h("button.tab", { type: "button", role: "tab", "aria-selected": String(tab === k),
                        onclick: () => { tab = k; draw(); } }, label)));
    el.replaceChildren(
      h("div.stack",
        h("div.row", h("h1", g.name), statusPill(m.status), h("span.muted", m.company_name)),
        tracker, callout),
      tabs,
      tab === "connect" ? connectTab() : useTab());
  }

  // ------------------------------------------------------------ tab 1: connect data
  function connectTab() {
    const wrap = h("div.stack");
    const err = h("div");
    const inputs = g.task_spec.input_schema;
    const cols = (m.data && m.data.columns) || [];
    const editable = m.status === "schema_proposed";
    const past = STEPS.indexOf(m.status) > STEPS.indexOf("schema_proposed");

    wrap.append(err);
    if (m.status === "requested" || m.status === "rejected" || m.status === "left") return wrap;

    // Data source: a CSV read here in the browser; only its shape and statistics are sent
    const canUpload = ["approved", "node_connected", "schema_proposed"].includes(m.status);
    const upload = () => {
      const status = h("p.faint");
      const input = h("input", { type: "file", accept: ".csv,text/csv", hidden: true });
      const zone = h("label.dropzone", input,
        h("strong", m.data ? "Replace the file" : "Choose a CSV file"), h("span.faint", "or drop it here"));
      const take = async (file) => {
        if (!file) return;
        err.replaceChildren();
        status.textContent = `Reading ${file.name} on this computer…`;
        try {
          const table = parseCsv(await file.text());
          if (!table.columns.length || !table.rows.length) throw new Error("This file has no header row or no data rows.");
          status.textContent = `${table.rows.length} rows read. Sending column statistics to your agent…`;
          await api.post(`/memberships/${m.id}/data`, {
            file_name: file.name, row_count: table.rows.length, columns: table.columns, profile: profile(table) });
          await load();
        } catch (e) { status.textContent = ""; err.replaceChildren(errorBox(e)); }
      };
      input.onchange = () => take(input.files[0]);
      zone.ondragover = (e) => { e.preventDefault(); zone.classList.add("over"); };
      zone.ondragleave = () => zone.classList.remove("over");
      zone.ondrop = (e) => { e.preventDefault(); zone.classList.remove("over"); take(e.dataTransfer.files[0]); };
      return [zone, status,
        h("p.faint", "No file at hand? ", h("a", { href: "demo-data/rustavi_clinic.csv", download: "" }, "Download a sample hospital export"), ".")];
    };
    if (!m.data) {
      wrap.append(panel("Connect your data", [
        h("p.muted", "Upload a CSV export of your database. The file is read here, in your browser: " +
                     "rows never leave this computer. Only column names and statistics are sent."),
        canUpload ? upload() : h("p.faint", "Data can be connected once the admin approves you."),
      ]));
      return wrap;
    }
    wrap.append(panel("Data source", [
      facts([
        ["file", h("span.mono", m.node && m.node.connector ? m.node.connector.display : "—")],
        ["agent", h("span.mono", m.node && m.node.agent ? `${m.node.agent.provider} / ${m.node.agent.model}` : "—")],
        ["connected", h("span.mono", relTime(m.node && m.node.last_seen))],
      ]),
      canUpload ? upload() : null,
    ]));

    // Data profile
    const prof = (m.data && m.data.profile) || {};
    const v = (c, k) => (prof[c] && prof[c][k] !== undefined && prof[c][k] !== null ? prof[c][k] : "—");
    wrap.append(panel("Your data (never leaves your company)", [
      h("div", h("span.mono", m.data ? m.data.row_count : 0), " rows"),
      cols.length ? h("div.table-wrap", h("table.table",
        h("thead", h("tr", ["column", "min", "median", "mean", "max", "missing"].map((x) => h("th", x)))),
        h("tbody", cols.map((c) => h("tr.mono", h("td", c), ["min", "median", "mean", "max", "missing"].map((k) => h("td", v(c, k))))))))
        : h("p.faint", "No columns reported yet."),
      h("p.faint", "Only these statistics were sent. No row ever leaves your machine."),
    ]));

    // Schema mapping
    if (m.schema) {
      const colSelect = (value, name) => h("select.mono", { name, disabled: !editable },
        h("option", { value: "" }, "— not mapped —"),
        cols.map((c) => h("option", { value: c, selected: c === value }, c)));
      const rows = inputs.map((f) => {
        const mp = (m.schema.mapping || []).find((x) => x.feature === f.name) || {};
        return { f, source: colSelect((mp.source || [])[0] || "", `src_${f.name}`),
                 unit: h("input", { value: mp.unit ?? f.unit ?? "", disabled: !editable, style: "width:100px" }),
                 transform: h("input.mono", { value: mp.transform ?? "x", disabled: !editable, style: "width:140px" }) };
      });
      const label = colSelect(m.schema.label || "", "label");
      const form = h("form.stack", { onsubmit: async (e) => {
        e.preventDefault();
        const btn = form.querySelector("button[type=submit]");
        btn.disabled = true; btn.textContent = "…"; err.replaceChildren();
        try {
          await api.post(`/memberships/${m.id}/schema/confirm`, {
            label: label.value,
            mapping: rows.map((r) => ({ feature: r.f.name, source: r.source.value ? [r.source.value] : [],
                                        unit: r.unit.value, transform: r.transform.value })),
          });
          await load();
        } catch (e2) {
          if (!fieldErrors(form, e2)) err.replaceChildren(errorBox(e2));
          btn.disabled = false; btn.textContent = "Confirm mapping";
        }
      } },
        h("div.table-wrap", h("table.table",
          h("thead", h("tr", ["group feature", "your column(s)", "your unit", "transform"].map((x) => h("th", x)))),
          h("tbody",
            rows.map((r) => h("tr", h("td", h("span.mono", r.f.name), " ", h("span.faint", r.f.unit || "")),
                                h("td", r.source), h("td", r.unit), h("td", r.transform))),
            h("tr", h("td.muted", "label column"), h("td", { colspan: 3 }, label))))),
        editable ? h("div", h("button.btn.primary", { type: "submit" }, "Confirm mapping")) : null,
        past ? h("p.ok-text", "Confirmed") : null);
      wrap.append(panel([h("h2", "Schema mapping — proposed by your agent"), madeBy(m.schema.made_by)],
        [h("blockquote.quote", `“${m.schema.reasoning}”`), form]));
    }

    // Validation
    if (m.validation) {
      wrap.append(panel("Validation", h("ul.stack", { style: "list-style:none;margin:0;padding:0;gap:8px" },
        m.validation.checks.map((c) => h("li.row", { style: "gap:8px" },
          h("span", { class: c.ok ? "ok-text" : "bad-text" }, c.ok ? "✓" : "✗"),
          h("span.mono", c.name), h("span.muted", c.detail))))));
    }
    return wrap;
  }

  // ------------------------------------------------------------ tab 2: use model
  function useTab() {
    const versions = [...(g.versions || [])].sort((a, b) => b.version - a.version);
    if (!versions.length) {
      return panel(null, h("p.faint", "No model yet: the admin starts training once enough companies are ready."));
    }
    const spec = g.task_spec;
    let v = versions[0];
    const wrap = h("div.stack");
    const chips = h("div.row", { style: "gap:8px" });
    const result = h("div");
    const err = h("div");
    const factsSlot = h("div");

    function drawChips() {
      chips.replaceChildren(...versions.map((x) => h("button.btn.ghost.small.mono", {
        type: "button", style: x.id === v.id ? "border-color:var(--coord);color:var(--coord)" : null,
        onclick: () => { v = x; drawChips(); drawFacts(); },
      }, `v${x.version} · AUC ${auc(x).toFixed(3)}`)));
    }
    function drawFacts() {
      const p = spec.privacy || {};
      const priv = [p.secure_aggregation && "secure aggregation",
                    p.dp && p.dp.enabled && `differential privacy ε = ${p.dp.epsilon_per_round} per round`].filter(Boolean).join(" + ");
      factsSlot.replaceChildren(panel("Model facts", facts([
        ["AUC", h("span.mono", auc(v).toFixed(3))],
        ["trained by", h("span.mono", `${(v.contributors || []).length} companies`)],
        ["rounds", h("span.mono", spec.training.rounds)],
        ["privacy", priv || "none"],
        ["sha256", h("span.mono", (v.sha256 || "").slice(0, 12))],
      ])));
    }

    const fields = spec.input_schema.map((f) => {
      const numeric = f.type === "number" || f.type === "integer";
      const input = numeric
        ? h("input.mono", { type: "number", name: f.name, step: f.type === "integer" ? "1" : "any" })
        : f.type === "category"
          ? h("select", { name: f.name }, h("option", { value: "" }, "—"), (f.categories || []).map((c) => h("option", c)))
          : h("input", { name: f.name, disabled: true, placeholder: "coming soon" });
      return { f, input, el: h("div.field", h("label", h("span.mono", { style: "color:var(--text)" }, f.name), " ", f.unit || ""),
                               input, h("span.hint", numeric || f.type === "category" ? f.description || "" : "coming soon")) };
    });

    const form = h("form.stack", { onsubmit: async (e) => {
      e.preventDefault();
      const btn = form.querySelector("button[type=submit]");
      btn.disabled = true; btn.textContent = "…"; err.replaceChildren(); fieldErrors(form, null);
      const inputs = {};
      for (const { f, input } of fields) {
        if (input.disabled) continue;
        const raw = input.value;
        inputs[f.name] = f.type === "number" || f.type === "integer" ? (raw === "" ? "" : Number(raw)) : raw;
      }
      try {
        const r = await api.post(`/groups/${encodeURIComponent(m.group_id)}/predict`, { inputs, version_id: v.id });
        result.replaceChildren(resultPanel(r, inputs));
      } catch (e2) {
        result.replaceChildren();
        if (!fieldErrors(form, e2)) err.replaceChildren(errorBox(e2));
      }
      btn.disabled = false; btn.textContent = "Predict";
    } },
      err,
      h("div.fields", fields.map((x) => x.el)),
      h("div", h("button.btn.primary", { type: "submit" }, "Predict")));

    const fill = h("button.btn.ghost", { type: "button", onclick: () => {
      for (const { f, input } of fields) if (f.name in EXAMPLE) input.value = EXAMPLE[f.name];
    } }, "Fill example");

    function resultPanel(r, inputs) {
      const positive = spec.label && r.prediction === spec.label.positive;
      const expl = [...(r.explanation || [])].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
      const maxC = Math.max(...expl.map((e) => Math.abs(e.contribution)), 1e-4);
      return panel("Result", [
        h("div.row", { style: "align-items:flex-end;gap:24px" },
          h("div.mono", { class: positive ? "bad-text" : "ok-text", style: "font-size:40px;line-height:1" }, r.prediction),
          h("div.stack", { style: "flex:1;min-width:160px;gap:4px" },
            h("div.mono", `${Math.round(r.probability * 100)}%`),
            h("div.bar", h("span", { style: `width:${(r.probability * 100).toFixed(1)}%` })))),
        h("h2", "Why"),
        h("div.stack", { style: "gap:8px" }, expl.map((e) => {
          const w = (Math.abs(e.contribution) / maxC) * 50;
          const pos = e.contribution > 0;
          return h("div", { style: "display:grid;grid-template-columns:minmax(0,180px) 1fr;gap:12px;align-items:center" },
            h("span.mono", { style: "overflow:hidden;text-overflow:ellipsis;white-space:nowrap" },
              `${e.feature} = ${e.value ?? inputs[e.feature] ?? "—"}`),
            h("div", { style: "position:relative;height:12px" },
              h("div", { style: "position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--border-strong)" }),
              h("div", { style: `position:absolute;top:0;height:12px;border-radius:4px;width:${w}%;` +
                                (pos ? "left:50%;background:var(--danger)" : "right:50%;background:var(--ok)") })));
        })),
        h("p.faint", `Computed by the shared model v${v.version}. Your inputs are not stored.`),
      ]);
    }

    drawChips();
    drawFacts();
    wrap.append(chips, panel([h("h2", "Predict"), fill], form), result, factsSlot);
    return wrap;
  }

  el.replaceChildren(h("p.loading", "Loading…"));
  try { await load(); } catch (err) { el.replaceChildren(errorBox(err)); }
}

// ------------------------------------------------------------ helpers

function panel(title, children) {
  const head = title === null ? null
    : Array.isArray(title) ? h("div.panel-head", title)
    : h("h2", title);
  return h("section.panel.stack", head, children);
}

/** A small RFC 4180 reader: quoted fields, doubled quotes, \r\n or \n line ends. */
function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (c === '"') quoted = false;
      else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field); field = "";
      if (row.some((x) => x.trim() !== "")) rows.push(row);
      row = [];
    } else field += c;
  }
  row.push(field);
  if (row.some((x) => x.trim() !== "")) rows.push(row);
  const [header = [], ...body] = rows;
  return { columns: header.map((x) => x.trim()), rows: body };
}

/** Per-column statistics: what the Schema agent sees instead of the rows. */
function profile({ columns, rows }) {
  const round = (x) => Math.round(x * 1000) / 1000;
  const out = {};
  columns.forEach((col, j) => {
    const raw = rows.map((r) => (r[j] ?? "").trim());
    const present = raw.filter((x) => x !== "" && x.toLowerCase() !== "na" && x.toLowerCase() !== "null");
    const nums = present.map(Number);
    const missing = raw.length - present.length;
    if (!present.length || nums.some((x) => !Number.isFinite(x))) { out[col] = { missing }; return; }
    nums.sort((a, b) => a - b);
    const mid = Math.floor(nums.length / 2);
    out[col] = { min: nums[0], median: round(nums.length % 2 ? nums[mid] : (nums[mid - 1] + nums[mid]) / 2),
                 mean: round(nums.reduce((s, x) => s + x, 0) / nums.length), max: nums[nums.length - 1], missing };
  });
  return out;
}

function facts(items) {
  return h("dl.facts", items.flatMap(([k, v]) => [h("dt", k), h("dd", v)]));
}

function madeBy(v) {
  return h("span.made-by", { class: v === "llm" ? "llm" : "" }, v === "llm" ? "LLM" : "rule");
}

function auc(version) {
  const a = version.metrics ? version.metrics.auc : version.auc;
  return typeof a === "number" ? a : NaN;
}

function relTime(iso) {
  if (!iso) return "—";
  const s = Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 1000));
  if (s < 60) return `${s} s ago`;
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
}
