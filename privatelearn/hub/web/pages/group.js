// #/groups/{id} — the admin's view of one group (PLATFORM-API §5.3–5.6, §5.8): start a
// training or fine-tune job, the members with their membership and node status, demo-node
// buttons, model versions side by side, and the job history.

import { api } from "../api.js";
import { errorBox, h, statusPill } from "../app.js";

const fmt = (s) => (s ? new Date(s).toISOString().slice(0, 16).replace("T", " ") : "—");
const auc = (v) => (v.metrics && typeof v.metrics.auc === "number" ? v.metrics.auc : NaN);

export async function render({ el, params, navigate }) {
  const id = params.id;
  const err = h("div");
  // replaceChildren() would print a skipped (null) part as the text "null"
  const show = (...parts) => el.replaceChildren(...parts.filter((p) => p !== null && p !== undefined && p !== false));

  async function load() {
    const [g, jobs] = await Promise.all([api.get(`/groups/${encodeURIComponent(id)}`),
                                         api.get(`/groups/${encodeURIComponent(id)}/jobs`)]);
    draw(g, jobs.items || []);
  }

  function draw(g, jobs) {
    const members = g.members || [];
    const versions = [...(g.versions || [])].sort((a, b) => a.version - b.version);
    const spec = g.task_spec || {};
    const minP = (spec.training && spec.training.min_participants) || 2;
    const readyCount = members.filter((m) => m.status === "ready" || m.status === "active").length;
    const newReady = members.filter((m) => m.status === "ready").length;
    const trainReason = versions.length ? "A version already exists: use fine-tune."
      : readyCount < minP ? `Needs ${minP} ready members (${readyCount} now).` : "";
    const ftReason = !versions.length ? "Train a first version before fine-tuning."
      : newReady < 1 ? "No new member is ready yet." : "";
    const running = jobs.find((j) => j.status === "running" || j.status === "queued");
    const coName = (cid) => (members.find((m) => m.company_id === cid) || {}).company_name || cid;
    const maxAuc = Math.max(...versions.map(auc).filter(Number.isFinite), 1e-4);

    const jobButton = (label, cls, kind, reason) => {
      const b = h(`button.${cls}`, { type: "button", disabled: !!reason, title: reason || null, onclick: async () => {
        b.disabled = true; b.textContent = "…"; err.replaceChildren();
        try { const job = await api.post(`/groups/${encodeURIComponent(id)}/jobs`, { kind }); navigate(`/jobs/${job.id}`); }
        catch (e) { err.replaceChildren(errorBox(e)); b.disabled = false; b.textContent = label; }
      } }, label);
      return h("div.row", b, reason ? h("span.faint", reason) : null);
    };
    const demoButton = (m) => {
      const online = m.node && m.node.status === "online";
      const label = online ? "Stop demo node" : "Start demo node";
      const b = h("button.btn.ghost.small", { type: "button", onclick: async () => {
        b.disabled = true; b.textContent = "…"; err.replaceChildren();
        try { await api.post(`/demo/nodes/${online ? "stop" : "start"}`, { membership_id: m.id }); await load(); }
        catch (e) { err.replaceChildren(errorBox(e)); b.disabled = false; b.textContent = label; }
      } }, label);
      return b;
    };

    show(
      h("div.stack", { style: "gap:8px" },
        h("div.row", h("h1", g.name), statusPill(g.status),
          g.status === "draft" ? h("a", { href: `#/groups/${g.id}/edit`, style: "color:var(--coord)" }, "Edit spec") : null),
        spec.question ? h("p.muted", `“${spec.question}”`) : null),
      err,
      running ? h("div.row", { class: "panel", style: "border-color:var(--coord);color:var(--coord)" },
        "Training in progress", h("a", { href: `#/jobs/${running.id}`, style: "color:var(--coord)" }, "Watch live")) : null,
      h("section.panel.stack", h("h2", "Actions"),
        jobButton("Start training", "btn.primary", "train", trainReason),
        jobButton("Fine-tune with new members", "btn.ghost", "finetune", ftReason)),
      h("section.panel.stack", h("h2", "Members"),
        members.length === 0 ? h("p.faint", "No company has joined yet.") :
          h("div.table-wrap", h("table.table",
            h("thead", h("tr", ["company", "membership", "node", "rows", "requested", ""].map((x) => h("th", x)))),
            h("tbody", members.map((m) => h("tr",
              h("td", m.company_name),
              h("td", statusPill(m.status)),
              h("td", m.node ? statusPill(m.node.status) : "—"),
              h("td.mono", (m.contribution && m.contribution.rows) || 0),
              h("td.mono", { style: "white-space:nowrap" }, fmt(m.requested_at)),
              h("td", demoButton(m)))))))),
      h("section.panel.stack", h("h2", "Versions"),
        versions.length === 0 ? h("p.faint", "No versions yet.") : [
          h("div.row", { style: "align-items:stretch" }, versions.map((v) => h("div.stack", {
            style: "background:var(--panel-2);border:1px solid var(--border);border-radius:8px;padding:16px;gap:8px;min-width:200px;flex:1",
          },
            h("div.mono.muted", `v${v.version}`),
            h("div.mono", { style: "font-size:28px;line-height:1.1" }, auc(v).toFixed(3), h("span.faint", " AUC")),
            h("div.muted", `${(v.contributors || []).length} contributors: ${(v.contributors || []).map((c) => coName(c.company_id)).join(", ")}`),
            v.parent_version_id ? h("div.faint", `fine-tuned from v${(versions.find((x) => x.id === v.parent_version_id) || {}).version ?? "?"}`) : null,
            h("div.mono.faint", `${(v.sha256 || "").slice(0, 12)} · ${fmt(v.created_at)}`)))),
          versions.length >= 2 ? h("div.stack", { style: "gap:8px" }, versions.map((v) => h("div.row", { style: "gap:8px;flex-wrap:nowrap" },
            h("span.mono", { style: "width:32px" }, `v${v.version}`),
            h("div.bar", { style: "flex:1" }, h("span", { style: `width:${(auc(v) / maxAuc) * 100}%` })),
            h("span.mono", auc(v).toFixed(3))))) : null,
        ]),
      h("section.panel.stack", h("h2", "Jobs history"),
        jobs.length === 0 ? h("p.faint", "No jobs yet.") :
          h("div.table-wrap", h("table.table",
            h("thead", h("tr", ["kind", "status", "rounds", "started", "finished"].map((x) => h("th", x)))),
            h("tbody", jobs.map((j) => h("tr",
              h("td", h("a.mono", { href: `#/jobs/${j.id}` }, j.kind)),
              h("td", statusPill(j.status)),
              h("td.mono", `${j.round ?? 0}/${j.rounds ?? "?"}`),
              h("td.mono", { style: "white-space:nowrap" }, fmt(j.started_at)),
              h("td.mono", { style: "white-space:nowrap" }, fmt(j.finished_at)))))))));
  }

  el.replaceChildren(h("p.loading", "Loading…"));
  try { await load(); } catch (e) { el.replaceChildren(errorBox(e)); }
}
