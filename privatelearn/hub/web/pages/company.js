// #/home — a company's start page (PLATFORM-API §5.3–5.4): its memberships with their
// status, and every group it can see, with a "Request to join" button where joining is
// possible. Joining refreshes both lists in place; the admin approves the request (E3).

import { api } from "../api.js";
import { errorBox, h, setSession, statusPill } from "../app.js";

// The server's join rules (hub/server/routes/memberships.py): only these group statuses take
// members, and a company may ask again only once its earlier membership is closed. The server
// enforces both; the page uses them to decide which button to show, not to grant anything.
const JOINABLE_GROUP_STATUS = new Set(["open", "training", "trained"]);
const CLOSED_MEMBERSHIP_STATUS = new Set(["rejected", "left"]);

const STYLE_ID = "page-company-style";
const STYLE = `
.page[data-page="company"] .section { display: flex; flex-direction: column; gap: 12px; }
.page[data-page="company"] .section-head { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
.page[data-page="company"] .flash {
  color: var(--text-dim); border: 1px solid var(--border-strong); border-radius: 8px; padding: 8px 12px;
}
.page[data-page="company"] .table-wrap { overflow-x: auto; }
.page[data-page="company"] table { width: 100%; border-collapse: collapse; }
.page[data-page="company"] th {
  text-align: left; font-size: 12px; font-weight: 500; color: var(--text-faint);
  padding: 8px 12px; border-bottom: 1px solid var(--border);
}
.page[data-page="company"] td { padding: 8px 12px; border-bottom: 1px solid var(--border); vertical-align: top; }
.page[data-page="company"] tr:last-child td { border-bottom: 0; }
.page[data-page="company"] .cards {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(min(320px, 100%), 1fr)); gap: 16px;
}
.page[data-page="company"] .card { display: flex; flex-direction: column; gap: 12px; min-width: 0; }
.page[data-page="company"] .card-title { display: flex; align-items: flex-start; justify-content: space-between; gap: 8px; }
.page[data-page="company"] .card-title h3 { font-size: 14px; font-weight: 600; overflow-wrap: anywhere; }
.page[data-page="company"] .question { overflow-wrap: anywhere; }
.page[data-page="company"] .facts { display: grid; grid-template-columns: auto 1fr; gap: 4px 12px; margin: 0; }
.page[data-page="company"] .facts dt { font-size: 12px; color: var(--text-faint); }
.page[data-page="company"] .facts dd { margin: 0; overflow-wrap: anywhere; }
.page[data-page="company"] .card-actions { margin-top: auto; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.page[data-page="company"] td.when { white-space: nowrap; }
@media (max-width: 640px) {
  /* one membership per block: group name on its own line, status · node · date below */
  .page[data-page="company"] thead { display: none; }
  .page[data-page="company"] tr { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 8px; padding: 8px 0; border-bottom: 1px solid var(--border); }
  .page[data-page="company"] tr:last-child { border-bottom: 0; }
  .page[data-page="company"] td { padding: 0; border-bottom: 0; }
  .page[data-page="company"] td:first-child { flex-basis: 100%; }
}
`;

function ensureStyle() {
  if (document.getElementById(STYLE_ID)) return;
  document.head.append(h("style", { id: STYLE_ID }, STYLE));
}

/** "2026-10-18T11:00:00Z" → "2026-10-18 11:00 UTC"; anything unparseable is shown as sent. */
function when(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? String(iso) : `${d.toISOString().slice(0, 16).replace("T", " ")} UTC`;
}

const words = (value) => (value ? String(value).replaceAll("_", " ") : "—");

const timeOf = (iso) => {
  const t = new Date(iso).getTime();
  return Number.isNaN(t) ? 0 : t;
};

/** Newest request first, whatever order the server sent. */
const newestFirst = (memberships) =>
  memberships.slice().sort((a, b) => timeOf(b.requested_at) - timeOf(a.requested_at));

/** The membership that currently ties this company to a group, if any (the server allows one). */
function openMembership(memberships, groupId) {
  return memberships.find((m) => m.group_id === groupId && !CLOSED_MEMBERSHIP_STATUS.has(m.status)) || null;
}

/** The latest closed membership (`memberships` is newest first), so a rejected company sees why. */
function lastClosedMembership(memberships, groupId) {
  return memberships.find((m) => m.group_id === groupId && CLOSED_MEMBERSHIP_STATUS.has(m.status)) || null;
}

function versionText(v) {
  if (!v) return "none yet";
  const auc = v.metrics && typeof v.metrics.auc === "number" ? ` · AUC ${v.metrics.auc.toFixed(3)}` : "";
  return `v${v.version}${auc}`;
}

function membershipLink(m, label) {
  return h("a", { href: `#/memberships/${encodeURIComponent(m.id)}` }, label);
}

function membershipsSection(memberships) {
  const head = h("div.section-head", h("h2", "My memberships"),
    h("span.faint", "One membership per group you joined. Open one to connect your data."));
  if (!memberships.length) {
    return h("section.section", head,
      h("p.panel.muted", "You have not asked to join a group yet. Pick one below."));
  }
  const rows = memberships.map((m) => h("tr", { dataset: { membership: m.id } },
    h("td", membershipLink(m, m.group_name || m.group_id)),
    h("td", statusPill(m.status),
      m.status_reason ? h("p.faint", m.status_reason) : null),
    h("td", m.node ? statusPill(m.node.status) : h("span.faint", "not connected")),
    h("td.mono.when", when(m.requested_at))));
  return h("section.section", head,
    h("div.panel.table-wrap",
      h("table",
        h("thead", h("tr", h("th", "Group"), h("th", "Status"), h("th", "Node"), h("th", "Requested"))),
        h("tbody", rows))));
}

function groupCard(group, memberships, company, onJoin) {
  const spec = group.task_spec || {};
  const inputs = Array.isArray(spec.input_schema) ? spec.input_schema.map((f) => f.name).filter(Boolean) : [];
  const facts = [
    ["Model", spec.model && spec.model.type ? words(spec.model.type) : "—"],
    ["Task", words(spec.task_type)],
    ["Inputs", inputs.length ? inputs.join(", ") : "—"],
    ["Members", String(group.member_count ?? 0)],
    ["Current model", versionText(group.current_version)],
  ];

  const actions = h("div.card-actions");
  const status = h("div");
  const open = openMembership(memberships, group.id);
  if (open) {
    actions.append(statusPill(open.status), membershipLink(open, "Open membership"));
  } else if (!JOINABLE_GROUP_STATUS.has(group.status)) {
    actions.append(h("span.faint", "This group is not taking new members."));
  } else if (!company || company.status !== "approved") {
    const why = !company || company.status === "pending"
      ? "An admin must approve your company first."
      : `Your company is ${words(company.status)}; only approved companies can join.`;
    actions.append(h("button.btn.primary", { type: "button", disabled: true }, "Request to join"),
      h("span.faint", why));
  } else {
    const button = h("button.btn.primary", { type: "button" }, "Request to join");
    button.addEventListener("click", () => onJoin(group, button, status));
    actions.append(button);
    const closed = lastClosedMembership(memberships, group.id);
    if (closed && closed.status === "rejected") {
      actions.append(h("span.faint", `Your earlier request was rejected${closed.status_reason
        ? `: ${closed.status_reason}` : "."}`));
    }
  }

  return h("article.panel.card", { dataset: { group: group.id } },
    h("div.card-title", h("h3", group.name), statusPill(group.status)),
    spec.question ? h("p.question", `“${spec.question}”`) : null,
    group.description ? h("p.muted", group.description) : null,
    h("dl.facts", facts.map(([k, v]) => [h("dt", k), h("dd", v)])),
    status,
    actions);
}

export async function render({ el }) {
  ensureStyle();
  let flash = null; // the outcome of the last join, kept across the redraw it triggers
  let loads = 0;

  async function load() {
    // two quick joins on different cards overlap their reloads: only the newest may draw,
    // or an older answer would hide the second membership
    const gen = ++loads;
    // /auth/me too: the admin may have approved the company since it signed in, and the
    // shell's "waiting for approval" banner should follow
    const [me, groups, memberships] = await Promise.all([
      api.get("/auth/me"), api.get("/groups"), api.get("/memberships")]);
    if (gen !== loads) return;
    setSession(me);
    draw(me.company, groups.items || [], newestFirst(memberships.items || []));
  }

  async function join(group, button, status) {
    button.disabled = true;
    button.textContent = "Requesting…";
    status.replaceChildren();
    try {
      await api.post(`/groups/${encodeURIComponent(group.id)}/join`);
      flash = `Asked to join ${group.name}. An admin reviews the request; its status shows above.`;
    } catch (err) {
      button.disabled = false;
      button.textContent = "Request to join";
      status.replaceChildren(errorBox(err));
      return;
    }
    try {
      await load();
    } catch (err) {
      el.replaceChildren(errorBox(err));
    }
  }

  function draw(co, groups, memberships) {
    const cards = groups.map((g) => groupCard(g, memberships, co, join));
    // native replaceChildren would print a null child as "null": drop the absent ones
    el.replaceChildren(...[
      h("div.section-head", h("h1", "Groups"),
        h("span.muted", "Each group trains one shared model. Your rows never leave your company.")),
      flash ? h("p.flash", { role: "status" }, flash) : null,
      membershipsSection(memberships),
      h("section.section",
        h("div.section-head", h("h2", "Browse groups"),
          h("span.faint", `${groups.length} group${groups.length === 1 ? "" : "s"}`)),
        cards.length ? h("div.cards", cards)
          : h("p.panel.muted", "No group is open yet. An admin publishes groups when they are ready.")),
    ].filter(Boolean));
  }

  await load();
}
