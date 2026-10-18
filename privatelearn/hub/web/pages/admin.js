// #/admin — the admin's approval queues (PLATFORM-API §5.2, §5.4): companies waiting to be
// approved and membership requests. Approve and reject act inline; the queues refresh in
// place (after every decision and every few seconds), so the page never needs a reload.
//
// Rows are keyed by id and kept across refreshes: a refresh adds new requests and drops the
// ones decided elsewhere, but never redraws a row the admin may be typing a reason into.

import { api } from "../api.js";
import { errorBox, h, statusPill } from "../app.js";

const POLL_MS = 10000;
const REASON_MAX = 500; // the server's limit on `reason` (hub/server/routes/companies.py)

const STYLE_ID = "page-admin-style";
const STYLE = `
.page[data-page="admin"] .section { display: flex; flex-direction: column; gap: 12px; }
.page[data-page="admin"] .section-head { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
.page[data-page="admin"] .section-head .btn { margin-left: auto; }
.page[data-page="admin"] .decisions { display: flex; flex-direction: column; gap: 4px; margin: 0; padding: 0; list-style: none; }
.page[data-page="admin"] .decisions li { color: var(--text-dim); }
.page[data-page="admin"] .table-wrap { overflow-x: auto; }
.page[data-page="admin"] table { width: 100%; border-collapse: collapse; }
.page[data-page="admin"] th {
  text-align: left; font-size: 12px; font-weight: 500; color: var(--text-faint);
  padding: 8px 12px; border-bottom: 1px solid var(--border);
}
.page[data-page="admin"] td { padding: 8px 12px; border-bottom: 1px solid var(--border); vertical-align: top; }
.page[data-page="admin"] tbody tr:last-child td { border-bottom: 0; }
.page[data-page="admin"] td.when { white-space: nowrap; }
.page[data-page="admin"] td.actions { width: 1%; }
.page[data-page="admin"] .decisions:empty, .page[data-page="admin"] .refresh-error:empty { display: none; }
.page[data-page="admin"] .row-actions { display: flex; flex-direction: column; align-items: flex-end; gap: 4px; }
.page[data-page="admin"] .row-actions .buttons { display: flex; gap: 8px; }
.page[data-page="admin"] .row-actions .why { max-width: 240px; text-align: right; }
.page[data-page="admin"] .reason { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
.page[data-page="admin"] .reason input { width: 240px; max-width: 100%; }
.page[data-page="admin"] .row-status:empty { display: none; }
.page[data-page="admin"] .row-status { margin-top: 8px; }
.page[data-page="admin"] .cell-title { overflow-wrap: anywhere; }
@media (max-width: 640px) {
  .page[data-page="admin"] thead { display: none; }
  .page[data-page="admin"] tbody tr {
    display: flex; flex-wrap: wrap; align-items: center; gap: 4px 8px; padding: 8px 0;
    border-bottom: 1px solid var(--border);
  }
  .page[data-page="admin"] tbody tr:last-child { border-bottom: 0; }
  .page[data-page="admin"] td { padding: 0; border-bottom: 0; }
  .page[data-page="admin"] td:first-child, .page[data-page="admin"] td.actions { flex-basis: 100%; width: auto; }
  .page[data-page="admin"] .row-actions { align-items: flex-start; }
  .page[data-page="admin"] .row-actions .why { max-width: none; text-align: left; }
  .page[data-page="admin"] .reason { justify-content: flex-start; }
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

const timeOf = (iso) => {
  const t = new Date(iso).getTime();
  return Number.isNaN(t) ? 0 : t;
};

/**
 * One queue: a table whose rows are keyed by item id. `sync(items)` adds rows for new items
 * (in `items` order), refreshes the rows that stay, and removes rows for items that left.
 */
function queue({ title, empty, columns, makeRow }) {
  const count = h("span.faint");
  const tbody = h("tbody");
  const table = h("div.panel.table-wrap",
    h("table", h("thead", h("tr", columns.map((c) => h("th", c)), h("th", { "aria-label": "Decision" }))), tbody));
  const none = h("p.panel.muted", empty);
  const section = h("section.section", h("div.section-head", h("h2", title), count), table, none);
  const rows = new Map(); // id → {tr, update(item)}

  function showCount() {
    count.textContent = rows.size ? `${rows.size} waiting` : "";
    table.hidden = !rows.size;
    none.hidden = Boolean(rows.size);
  }

  function sync(items) {
    const keep = new Set(items.map((x) => x.id));
    for (const [id, row] of rows) {
      if (!keep.has(id)) {
        row.tr.remove();
        rows.delete(id);
      }
    }
    let previous = null;
    for (const item of items) {
      let row = rows.get(item.id);
      if (!row) {
        row = makeRow(item);
        rows.set(item.id, row);
      } else {
        row.update(item);
      }
      // keep DOM order equal to `items` order without touching rows already in place
      const expected = previous ? previous.nextSibling : tbody.firstChild;
      if (expected !== row.tr) tbody.insertBefore(row.tr, expected);
      previous = row.tr;
    }
    showCount();
  }

  /** Remove one row right away (its request was just decided here). */
  function drop(id) {
    const row = rows.get(id);
    if (!row) return;
    row.tr.remove();
    rows.delete(id);
    showCount();
  }

  return { section, sync, drop };
}

/**
 * The Approve / Reject controls of one row. Reject opens an inline, optional reason field
 * (Enter confirms, Escape cancels). `act(kind, body)` performs the call; on failure the error
 * shows under the controls and they come back, on success the caller removes the row.
 */
function decisionControls({ label, act, canApprove }) {
  const cell = h("div");
  const status = h("div.row-status");
  const controls = h("div.row-actions");
  let busy = false;
  let shownWhy; // the canApprove() answer the buttons on screen were built for

  function setBusy(on) {
    busy = on;
    cell.querySelectorAll("button, input").forEach((n) => { n.disabled = on; });
    // the loop above re-enabled Approve too: rebuild so it follows canApprove() again
    if (!on && !controls.querySelector(".reason")) showButtons();
  }

  async function run(kind, body) {
    if (busy) return;
    status.replaceChildren();
    setBusy(true);
    try {
      await act(kind, body);
    } catch (err) {
      status.replaceChildren(errorBox(err));
      setBusy(false);
    }
  }

  function showButtons() {
    const why = canApprove();
    shownWhy = why;
    const approve = h("button.btn.primary.small", { type: "button", disabled: Boolean(why), title: why || null,
                                                    "aria-label": `Approve ${label()}` }, "Approve");
    approve.addEventListener("click", () => run("approve"));
    const reject = h("button.btn.ghost.small", { type: "button", "aria-label": `Reject ${label()}` }, "Reject");
    reject.addEventListener("click", showReason);
    controls.replaceChildren(...[h("div.buttons", approve, reject), why ? h("p.faint.why", why) : null]
      .filter(Boolean));
  }

  function showReason() {
    const input = h("input", { type: "text", maxlength: String(REASON_MAX), placeholder: "Reason (optional)",
                               "aria-label": `Reason for rejecting ${label()}` });
    const confirm = h("button.btn.primary.small", { type: "button" }, "Confirm reject");
    const cancel = h("button.btn.ghost.small", { type: "button" }, "Cancel");
    const submit = () => {
      const reason = input.value.trim();
      run("reject", reason ? { reason } : {});
    };
    confirm.addEventListener("click", submit);
    cancel.addEventListener("click", () => { status.replaceChildren(); showButtons(); });
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); submit(); }
      if (e.key === "Escape") { e.preventDefault(); cancel.click(); }
    });
    controls.replaceChildren(h("div.reason", input, confirm, cancel));
    input.focus();
  }

  /** Re-evaluate whether Approve is possible (the company may have been approved meanwhile). */
  function update() {
    if (busy || controls.querySelector(".reason")) return; // never interrupt a decision in progress
    // rebuild only when Approve's state changed: replacing the buttons on every 10 s poll
    // dropped keyboard focus and could swallow a click landing during the swap
    if (canApprove() !== shownWhy || !controls.querySelector("button")) showButtons();
  }

  showButtons();
  cell.append(controls, status);
  return { cell, update };
}

// Every group with a link to its page (E8), so the admin can reach training and versions.
function groupsSection() {
  const list = h("div.row", h("span.faint", "Loading…"));
  api.get("/groups").then(({ items }) => {
    list.replaceChildren(...(items.length ? items.map((g) => h("a.btn.ghost", { href: `#/groups/${encodeURIComponent(g.id)}` },
      g.name, " ", statusPill(g.status))) : [h("span.faint", "No groups yet.")]),
      h("a.btn.primary", { href: "#/groups/new" }, "+ New group"));
  }, (err) => list.replaceChildren(errorBox(err)));
  return h("section.panel.stack", h("h2", "Groups"), list);
}

export async function render({ el }) {
  ensureStyle();

  const state = { companies: new Map() }; // id → Company, every company (for membership rows)
  let version = 0; // bumped by every decision: a refresh that started before one is stale
  let inFlight = null;
  const decisions = h("ul.decisions", { role: "status" });

  function remember(text) {
    decisions.prepend(h("li", text));
    while (decisions.children.length > 5) decisions.lastElementChild.remove();
  }

  async function decide(path, body, done) {
    await api.post(path, body);
    version++;
    done();
    refresh().catch((err) => console.warn("refresh after a decision failed:", err));
  }

  // ---------------------------------------------------------------- companies

  const companies = queue({
    title: "Companies waiting for approval",
    empty: "No company is waiting. New registrations appear here.",
    columns: ["Company", "Contact", "Registered"],
    makeRow(item) {
      let company = item;
      const name = h("div.cell-title");
      const contact = h("td.mono");
      const created = h("td.mono.when");
      const controls = decisionControls({
        label: () => company.name,
        canApprove: () => null,
        act: (kind, body) => decide(`/companies/${encodeURIComponent(company.id)}/${kind}`, body, () => {
          companies.drop(company.id);
          remember(`${kind === "approve" ? "Approved" : "Rejected"} company ${company.name}`);
        }),
      });
      const tr = h("tr", { dataset: { id: item.id } },
        h("td", name), contact, created, h("td.actions", controls.cell));
      function update(next) {
        company = next;
        name.replaceChildren(h("strong", company.name), " ", h("span.faint", company.field || ""));
        contact.textContent = company.contact_email || "—";
        created.textContent = when(company.created_at);
        controls.update();
      }
      update(item);
      return { tr, update };
    },
  });

  // ---------------------------------------------------------------- membership requests

  function companyBlock(m) {
    const co = state.companies.get(m.company_id);
    if (!co || co.status === "approved") return null;
    return `${co.name} is ${co.status.replaceAll("_", " ")}: approve the company first`;
  }

  const requests = queue({
    title: "Membership requests",
    empty: "No membership request is waiting.",
    columns: ["Company", "Group", "Requested"],
    makeRow(item) {
      let m = item;
      const who = h("div.cell-title");
      const group = h("td");
      const requested = h("td.mono.when");
      const controls = decisionControls({
        label: () => `${m.company_name} in ${m.group_name}`,
        canApprove: () => companyBlock(m),
        act: (kind, body) => decide(`/memberships/${encodeURIComponent(m.id)}/${kind}`, body, () => {
          requests.drop(m.id);
          remember(`${kind === "approve" ? "Approved" : "Rejected"} ${m.company_name} in ${m.group_name}`);
        }),
      });
      const tr = h("tr", { dataset: { id: item.id } },
        h("td", who), group, requested, h("td.actions", controls.cell));
      function update(next) {
        m = next;
        const co = state.companies.get(m.company_id);
        // native replaceChildren prints arrays and null as text: pass nodes and strings only
        who.replaceChildren(h("strong", m.company_name || m.company_id),
          ...(co && co.status !== "approved" ? [" ", statusPill(co.status)] : []));
        group.replaceChildren(h("a", { href: `#/groups/${encodeURIComponent(m.group_id)}` }, m.group_name || m.group_id));
        requested.textContent = when(m.requested_at);
        controls.update();
      }
      update(item);
      return { tr, update };
    },
  });

  // ---------------------------------------------------------------- refresh

  async function fetchAndSync() {
    const started = version;
    const [all, pending] = await Promise.all([
      api.get("/companies"), api.get("/memberships", { status: "requested" })]);
    if (started !== version) return false; // a decision landed meanwhile: this answer is stale
    state.companies = new Map((all.items || []).map((c) => [c.id, c]));
    const oldestFirst = (a, b, key) => timeOf(a[key]) - timeOf(b[key]);
    companies.sync((all.items || []).filter((c) => c.status === "pending")
      .sort((a, b) => oldestFirst(a, b, "created_at")));
    requests.sync((pending.items || []).sort((a, b) => oldestFirst(a, b, "requested_at")));
    return true;
  }

  /** Re-read both queues; concurrent calls share one request, a stale answer is retried once. */
  function refresh() {
    if (!inFlight) {
      inFlight = (async () => {
        try {
          if (!(await fetchAndSync())) await fetchAndSync();
        } finally {
          inFlight = null;
        }
      })();
    }
    return inFlight;
  }

  const refreshError = h("div.refresh-error");
  const refreshButton = h("button.btn.ghost.small", { type: "button" }, "Refresh");
  async function manualRefresh() {
    refreshButton.disabled = true;
    refreshError.replaceChildren();
    try {
      await refresh();
    } catch (err) {
      refreshError.replaceChildren(errorBox(err));
    } finally {
      refreshButton.disabled = false;
    }
  }
  refreshButton.addEventListener("click", manualRefresh);

  await refresh(); // a failure here is shown by the shell instead of the page

  el.replaceChildren(
    h("div.section-head", h("h1", "Approvals"),
      h("span.muted", "Companies and membership requests waiting for a decision."), refreshButton),
    refreshError,
    decisions,
    companies.section,
    requests.section,
    groupsSection());

  const timer = setInterval(() => {
    if (document.hidden) return;
    refresh().then(() => refreshError.replaceChildren(),
                   (err) => refreshError.replaceChildren(errorBox(err)));
  }, POLL_MS);
  return () => clearInterval(timer);
}
