// In-browser mock of the platform API (PLATFORM-API.md §5), used by api.js under ?mock=1.
// It holds the db.json fixtures in memory, enforces the roles and the state machines
// of §3 the way the real server must, and answers with the §1.1 error shape. A reload
// resets it. Fixtures: db.json (objects) and events.json (one finished job's §6 events);
// every number in them comes from privatelearn/trace.json.

let db = null;
let eventLog = null;

async function load() {
  if (!db) {
    const [d, e] = await Promise.all([
      fetch(new URL("./db.json", import.meta.url)).then((r) => r.json()),
      fetch(new URL("./events.json", import.meta.url)).then((r) => r.json()),
    ]);
    db = d;
    eventLog = e.items;
    db.sessions = {}; // token -> user id
  }
  return db;
}

/** The mock's sign-in emails, for the login page's one-click list (mock mode only). */
export async function demoAccounts() {
  await load();
  return db.users.map((u) => {
    const c = db.companies.find((x) => x.id === u.company_id);
    return { email: u.email, label: c ? `${c.name} (${c.status})` : "Platform admin" };
  });
}

export async function events(jobId) {
  await load();
  return eventLog.filter((e) => e.job_id === jobId);
}

// ---------------------------------------------------------------- helpers

const now = () => new Date().toISOString().replace(/\.\d+Z$/, "Z");
let counter = 100;
const newId = (prefix) => `${prefix}_${++counter}`;

class Fail extends Error {
  constructor(status, code, message, field = null) {
    super(message);
    this.status = status;
    this.body = { error: { code, message, field } };
  }
}
const notFound = (what) => new Fail(404, "not_found", `${what} not found`);
const invalid = (message) => new Fail(409, "invalid_transition", message);
const forbidden = (message = "not allowed for this account") => new Fail(403, "forbidden", message);

const find = (list, id, what) => {
  const item = list.find((x) => x.id === id);
  if (!item) throw notFound(`${what} ${id}`);
  return item;
};
const companyOf = (user) => (user && user.company_id ? db.companies.find((c) => c.id === user.company_id) : null);

function requireUser(ctx) {
  if (!ctx.user) throw new Fail(401, "unauthorized", "sign in first");
  return ctx.user;
}
function requireAdmin(ctx) {
  if (requireUser(ctx).role !== "admin") throw forbidden("admin only");
}
function requireCompany(ctx) {
  if (requireUser(ctx).role !== "company") throw forbidden("company accounts only");
  return companyOf(ctx.user);
}
function requireOwnMembership(ctx, m) {
  const co = requireCompany(ctx);
  if (m.company_id !== co.id) throw forbidden("not your company");
}
function requireMemberOfGroup(ctx, groupId) {
  const user = requireUser(ctx);
  if (user.role === "admin") return;
  if (!db.memberships.some((m) => m.group_id === groupId && m.company_id === user.company_id)) {
    throw forbidden("not a member of this group");
  }
}
function transition(m, from, to) {
  if (!from.includes(m.status)) {
    throw invalid(`membership ${m.id} is '${m.status}'; '${to}' needs one of ${from.join(", ")}`);
  }
  m.status = to;
  return m;
}
const ORDER = ["requested", "approved", "node_connected", "schema_proposed", "schema_confirmed",
               "ready", "training", "active"];
const atLeast = (status, floor) => ORDER.indexOf(status) >= ORDER.indexOf(floor);

function groupView(g, user) {
  const members = db.memberships.filter((m) => m.group_id === g.id &&
    (user.role === "admin" || m.company_id === user.company_id));
  const versions = db.versions.filter((v) => v.group_id === g.id);
  return { ...g, members, versions };
}

// ---------------------------------------------------------------- routes
// [method, pattern, handler(ctx, params, body, query)]; ":x" segments become params.x

const routes = [
  ["GET", "/meta", () => db.meta],

  ["POST", "/auth/login", (ctx, p, body) => {
    const email = String(body.email || "").trim().toLowerCase();
    const user = db.users.find((u) => u.email === email);
    if (!user) throw new Fail(401, "unauthorized", `no account for ${email || "that email"}`);
    const token = `mock_${user.id}_${Date.now()}`;
    db.sessions[token] = user.id;
    return { token, user, company: companyOf(user) };
  }],
  ["POST", "/auth/logout", (ctx) => { delete db.sessions[ctx.token]; return null; }],
  ["GET", "/auth/me", (ctx) => ({ user: requireUser(ctx), company: companyOf(ctx.user) })],

  ["POST", "/companies/register", (ctx, p, body) => {
    for (const field of ["name", "field", "contact_email"]) {
      if (!String(body[field] || "").trim()) {
        throw new Fail(422, "validation_error", `${field} is required`, field);
      }
    }
    const email = body.contact_email.trim().toLowerCase();
    if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) {
      throw new Fail(422, "validation_error", "not an email address", "contact_email");
    }
    if (db.users.some((u) => u.email === email)) {
      throw new Fail(422, "validation_error", "an account with this email exists; sign in", "contact_email");
    }
    const company = { id: newId("co"), name: body.name.trim(), field: body.field.trim(), contact_email: email,
                      status: "pending", wallet_address: body.wallet_address || null, created_at: now() };
    const user = { id: newId("usr"), email, role: "company", company_id: company.id };
    db.companies.push(company);
    db.users.push(user);
    const token = `mock_${user.id}_${Date.now()}`;
    db.sessions[token] = user.id;
    return { token, user, company };
  }],
  ["GET", "/companies", (ctx, p, b, q) => {
    requireAdmin(ctx);
    return { items: db.companies.filter((c) => !q.status || c.status === q.status) };
  }],
  ["GET", "/companies/:id", (ctx, p) => {
    const user = requireUser(ctx);
    if (user.role !== "admin" && user.company_id !== p.id) throw forbidden("not your company");
    return find(db.companies, p.id, "company");
  }],
  ["POST", "/companies/:id/approve", (ctx, p) => {
    requireAdmin(ctx);
    const c = find(db.companies, p.id, "company");
    if (!["pending", "suspended"].includes(c.status)) throw invalid(`company ${c.id} is '${c.status}'`);
    c.status = "approved";
    return c;
  }],
  ["POST", "/companies/:id/reject", (ctx, p) => {
    requireAdmin(ctx);
    const c = find(db.companies, p.id, "company");
    if (c.status !== "pending") throw invalid(`company ${c.id} is '${c.status}'`);
    c.status = "rejected";
    return c;
  }],

  ["GET", "/groups", (ctx) => {
    const user = requireUser(ctx);
    const mine = new Set(db.memberships.filter((m) => m.company_id === user.company_id).map((m) => m.group_id));
    return { items: db.groups.filter((g) => user.role === "admin" || ["open", "trained"].includes(g.status) ||
                                            mine.has(g.id)) };
  }],
  ["POST", "/groups/draft-spec", (ctx, p, body) => {
    requireAdmin(ctx);
    const template = structuredClone(db.groups[0].task_spec);
    template.question = String(body.question || template.question);
    return { task_spec: template, made_by: "rule",
             reasoning: "mock: the architect agent is not running, so this is the template TaskSpec" };
  }],
  ["GET", "/groups/:id", (ctx, p) => {
    requireUser(ctx);
    return groupView(find(db.groups, p.id, "group"), ctx.user);
  }],
  ["POST", "/groups", (ctx, p, body) => {
    requireAdmin(ctx);
    if (!String(body.name || "").trim()) throw new Fail(422, "validation_error", "name is required", "name");
    const g = { id: newId("grp"), name: body.name.trim(), description: body.description || "", status: "draft",
                task_spec: body.task_spec || null, pricing: null, member_count: 0, current_version: null,
                created_at: now() };
    db.groups.push(g);
    return g;
  }],
  ["PATCH", "/groups/:id", (ctx, p, body) => {
    requireAdmin(ctx);
    const g = find(db.groups, p.id, "group");
    if (body.task_spec && g.status !== "draft") throw invalid("task_spec is editable only while the group is a draft");
    for (const k of ["name", "description", "task_spec"]) if (k in body) g[k] = body[k];
    return g;
  }],
  ["POST", "/groups/:id/publish", (ctx, p) => {
    requireAdmin(ctx);
    const g = find(db.groups, p.id, "group");
    if (g.status !== "draft") throw invalid(`group ${g.id} is '${g.status}'; only a draft can be published`);
    g.status = "open";
    return g;
  }],
  ["POST", "/groups/:id/join", (ctx, p) => {
    const co = requireCompany(ctx);
    const g = find(db.groups, p.id, "group");
    if (co.status !== "approved") throw invalid(`company ${co.id} is '${co.status}'; wait for approval`);
    if (!["open", "trained"].includes(g.status)) throw invalid(`group ${g.id} is '${g.status}'`);
    if (db.memberships.some((m) => m.group_id === g.id && m.company_id === co.id && m.status !== "left")) {
      throw invalid("this company already has a membership in this group");
    }
    const m = { id: newId("mem"), company_id: co.id, company_name: co.name, group_id: g.id, group_name: g.name,
                status: "requested", node: null, data: null, schema: null, validation: null,
                contribution: { rows: 0, rounds: 0 }, requested_at: now() };
    db.memberships.push(m);
    g.member_count += 1;
    return m;
  }],

  ["GET", "/memberships", (ctx, p, b, q) => {
    const user = requireUser(ctx);
    return { items: db.memberships.filter((m) => (user.role === "admin" || m.company_id === user.company_id) &&
                                                 (!q.group_id || m.group_id === q.group_id) &&
                                                 (!q.status || m.status === q.status)) };
  }],
  ["GET", "/memberships/:id", (ctx, p) => {
    const m = find(db.memberships, p.id, "membership");
    if (requireUser(ctx).role !== "admin") requireOwnMembership(ctx, m);
    return m;
  }],
  ["POST", "/memberships/:id/approve", (ctx, p) => {
    requireAdmin(ctx);
    return transition(find(db.memberships, p.id, "membership"), ["requested"], "approved");
  }],
  ["POST", "/memberships/:id/reject", (ctx, p) => {
    requireAdmin(ctx);
    return transition(find(db.memberships, p.id, "membership"), ["requested"], "rejected");
  }],
  ["POST", "/memberships/:id/node-token", (ctx, p) => {
    const m = find(db.memberships, p.id, "membership");
    requireOwnMembership(ctx, m);
    if (!atLeast(m.status, "approved")) throw invalid(`membership ${m.id} is '${m.status}'; approve it first`);
    const token = `ntk_mock_${Math.random().toString(36).slice(2, 10)}`;
    return { node_token: token,
             command: `python hub/node/node.py --server ${location.origin} --token ${token} --db <your connection>` };
  }],
  ["POST", "/memberships/:id/schema/confirm", (ctx, p, body) => {
    const m = find(db.memberships, p.id, "membership");
    requireOwnMembership(ctx, m);
    transition(m, ["schema_proposed"], "schema_confirmed");
    if (body.mapping) m.schema.mapping = body.mapping;
    if (body.label) m.schema.label = body.label;
    m.schema.confirmed = true;
    return m;
  }],
  ["POST", "/memberships/:id/leave", (ctx, p) => {
    const m = find(db.memberships, p.id, "membership");
    requireOwnMembership(ctx, m);
    return transition(m, ORDER.filter((s) => s !== "training"), "left");
  }],

  ["POST", "/groups/:id/jobs", (ctx, p, body) => {
    requireAdmin(ctx);
    const g = find(db.groups, p.id, "group");
    if (db.jobs.some((j) => j.group_id === g.id && ["queued", "running"].includes(j.status))) {
      throw invalid("a job is already running for this group");
    }
    const ready = db.memberships.filter((m) => m.group_id === g.id && ["ready", "active"].includes(m.status) &&
                                               (!body.membership_ids || body.membership_ids.includes(m.id)));
    const min = g.task_spec?.training?.min_participants ?? 2;
    if (ready.length < min) throw invalid(`${ready.length} participant(s) ready; this group needs ${min}`);
    const job = { id: newId("job"), group_id: g.id, kind: body.kind || "train", status: "queued",
                  step: "waiting_for_nodes", round: 0, rounds: g.task_spec?.training?.rounds ?? 4,
                  parent_version_id: body.kind === "finetune" ? g.current_version?.id ?? null : null,
                  participants: ready.map((m, i) => ({ membership_id: m.id, node_id: m.node?.id ?? null,
                                                       company_name: m.company_name, state: "waiting",
                                                       color: db.jobs[0].participants[i % 3].color })),
                  metrics: { auc_series: [], loss_series: [], eps_total: 0 }, started_at: null,
                  finished_at: null, error: null };
    db.jobs.push(job);
    return job;
  }],
  ["GET", "/groups/:id/jobs", (ctx, p) => {
    requireMemberOfGroup(ctx, p.id);
    return { items: db.jobs.filter((j) => j.group_id === p.id) };
  }],
  ["GET", "/jobs/:id", (ctx, p) => {
    const j = find(db.jobs, p.id, "job");
    requireMemberOfGroup(ctx, j.group_id);
    return j;
  }],
  ["POST", "/jobs/:id/cancel", (ctx, p) => {
    requireAdmin(ctx);
    const j = find(db.jobs, p.id, "job");
    if (!["queued", "running"].includes(j.status)) throw invalid(`job ${j.id} is '${j.status}'`);
    j.status = "cancelled";
    return j;
  }],
  ["GET", "/jobs/:id/events", (ctx, p, b, q) => {
    const j = find(db.jobs, p.id, "job");
    requireMemberOfGroup(ctx, j.group_id);
    const since = Number(q.since || 0);
    return { items: eventLog.filter((e) => e.job_id === j.id && e.seq > since) };
  }],

  ["GET", "/groups/:id/versions", (ctx, p) => {
    requireMemberOfGroup(ctx, p.id);
    return { items: db.versions.filter((v) => v.group_id === p.id) };
  }],
  ["POST", "/groups/:id/predict", (ctx, p, body) => {
    const user = requireUser(ctx);
    const g = find(db.groups, p.id, "group");
    if (user.role !== "admin" && !db.memberships.some((m) => m.group_id === g.id &&
                                                          m.company_id === user.company_id && m.status === "active")) {
      throw forbidden("only active members can use this model");
    }
    const versionId = body.version_id || g.current_version?.id;
    const model = versionId && db.models[versionId];
    if (!model) throw invalid(`group ${g.id} has no trained model yet`);
    const inputs = body.inputs || {};
    const fields = g.task_spec.input_schema;
    for (const f of fields) {
      const v = inputs[f.name];
      if (v === undefined || v === null || v === "") {
        if (f.required) throw new Fail(422, "validation_error", `${f.name} is required`, `inputs.${f.name}`);
      } else if (typeof v !== "number" || !Number.isFinite(v) || (f.type === "integer" && !Number.isInteger(v))) {
        throw new Fail(422, "validation_error", `${f.name} must be ${f.type === "integer" ? "an integer" : "a number"}`,
                       `inputs.${f.name}`);
      }
    }
    // logistic: p = σ(b + Σ w_i·(x_i − mean_i)/std_i); contribution_i = w_i·z_i
    const [bias, ...w] = model.weights;
    const explanation = model.features.map((name, i) => {
      const x = Number(inputs[name] ?? model.norm.mean[i]);
      return { feature: name, contribution: w[i] * ((x - model.norm.mean[i]) / model.norm.std[i]) };
    });
    const logit = bias + explanation.reduce((s, e) => s + e.contribution, 0);
    const prob = 1 / (1 + Math.exp(-logit));
    const [neg, pos] = g.task_spec.label.classes;
    return { prediction: prob >= 0.5 ? pos : neg, probability: prob, scores: { [neg]: 1 - prob, [pos]: prob },
             version_id: versionId, explanation };
  }],

  ["GET", "/usage", (ctx, p, b, q) => {
    const user = requireUser(ctx);
    const companyId = user.role === "admin" ? q.company_id : user.company_id;
    const items = db.usage.filter((u) => (!companyId || u.company_id === companyId) &&
                                         (!q.group_id || u.group_id === q.group_id));
    const totals = {};
    for (const u of items) totals[u.kind] = (totals[u.kind] || 0) + u.units;
    return { items, totals };
  }],

  ["POST", "/demo/nodes/start", (ctx) => { requireAdmin(ctx); return { pid: 4242 }; }],
  ["POST", "/demo/nodes/stop", (ctx) => { requireAdmin(ctx); return null; }],
  ["POST", "/demo/reset", (ctx) => { requireAdmin(ctx); db = null; return null; }],
];

function match(pattern, path) {
  const a = pattern.split("/");
  const b = path.split("/");
  if (a.length !== b.length) return null;
  const params = {};
  for (let i = 0; i < a.length; i++) {
    if (a[i].startsWith(":")) params[a[i].slice(1)] = decodeURIComponent(b[i]);
    else if (a[i] !== b[i]) return null;
  }
  return params;
}

/** Answer one API call: → {status, body}, exactly what the real server would send. */
export async function handle(method, fullPath, body, token) {
  await load();
  const [path, qs] = fullPath.split("?");
  const query = Object.fromEntries(new URLSearchParams(qs || ""));
  const ctx = { token, user: null };
  const userId = token ? db.sessions[token] : null;
  if (userId) ctx.user = db.users.find((u) => u.id === userId) || null;
  else if (token && token.startsWith("mock_")) {
    // a reload wipes the sessions: keep a seeded user signed in (registered ones are gone)
    const id = token.split("_").slice(1, 3).join("_");
    ctx.user = db.users.find((u) => u.id === id) || null;
    if (ctx.user) db.sessions[token] = ctx.user.id;
  }
  try {
    let pathMatched = false;
    for (const [m, pattern, fn] of routes) {
      const params = match(pattern, path);
      if (!params) continue;
      pathMatched = true;
      if (m !== method) continue;
      const result = await fn(ctx, params, body || {}, query);
      return { status: result === null ? 204 : 200, body: structuredClone(result) };
    }
    if (pathMatched) throw new Fail(400, "bad_request", `${method} is not supported on ${path}`);
    throw notFound(`route ${path}`);
  } catch (err) {
    if (err instanceof Fail) return { status: err.status, body: err.body };
    return { status: 500, body: { error: { code: "internal", message: String(err), field: null } } };
  }
}
