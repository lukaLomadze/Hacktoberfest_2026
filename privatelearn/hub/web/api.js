// API client for the platform (PLATFORM-API.md v0.1). Every page talks to the server
// through this module only, so the bearer token, the §1.1 error shape and the mock switch
// live in one place.
//
//   ?mock=1   serve every call from hub/web/mock/ (the contract's examples) — no backend needed

const BASE = "/api/v1";
const TOKEN_KEY = "privatelearn.token";

const params = new URLSearchParams(location.search);
// ?mock=1 forces the mock, ?mock=0 forces the server. Without either, the mock takes over when
// no platform server answers /meta (a plain static server, or the page opened on its own).
const MOCK = params.has("mock") ? params.get("mock") !== "0" : !(await serverAnswers());
const MOCK_FORCED = params.has("mock") && MOCK;

async function serverAnswers() {
  try {
    const r = await fetch(`${BASE}/meta`, { headers: { Accept: "application/json" } });
    return r.ok && (await r.json()).version !== undefined;
  } catch {
    return false;
  }
}

// localStorage can throw (private window, blocked storage): the session then lives in memory.
let memoryToken = null;
function readToken() {
  try { return localStorage.getItem(TOKEN_KEY); } catch { return memoryToken; }
}
function writeToken(token) {
  memoryToken = token;
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch { /* memory only */ }
}

/** A non-2xx response, in the §1.1 shape: {error: {code, message, field}}. */
export class ApiError extends Error {
  constructor(status, code, message, field = null) {
    super(message);
    this.name = "ApiError";
    this.status = status; // 0 = the request never reached the server
    this.code = code;
    this.field = field;
  }
}

function errorFrom(status, body) {
  const e = body && typeof body === "object" ? body.error : null;
  if (e && typeof e.code === "string") {
    return new ApiError(status, e.code, e.message || e.code, e.field ?? null);
  }
  // a proxy page, a plain static server or a crash: still an ApiError, so callers handle one
  // shape. Most often the platform server is simply not running.
  return new ApiError(status, `http_${status}`,
    `HTTP ${status} without an API error body: is the platform server running? (?mock=1 needs none)`);
}

let mockServer = null;
async function mock() {
  if (!mockServer) mockServer = await import("./mock/server.js");
  return mockServer;
}

async function request(method, path, body) {
  const token = readToken();
  let status;
  let data = null;
  if (MOCK) {
    ({ status, body: data } = await (await mock()).handle(method, path, body, token));
  } else {
    const headers = { Accept: "application/json" };
    if (body !== undefined) headers["Content-Type"] = "application/json";
    if (token) headers.Authorization = `Bearer ${token}`;
    let res;
    try {
      res = await fetch(BASE + path, {
        method, headers, body: body === undefined ? undefined : JSON.stringify(body),
      });
    } catch (err) {
      throw new ApiError(0, "network_error", `cannot reach the server (${err.message})`);
    }
    status = res.status;
    const text = await res.text();
    if (text) {
      try { data = JSON.parse(text); } catch { data = null; }
    }
  }
  if (status === 401 && token) {
    // the session expired or the server was reset: forget it and let the app send us to login
    writeToken(null);
    window.dispatchEvent(new CustomEvent("api:unauthorized"));
  }
  if (status < 200 || status >= 300) throw errorFrom(status, data);
  return status === 204 ? null : data;
}

function query(params) {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params || {})) {
    if (v !== undefined && v !== null && v !== "") q.set(k, v);
  }
  const s = q.toString();
  return s ? `?${s}` : "";
}

export const api = {
  isMock: MOCK,
  isMockForced: MOCK_FORCED,
  get token() { return readToken(); },
  setToken: writeToken,

  get: (path, params) => request("GET", path + query(params)),
  post: (path, body = {}) => request("POST", path, body),
  patch: (path, body = {}) => request("PATCH", path, body),
  del: (path) => request("DELETE", path),

  /**
   * Live job events (§6). Calls onEvent(event) for each event in seq order, catching up from
   * `since` first. Returns a stop() function. Real mode uses EventSource (which cannot send
   * headers, hence ?token=); the browser resends Last-Event-ID on reconnect by itself.
   * Mock mode replays mock/events.json on a timer.
   */
  stream(jobId, onEvent, { since = 0, intervalMs = 400 } = {}) {
    if (MOCK) {
      let stopped = false;
      (async () => {
        const items = (await (await mock()).events(jobId)).filter((e) => e.seq > since);
        for (const e of items) {
          if (stopped) return;
          onEvent(e);
          await new Promise((r) => setTimeout(r, intervalMs));
        }
      })();
      return () => { stopped = true; };
    }
    const q = query({ token: readToken(), since: since || undefined });
    const source = new EventSource(`${BASE}/jobs/${encodeURIComponent(jobId)}/stream${q}`);
    let last = since;
    const handler = (msg) => {
      let e;
      try { e = JSON.parse(msg.data); } catch { return; }
      if (typeof e.seq === "number" && e.seq <= last) return; // duplicate after a reconnect
      last = e.seq ?? last;
      onEvent(e);
    };
    // the server names each SSE message after the event type; listen to the known ones and
    // to unnamed messages, so a new event type still arrives (and is shown as text)
    source.onmessage = handler;
    for (const t of ["job.status", "job.step", "participant.state", "agent.message", "packet",
                     "round.decision", "round.completed", "model.published"]) {
      source.addEventListener(t, handler);
    }
    return () => source.close();
  },
};
