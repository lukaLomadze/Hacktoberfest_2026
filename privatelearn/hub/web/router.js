// Hash router: "#/groups/grp_1" → the "group" page with params {id: "grp_1"}.
//
// A page is a module pages/<name>.js exporting `render(ctx)`, where ctx is
// {el, params, query, session, navigate}. render may return a cleanup function that runs
// when the user leaves the page (close an EventSource, stop a timer). Pages load lazily,
// so a later task (E2–E9) only adds its file: no change here or in app.js. A route whose
// page does not exist yet shows which task brings it.

export const ROUTES = [
  // pattern            page          who may open it   task that builds it
  ["/login",            "login",      "guest",          "E1"],
  ["/register",         "register",   "guest",          "E1"],
  ["/home",             "company",    "company",        "E2"],
  ["/admin",            "admin",      "admin",          "E3"],
  ["/groups/new",       "group-edit", "admin",          "E4"],
  ["/groups/:id/edit",  "group-edit", "admin",          "E4"],
  ["/groups/:id",       "group",      "admin",          "E8"],
  ["/memberships/:id",  "membership", "company",        "E5"],
  ["/jobs/:id",         "job",        "any",            "E6"],
  ["/usage",            "usage",      "any",            "E9"],
];

/** The landing route for a role: where sign-in and "#/" go. */
export const homeFor = (role) => (role === "admin" ? "/admin" : role === "company" ? "/home" : "/login");

export function parseHash(hash = location.hash) {
  const raw = hash.replace(/^#/, "") || "/";
  const [path, qs] = raw.split("?");
  return { path: path || "/", query: Object.fromEntries(new URLSearchParams(qs || "")) };
}

export function matchRoute(path) {
  const parts = path.split("/").filter(Boolean);
  for (const [pattern, page, access, task] of ROUTES) {
    const want = pattern.split("/").filter(Boolean);
    if (want.length !== parts.length) continue;
    const params = {};
    let ok = true;
    for (let i = 0; i < want.length && ok; i++) {
      if (!want[i].startsWith(":")) ok = want[i] === parts[i];
      else {
        try { params[want[i].slice(1)] = decodeURIComponent(parts[i]); } catch { ok = false; } // "%E0": not found
      }
    }
    if (ok) return { pattern, page, access, task, params };
  }
  return null;
}

export function navigate(path) {
  const target = path.startsWith("#") ? path : `#${path}`;
  if (location.hash === target) window.dispatchEvent(new HashChangeEvent("hashchange"));
  else location.hash = target;
}

const pageCache = new Map();

/** Import pages/<name>.js; null when the file does not exist yet. */
async function loadPage(name) {
  if (!pageCache.has(name)) {
    pageCache.set(name, import(`./pages/${name}.js`).catch((err) => {
      pageCache.delete(name); // retry next time (a dev may be adding the file right now)
      if (err instanceof SyntaxError) throw err; // a broken page is a bug: show it
      return null;
    }));
  }
  return pageCache.get(name);
}

/**
 * Start routing. `resolve(route)` returns a path to redirect to (auth / role checks) or
 * null to render; `mount(route, module)` renders it and returns an optional cleanup.
 */
export function startRouter({ resolve, mount }) {
  let cleanup = null;
  let generation = 0;

  async function render() {
    const gen = ++generation;
    const { path, query } = parseHash();
    const route = matchRoute(path);
    const redirect = resolve(route, path);
    if (redirect && redirect !== path) {
      navigate(redirect);
      return;
    }
    if (typeof cleanup === "function") {
      try { cleanup(); } catch (err) { console.error(err); }
    }
    cleanup = null;
    let module = null;
    let error = null;
    try {
      module = route ? await loadPage(route.page) : null;
    } catch (err) {
      error = err;
    }
    if (gen !== generation) return; // the user navigated again while the page loaded
    const done = await mount({ ...route, path, query, error }, module);
    if (gen === generation) {
      cleanup = done;
    } else if (typeof done === "function") {
      // the user navigated again while render() was still awaiting: the newer page owns
      // `cleanup` now, so release this page's streams and timers right away
      try { done(); } catch (err) { console.error(err); }
    }
  }

  window.addEventListener("hashchange", render);
  render();
  return render;
}
