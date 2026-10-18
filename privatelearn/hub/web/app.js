// The platform web app shell: session, role-based navigation, routing, and the small
// shared components every page uses (h, statusPill, errorBox, fieldErrors).
// Pages import what they need from here: `import { h, statusPill } from "../app.js";`

import { api, ApiError } from "./api.js";
import { homeFor, navigate, startRouter } from "./router.js";

// ---------------------------------------------------------------- tiny DOM helper

/**
 * h("button.btn.primary", {onclick, type: "submit"}, "Save") → an element.
 * Children may be strings (always set as text, never HTML), nodes, arrays, or null.
 */
export function h(tag, attrs = {}, ...children) {
  const [name, ...classes] = tag.split(".");
  const el = document.createElement(name || "div");
  if (classes.length) el.className = classes.join(" ");
  // anything but a plain attribute object is the first child (h("td", 3) shows "3")
  if (attrs !== null && (typeof attrs !== "object" || attrs instanceof Node || Array.isArray(attrs))) {
    children.unshift(attrs);
    attrs = {};
  }
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (k === "class") el.className = [el.className, v].filter(Boolean).join(" ");
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, v);
  }
  const add = (c) => {
    if (c === null || c === undefined || c === false) return;
    if (Array.isArray(c)) c.forEach(add);
    else el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  };
  children.forEach(add);
  return el;
}

// ---------------------------------------------------------------- status pill

// Tone per known enum value (PLATFORM-API §2). Anything not listed — including enum values
// added after this file was written — renders as a plain neutral pill showing its raw text.
const TONES = {
  ok: ["approved", "active", "succeeded", "online", "trained", "ready", "done", "published", "sent"],
  bad: ["rejected", "failed", "offline", "suspended", "cancelled", "excluded"],
  busy: ["training", "running", "busy", "probing", "planning", "masking", "evaluating", "key_exchange",
         "schema_negotiation", "normalization"],
  wait: ["pending", "requested", "queued", "waiting", "waiting_for_nodes", "node_connected", "schema_proposed",
         "schema_confirmed", "draft"],
};
const TONE_OF = new Map(Object.entries(TONES).flatMap(([tone, values]) => values.map((v) => [v, tone])));

/** A pill for any status/enum value. Unknown values are shown as text, never dropped. */
export function statusPill(value) {
  const text = value === null || value === undefined || value === "" ? "—" : String(value);
  const tone = TONE_OF.get(text) || "unknown";
  return h("span.pill", { class: `pill-${tone}`, title: text }, text.replaceAll("_", " "));
}

// ---------------------------------------------------------------- errors

/** A readable message for any thrown value (ApiError carries the §1.1 code). */
export function errorText(err) {
  if (err instanceof ApiError) {
    return err.status && !err.code.startsWith("http_") ? `${err.message} (${err.code})` : err.message;
  }
  return err && err.message ? err.message : String(err);
}

export function errorBox(err) {
  return h("div.error-box", { role: "alert" }, errorText(err));
}

/**
 * Show a 422's `field` next to the matching input of `form` (by name; "a.b.c" also tries "c").
 * Returns true when it found the input, so the caller can skip the generic message.
 */
export function fieldErrors(form, err) {
  form.querySelectorAll(".field-error").forEach((n) => n.remove());
  form.querySelectorAll("[aria-invalid]").forEach((n) => n.removeAttribute("aria-invalid"));
  if (!(err instanceof ApiError) || !err.field) return false;
  const input = form.elements.namedItem(err.field) || form.elements.namedItem(err.field.split(".").pop());
  if (!input || !(input instanceof HTMLElement)) return false;
  input.setAttribute("aria-invalid", "true");
  input.insertAdjacentElement("afterend", h("p.field-error", err.message));
  input.focus();
  return true;
}

// ---------------------------------------------------------------- session + meta

export const session = { user: null, company: null };
let metaPromise = null;

/** GET /meta, once. Pages read enums and available_model_types from here, never hardcode them. */
export function meta() {
  if (!metaPromise) metaPromise = api.get("/meta").catch((err) => { metaPromise = null; throw err; });
  return metaPromise;
}

export function setSession({ token, user, company }) {
  if (token) api.setToken(token);
  session.user = user || null;
  session.company = company || null;
  renderNav();
}

export async function signOut() {
  try { await api.post("/auth/logout"); } catch { /* the token may already be gone */ }
  api.setToken(null);
  setSession({ user: null, company: null });
  navigate("/login");
}

async function restoreSession() {
  if (!api.token) return;
  try {
    setSession(await api.get("/auth/me"));
  } catch (err) {
    if (!(err instanceof ApiError) || err.status !== 401) console.warn("session check failed:", err);
    api.setToken(null);
  }
}

window.addEventListener("api:unauthorized", () => {
  if (session.user) {
    setSession({ user: null, company: null });
    navigate("/login");
  }
});

// ---------------------------------------------------------------- shell

const NAV = {
  admin: [["/admin", "Approvals"], ["/groups/new", "New group"]],
  company: [["/home", "Groups"]],
};

function renderNav() {
  const nav = document.getElementById("nav");
  const who = document.getElementById("who");
  const banner = document.getElementById("banner");
  if (!nav) return;
  const { user, company } = session;
  const current = location.hash.replace(/^#/, "").split("?")[0];
  nav.replaceChildren(...(user ? NAV[user.role] || [] : []).map(([path, label]) =>
    h("a.nav-link", { href: `#${path}`, "aria-current": current === path ? "page" : null }, label)));
  who.replaceChildren(...(user ? [
    h("span.who-name", company ? company.name : "Platform admin"),
    h("span.who-email", user.email),
    h("button.btn.ghost.small", { type: "button", onclick: signOut }, "Sign out"),
  ] : []));
  const pending = user && user.role === "company" && company && company.status !== "approved";
  banner.hidden = !pending;
  if (pending) {
    banner.replaceChildren(statusPill(company.status), " ",
      company.status === "pending"
        ? "An admin has to approve your company before you can join groups. You can look around meanwhile."
        : `Your company is ${company.status.replaceAll("_", " ")}; contact the platform admin.`);
  }
}

const TASK_ISSUES = { E2: 67, E3: 68, E4: 69, E5: 70, E6: 71, E7: 72, E8: 73 };

function placeholder(route) {
  const issue = TASK_ISSUES[route.task];
  return h("section.panel.placeholder",
    h("h2", "Not built yet"),
    h("p", `This page (${route.page}) arrives with task ${route.task}`,
      issue ? [" — ", h("a", { href: `https://github.com/lukaLomadze/Hacktoberfest_2026/issues/${issue}`,
                               target: "_blank", rel: "noopener" }, `#${issue}`)] : null, "."));
}

function resolve(route, path) {
  const user = session.user;
  if (!route) return path === "/" ? homeFor(user && user.role) : null; // unknown path → not-found page
  if (route.access === "guest") return user ? homeFor(user.role) : null;
  if (!user) return "/login";
  if (route.access !== "any" && route.access !== user.role) return homeFor(user.role);
  return null;
}

async function mount(route, module) {
  const main = document.getElementById("main");
  renderNav();
  if (route.error) {
    main.replaceChildren(errorBox(route.error));
    return null;
  }
  if (!route.page) {
    main.replaceChildren(h("section.panel.placeholder", h("h2", "Page not found"),
      h("p", `Nothing lives at #${route.path}. `, h("a", { href: "#/" }, "Go home"), ".")));
    return null;
  }
  if (!module || typeof module.render !== "function") {
    main.replaceChildren(placeholder(route));
    return null;
  }
  const el = h("div.page", { dataset: { page: route.page } });
  main.replaceChildren(el);
  try {
    return await module.render({ el, params: route.params, query: route.query, session, navigate });
  } catch (err) {
    console.error(err);
    el.replaceChildren(errorBox(err));
    return null;
  }
}

async function boot() {
  // the badge shows only when ?mock=1 was asked for; ?rec=1 hides it for screen recordings
  if (api.isMockForced && !new URLSearchParams(location.search).has("rec")) document.getElementById("mock-badge").hidden = false;
  await restoreSession();
  startRouter({ resolve, mount });
}

boot();
