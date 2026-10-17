// #/login — demo sign-in (PLATFORM-API §5.1): an email, no password.

import { api } from "../api.js";
import { errorBox, fieldErrors, h, setSession } from "../app.js";
import { homeFor } from "../router.js";

async function demoAccounts() {
  // only the mock lists its accounts; the real server has no endpoint that lists users, on
  // purpose (its seeded demo emails are in the server's seed data)
  if (!api.isMock) return [];
  const mock = await import("../mock/server.js");
  return mock.demoAccounts();
}

export async function render({ el, navigate }) {
  const status = h("div");
  const email = h("input", { name: "email", type: "email", autocomplete: "email", required: true,
                             placeholder: "you@company.example", id: "login-email" });
  const submit = h("button.btn.primary", { type: "submit" }, "Sign in");
  const form = h("form.form.panel", { novalidate: true },
    h("div.field", h("label", { for: "login-email" }, "Email"), email,
      h("p.hint", "Demo sign-in: no password. Use a seeded or registered email.")),
    status,
    h("div.form-actions", submit, h("span.muted", "New company? ", h("a", { href: "#/register" }, "Register"))));

  async function signIn(address) {
    status.replaceChildren();
    submit.disabled = true;
    try {
      const res = await api.post("/auth/login", { email: address.trim() });
      setSession(res);
      navigate(homeFor(res.user.role));
    } catch (err) {
      if (!fieldErrors(form, err)) status.replaceChildren(errorBox(err));
    } finally {
      submit.disabled = false;
    }
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    if (!email.value.trim()) {
      email.focus();
      return;
    }
    signIn(email.value);
  });

  const accounts = await demoAccounts();
  el.replaceChildren(h("section.auth",
    h("div", h("h1", "Sign in"), h("p.muted", "Companies train one shared model; no data row leaves a company.")),
    form,
    accounts.length ? h("div.panel.demo-accounts",
      h("h2", "Demo accounts"),
      accounts.map((a) => h("button.btn.ghost.demo-account", { type: "button", onclick: () => signIn(a.email) },
        h("span", a.label), h("span.mono", a.email)))) : null));
  email.focus();
}
