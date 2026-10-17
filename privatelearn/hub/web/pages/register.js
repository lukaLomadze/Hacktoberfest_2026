// #/register — a company signs up (PLATFORM-API §5.2) and waits for admin approval.

import { api } from "../api.js";
import { errorBox, fieldErrors, h, setSession } from "../app.js";

function field(name, label, attrs = {}, hint = null) {
  const id = `reg-${name}`;
  return h("div.field",
    h("label", { for: id }, label),
    h("input", { id, name, ...attrs }),
    hint ? h("p.hint", hint) : null);
}

export async function render({ el, navigate }) {
  const status = h("div");
  const submit = h("button.btn.primary", { type: "submit" }, "Register company");
  const form = h("form.form.panel", { novalidate: true },
    field("name", "Company name", { required: true, autocomplete: "organization" }),
    field("field", "Field", { required: true, placeholder: "healthcare, finance, retail …" },
      "Any field: the platform is not tied to hospitals."),
    field("contact_email", "Contact email", { type: "email", required: true, autocomplete: "email" },
      "You sign in with this email."),
    field("wallet_address", "Wallet address (optional)", { autocomplete: "off", spellcheck: "false" },
      "Reserved for payments, which are not live yet. Nothing reads it today."),
    status,
    h("div.form-actions", submit, h("span.muted", "Already registered? ", h("a", { href: "#/login" }, "Sign in"))));

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    status.replaceChildren();
    const data = Object.fromEntries(new FormData(form));
    const body = { name: data.name.trim(), field: data.field.trim(), contact_email: data.contact_email.trim() };
    if (data.wallet_address.trim()) body.wallet_address = data.wallet_address.trim();
    submit.disabled = true;
    try {
      const res = await api.post("/companies/register", body);
      setSession(res); // the company is `pending`; the shell shows the waiting banner
      navigate("/home");
    } catch (err) {
      if (!fieldErrors(form, err)) status.replaceChildren(errorBox(err));
    } finally {
      submit.disabled = false;
    }
  });

  el.replaceChildren(h("section.auth",
    h("div", h("h1", "Register your company"),
      h("p.muted", "An admin approves new companies. Your data stays with you: you connect it later " +
                   "through your own agent node, which sends only column statistics and masked updates.")),
    form));
  form.elements.namedItem("name").focus();
}
