"""D2: registration and company approval."""

from __future__ import annotations

import threading
import unittest

from hubtest import GORI, TBILISI, err, login, make_client

WALLET = "9xQeWvG816bUx9EPjHmaT23yvVM2ZWbrrpZb9PusVFin"


def register(c, name="Acme Logistics", email="ops@acme.example", **extra):
    return c.post("/api/v1/companies/register",
                  json={"name": name, "field": "logistics", "contact_email": email, **extra})


class RegisterTest(unittest.TestCase):
    def setUp(self):
        self.c = make_client()

    def test_register_signs_in_as_pending(self):
        r = register(self.c, wallet_address=WALLET)
        self.assertEqual(r.status_code, 201, r.text)
        body = r.json()
        self.assertEqual(body["company"]["status"], "pending")
        self.assertEqual(body["company"]["wallet_address"], WALLET)
        me = self.c.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {body['token']}"}).json()
        self.assertEqual(me["company"]["id"], body["company"]["id"])

    def test_duplicates_are_409(self):
        register(self.c)
        r = register(self.c, name="  acme   LOGISTICS ", email="other@acme.example")
        self.assertEqual((r.status_code, err(r)["code"], err(r)["field"]), (409, "conflict", "name"))
        r = register(self.c, name="Acme 2", email="OPS@acme.example")
        self.assertEqual((r.status_code, err(r)["field"]), (409, "contact_email"))

    def test_bad_fields(self):
        cases = [({"name": "A"}, "name"), ({"contact_email": "x"}, "contact_email"),
                 ({"wallet_address": "0xabc"}, "wallet_address"), ({"wallet_address": "O" * 40}, "wallet_address"),
                 ({"field": ""}, "field"), ({"name": 5}, "name"), ({"extra": 1}, "extra")]
        for change, field in cases:
            body = {"name": "Good Name", "field": "retail", "contact_email": "a@b.example", **change}
            r = self.c.post("/api/v1/companies/register", json=body)
            self.assertEqual((r.status_code, err(r)["field"]), (422, field), change)

    def test_concurrent_registration_one_wins(self):
        codes = []

        def go():
            codes.append(register(self.c).status_code)

        ts = [threading.Thread(target=go) for _ in range(8)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.assertEqual(sorted(codes), [201] + [409] * 7)


class ApprovalTest(unittest.TestCase):
    def setUp(self):
        self.c = make_client()
        self.admin = login(self.c)
        pending = self.c.get("/api/v1/companies?status=pending", headers=self.admin).json()["items"]
        self.gori = pending[0]["id"]

    def post(self, action, cid=None, **kw):
        return self.c.post(f"/api/v1/companies/{cid or self.gori}/{action}", headers=self.admin, **kw)

    def test_lifecycle(self):
        self.assertEqual(self.post("approve").json()["status"], "approved")
        r = self.post("suspend", json={"reason": "unpaid"})
        self.assertEqual((r.json()["status"], r.json()["status_reason"]), ("suspended", "unpaid"))
        self.assertEqual(self.post("approve").json()["status"], "approved")
        r = self.post("reject")
        self.assertEqual((r.status_code, err(r)["code"]), (409, "invalid_transition"))

    def test_rejected_is_final(self):
        self.assertEqual(self.post("reject").json()["status"], "rejected")
        for action in ("approve", "suspend", "reject"):
            self.assertEqual(self.post(action).status_code, 409, action)

    def test_admin_only_and_own_company_only(self):
        gori = login(self.c, GORI)
        tbilisi = login(self.c, TBILISI)
        self.assertEqual(self.c.post(f"/api/v1/companies/{self.gori}/approve", headers=gori).status_code, 403)
        self.assertEqual(self.c.get("/api/v1/companies", headers=gori).status_code, 403)
        self.assertEqual(self.c.get(f"/api/v1/companies/{self.gori}", headers=gori).status_code, 200)
        self.assertEqual(self.c.get(f"/api/v1/companies/{self.gori}", headers=tbilisi).status_code, 403)

    def test_unknown_company_and_bad_filter(self):
        self.assertEqual(self.post("approve", cid="co_nope").status_code, 404)
        r = self.c.get("/api/v1/companies?status=nope", headers=self.admin)
        self.assertEqual((r.status_code, err(r)["field"]), (422, "status"))

    def test_reason_validation(self):
        r = self.post("reject", json={"reason": "x" * 501})
        self.assertEqual((r.status_code, err(r)["field"]), (422, "reason"))
        r = self.post("reject", json={"why": "x"})
        self.assertEqual((r.status_code, err(r)["field"]), (422, "why"))


if __name__ == "__main__":
    unittest.main()
