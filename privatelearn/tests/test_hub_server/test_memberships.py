"""D4: memberships and their state machine."""

from __future__ import annotations

import threading
import unittest

from hubtest import BATUMI, GORI, RUSTAVI, TBILISI, err, login, make_client


class MembershipTest(unittest.TestCase):
    def setUp(self):
        self.c = make_client()
        self.admin = login(self.c)
        self.rustavi = login(self.c, RUSTAVI)
        groups = self.c.get("/api/v1/groups", headers=self.admin).json()["items"]
        self.gid = next(g["id"] for g in groups if g["status"] == "open")
        self.draft = next(g["id"] for g in groups if g["status"] == "draft")

    def join(self, headers=None):
        return self.c.post(f"/api/v1/groups/{self.gid}/join", headers=headers or self.rustavi)

    def test_join_approve_token_leave(self):
        r = self.join()
        self.assertEqual(r.status_code, 201, r.text)
        m = r.json()
        self.assertEqual((m["status"], m["node"], m["data"], m["has_node_token"]), ("requested", None, None, False))
        self.assertEqual(self.join().status_code, 409)  # one open membership per group
        # no node token before approval
        self.assertEqual(self.c.post(f"/api/v1/memberships/{m['id']}/node-token", headers=self.rustavi).status_code, 409)
        self.assertEqual(self.c.post(f"/api/v1/memberships/{m['id']}/approve", headers=self.admin).json()["status"],
                         "approved")
        tok = self.c.post(f"/api/v1/memberships/{m['id']}/node-token", headers=self.rustavi).json()
        self.assertIn(tok["node_token"], tok["command"])
        self.assertTrue(tok["command"].startswith("python hub/node/node.py --server http://testserver "))
        again = self.c.post(f"/api/v1/memberships/{m['id']}/node-token", headers=self.rustavi).json()
        self.assertNotEqual(tok["node_token"], again["node_token"])
        left = self.c.post(f"/api/v1/memberships/{m['id']}/leave", headers=self.rustavi).json()
        self.assertEqual((left["status"], left["has_node_token"]), ("left", False))
        # after leaving, a fresh request is allowed
        self.assertEqual(self.join().status_code, 201)

    def test_reject_then_rejoin(self):
        m = self.join().json()
        r = self.c.post(f"/api/v1/memberships/{m['id']}/reject", headers=self.admin, json={"reason": "not a fit"})
        self.assertEqual((r.json()["status"], r.json()["status_reason"]), ("rejected", "not a fit"))
        self.assertEqual(self.c.post(f"/api/v1/memberships/{m['id']}/approve", headers=self.admin).status_code, 409)
        self.assertEqual(self.c.post(f"/api/v1/memberships/{m['id']}/leave", headers=self.rustavi).status_code, 409)
        self.assertEqual(self.join().status_code, 201)

    def test_who_may_join(self):
        gori = login(self.c, GORI)  # pending company
        r = self.join(gori)
        self.assertEqual((r.status_code, err(r)["code"]), (409, "invalid_transition"))
        self.assertEqual(self.join(self.admin).status_code, 403)
        r = self.c.post(f"/api/v1/groups/{self.draft}/join", headers=self.rustavi)
        self.assertEqual(r.status_code, 404)  # drafts are invisible
        self.assertEqual(self.c.post("/api/v1/groups/grp_missing/join", headers=self.rustavi).status_code, 404)

    def test_suspended_company_cannot_be_approved_into_a_group(self):
        m = self.join().json()
        rid = self.c.get("/api/v1/auth/me", headers=self.rustavi).json()["company"]["id"]
        self.c.post(f"/api/v1/companies/{rid}/suspend", headers=self.admin)
        r = self.c.post(f"/api/v1/memberships/{m['id']}/approve", headers=self.admin)
        self.assertEqual((r.status_code, err(r)["code"]), (409, "invalid_transition"))

    def test_privacy_between_companies(self):
        m = self.join().json()
        tbilisi = login(self.c, TBILISI)
        self.assertEqual(self.c.get(f"/api/v1/memberships/{m['id']}", headers=tbilisi).status_code, 403)
        self.assertEqual(self.c.post(f"/api/v1/memberships/{m['id']}/leave", headers=tbilisi).status_code, 403)
        self.assertEqual(self.c.post(f"/api/v1/memberships/{m['id']}/node-token", headers=tbilisi).status_code, 403)
        own = self.c.get("/api/v1/memberships", headers=tbilisi).json()["items"]
        self.assertEqual({x["company_name"] for x in own}, {"Tbilisi Central Hospital"})
        self.assertEqual(len(self.c.get("/api/v1/memberships", headers=self.admin).json()["items"]), 4)
        self.assertEqual(self.c.post(f"/api/v1/memberships/{m['id']}/leave", headers=self.admin).status_code, 403)

    def test_filters(self):
        self.join()
        r = self.c.get(f"/api/v1/memberships?group_id={self.gid}&status=requested", headers=self.admin)
        self.assertEqual(len(r.json()["items"]), 1)
        r = self.c.get("/api/v1/memberships?status=bogus", headers=self.admin)
        self.assertEqual((r.status_code, err(r)["field"]), (422, "status"))

    def test_seeded_ready_membership_shape(self):
        batumi = login(self.c, BATUMI)
        m = self.c.get("/api/v1/memberships", headers=batumi).json()["items"][0]
        self.assertEqual(m["status"], "ready")
        self.assertEqual(m["data"]["row_count"], 120)
        self.assertIn("glucose_mmol_l", m["data"]["columns"])
        self.assertEqual(m["schema"]["missing_required"], [])
        bmi = next(x for x in m["schema"]["mapping"] if x["feature"] == "bmi")
        self.assertEqual((bmi["source"], bmi["factor"]), (["weight_kg", "height_m"], None))

    def test_concurrent_joins_one_wins(self):
        codes = []

        def go():
            codes.append(self.join().status_code)

        ts = [threading.Thread(target=go) for _ in range(8)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.assertEqual(sorted(codes), [201] + [409] * 7)


if __name__ == "__main__":
    unittest.main()
