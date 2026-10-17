"""D1: app scaffold, error shape, demo auth, meta, seed, database."""

from __future__ import annotations

import os
import pathlib
import shutil
import tempfile
import threading
import unittest
from unittest import mock

from hubtest import ADMIN_EMAIL, TBILISI, err, login, make_client

from hub.server import enums, paths
from hub.server.app import create_app
from hub.server.context import Settings
from hub.server.db import Database
from hub.server.seed import seed


class ErrorShapeTest(unittest.TestCase):
    def setUp(self):
        self.c = make_client()

    def test_invalid_json_is_400(self):
        r = self.c.post("/api/v1/auth/login", content=b"{bad", headers={"content-type": "application/json"})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(err(r)["code"], "bad_request")

    def test_non_object_body_is_400(self):
        for body in ([1], "x", 3, None):
            r = self.c.post("/api/v1/auth/login", json=body)
            self.assertEqual(r.status_code, 400, body)
            self.assertEqual(err(r)["code"], "bad_request")

    def test_unknown_route_and_method(self):
        r = self.c.get("/api/v1/does-not-exist")
        self.assertEqual((r.status_code, err(r)["code"]), (404, "not_found"))
        r = self.c.get("/api/v1/auth/login")
        self.assertEqual((r.status_code, err(r)["code"]), (405, "method_not_allowed"))

    def test_validation_names_the_field(self):
        r = self.c.post("/api/v1/auth/login", json={"email": "not-an-email"})
        e = err(r)
        self.assertEqual((r.status_code, e["code"], e["field"]), (422, "validation_error", "email"))


class WebMountTest(unittest.TestCase):
    """With hub/web present the static mount at / must not swallow API errors."""

    def setUp(self):
        self.web = tempfile.mkdtemp()
        with open(os.path.join(self.web, "index.html"), "w", encoding="utf-8") as fh:
            fh.write("<!doctype html><title>web</title>")
        self.addCleanup(shutil.rmtree, self.web, True)
        patcher = mock.patch.object(paths, "WEB", pathlib.Path(self.web))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.c = make_client()

    def test_web_is_served_and_api_errors_keep_their_shape(self):
        r = self.c.get("/")
        self.assertEqual((r.status_code, r.text), (200, "<!doctype html><title>web</title>"))
        r = self.c.get("/api/v1/auth/login")  # the path exists, the method does not
        self.assertEqual((r.status_code, err(r)["code"]), (405, "method_not_allowed"))
        for path in ("/api/v1/does-not-exist", "/api", "/api/index.html"):
            r = self.c.get(path)
            self.assertEqual((r.status_code, err(r)["code"]), (404, "not_found"), path)
            self.assertIn("route /api", err(r)["message"])  # ours, not the static files' 404


class AuthTest(unittest.TestCase):
    def setUp(self):
        self.c = make_client()

    def test_login_me_logout(self):
        h = login(self.c)
        me = self.c.get("/api/v1/auth/me", headers=h).json()
        self.assertEqual(me["user"]["role"], "admin")
        self.assertIsNone(me["company"])
        self.assertEqual(self.c.post("/api/v1/auth/logout", headers=h).status_code, 204)
        self.assertEqual(self.c.get("/api/v1/auth/me", headers=h).status_code, 401)
        self.assertEqual(self.c.post("/api/v1/auth/logout", headers=h).status_code, 401)

    def test_company_login_returns_its_company(self):
        r = self.c.post("/api/v1/auth/login", json={"email": "  " + TBILISI.upper() + " "})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["company"]["status"], "approved")
        self.assertEqual(r.json()["user"]["role"], "company")

    def test_unknown_email_is_401(self):
        r = self.c.post("/api/v1/auth/login", json={"email": "nobody@nowhere.example"})
        self.assertEqual((r.status_code, err(r)["code"]), (401, "unauthorized"))

    def test_bad_or_missing_token(self):
        for headers in ({}, {"Authorization": "Bearer nope"}, {"Authorization": "Token abc"},
                        {"Authorization": "Bearer "}, {"Authorization": "bearer"}):
            r = self.c.get("/api/v1/auth/me", headers=headers)
            self.assertEqual(r.status_code, 401, headers)

    def test_scheme_is_case_insensitive(self):
        token = login(self.c)["Authorization"].split()[1]
        r = self.c.get("/api/v1/auth/me", headers={"Authorization": f"bEaReR {token}"})
        self.assertEqual(r.status_code, 200)

    def test_expired_session(self):
        c = make_client(settings=Settings(session_ttl_s=-1))
        h = login(c)
        r = c.get("/api/v1/auth/me", headers=h)
        self.assertEqual(r.status_code, 401)
        self.assertIn("expired", err(r)["message"])


class MetaTest(unittest.TestCase):
    def test_meta(self):
        body = make_client().get("/api/v1/meta").json()
        self.assertEqual(body["version"], enums.API_VERSION)
        self.assertEqual(body["available_model_types"], ["logistic"])
        self.assertEqual(body["enums"]["job_step"][0], "waiting_for_nodes")
        self.assertEqual(body["enums"]["job_step"][-1], "published")
        self.assertIn("payment_pending", body["enums"]["membership_status"])


class SeedTest(unittest.TestCase):
    def test_dev_seed(self):
        db = make_client("dev").app.state.hub.db
        self.assertEqual(db.one("SELECT count(*) AS n FROM users")["n"], 6)
        self.assertEqual(db.one("SELECT count(*) AS n FROM companies WHERE status='pending'")["n"], 1)
        self.assertEqual([r["status"] for r in db.all("SELECT status FROM ml_groups ORDER BY name")],
                         ["draft", "open"])
        ready = db.all("SELECT * FROM memberships WHERE status='ready'")
        self.assertEqual(len(ready), 3)
        for m in ready:
            self.assertIn('"ok":true', m["validation"])
            self.assertNotIn('"missing_required":["', m["schema_"])

    def test_demo_seed_leaves_wiring_to_the_nodes(self):
        db = make_client("demo").app.state.hub.db
        rows = db.all("SELECT status, data, schema_ FROM memberships")
        self.assertEqual({r["status"] for r in rows}, {"approved"})
        self.assertTrue(all(r["data"] is None and r["schema_"] is None for r in rows))

    def test_none_seed_and_bad_mode(self):
        db = make_client("none").app.state.hub.db
        self.assertEqual(db.one("SELECT count(*) AS n FROM users")["n"], 0)
        with self.assertRaises(ValueError):
            seed(Database(":memory:"), "prod")

    def test_seed_is_idempotent_across_restarts(self):
        d = tempfile.mkdtemp()
        try:
            path = os.path.join(d, "hub.db")
            for _ in range(2):
                app = create_app(path, seed_mode="dev")
                n = app.state.hub.db.one("SELECT count(*) AS n FROM companies")["n"]
                app.state.hub.db.close()
            self.assertEqual(n, 5)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class DatabaseTest(unittest.TestCase):
    def test_nested_tx_rolls_back_together(self):
        db = Database(":memory:")
        with self.assertRaises(RuntimeError):
            with db.tx() as c:
                c.execute("INSERT INTO companies (id,name,name_key,field,contact_email,status,created_at) "
                          "VALUES ('co_1','A','a','f','a@b.example','pending','t')")
                with db.tx() as c2:
                    c2.execute("UPDATE companies SET status='approved'")
                raise RuntimeError("abort")
        self.assertIsNone(db.one("SELECT * FROM companies"))

    def test_concurrent_logins(self):
        c = make_client()
        results = []

        def go():
            results.append(c.post("/api/v1/auth/login", json={"email": ADMIN_EMAIL}).status_code)

        threads = [threading.Thread(target=go) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(results, [200] * 20)


if __name__ == "__main__":
    unittest.main()
