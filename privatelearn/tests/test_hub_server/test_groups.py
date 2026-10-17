"""D3: groups and TaskSpec validation."""

from __future__ import annotations

import copy
import unittest

from hubtest import GORI, RUSTAVI, TBILISI, err, login, make_client

from hub.server import taskspec
from hub.server.db import now_iso
from hub.server.errors import ApiError
from hub.server.seed import DIABETES_SPEC

AVAILABLE = ("logistic",)
ALL = ("logistic", "mlp", "cnn", "vit", "llm_lora")


def spec(**changes):
    s = copy.deepcopy(DIABETES_SPEC)
    s.update(changes)
    return s


class TaskSpecTest(unittest.TestCase):
    def bad(self, s, field, available=AVAILABLE):
        with self.assertRaises(ApiError) as cm:
            taskspec.normalize(s, available)
        self.assertEqual(cm.exception.status, 422)
        self.assertEqual(cm.exception.field, field)

    def test_defaults_are_filled(self):
        out = taskspec.normalize({"question": "Will this order be late?", "task_type": "binary_classification",
                                  "label": {"name": "late", "classes": ["no", "yes"], "positive": "yes"},
                                  "input_schema": [{"name": "distance_km", "type": "number"}],
                                  "model": {"type": "logistic"}}, AVAILABLE)
        self.assertEqual(out["training"], {"rounds": 4, "min_participants": 2, "aggregation": "adaptive"})
        self.assertEqual(out["privacy"]["dp"], {"enabled": True, "epsilon_per_round": 1.5, "delta": 1e-5})
        self.assertEqual(out["metric"], "auc")
        self.assertEqual(out["input_schema"][0], {"name": "distance_km", "type": "number", "unit": "",
                                                  "description": "", "required": True})
        self.assertEqual(taskspec.normalize(out, AVAILABLE), out)  # normalising is idempotent

    def test_type_traps(self):
        self.bad(spec(training={"rounds": True}), "task_spec.training.rounds")      # bool is not an int
        self.bad(spec(training={"rounds": 2.0}), "task_spec.training.rounds")
        self.bad(spec(privacy={"dp": {"epsilon_per_round": float("nan")}}), "task_spec.privacy.dp.epsilon_per_round")
        self.bad(spec(privacy={"dp": {"epsilon_per_round": 0}}), "task_spec.privacy.dp.epsilon_per_round")
        self.bad(spec(model={"type": "logistic", "hyperparams": {"lr": float("inf")}}),
                 "task_spec.model.hyperparams.lr")
        self.bad(spec(model={"type": "logistic", "hyperparams": {"custom": [1, float("nan")]}}),
                 "task_spec.model.hyperparams.custom[1]")

    def test_label_rules(self):
        self.bad(spec(label={"name": "y", "classes": ["no", "yes", "maybe"], "positive": "yes"}),
                 "task_spec.label.classes")
        self.bad(spec(label={"name": "y", "classes": ["Yes", "yes"], "positive": "yes"}), "task_spec.label.classes")
        self.bad(spec(label={"name": "y", "classes": ["no", "yes"], "positive": "maybe"}), "task_spec.label.positive")
        self.bad(spec(label={"name": "Has Diabetes", "classes": ["no", "yes"], "positive": "yes"}),
                 "task_spec.label.name")
        s = spec()
        s["input_schema"] = s["input_schema"] + [{"name": "has_diabetes", "type": "number"}]
        self.bad(s, "task_spec.input_schema[6].name")  # the label may not be an input

    def test_inputs(self):
        s = spec()
        s["input_schema"] = s["input_schema"] + [{"name": "glucose", "type": "number"}]
        self.bad(s, "task_spec.input_schema[6].name")
        s = spec()
        s["input_schema"] = s["input_schema"] + [{"name": "glucose\n", "type": "number"}]
        self.bad(s, "task_spec.input_schema[6].name")  # `$` would let a trailing newline through
        self.bad(spec(input_schema=[]), "task_spec.input_schema")
        self.bad(spec(input_schema=[{"name": "a", "type": "number", "required": False}]), "task_spec.input_schema")
        self.bad(spec(input_schema=[{"name": "a", "type": "number", "shape": [2, 2]}]),
                 "task_spec.input_schema[0].shape")

    def test_model_compatibility(self):
        self.bad(spec(model={"type": "mlp"}), "task_spec.model.type")  # not available yet
        self.bad(spec(input_schema=[{"name": "kind", "type": "category", "categories": ["a", "b"]}]),
                 "task_spec.input_schema[0].type")  # logistic takes numbers only
        self.bad(spec(task_type="regression", label={"name": "cost", "classes": None, "positive": None}),
                 "task_spec.task_type")
        img = spec(task_type="image_classification", model={"type": "cnn"},
                   label={"name": "digit", "classes": [str(i) for i in range(10)]},
                   input_schema=[{"name": "pixels", "type": "image", "shape": [8, 8], "channels": 1},
                                 {"name": "other", "type": "image", "shape": [8, 8], "channels": 1}], metric=None)
        self.bad(img, "task_spec.input_schema", available=ALL)
        img["input_schema"] = img["input_schema"][:1]
        self.assertEqual(taskspec.normalize(img, ALL)["metric"], "accuracy")

    def test_secure_aggregation_needs_two(self):
        self.bad(spec(training={"min_participants": 1}), "task_spec.training.min_participants")
        ok = taskspec.normalize(spec(training={"min_participants": 1},
                                     privacy={"secure_aggregation": False}), AVAILABLE)
        self.assertEqual(ok["training"]["min_participants"], 1)

    def test_dp_off_drops_the_budget(self):
        out = taskspec.normalize(spec(privacy={"dp": {"enabled": False, "epsilon_per_round": 3}}), AVAILABLE)
        self.assertEqual(out["privacy"]["dp"], {"enabled": False, "epsilon_per_round": None, "delta": None})

    def test_unknown_fields(self):
        self.bad(spec(colour="red"), "task_spec.colour")
        self.bad(spec(training={"epochs": 3}), "task_spec.training.epochs")


class GroupRoutesTest(unittest.TestCase):
    def setUp(self):
        self.c = make_client()
        self.admin = login(self.c)
        self.tbilisi = login(self.c, TBILISI)
        groups = self.c.get("/api/v1/groups", headers=self.admin).json()["items"]
        self.diabetes = next(g for g in groups if g["status"] == "open")
        self.draft = next(g for g in groups if g["status"] == "draft")

    def test_company_sees_public_and_own_groups_only(self):
        ids = [g["id"] for g in self.c.get("/api/v1/groups", headers=self.tbilisi).json()["items"]]
        self.assertEqual(ids, [self.diabetes["id"]])
        r = self.c.get(f"/api/v1/groups/{self.draft['id']}", headers=self.tbilisi)
        self.assertEqual(r.status_code, 404)  # a draft does not leak

    def test_group_detail_members(self):
        admin_view = self.c.get(f"/api/v1/groups/{self.diabetes['id']}", headers=self.admin).json()
        self.assertEqual(len(admin_view["members"]), 3)
        self.assertEqual(admin_view["member_count"], 3)
        own = self.c.get(f"/api/v1/groups/{self.diabetes['id']}", headers=self.tbilisi).json()
        self.assertEqual([m["company_name"] for m in own["members"]], ["Tbilisi Central Hospital"])
        gori = login(self.c, GORI)
        outsider = self.c.get(f"/api/v1/groups/{self.diabetes['id']}", headers=gori).json()
        self.assertEqual((outsider["members"], outsider["versions"]), ([], []))

    def test_a_pending_join_request_does_not_reveal_versions(self):
        gid, hub = self.diabetes["id"], self.c.app.state.hub
        with hub.db.tx() as c:
            c.execute("INSERT INTO model_versions (id, group_id, version, job_id, model_type, features, norm, "
                      "weights, metrics, contributors, sha256, created_at) VALUES "
                      "('mv_t', ?, 1, 'job_t', 'logistic', '[]', '{}', '[]', '{}', '[]', 'x', ?)", (gid, now_iso()))
            company = c.execute("SELECT company_id FROM users WHERE email = ?", (RUSTAVI,)).fetchone()[0]
            c.execute("INSERT INTO memberships (id, company_id, group_id, status, requested_at, updated_at) "
                      "VALUES ('mem_t', ?, ?, 'requested', ?, ?)", (company, gid, now_iso(), now_iso()))
        rustavi = login(self.c, RUSTAVI)
        self.assertEqual(self.c.get(f"/api/v1/groups/{gid}", headers=rustavi).json()["versions"], [])
        hub.db.execute("UPDATE memberships SET status = 'approved' WHERE id = 'mem_t'")
        versions = self.c.get(f"/api/v1/groups/{gid}", headers=rustavi).json()["versions"]
        self.assertEqual([v["id"] for v in versions], ["mv_t"])

    def test_create_publish_archive(self):
        r = self.c.post("/api/v1/groups", headers=self.admin,
                        json={"name": "Diabetes v2", "task_spec": DIABETES_SPEC,
                              "pricing": {"enrollment_fee": 0.5, "revenue_share": "by_rows"}})
        self.assertEqual(r.status_code, 201, r.text)
        g = r.json()
        self.assertEqual((g["status"], g["pricing"]["currency"]), ("draft", "SOL"))
        gid = g["id"]
        r = self.c.patch(f"/api/v1/groups/{gid}", headers=self.admin, json={"task_spec": spec(question="Diabetic?")})
        self.assertEqual(r.json()["task_spec"]["question"], "Diabetic?")
        self.assertEqual(self.c.post(f"/api/v1/groups/{gid}/publish", headers=self.admin).json()["status"], "open")
        r = self.c.patch(f"/api/v1/groups/{gid}", headers=self.admin, json={"task_spec": DIABETES_SPEC})
        self.assertEqual((r.status_code, err(r)["code"]), (409, "invalid_transition"))
        self.assertEqual(self.c.patch(f"/api/v1/groups/{gid}", headers=self.admin,
                                      json={"name": "Renamed"}).json()["name"], "Renamed")
        self.assertEqual(self.c.post(f"/api/v1/groups/{gid}/publish", headers=self.admin).status_code, 409)
        self.assertEqual(self.c.post(f"/api/v1/groups/{gid}/archive", headers=self.admin).json()["status"],
                         "archived")
        self.assertEqual(self.c.patch(f"/api/v1/groups/{gid}", headers=self.admin,
                                      json={"name": "x2"}).status_code, 409)
        self.assertEqual(self.c.post(f"/api/v1/groups/{gid}/archive", headers=self.admin).status_code, 409)

    def test_admin_only_and_validation(self):
        r = self.c.post("/api/v1/groups", headers=self.tbilisi, json={"name": "x", "task_spec": DIABETES_SPEC})
        self.assertEqual(r.status_code, 403)
        r = self.c.post("/api/v1/groups", headers=self.admin, json={"name": "Ok name", "task_spec": {}})
        self.assertEqual((r.status_code, err(r)["field"]), (422, "task_spec.question"))
        r = self.c.patch(f"/api/v1/groups/{self.draft['id']}", headers=self.admin, json={"status": "open"})
        self.assertEqual((r.status_code, err(r)["field"]), (422, "status"))
        r = self.c.post("/api/v1/groups", headers=self.admin,
                        json={"name": "Ok name", "task_spec": DIABETES_SPEC, "pricing": {"per_prediction": -1}})
        self.assertEqual((r.status_code, err(r)["field"]), (422, "pricing.per_prediction"))


if __name__ == "__main__":
    unittest.main()
