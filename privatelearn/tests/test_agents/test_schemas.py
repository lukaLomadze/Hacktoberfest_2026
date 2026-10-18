"""Contract validators accept any participant id (X2), not only h0/h1/h2."""

import copy
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from shared import schemas

SAMPLE = json.loads((ROOT / "data" / "sample_trace.json").read_text(encoding="utf-8"))


def update(hospital):
    return {"v": "1.0", "round": 1, "hospital": hospital, "delta_w": [0.0] * 7, "loss": 0.5, "n": 10,
            "privacy": {"masked": True, "dp_eps": 1.5, "dp_mechanism": "gaussian", "dp_sigma": 0.1}}


def renamed(trace, mapping):
    """The trace with every hospital id replaced through `mapping`."""
    t = copy.deepcopy(trace)
    for n in t["nodes"]:
        n["id"] = mapping.get(n["id"], n["id"])
    for r in t["rounds"]:
        for u in r["updates"]:
            u["hospital"] = mapping[u["hospital"]]
    t["attack"]["target"] = mapping[t["attack"]["target"]]
    return t


class TestParticipantIds(unittest.TestCase):
    def test_opaque_ids_pass_the_update_validator(self):
        for hid in ("h0", "node_9f3a", "acme-clinic", "Z.lab", "h12"):
            schemas.validate_update(update(hid))
            self.assertTrue(schemas.is_participant_id(hid))

    def test_bad_ids_fail(self):
        for hid in ("", "coord", "-x", "a b", "x" * 65, 3, None):
            with self.assertRaises(ValueError):
                schemas.validate_update(update(hid))
            self.assertFalse(schemas.is_participant_id(hid))

    def test_trace_with_opaque_ids_is_valid(self):
        schemas.validate_trace(renamed(SAMPLE, {"h0": "node_9f3a", "h1": "node_2b71", "h2": "acme-clinic"}))

    def test_trace_rejects_an_update_from_an_unknown_node(self):
        t = copy.deepcopy(SAMPLE)
        t["rounds"][0]["updates"][0]["hospital"] = "node_9f3a"
        with self.assertRaises(ValueError):
            schemas.validate_trace(t)

    def test_trace_rejects_attack_on_unknown_node_and_missing_coordinator(self):
        t = copy.deepcopy(SAMPLE)
        t["attack"]["target"] = "node_9f3a"
        with self.assertRaises(ValueError):
            schemas.validate_trace(t)
        t = copy.deepcopy(SAMPLE)
        t["nodes"] = [n for n in t["nodes"] if n["role"] != "coordinator"]
        with self.assertRaises(ValueError):
            schemas.validate_trace(t)

    def test_node_count_must_match_meta(self):
        t = copy.deepcopy(SAMPLE)
        t["meta"]["hospitals"] = 4
        with self.assertRaises(ValueError):
            schemas.validate_trace(t)


if __name__ == "__main__":
    unittest.main(verbosity=2)
