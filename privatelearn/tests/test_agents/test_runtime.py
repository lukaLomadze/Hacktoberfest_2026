"""Tests for the agent runtime and message bus (stdlib unittest, no pytest)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from agents.runtime import BROADCAST, Agent, Message, MessageBus, sanitize


def make_update(hospital="h0", round_=1):
    return {
        "v": "1.0", "round": round_, "hospital": hospital,
        "delta_w": [0.031, -0.012, 0.004, 0.021, -0.008, 0.017, 0.002],
        "loss": 0.512, "n": 120,
        "privacy": {"masked": True, "mask_seed": "b3f8", "dp_eps": 1.5,
                    "dp_mechanism": "gaussian", "dp_sigma": 0.008},
    }


def make_broadcast(round_=1):
    return {
        "v": "1.0", "round": round_, "w": [0.41, 0.88, -0.12, 1.02, 0.31, -0.04, 0.05],
        "strategy": "fedavg", "eps_total": 1.5, "auc": 0.84,
        "losses": [0.51, 0.53, 0.49], "converged": False,
    }


class Hospital(Agent):
    """Sends one update per broadcast it receives (and one at start)."""

    def __init__(self, agent_id, rounds=2):
        super().__init__(agent_id)
        self.rounds = rounds
        self.seen = []
        self.started = False

    def step(self, messages):
        self.seen.extend(messages)
        if not self.started:
            self.started = True
            return [self.send("coord", "update", make_update(self.id, 1))]
        out = []
        for m in messages:
            if m.kind == "broadcast" and m.payload["round"] < self.rounds:
                out.append(self.send("coord", "update", make_update(self.id, m.payload["round"] + 1)))
        return out


class Coordinator(Agent):
    """Broadcasts once it has an update from every hospital for a round."""

    def __init__(self, hospitals):
        super().__init__("coord")
        self.hospitals = set(hospitals)
        self.by_round = {}
        self.broadcasts = []

    def step(self, messages):
        out = []
        for m in messages:
            if m.kind != "update":
                continue
            r = m.payload["round"]
            self.by_round.setdefault(r, set()).add(m.sender)
            if self.by_round[r] == self.hospitals:
                self.broadcasts.append(r)
                out.append(self.send(BROADCAST, "broadcast", make_broadcast(r)))
        return out


class TestSanitize(unittest.TestCase):
    def test_valid_update_and_broadcast_pass(self):
        self.assertEqual(sanitize("update", make_update())["hospital"], "h0")
        self.assertEqual(sanitize("broadcast", make_broadcast())["strategy"], "fedavg")

    def test_invalid_update_rejected_by_contract(self):
        bad = make_update()
        bad["delta_w"] = [1.0]  # the contract's weight count is negotiated, so too short is invalid
        with self.assertRaisesRegex(ValueError, "delta_w"):
            sanitize("update", bad)

    def test_invalid_broadcast_rejected_by_contract(self):
        bad = make_broadcast()
        bad["strategy"] = "magic"
        with self.assertRaisesRegex(ValueError, "strategy"):
            sanitize("broadcast", bad)

    def test_raw_data_keys_rejected_at_any_depth(self):
        with self.assertRaisesRegex(ValueError, "raw-data"):
            sanitize("schema", {"mapping": {"age": "Age"}, "extra": [{"rows": [[1, 2]]}]})
        bad = make_update()
        bad["X"] = [[0.1] * 6]
        with self.assertRaisesRegex(ValueError, "raw-data"):
            sanitize("update", bad)

    def test_non_json_payload_rejected(self):
        with self.assertRaisesRegex(ValueError, "JSON"):
            sanitize("schema", {"obj": object()})
        with self.assertRaisesRegex(ValueError, "JSON"):
            sanitize("schema", {"x": float("nan")})

    def test_payload_is_deep_copied(self):
        src = {"mapping": {"age": "Age"}}
        out = sanitize("schema", src)
        src["mapping"]["age"] = "changed"
        self.assertEqual(out["mapping"]["age"], "Age")

    def test_free_form_kind_passes(self):
        self.assertEqual(sanitize("plan", {"epochs": 4}), {"epochs": 4})


class TestBus(unittest.TestCase):
    def setUp(self):
        self.bus = MessageBus()
        self.hospitals = [self.bus.register(Hospital(h)) for h in ("h0", "h1", "h2")]
        self.coord = self.bus.register(Coordinator(["h0", "h1", "h2"]))

    def test_round_trip_two_rounds(self):
        ticks = self.bus.run(max_ticks=20)
        self.assertEqual(self.coord.broadcasts, [1, 2])
        for h in self.hospitals:
            got = [m.payload["round"] for m in h.seen if m.kind == "broadcast"]
            self.assertEqual(got, [1, 2])
        # tick 0 start, 1 coord broadcasts r1, 2 hospitals send r2, 3 coord broadcasts r2,
        # 4 hospitals receive r2 and stop, quiet.
        self.assertEqual(ticks, 5)

    def test_broadcast_skips_sender(self):
        self.bus.run()
        self.assertFalse(any(m.recipient == "coord" and m.kind == "broadcast" for m in self.bus.log))

    def test_messages_delivered_next_tick_with_ordered_seq(self):
        self.bus.run()
        self.assertEqual([m.seq for m in self.bus.log], list(range(len(self.bus.log))))
        first = [m for m in self.bus.log if m.tick == 1]
        self.assertEqual([m.sender for m in first], ["h0", "h1", "h2"])
        self.assertTrue(all(m.kind == "update" for m in first))

    def test_deterministic(self):
        self.bus.run()
        bus2 = MessageBus()
        for h in ("h0", "h1", "h2"):
            bus2.register(Hospital(h))
        bus2.register(Coordinator(["h0", "h1", "h2"]))
        bus2.run()
        self.assertEqual([m.to_dict() for m in self.bus.log], [m.to_dict() for m in bus2.log])

    def test_recipients_get_independent_copies(self):
        class Mutator(Agent):
            def step(self, messages):
                for m in messages:
                    m.payload["w"][0] = 999.0
                return []

        bus = MessageBus()
        sender = bus.register(Agent("coord"))
        bus.register(Mutator("h0"))
        bus.register(Agent("h1"))
        bus.post(sender.send(BROADCAST, "broadcast", make_broadcast()))
        bus.run_tick()
        to_h1 = [m for m in bus.log if m.recipient == "h1"][0]
        self.assertEqual(to_h1.payload["w"][0], 0.41)

    def test_invalid_message_from_agent_raises(self):
        class Leaky(Agent):
            def step(self, messages):
                return [self.send("coord", "update", {"rows": [[1] * 6]})]

        bus = MessageBus()
        bus.register(Leaky("h0"))
        bus.register(Agent("coord"))
        with self.assertRaisesRegex(ValueError, "raw-data"):
            bus.run_tick()

    def test_unknown_recipient_and_spoofed_sender(self):
        with self.assertRaisesRegex(ValueError, "recipient"):
            self.bus.post(Message("h0", "nobody", "plan", {}))
        with self.assertRaisesRegex(ValueError, "sender"):
            self.bus.post(Message("ghost", "coord", "plan", {}))

        class Spoofer(Agent):
            def step(self, messages):
                return [Message("coord", "h0", "plan", {})]

        bus = MessageBus()
        bus.register(Spoofer("h0"))
        bus.register(Agent("coord"))
        with self.assertRaisesRegex(ValueError, "send as"):
            bus.run_tick()

    def test_duplicate_and_reserved_ids(self):
        with self.assertRaises(ValueError):
            self.bus.register(Agent("h0"))
        with self.assertRaises(ValueError):
            Agent(BROADCAST)

    def test_run_raises_if_never_quiet(self):
        class Chatty(Agent):
            def step(self, messages):
                return [self.send("h1", "plan", {"ping": 1})]

        bus = MessageBus()
        bus.register(Chatty("h0"))
        bus.register(Agent("h1"))
        with self.assertRaises(RuntimeError):
            bus.run(max_ticks=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
