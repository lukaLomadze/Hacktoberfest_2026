"""Unit tests for the numeric core: data, contract, models, privacy, metrics."""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fedheart import config, data, features, metrics, models, privacy  # noqa: E402

CONTRACT = {"features": [{"name": "age"}, {"name": "sex"}, {"name": "cp", "encoding": "categorical"},
                         {"name": "chol", "missing": "impute_mean_with_indicator"}],
            "quality_rules": [{"column": "chol", "rule": "zero_means_missing"}]}


class TestData(unittest.TestCase):
    def test_sites_load_with_documented_sizes(self):
        sizes = {s["id"]: len(data.load_site(s["id"])[1]) for s in config.SITES}
        self.assertEqual(sizes, {"cleveland": 303, "hungarian": 294, "switzerland": 123, "va": 200})

    def test_split_is_stratified_disjoint_and_reproducible(self):
        _, y = data.load_site("switzerland")
        a = data.stratified_split(y, 7)
        b = data.stratified_split(y, 7)
        self.assertTrue(all(np.array_equal(a[k], b[k]) for k in a))
        allidx = np.concatenate(list(a.values()))
        self.assertEqual(len(allidx), len(set(allidx.tolist())))
        self.assertEqual(len(allidx), len(y))
        self.assertGreater(y[a["test"]].sum(), 0)


class TestContract(unittest.TestCase):
    def test_rejects_unknown_columns_and_destructive_rules(self):
        with self.assertRaises(ValueError):
            features.validate_contract({"features": [{"name": "weight"}]})
        with self.assertRaises(ValueError):  # 0 is a real value for sex
            features.validate_contract({"features": [{"name": "age"}],
                                        "quality_rules": [{"column": "sex", "rule": "zero_means_missing"}]})

    def test_switzerland_cholesterol_zeros_become_missing(self):
        t, _ = data.load_site("switzerland")
        c = features.validate_contract(CONTRACT)
        self.assertEqual(features.contract_impact(t, c)["features"]["chol"]["missing_rate_after_rules"], 1.0)

    def test_federated_stats_equal_pooled_stats(self):
        c = features.validate_contract(CONTRACT)
        tot, pooled = None, []
        for s in config.SITES:
            t, y = data.load_site(s["id"])
            v = features.local_stats(t, y, c)
            tot = v if tot is None else tot + v
            pooled.append(features.apply_rules(t, c["quality_rules"])["chol"])
        norm = features.finalize_norm(c, tot)
        allchol = np.concatenate(pooled)
        allchol = allchol[~np.isnan(allchol)]
        self.assertAlmostEqual(norm["features"]["chol"]["mean"], allchol.mean(), places=8)
        self.assertAlmostEqual(norm["features"]["chol"]["std"], allchol.std(), places=6)

    def test_encoding_shape_and_imputation(self):
        c = features.validate_contract(CONTRACT)
        t, y = data.load_site("switzerland")
        norm = features.finalize_norm(c, features.local_stats(t, y, c) + features.local_stats(
            *data.load_site("cleveland"), c))
        X = features.encode(t, c, norm)
        self.assertEqual(X.shape, (123, len(features.encoded_names(c))))
        self.assertTrue(np.all(X[:, features.encoded_names(c).index("chol__missing")] == 1.0))
        self.assertFalse(np.isnan(X).any())


class TestModels(unittest.TestCase):
    def _gradcheck(self, spec):
        rng = np.random.default_rng(0)
        X, y = rng.normal(size=(7, 4)), (rng.random(7) > 0.5).astype(float)
        m = models.make({**spec, "l2": 0.0}, 4)
        w = m.init(1) + rng.normal(0, 0.1, m.n_params)
        g = m.per_example_grads(w, X, y).mean(axis=0)
        num = np.zeros_like(w)
        for i in range(len(w)):
            e = np.zeros_like(w)
            e[i] = 1e-6
            num[i] = (m.loss(w + e, X, y) - m.loss(w - e, X, y)) / 2e-6
        np.testing.assert_allclose(g, num, atol=1e-6)

    def test_logistic_gradient(self):
        self._gradcheck({"type": "logistic"})

    def test_mlp_gradient(self):
        self._gradcheck({"type": "mlp", "hidden": 3})

    def test_dp_train_without_noise_reduces_loss(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 3))
        y = (X[:, 0] > 0).astype(float)
        m = models.make({"type": "logistic", "l2": 0.0}, 3)
        w, _ = models.dp_train(m, m.init(0), X, y, 20, 0.5, 1.0, 0.0, rng)
        self.assertLess(m.loss(w, X, y), 0.5)


class TestPrivacy(unittest.TestCase):
    def test_masks_cancel_and_hide(self):
        ids = ["a", "b", "c"]
        keys = {}
        kp = {i: privacy.dh_keypair() for i in ids}
        for i in ids:
            keys[i] = {j: privacy.pair_key(kp[i][0], kp[j][1], i, j) for j in ids if j != i}
        xs = {i: np.array([0.1 * k, -2.5, 1e3]) for k, i in enumerate(ids)}
        masked = {i: privacy.mask_vector(xs[i], i, ids, keys[i], "round-1") for i in ids}
        np.testing.assert_allclose(privacy.unmask_sum(masked.values()), sum(xs.values()), atol=1e-6)
        # one message alone decodes to garbage, not to the update
        self.assertGreater(np.abs(privacy.decode_fixed(masked["a"]) - xs["a"]).min(), 1e3)
        # a different purpose gives different masks (no reuse across rounds)
        again = privacy.mask_vector(xs["a"], "a", ids, keys["a"], "round-2")
        self.assertFalse(np.array_equal(again, masked["a"]))

    def test_missing_participant_breaks_the_sum(self):
        ids = ["a", "b", "c"]
        kp = {i: privacy.dh_keypair() for i in ids}
        keys = {i: {j: privacy.pair_key(kp[i][0], kp[j][1], i, j) for j in ids if j != i} for i in ids}
        two = [privacy.mask_vector(np.ones(3), i, ids, keys[i], "r") for i in ("a", "b")]
        self.assertGreater(np.abs(privacy.unmask_sum(two) - 2).max(), 1e3)

    def test_accounting_round_trip(self):
        rho = privacy.rho_from_epsilon(8.0, 1e-5)
        self.assertAlmostEqual(privacy.epsilon_from_rho(rho, 1e-5), 8.0, places=9)
        # epochs steps at sigma_for cost exactly rho_round
        s = privacy.sigma_for(180, 1.0, 5, rho / 40)
        self.assertAlmostEqual(5 * privacy.step_rho(180, 1.0, s), rho / 40, places=12)


class TestHospitalNeverReusesAMask(unittest.TestCase):
    """Two answers under one mask purpose would let the coordinator subtract them (the
    masks cancel) and see one hospital's update, so a hospital answers each purpose once."""

    def setUp(self):
        import tempfile

        from fedheart import hospital
        from fedheart.transport import RemoteError
        self.RemoteError = RemoteError
        self.tmp = tempfile.TemporaryDirectory()
        self.h = hospital.Hospital("cleveland", pathlib.Path(self.tmp.name), "rule", config.DEFAULT_MODEL,
                                   "low", config.SEED, noise_seed=1)
        _, peer_public = privacy.dh_keypair()
        self.participants = ["node_cleveland", "node_va"]
        self.h.roster_({"job_id": "job-test", "privacy": {"epsilon": 8.0, "delta": 1e-5, "max_rounds": 10,
                                                          "clip": 1.0},
                        "participants": [{"node_id": "node_cleveland", "dh_public": format(self.h.dh_public, "x")},
                                         {"node_id": "node_va", "dh_public": format(peer_public, "x")}]})
        self.contract = features.validate_contract(CONTRACT)

    def tearDown(self):
        self.tmp.cleanup()

    def test_masked_statistics_are_sent_once(self):
        req = {"contract": self.contract, "participants": self.participants}
        self.h.stats(req)
        with self.assertRaises(self.RemoteError) as e:
            self.h.stats(req)
        self.assertEqual(e.exception.status, 409)

    def test_a_round_is_trained_and_sent_once(self):
        t, y = self.h._split["train"]
        self.h.norm_({"contract": self.contract,
                      "norm": features.finalize_norm(self.contract, features.local_stats(t, y, self.contract))})
        spec = models.validate_spec({"type": "logistic", "l2": 0.01})
        params = models.make(spec, len(self.h.input_names)).init(0)
        self.h.plans[1] = {"participate": True, "epochs": 2, "lr": 0.1, "reasoning": "test"}
        req = {"round": 1, "params": [float(v) for v in params], "model": spec,
               "participants": self.participants, "weight": 0.5}
        self.h.train(req)
        spent = self.h.rho_spent
        req2 = dict(req, params=[0.1] * len(params))  # a different model: the leak a repeat would allow
        with self.assertRaises(self.RemoteError) as e:
            self.h.train(req2)
        self.assertEqual(e.exception.status, 409)
        self.assertEqual(self.h.rho_spent, spent)  # refused before any privacy was spent


class TestMetrics(unittest.TestCase):
    def test_histogram_auc_matches_exact(self):
        rng = np.random.default_rng(3)
        y = (rng.random(400) > 0.4).astype(float)
        p = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.2, 400), 0, 1)
        exact = metrics.auc(y, p)
        hp, hn = metrics.histograms(y, p)
        self.assertAlmostEqual(metrics.auc_from_histograms(hp, hn), exact, delta=0.01)

    def test_pool_combines_sites(self):
        a = metrics.site_report([1, 0, 1], [0.9, 0.2, 0.6])
        b = metrics.site_report([1, 1], [0.8, 0.3])
        pooled = metrics.pool({"a": a, "b": b})
        self.assertEqual(pooled["n"], 5)
        self.assertIsNone(b["auc"])  # one class only: no AUC, never a fake 0.5


if __name__ == "__main__":
    unittest.main()
