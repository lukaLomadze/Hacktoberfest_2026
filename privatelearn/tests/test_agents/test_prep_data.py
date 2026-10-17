"""Tests for T1 data prep (stdlib unittest). Uses the committed data/raw/diabetes.csv."""

import csv
import filecmp
import os
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from pipeline import prep_data as pd_

RAW = ROOT / "data" / "raw" / "diabetes.csv"

LOCAL_HEADERS = {
    0: ["Age", "BMI", "Glucose", "BloodPressure", "Insulin", "Pregnancies", "Outcome"],
    1: ["age_months", "glucose_mmol_l", "bp_kpa", "height_m", "weight_kg", "insulin_pmol_l",
        "pregnancies", "outcome"],
    2: ["age", "bmi", "glucose", "bp_sys", "insulin_pmol_l", "num_pregnancies", "outcome"],
}


def to_canonical(h, r):
    """Reference local→canonical conversion straight from CONTRACTS.md §4."""
    f = float
    if h == 0:
        return [f(r["Pregnancies"]), f(r["Glucose"]), f(r["BloodPressure"]), f(r["BMI"]),
                f(r["Insulin"]), f(r["Age"])], int(r["Outcome"])
    if h == 1:
        return [f(r["pregnancies"]), f(r["glucose_mmol_l"]) * 18.0182, f(r["bp_kpa"]) * 7.50062,
                f(r["weight_kg"]) / f(r["height_m"]) ** 2, f(r["insulin_pmol_l"]) / 6.945,
                f(r["age_months"]) / 12], int(r["outcome"])
    return [f(r["num_pregnancies"]), f(r["glucose"]), f(r["bp_sys"]), f(r["bmi"]),
            f(r["insulin_pmol_l"]) / 6.945, f(r["age"])], int(r["outcome"])


def read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return reader.fieldnames, list(reader)


class TestPrepData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = pathlib.Path(cls.tmp.name)
        cls.manifest = pd_.prepare(RAW, cls.out)
        cls.raw = pd_.load_raw(RAW)
        cls.clean, _ = pd_.impute(cls.raw)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def hospital(self, h):
        return read(self.out / "hospitals" / f"hospital_{h}.csv")

    def test_three_hospitals_of_120_rows_with_local_headers(self):
        for h in range(3):
            header, rows = self.hospital(h)
            self.assertEqual(header, LOCAL_HEADERS[h])
            self.assertEqual(len(rows), 120)

    def test_deterministic_byte_identical(self):
        with tempfile.TemporaryDirectory() as other:
            pd_.prepare(RAW, other)
            for rel in ["hospitals/hospital_0.csv", "hospitals/hospital_1.csv",
                        "hospitals/hospital_2.csv", "hospitals/manifest.json", "holdout/test.csv"]:
                self.assertTrue(filecmp.cmp(self.out / rel, os.path.join(other, rel), shallow=False), rel)

    def test_rows_are_disjoint_and_holdout_is_20_percent(self):
        ids = [set(self.manifest["row_ids"][f"h{h}"]) for h in range(3)]
        hold = set(self.manifest["holdout"]["row_ids"])
        self.assertEqual(len(set.union(*ids)), 360)
        self.assertEqual(len(hold), 154)
        self.assertFalse(hold & set.union(*ids))
        _, test_rows = read(self.out / "holdout" / "test.csv")
        self.assertEqual(len(test_rows), 154)

    def test_split_by_age_bands_not_label(self):
        ages = [self.clean["Age"][self.manifest["row_ids"][f"h{h}"]] for h in range(3)]
        self.assertLessEqual(ages[0].max(), ages[1].min())
        self.assertLessEqual(ages[1].max(), ages[2].min())
        for h in range(3):
            b = self.manifest["per_hospital_class_balance"][f"h{h}"]
            self.assertGreater(b["positive"], 0, f"h{h} has no positives")
            self.assertGreater(b["negative"], 0, f"h{h} has no negatives")

    def test_local_units_convert_back_to_canonical_within_1e6(self):
        for h in range(3):
            _, rows = self.hospital(h)
            ids = self.manifest["row_ids"][f"h{h}"]
            expected = np.column_stack([self.clean[c][ids] for c in pd_.CANONICAL])
            labels = self.clean[pd_.LABEL][ids]
            got = [to_canonical(h, r) for r in rows]
            np.testing.assert_allclose(np.array([v for v, _ in got]), expected, rtol=0, atol=1e-6)
            np.testing.assert_array_equal(np.array([y for _, y in got]), labels)

    def test_imputation_removes_zeros(self):
        for c in pd_.IMPUTE_ZEROS:
            self.assertTrue(np.all(self.clean[c] != 0), c)
        self.assertTrue((self.raw["Insulin"] == 0).any())  # the raw data really had zeros

    def test_manifest_stats_are_full_dataset_canonical_order(self):
        X = np.column_stack([self.clean[c] for c in pd_.CANONICAL])
        np.testing.assert_allclose(self.manifest["feature_mean"], X.mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(self.manifest["feature_std"], X.std(axis=0), atol=1e-12)
        self.assertEqual(self.manifest["canonical_features"],
                         ["pregnancies", "glucose", "blood_pressure", "bmi", "insulin", "age"])
        for key in ("seed", "n_per_hospital", "feature_mean", "feature_std",
                    "per_hospital_class_balance", "row_ids"):
            self.assertIn(key, self.manifest)

    def test_committed_outputs_match_regeneration(self):
        def text(p):  # line endings differ by OS / git checkout; content must not
            return p.read_bytes().replace(b"\r\n", b"\n")

        committed = ROOT / "data"
        for rel in ["hospitals/hospital_0.csv", "hospitals/hospital_1.csv",
                    "hospitals/hospital_2.csv", "hospitals/manifest.json", "holdout/test.csv"]:
            self.assertTrue((committed / rel).exists(), f"{rel} not committed; run prep_data.py")
            self.assertEqual(text(self.out / rel), text(committed / rel), rel)


if __name__ == "__main__":
    unittest.main(verbosity=2)
