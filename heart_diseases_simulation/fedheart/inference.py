"""Use the published model: load models/heart_model.json and score a patient.

    from fedheart.inference import HeartModel
    m = HeartModel.load("models/heart_model.json")
    m.predict({"age": 63, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233, ...})

Missing values (None or an absent key) are handled exactly as in training: the contract's
quality rules, federated-mean imputation, and missing indicators if the contract has them.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np

from . import config, features, models


class HeartModel:
    def __init__(self, artifact: dict):
        if artifact.get("format") != "fedheart-model/1":
            raise ValueError("not a fedheart model file")
        self.artifact = artifact
        self.contract = artifact["contract"]
        self.norm = {"features": artifact["normalization"]}
        self.inputs = artifact["inputs"]
        self.model = models.make(artifact["model"], len(self.inputs))
        self.params = np.asarray(artifact["params"], dtype=float)
        digest = hashlib.sha256(json.dumps([round(float(v), 10) for v in self.params]).encode()).hexdigest()
        if digest != artifact["sha256"]:
            raise ValueError("model parameters do not match their sha256: the file was altered")

    @classmethod
    def load(cls, path=None):
        path = pathlib.Path(path or config.MODELS_DIR / "heart_model.json")
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def predict_proba(self, record: dict) -> float:
        unknown = set(record) - set(config.FEATURE_NAMES)
        if unknown:
            raise ValueError(f"unknown fields {sorted(unknown)}; fields are {config.FEATURE_NAMES}")
        X = features.encode_record(record, self.contract, self.norm)
        return float(self.model.predict_proba(self.params, X)[0])

    def explain(self, record: dict):
        """Per-input contribution to the log-odds (logistic models only)."""
        if self.artifact["model"]["type"] != "logistic":
            return None
        x = features.encode_record(record, self.contract, self.norm)[0]
        contrib = self.params[1:] * x
        order = np.argsort(-np.abs(contrib))
        return [{"input": self.inputs[i], "contribution": round(float(contrib[i]), 4)} for i in order if contrib[i]]

    def predict(self, record: dict, threshold: float = 0.5) -> dict:
        p = self.predict_proba(record)
        out = {"probability_heart_disease": round(p, 4), "prediction": "heart disease" if p >= threshold else "no heart disease",
               "threshold": threshold, "model": self.artifact["model"], "model_sha256": self.artifact["sha256"]}
        ex = self.explain(record)
        if ex is not None:
            out["top_contributions"] = ex[:5]
        return out
