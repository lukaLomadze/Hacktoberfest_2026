"""A deterministic test fixture that maps the three Pima hospitals' columns.

This lives under tests/ — NOT in the runtime. The runtime has no dataset knowledge;
the agents (LLM) do this. Tests need a deterministic stand-in, so here it is.
"""

import re

from agents.decider import RuleDecider


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


class PimaDecider(RuleDecider):
    """Maps the Pima hospitals' local column names to semantic names (name + factor)."""

    LABELS = {"outcome", "label", "diabetic", "diabetes"}
    ALIASES = {
        "pregnancies": ("pregnancies", "count", 1.0),
        "numpregnancies": ("pregnancies", "count", 1.0),
        "glucose": ("glucose", "mg/dL", 1.0),
        "glucosemgdl": ("glucose", "mg/dL", 1.0),
        "glucosemmoll": ("glucose", "mmol/L", 18.0182),
        "bloodpressure": ("blood_pressure", "mmHg", 1.0),
        "bpsys": ("blood_pressure", "mmHg", 1.0),
        "bpkpa": ("blood_pressure", "kPa", 7.50062),
        "bmi": ("bmi", "kg/m^2", 1.0),
        "insulin": ("insulin", "uU/mL", 1.0),
        "insulinpmoll": ("insulin", "pmol/L", 1.0 / 6.945),
        "age": ("age", "years", 1.0),
        "agemonths": ("age", "months", 1.0 / 12.0),
    }

    def describe(self, columns, fallback, profile=None):
        label = None
        cols = []
        by_norm = {_norm(c): c for c in columns}
        for n, c in by_norm.items():
            if n in self.LABELS:
                label = c
            elif n in self.ALIASES:
                sem, unit, factor = self.ALIASES[n]
                cols.append({"source": [c], "semantic": sem, "unit": unit, "factor": factor})
        # BMI derived from weight + height (h1)
        h = [c for n, c in by_norm.items() if n == "heightm"]
        w = [c for n, c in by_norm.items() if n == "weightkg"]
        if h and w and not any(c["semantic"] == "bmi" for c in cols):
            cols.append({"source": [w[0], h[0]], "semantic": "bmi", "unit": "kg, m", "factor": None})
        if label is None:
            label = columns[-1]
        return {"label": label, "columns": cols, "reasoning": "test fixture", "made_by": "rule"}

    def contract(self, proposals, fallback, feature_names=None):
        return fallback()
