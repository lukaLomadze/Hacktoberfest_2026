"""Score a patient with the published model (models/heart_model.json).

    python predict.py --example
    python predict.py --age 63 --sex 1 --cp 4 --trestbps 145 --chol 233 --thalach 150 --exang 0 --oldpeak 2.3
    python predict.py --json patient.json            # {"age": 63, "sex": 1, ...}; omitted fields = missing
    python predict.py --model runs/<id>/model.json --example

Not a medical device: a hackathon model trained on 550 patients from 1980s cohorts.
"""

from __future__ import annotations

import argparse
import json
import sys

from fedheart import config
from fedheart.inference import HeartModel

EXAMPLE = {"age": 63, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2, "thalach": 150,
           "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6}


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=None, help="model file (default models/heart_model.json)")
    ap.add_argument("--json", default=None, help="a JSON file with one patient")
    ap.add_argument("--example", action="store_true", help="score a built-in example patient")
    for c in config.COLUMNS:
        ap.add_argument(f"--{c['name']}", type=float, default=None, help=f"{c['meaning']} ({c['unit']})")
    args = ap.parse_args(argv)

    if args.json:
        with open(args.json, encoding="utf-8") as fh:
            record = json.load(fh)
    elif args.example:
        record = dict(EXAMPLE)
    else:
        record = {c: getattr(args, c) for c in config.FEATURE_NAMES if getattr(args, c) is not None}
        if not record:
            ap.error("give patient values, --json or --example")
    model = HeartModel.load(args.model)
    out = model.predict(record)
    used = [f["name"] for f in model.contract["features"]]
    out["inputs_used_by_model"] = used
    out["ignored_fields"] = sorted(set(record) - set(used))
    print(json.dumps({"patient": record, **out}, indent=2))


if __name__ == "__main__":
    main()
