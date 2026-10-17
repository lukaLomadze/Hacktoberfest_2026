"""Seed data: one admin, five companies, two groups.

Modes:
  dev   the three diabetes hospitals are already `ready`: their column profile, schema
        mapping and validation are computed from the real hospital CSVs with the engine's
        own profiler and rule-based Schema agent, exactly what a node would post. Lets
        the frontend and the fake federation run a job before any node exists.
  demo  the three hospitals are only `approved`, so the judges watch each node connect,
        map its columns and validate, live.
  none  empty database.

Seeding only runs on an empty database, so restarting the server never duplicates rows.
Sign in with any email below (demo login, no password).
"""

from __future__ import annotations

from . import paths
from .db import dumps, new_id, now_iso
from .enums import AVAILABLE_MODEL_TYPES
from .membership import missing_required
from .taskspec import normalize

ADMIN_EMAIL = "admin@privatelearn.example"

COMPANIES = [  # (name, field, email, status)
    ("Tbilisi Central Hospital", "healthcare", "it@tbilisi-central.example", "approved"),
    ("Batumi Medical Center", "healthcare", "data@batumi-medical.example", "approved"),
    ("Kutaisi Regional Clinic", "healthcare", "it@kutaisi-clinic.example", "approved"),
    ("Rustavi Family Clinic", "healthcare", "it@rustavi-family.example", "approved"),  # the late joiner
    ("Gori Cardio Lab", "healthcare", "lab@gori-cardio.example", "pending"),
]
HOSPITALS = COMPANIES[:3]  # hospital_0/1/2.csv, in that order

DIABETES_SPEC = {
    "question": "Does this patient have diabetes?",
    "task_type": "binary_classification",
    "label": {"name": "has_diabetes", "classes": ["no", "yes"], "positive": "yes"},
    "input_schema": [
        {"name": "pregnancies", "type": "integer", "unit": "count", "description": "number of pregnancies"},
        {"name": "glucose", "type": "number", "unit": "mg/dL", "description": "plasma glucose"},
        {"name": "blood_pressure", "type": "number", "unit": "mmHg", "description": "diastolic blood pressure"},
        {"name": "bmi", "type": "number", "unit": "kg/m^2", "description": "body-mass index"},
        {"name": "insulin", "type": "number", "unit": "uU/mL", "description": "2-hour serum insulin"},
        {"name": "age", "type": "number", "unit": "years", "description": "age"},
    ],
    "model": {"type": "logistic", "hyperparams": {"epochs": 4, "lr": 0.1}},
    "training": {"rounds": 4, "min_participants": 2, "aggregation": "adaptive"},
    "privacy": {"secure_aggregation": True, "dp": {"enabled": True, "epsilon_per_round": 1.5, "delta": 1e-5}},
    "metric": "auc",
}
CANCER_SPEC = {
    "question": "Is this breast tumour malignant?",
    "task_type": "binary_classification",
    "label": {"name": "malignant", "classes": ["benign", "malignant"], "positive": "malignant"},
    "input_schema": [
        {"name": "radius_mean", "type": "number", "unit": "mm", "description": "mean cell-nucleus radius"},
        {"name": "texture_mean", "type": "number", "unit": "", "description": "grey-level standard deviation"},
        {"name": "perimeter_mean", "type": "number", "unit": "mm", "description": "mean nucleus perimeter"},
        {"name": "area_mean", "type": "number", "unit": "mm^2", "description": "mean nucleus area"},
        {"name": "smoothness_mean", "type": "number", "unit": "", "description": "local radius variation"},
    ],
    "model": {"type": "logistic", "hyperparams": {}},
}


# Demo-scenario data, not runtime knowledge (docs/AGENT-DECISIONS.md): the mapping each
# hospital's Schema agent proposes for its own CSV in the recorded LLM run. The engine's
# offline Schema agent deliberately infers nothing, so the dev seed states the mapping.
# (feature, source columns, local unit, factor to the canonical unit, transform)
_SAME = lambda col, unit: ([col], unit, 1.0, "rename")  # noqa: E731
SEED_MAPPINGS = [
    {"pregnancies": _SAME("Pregnancies", "count"), "glucose": _SAME("Glucose", "mg/dL"),
     "blood_pressure": _SAME("BloodPressure", "mmHg"), "bmi": _SAME("BMI", "kg/m^2"),
     "insulin": _SAME("Insulin", "uU/mL"), "age": _SAME("Age", "years")},
    {"pregnancies": _SAME("pregnancies", "count"),
     "glucose": (["glucose_mmol_l"], "mmol/L", 18.0182, "x * 18.0182"),
     "blood_pressure": (["bp_kpa"], "kPa", 7.50062, "x * 7.50062"),
     "bmi": (["weight_kg", "height_m"], "kg, m", None, "weight_kg / height_m^2"),
     "insulin": (["insulin_pmol_l"], "pmol/L", 1 / 6.945, "x / 6.945"),
     "age": (["age_months"], "months", 1 / 12, "x / 12")},
    {"pregnancies": _SAME("num_pregnancies", "count"), "glucose": _SAME("glucose", "mg/dL"),
     "blood_pressure": _SAME("bp_sys", "mmHg"), "bmi": _SAME("bmi", "kg/m^2"),
     "insulin": (["insulin_pmol_l"], "pmol/L", 1 / 6.945, "x / 6.945"), "age": _SAME("age", "years")},
]


def hospital_wiring(index: int, task_spec: dict):
    """What hospital_<index>.csv's node would post: (data, schema, validation), or None.

    Names and statistics only: the rows are read here and dropped. None (the membership
    is then seeded as `approved`) if the data or the engine is missing or has changed
    so that the seed mapping no longer fits the file.
    """
    try:
        paths.use_engine()
        from agents.hospital import column_profile, load_local_csv
        from shared import io
        columns, rows = load_local_csv(paths.hospital_csv(index))
        label = io.load_json(paths.MANIFEST)["label"][f"h{index}"]
    except (OSError, ImportError, ValueError, KeyError):
        return None
    seed_map = SEED_MAPPINGS[index]
    if not rows or label not in columns or any(s not in columns for m in seed_map.values() for s in m[0]):
        return None
    profile = {c: {**s, "missing": 0} for c, s in column_profile(columns, rows).items()}
    mapping = [{"feature": f, "source": list(seed_map[f][0]), "unit": seed_map[f][1],
                "factor": seed_map[f][2], "transform": seed_map[f][3]}
               for f in (i["name"] for i in task_spec["input_schema"]) if f in seed_map]
    positive = sum(1 for r in rows if r[label] == 1.0) / len(rows)
    missing = missing_required(task_spec, mapping)
    data = {"row_count": len(rows), "columns": columns, "profile": profile}
    schema = {"label": label, "mapping": mapping, "made_by": "llm",
              "reasoning": "seeded: the mapping this hospital's Schema agent proposed in the recorded run",
              "confirmed": True, "edited": False, "missing_required": missing}
    validation = {"ok": not missing and 0 < positive < 1, "checks": [
        {"name": "row_count", "ok": len(rows) >= 10, "detail": f"{len(rows)} rows"},
        {"name": "label_balance", "ok": 0 < positive < 1, "detail": f"{positive:.0%} positive"},
        {"name": "mapping_complete", "ok": not missing,
         "detail": "every required input is mapped" if not missing else f"missing {', '.join(missing)}"},
    ]}
    return data, schema, validation


def seed(db, mode: str = "dev", available=AVAILABLE_MODEL_TYPES) -> bool:
    """Seed an empty database. Returns True if it seeded anything."""
    if mode not in ("dev", "demo", "none"):
        raise ValueError(f"unknown seed mode {mode!r}; use dev, demo or none")
    if mode == "none" or db.one("SELECT 1 AS x FROM users LIMIT 1"):
        return False
    t = now_iso()
    diabetes = normalize(DIABETES_SPEC, available)
    cancer = normalize(CANCER_SPEC, available)
    with db.tx() as c:
        c.execute("INSERT INTO users (id, email, role, company_id, created_at) VALUES (?,?,?,?,?)",
                  (new_id("usr"), ADMIN_EMAIL, "admin", None, t))
        company_ids = []
        for name, field, email, status in COMPANIES:
            cid = new_id("co")
            company_ids.append(cid)
            c.execute("INSERT INTO companies (id, name, name_key, field, contact_email, status, created_at) "
                      "VALUES (?,?,?,?,?,?,?)", (cid, name, name.casefold(), field, email, status, t))
            c.execute("INSERT INTO users (id, email, role, company_id, created_at) VALUES (?,?,?,?,?)",
                      (new_id("usr"), email, "company", cid, t))
        gid = new_id("grp")
        c.execute("INSERT INTO ml_groups (id, name, description, status, task_spec, created_at, updated_at) "
                  "VALUES (?,?,?,?,?,?,?)",
                  (gid, "Diabetes risk", "Hospitals screening for type-2 diabetes risk", "open",
                   dumps(diabetes), t, t))
        c.execute("INSERT INTO ml_groups (id, name, description, status, task_spec, created_at, updated_at) "
                  "VALUES (?,?,?,?,?,?,?)",
                  (new_id("grp"), "Breast cancer screening", "Pathology labs classifying tumour biopsies",
                   "draft", dumps(cancer), t, t))
        for i in range(len(HOSPITALS)):
            wiring = hospital_wiring(i, diabetes) if mode == "dev" else None
            status = "ready" if wiring and wiring[2]["ok"] else "approved"
            data, schema, valid = wiring if status == "ready" else (None, None, None)
            c.execute("INSERT INTO memberships (id, company_id, group_id, status, data, schema_, validation, "
                      "requested_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                      (new_id("mem"), company_ids[i], gid, status,
                       dumps(data) if data else None, dumps(schema) if schema else None,
                       dumps(valid) if valid else None, t, t))
    return True
