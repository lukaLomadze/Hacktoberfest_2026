"""Fixed facts about the federation: the four hospitals, the data dictionary, defaults.

The four sites are the four real cohorts of the UCI Heart Disease study (Detrano et al.,
1989). Each one is run as its own OS process that can read only its own file.
"""

from __future__ import annotations

import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
# the test suite points these at temporary folders so it never touches real runs or the model
RUNS_DIR = pathlib.Path(os.environ.get("FEDHEART_RUNS_DIR") or ROOT / "runs")
MODELS_DIR = pathlib.Path(os.environ.get("FEDHEART_MODELS_DIR") or ROOT / "models")
# the platform mock (privatelearn/hub/web) shows real runs from this folder
HUB_MOCK_REAL_DIR = ROOT.parent / "privatelearn" / "hub" / "web" / "mock" / "real"

SITES = [
    {"id": "cleveland", "node_id": "node_cleveland", "name": "Cleveland Clinic",
     "place": "Cleveland Clinic Foundation, Cleveland, Ohio, USA", "file": "processed.cleveland.data", "color": "#4d8bff"},
    {"id": "hungarian", "node_id": "node_hungarian", "name": "Budapest Cardiology",
     "place": "Hungarian Institute of Cardiology, Budapest", "file": "processed.hungarian.data", "color": "#3fcf8e"},
    {"id": "switzerland", "node_id": "node_switzerland", "name": "Zurich & Basel",
     "place": "University Hospitals of Zurich and Basel, Switzerland", "file": "processed.switzerland.data", "color": "#ff6b6b"},
    {"id": "va", "node_id": "node_va", "name": "VA Long Beach",
     "place": "V.A. Medical Center, Long Beach, California, USA", "file": "processed.va.data", "color": "#b48cff"},
]
SITE_BY_ID = {s["id"]: s for s in SITES}
SITE_BY_NODE = {s["node_id"]: s for s in SITES}

# The 13 attributes every site exports (UCI "processed" format, '?' = missing) + the label.
# This is each hospital's own data dictionary; the coordinator never reads the files.
COLUMNS = [
    {"name": "age", "meaning": "age", "unit": "years", "kind": "numeric"},
    {"name": "sex", "meaning": "sex", "unit": "1 = male, 0 = female", "kind": "binary"},
    {"name": "cp", "meaning": "chest pain type", "kind": "categorical",
     "unit": "1 typical angina, 2 atypical angina, 3 non-anginal pain, 4 asymptomatic", "codes": [1, 2, 3, 4]},
    {"name": "trestbps", "meaning": "resting blood pressure on admission", "unit": "mmHg", "kind": "numeric"},
    {"name": "chol", "meaning": "serum cholesterol", "unit": "mg/dl", "kind": "numeric"},
    {"name": "fbs", "meaning": "fasting blood sugar > 120 mg/dl", "unit": "1 = true, 0 = false", "kind": "binary"},
    {"name": "restecg", "meaning": "resting electrocardiogram", "kind": "categorical",
     "unit": "0 normal, 1 ST-T wave abnormality, 2 left ventricular hypertrophy", "codes": [0, 1, 2]},
    {"name": "thalach", "meaning": "maximum heart rate achieved (exercise test)", "unit": "beats/min",
     "kind": "numeric"},
    {"name": "exang", "meaning": "exercise-induced angina", "unit": "1 = yes, 0 = no", "kind": "binary"},
    {"name": "oldpeak", "meaning": "ST depression induced by exercise relative to rest", "unit": "mm",
     "kind": "numeric"},
    {"name": "slope", "meaning": "slope of the peak exercise ST segment", "kind": "categorical",
     "unit": "1 upsloping, 2 flat, 3 downsloping", "codes": [1, 2, 3]},
    {"name": "ca", "meaning": "number of major vessels coloured by fluoroscopy", "unit": "count 0-3",
     "kind": "numeric"},
    {"name": "thal", "meaning": "thallium stress test result", "kind": "categorical",
     "unit": "3 normal, 6 fixed defect, 7 reversible defect", "codes": [3, 6, 7]},
]
FEATURE_NAMES = [c["name"] for c in COLUMNS]
COLUMN_BY_NAME = {c["name"]: c for c in COLUMNS}
LABEL = {"name": "num", "meaning": "angiographic heart disease (num > 0 = diameter narrowing > 50%)",
         "positive": "num > 0"}

# ---- defaults (every one can be overridden on the command line)
SEED = 2026
SPLIT = (0.6, 0.2, 0.2)  # train / validation / test, stratified, per site
MAX_ROUNDS = 40  # hard cap; the coordinator agent decides when to stop before it
MIN_ROUNDS = 3
EPSILON_TOTAL = 8.0  # record-level (ε, δ)-DP budget per hospital for the whole job
DELTA = 1e-5
CLIP = 1.0  # per-example gradient L2 clip
HIST_BINS = 50  # predicted-probability histogram bins for pooled metrics

SUPPORTED_MODELS = ("logistic", "mlp")

# ---- the agents' LLM: DeepSeek's OpenAI-compatible API (key in $DEEPSEEK_API_KEY)
DEEPSEEK_BASE_URL = "https://api.deepseek.com"  # $DEEPSEEK_BASE_URL overrides it (tests use a fake)
DEFAULT_MODEL = "deepseek-v4-pro"
EFFORTS = ("none", "low", "high", "max")  # DeepSeek reasoning_effort; none turns thinking off

# $/1M tokens (input cache miss, input cache hit, output) at PEAK rates, from DeepSeek's
# pricing page on 2026-10-18; off-peak is half. Only used for the cost estimate in usage.json.
PRICES = {
    "deepseek-v4-pro": (1.32, 0.044, 3.96),
    "deepseek-flash": (0.30, 0.006, 1.20),
}
