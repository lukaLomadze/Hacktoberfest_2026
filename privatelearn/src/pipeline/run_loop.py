"""T9 — end-to-end federated run → trace.json.

Three HospitalAgents (T7) and the CoordinatorAgent (T8) on the message bus (T6),
with the real privacy stack (T4): every hospital trains with per-example gradient
clipping, adds Gaussian noise calibrated to ε per round at PATIENT-level
sensitivity, and masks its update with pairwise secrets the coordinator never
holds. The coordinator sees only the masked sum.

The demo scenario (fixed, so the replay is reproducible):
  round 1  all three hospitals train
  round 2  Hospital C is offline              → aggregated over A and B
  round 3  Hospital C's export sends age in months under the same column name
           → its probe loss is an outlier → excluded before masking
  round 4  all three hospitals train

Baselines measured on the same holdout (data/holdout/test.csv, never trained on):
  solo         each hospital alone, trained to convergence on its own 120 rows
  centralized  all 360 rows pooled (the illegal upper bound), trained to convergence
  federated    this run's final global model

    python src/pipeline/run_loop.py            # writes trace.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agents.coordinator import CANONICAL, CoordinatorAgent  # noqa: E402
from agents.decider import build_decider  # noqa: E402
from agents.hospital import HospitalAgent, no_protection  # noqa: E402
from agents.runtime import MessageBus  # noqa: E402
from model import logistic  # noqa: E402
from model.attack import run_attack  # noqa: E402
from model.privacy import (  # noqa: E402
    DP_DELTA,
    make_pair_secrets,
    prepare_update,
    record_sensitivity,
    sigma_for_epsilon,
    train_dp,
)
from shared import io, schemas  # noqa: E402

SEED = 42
ROUNDS = 4
EPOCHS = 4
LR = 0.1
EXAMPLE_CLIP = 1.0
EPS_PER_ROUND = 1.5
SOLO_EPOCHS = 400  # "trained to convergence" for the solo / centralized baselines
FAULTS = {"h2": {2: "offline", 3: "age_in_months"}}

NODES = [
    {"id": "h0", "label": "Hospital A", "color": "#4d8bff", "role": "hospital"},
    {"id": "h1", "label": "Hospital B", "color": "#3fcf8e", "role": "hospital"},
    {"id": "h2", "label": "Hospital C", "color": "#ff6b6b", "role": "hospital"},
    {"id": "coord", "label": "Coordinator", "color": "#f5b94d", "role": "coordinator"},
]
HOSPITALS = ["h0", "h1", "h2"]
HONESTY = ("Real Pima Indians Diabetes records (360 of 768) split into three SIMULATED hospitals; "
           "the player replays this recorded run, not live networking. Masking, DP noise, the "
           "agents' decisions and every number are real outputs of the code. DP accounting is a "
           "stated-ε demonstration (analytic Gaussian, basic composition), not an audited accountant.")


def load_holdout(manifest):
    with open(ROOT / "data" / "holdout" / "test.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    order = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Insulin", "Age"]
    X = np.array([[float(r[c]) for c in order] for r in rows])
    y = np.array([int(r["Outcome"]) for r in rows], dtype=float)
    return (X - np.array(manifest["feature_mean"])) / np.array(manifest["feature_std"]), y


TRACE_DECIMALS = 10  # floats differ across OS/numpy builds in the ~16th digit; round them away


def round_floats(obj, digits: int = TRACE_DECIMALS):
    """Round every float in a JSON-like structure, so the trace is identical on every platform."""
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: round_floats(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, digits) for v in obj]
    return obj


def model_hash(w) -> str:
    return hashlib.sha256(json.dumps([round(float(v), 12) for v in w]).encode()).hexdigest()


def run(seed: int = SEED, out_path=None, private: bool = True, use_llm: bool = False) -> dict:
    """The full run. private=False replays the same scenario with masking and DP off
    (plain training, unprotected updates) to measure what privacy costs.
    use_llm=True lets the Schema / Planner / Coordinator agents make their decisions
    with DeepSeek (needs DEEPSEEK_API_KEY); every decision is recorded in the trace."""
    t0 = time.perf_counter()
    manifest = io.load_json(ROOT / "data" / "hospitals" / "manifest.json")
    if manifest["canonical_features"] != CANONICAL:
        raise ValueError("manifest feature order differs from the agreed canonical schema")
    Xt, yt = load_holdout(manifest)

    def evaluate(w, features=None):
        X = Xt if not features else Xt[:, [CANONICAL.index(f) for f in features]]
        return float(logistic.auc(yt, logistic.predict_proba(np.asarray(w), X)))

    # --- privacy stack: patient-level DP + pairwise masking --------------------------------
    n = manifest["n_per_hospital"]
    sensitivity = record_sensitivity(n, LR, EPOCHS, EXAMPLE_CLIP)
    sigma = sigma_for_epsilon(EPS_PER_ROUND, sensitivity, DP_DELTA)
    secrets = make_pair_secrets([0, 1, 2], seed=f"handshake-{seed}")  # hospital-to-hospital only

    if len(set(manifest["per_hospital_class_balance"][h]["n"] for h in HOSPITALS)) != 1:
        raise ValueError("the protector assumes equal n per hospital (weight = 1/participants)")

    def protector_for(idx):
        own = {k: v for k, v in secrets.items() if idx in k}  # each hospital holds only its pairs

        def protect(delta, hospital_index, participants, round_):
            # the hospital sends weight·Δw, whose patient-level sensitivity is weight·Δ, so the
            # noise scales by the same weight (equal n → weight = 1/len(participants)).
            # C is far above any real update norm: the DP bound comes from per-example clipping.
            s = sigma / len(participants)
            masked, _ = prepare_update(delta, hospital_index, participants, round_, own,
                                       noise_seed=f"dp-{seed}-h{idx}", C=10.0, sigma=s, dp_delta=DP_DELTA)
            return masked, {"masked": True, "mask_seed": f"round-{round_}", "dp_eps": EPS_PER_ROUND,
                            "dp_mechanism": "gaussian", "dp_sigma": float(s)}
        return protect

    def dp_train(w0, X, y, epochs, lr, mu=0.0, w_anchor=None):
        return train_dp(w0, X, y, epochs=epochs, lr=lr, example_clip=EXAMPLE_CLIP, mu=mu, w_anchor=w_anchor)

    decider = build_decider(use_llm)

    hospitals = [
        HospitalAgent.from_csv(h, ROOT / "data" / "hospitals" / f"hospital_{i}.csv",
                               protector=protector_for(i) if private else no_protection,
                               train_fn=dp_train if private else logistic.train, faults=FAULTS.get(h),
                               decider=decider)
        for i, h in enumerate(HOSPITALS)
    ]
    names = {nd["id"]: nd["label"] for nd in NODES}
    coord = CoordinatorAgent(HOSPITALS, evaluate, manifest["feature_mean"], manifest["feature_std"],
                             rounds=ROUNDS, names=names, decider=decider)

    bus = MessageBus()
    for h in hospitals:
        bus.register(h)
    bus.register(coord)
    bus.run(max_ticks=100)
    if not coord.done:
        raise RuntimeError("coordinator did not finish all rounds")

    # --- baselines on the same holdout, over the negotiated feature set ---------------------
    # each hospital's _X is already in the negotiated feature order, so no re-slicing here;
    # the holdout is sliced by name inside evaluate().
    n_w = len(coord.features) + 1
    solo = {}
    for h in hospitals:
        w = logistic.train(np.zeros(n_w), h._X, h._labels, epochs=SOLO_EPOCHS, lr=LR)
        solo[h.id] = round(evaluate(w, coord.features), 4)
    X_all = np.vstack([h._X for h in hospitals])
    y_all = np.concatenate([h._labels for h in hospitals])
    auc_central = round(evaluate(
        logistic.train(np.zeros(n_w), X_all, y_all, epochs=SOLO_EPOCHS, lr=LR), coord.features), 4)
    auc_fed = round(coord.auc_series[-1], 4)

    # --- gradient inversion on a real patient row of Hospital A ------------------------------
    attack = run_attack(np.zeros(n_w), hospitals[0]._X[0], hospitals[0]._labels[0], client_id=0,
                        participants=(0, 1, 2), round_=1, secrets=secrets, C=10.0, sigma=sigma, seed=seed)

    # --- every hospital ends with the same model ---------------------------------------------
    hashes = {h.id: model_hash(h.w_global) for h in hospitals}

    # loss_series[0]: the starting model's mean local loss, from the round-1 probes
    first_probe_loss = float(np.mean(list(coord.history[0]["probe_losses"].values())))
    rounds = [dict(r, auc_before=round(r["auc_before"], 4), auc_after=round(r["auc_after"], 4),
                   probe_losses={h: round(v, 4) for h, v in r["probe_losses"].items()})
              for r in coord.history]

    trace = {
        "v": "1.0",
        "meta": {
            "dataset": "Pima Indians Diabetes (NIDDK)",
            "hospitals": 3, "rows_per_hospital": n,
            "features": coord.features, "weights": coord.n_weights, "rounds": ROUNDS,
            "auc_solo": max(solo.values()),
            "auc_solo_by_hospital": solo,
            "auc_federated": auc_fed,
            "auc_centralized": auc_central,
            # same scenario with masking + DP switched off: the measured cost of privacy
            "auc_federated_no_privacy": (run(seed, None, private=False, use_llm=False)["meta"]["auc_federated"]
                                         if private else auc_fed),
            "eps_per_round": EPS_PER_ROUND, "dp_delta": DP_DELTA,
            "dp_sensitivity": sensitivity, "dp_sigma": sigma, "example_clip": EXAMPLE_CLIP,
            "local_epochs": EPOCHS, "lr": LR, "seed": seed,
            "model_hashes": hashes,
            "decision_maker": {
                **decider.info(),
                "schema": {h.id: {"made_by": h.schema.made_by, "reasoning": h.schema.reasoning}
                           for h in hospitals},
                "policy": "the LLM only sees column names, local column statistics "
                          "(min/p05/median/p95/max) and aggregate losses, never a patient row; "
                          "gradients, aggregation, masking and DP are deterministic code",
            },
            "honesty": HONESTY,
        },
        "nodes": NODES,
        "schema": coord.schema_report,
        "rounds": rounds,
        "metrics": {"auc_series": [round(v, 4) for v in coord.auc_series],
                    "loss_series": [round(first_probe_loss, 4)] + [round(v, 4) for v in coord.loss_series]},
        "attack": attack,
        "agents": {h.id: h.activity for h in hospitals},
        "patient_bytes_transferred": 0,
        "timeline_ms": 60000,
    }
    trace = round_floats(trace)
    schemas.validate_trace(trace)
    if out_path is not None:
        io.save_json(out_path, trace)
    run.last_runtime_s = time.perf_counter() - t0  # printed, not stored: the trace stays byte-stable
    return trace


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252 (no ε, →, ≤)
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Run PrivateLearn end to end and write trace.json")
    ap.add_argument("--out", default=str(ROOT / "trace.json"))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--llm", action="store_true",
                    help="let the agents decide with DeepSeek (needs DEEPSEEK_API_KEY); "
                         "decisions are recorded into the trace")
    args = ap.parse_args(argv)
    t = run(args.seed, args.out, use_llm=args.llm)
    m = t["meta"]
    print(f"rounds: {[r['strategy'] for r in t['rounds']]}")
    for r in t["rounds"]:
        print(f"  r{r['round']}: {r['decision']}  AUC {r['auc_before']:.3f} → {r['auc_after']:.3f}  ε {r['eps_total']}")
    print(f"AUC  solo best {m['auc_solo']:.3f} {m['auc_solo_by_hospital']}  federated {m['auc_federated']:.3f}  "
          f"centralized {m['auc_centralized']:.3f}")
    print(f"attack: {t['attack']}")
    print(f"model hashes identical: {len(set(m['model_hashes'].values())) == 1}  runtime {run.last_runtime_s:.2f} s")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
