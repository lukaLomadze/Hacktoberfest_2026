"""Bank dataset spec for PrivateLearn — the "swapped hospitals for banks" demo.

Turns the UCI "default of credit card clients" dataset (30,000 rows) into three
simulated BANKS that each hold the same customers but store them under different
column names, currencies and units:

    Bank A  native TWD, age in years    credit_limit_twd
    Bank B  USD, age in months          credit_limit_usd   (1 TWD ≈ 30.5 USD)
    Bank C  thousands of TWD            credit_limit_k     (1,000 TWD per unit)

The target is `default.payment.next.month` (binary). This is the exact same shape
`prep_data.py` produces for Pima — the runtime never sees a bank name, only the
manifest + local CSVs. Run with:

    python src/pipeline/prep_banks.py

The raw CSV (data/raw/default_of_credit_card_clients.csv) is the OpenML download of
the UCI dataset, renamed to its real column names; see the module `source` field.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.generic_prep import prepare_dataset  # noqa: E402

RAW = ROOT / "data" / "raw" / "default_of_credit_card_clients.csv"
OUT = ROOT / "data" / "banks"

# 1 TWD ≈ 30.5 USD at a fixed demo rate, so the currency conversion is a clean, auditable factor.
TWD_PER_USD = 30.5

SPEC = {
    "dataset": "Default of Credit Card Clients (UCI)",
    "source": "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
    "raw_target": "default.payment.next.month",
    "target_values": None,  # already 0/1
    "holdout_target": "default",
    "drop_columns": [],
    "features": [
        {"raw": "AGE", "name": "age", "unit": "years"},
        {"raw": "LIMIT_BAL", "name": "credit_limit", "unit": "TWD"},
        {"raw": "SEX", "name": "sex", "unit": "category"},
        {"raw": "EDUCATION", "name": "education", "unit": "category"},
        {"raw": "MARRIAGE", "name": "marriage", "unit": "category"},
        {"raw": "PAY_0", "name": "pay_status", "unit": "status"},
        {"raw": "BILL_AMT1", "name": "bill_amount", "unit": "TWD"},
        {"raw": "PAY_AMT1", "name": "pay_amount", "unit": "TWD"},
    ],
    "split_col": "age",
    "n_clients": 3,
    "n_per_client": 400,
    "holdout_fraction": 0.2,
    "clients": [
        {
            "id": "h0",
            "target": "default_flag",
            "columns": {
                "age": {"source": ["age_years"], "factor": 1.0, "unit": "years"},
                "credit_limit": {"source": ["credit_limit_twd"], "factor": 1.0, "unit": "TWD"},
                "sex": {"source": ["gender"], "factor": 1.0, "unit": "category"},
                "education": {"source": ["education"], "factor": 1.0, "unit": "category"},
                "marriage": {"source": ["marital_status"], "factor": 1.0, "unit": "category"},
                "pay_status": {"source": ["pay_status_sep"], "factor": 1.0, "unit": "status"},
                "bill_amount": {"source": ["bill_amt_twd"], "factor": 1.0, "unit": "TWD"},
                "pay_amount": {"source": ["pay_amt_twd"], "factor": 1.0, "unit": "TWD"},
            },
        },
        {
            "id": "h1",
            "target": "defaulted",
            "columns": {
                "age": {"source": ["age_months"], "factor": 1.0 / 12.0, "unit": "months"},
                "credit_limit": {"source": ["credit_limit_usd"], "factor": TWD_PER_USD, "unit": "USD"},
                "sex": {"source": ["gender"], "factor": 1.0, "unit": "category"},
                "education": {"source": ["education"], "factor": 1.0, "unit": "category"},
                "marriage": {"source": ["marital_status"], "factor": 1.0, "unit": "category"},
                "pay_status": {"source": ["pay_status"], "factor": 1.0, "unit": "status"},
                "bill_amount": {"source": ["bill_amt_usd"], "factor": TWD_PER_USD, "unit": "USD"},
                "pay_amount": {"source": ["pay_amt_usd"], "factor": TWD_PER_USD, "unit": "USD"},
            },
        },
        {
            "id": "h2",
            "target": "default_next_month",
            "columns": {
                "age": {"source": ["age"], "factor": 1.0, "unit": "years"},
                "credit_limit": {"source": ["credit_limit_k"], "factor": 1000.0, "unit": "kTWD"},
                "sex": {"source": ["sex"], "factor": 1.0, "unit": "category"},
                "education": {"source": ["education"], "factor": 1.0, "unit": "category"},
                "marriage": {"source": ["marriage"], "factor": 1.0, "unit": "category"},
                "pay_status": {"source": ["pay_status"], "factor": 1.0, "unit": "status"},
                "bill_amount": {"source": ["bill_amt_k"], "factor": 1000.0, "unit": "kTWD"},
                "pay_amount": {"source": ["pay_amt_k"], "factor": 1000.0, "unit": "kTWD"},
            },
        },
    ],
    "node_labels": {"h0": "Bank A", "h1": "Bank B", "h2": "Bank C"},
    # 8 rounds so the weaker per-feature signal of this dataset converges (Pima needs only 4).
    "rounds": 8,
    # round 2: Bank C is offline; round 3: Bank C's export silently switches credit_limit
    # from thousands-of-TWD back to raw TWD (×1000) → its probe loss is an outlier.
    "faults": {"h2": {2: "offline", 3: {"unit_swap": {"feature": "credit_limit", "factor": 1000.0}}}},
    "honesty": ("Real default-of-credit-card-clients records (1200 of 30000) split into three "
                "SIMULATED banks; the player replays this recorded run, not live networking. "
                "Masking, DP noise, the agents' decisions and every number are real outputs of "
                "the code. DP accounting is a stated-ε demonstration, not an audited accountant."),
}


def main(argv=None):
    m = prepare_dataset(RAW, SPEC, OUT)
    for h, b in m["per_hospital_class_balance"].items():
        print(f"{h}: n={b['n']} pos={b['positive']} neg={b['negative']} "
              f"{m['split_col']} {b['split_min']}-{b['split_max']}")
    hb = m["holdout"]["balance"]
    print(f"holdout: n={hb['n']} pos={hb['positive']} neg={hb['negative']}")
    print(f"wrote {OUT / 'hospitals' / 'manifest.json'}")


if __name__ == "__main__":
    main()
