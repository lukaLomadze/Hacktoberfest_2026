"""Run PrivateLearn end to end on the bank dataset and write data/banks/trace.json.

This is the "swapped hospitals for banks" driver: it loads the bank manifest produced
by `prep_banks.py` and calls the exact same `run_loop.run` the Pima demo uses, with the
manifest + data directory pointing at the banks. No dataset knowledge lives in the
runtime — every difference is carried by the manifest (features, label, schema mapping,
node labels, faults, honesty note).

    python src/pipeline/prep_banks.py        # write data/banks/ (once)
    python src/pipeline/run_banks.py         # write data/banks/trace.json (offline)
    python src/pipeline/run_banks.py --llm   # same, with DeepSeek decisions (needs key)
"""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import run_loop  # noqa: E402
from shared import io  # noqa: E402

DATA_DIR = ROOT / "data" / "banks"


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Run PrivateLearn end to end on the banks and write the trace")
    ap.add_argument("--out", default=str(DATA_DIR / "trace.json"))
    ap.add_argument("--seed", type=int, default=run_loop.SEED)
    ap.add_argument("--llm", action="store_true",
                    help="let the agents decide with DeepSeek (needs DEEPSEEK_API_KEY); "
                         "decisions are recorded into the trace")
    args = ap.parse_args(argv)

    manifest = io.load_json(DATA_DIR / "hospitals" / "manifest.json")
    # offline (no key): the manifest-provided schema drives reconciliation; --llm lets
    # DeepSeek re-derive the schema / plans / strategy (needs DEEPSEEK_API_KEY).
    t = run_loop.run(manifest=manifest, data_dir=DATA_DIR, out_path=args.out,
                     seed=args.seed, use_llm=args.llm)
    m = t["meta"]
    print(f"dataset: {m['dataset']}")
    print(f"rounds: {[r['strategy'] for r in t['rounds']]}")
    for r in t["rounds"]:
        print(f"  r{r['round']}: {r['decision']}  AUC {r['auc_before']:.3f} → {r['auc_after']:.3f}  ε {r['eps_total']}")
    print(f"AUC  solo best {m['auc_solo']:.3f} {m['auc_solo_by_hospital']}  federated {m['auc_federated']:.3f}  "
          f"centralized {m['auc_centralized']:.3f}")
    print(f"attack: {t['attack']}")
    print(f"model hashes identical: {len(set(m['model_hashes'].values())) == 1}  "
          f"runtime {run_loop.run.last_runtime_s:.2f} s")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
