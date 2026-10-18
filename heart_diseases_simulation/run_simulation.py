"""Run the federated heart-disease job: 4 hospital processes + 1 coordinator process.

    python run_simulation.py                  # real DeepSeek agents (needs DEEPSEEK_API_KEY)
    python run_simulation.py --agents rule    # offline scripted stand-in, no key, no cost

Each hospital is a separate OS process that reads only its own data file and talks HTTP on
127.0.0.1; the coordinator process holds no data. This launcher only starts the processes,
prints the job's events as they happen, and afterwards runs the experimenter's benchmark
and copies the published model to models/heart_model.json.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import threading
import time

from fedheart import benchmark, config, events, llm
from fedheart.transport import RemoteError, post

ROOT = pathlib.Path(__file__).resolve().parent


def parse_outages(spec: str) -> dict:
    """'switzerland:4,5;va:9' → {'switzerland': '4,5', 'va': '9'}"""
    out = {}
    for part in filter(None, (p.strip() for p in spec.split(";"))):
        site, rounds = part.split(":", 1)
        if site not in config.SITE_BY_ID:
            raise SystemExit(f"--outage: unknown site {site!r}")
        out[site] = rounds
    return out


class Printer(threading.Thread):
    """Tails runs/<id>/events.json and prints a readable line per event."""

    def __init__(self, path: pathlib.Path):
        super().__init__(daemon=True)
        self.path = path
        self.seen = 0
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            self.flush()
            time.sleep(0.5)
        self.flush()

    def flush(self):
        try:
            items = json.loads(self.path.read_text(encoding="utf-8"))["items"]
        except (OSError, ValueError, KeyError):
            return
        for e in items[self.seen:]:
            line = self.fmt(e)
            if line:
                print(line, flush=True)
        self.seen = len(items)

    @staticmethod
    def fmt(e):
        r = f"r{e['round']:<2}"
        t = e["type"]
        if t == "job.step":
            return f"\n=== {e['step'].replace('_', ' ').upper()} ==="
        if t == "agent.message":
            who = e["from"].get("company_name", "?")
            agent = e["from"].get("agent", "")
            tag = "" if agent == "coordinator" else f"·{agent}"
            return f"{r} [{e.get('made_by', '')}] {who}{tag}: {e['text']}"
        if t == "round.decision":
            return f"{r} DECISION {e['strategy']} with {len(e['participants'])} hospitals — {e['reasoning']}"
        if t == "round.completed":
            auc = e.get("auc_after")
            return (f"{r} ROUND DONE pooled val AUC {auc if auc is None else round(auc, 4)}  loss "
                    f"{e.get('loss') if e.get('loss') is None else round(e['loss'], 4)}  max ε {e['eps_total']}  "
                    f"applied={e['applied']}")
        if t == "participant.state" and e["state"] in ("offline", "excluded"):
            return f"{r} {e['node_id']} is {e['state']}: {e.get('detail', '')}"
        if t == "job.status":
            return f"JOB {e['status'].upper()}" + (f": {e['error']}" if e.get("error") else "")
        return None


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agents", choices=["deepseek", "rule"], default="deepseek")
    ap.add_argument("--model", default=config.DEFAULT_MODEL,
                    help="DeepSeek model for the coordinator (deepseek-v4-pro or deepseek-flash)")
    ap.add_argument("--hospital-model", default=None, help="DeepSeek model for the hospitals (default: --model)")
    ap.add_argument("--coordinator-effort", default="high", choices=config.EFFORTS,
                    help="DeepSeek reasoning_effort for the coordinator (none = no thinking)")
    ap.add_argument("--hospital-effort", default="low", choices=config.EFFORTS)
    ap.add_argument("--max-rounds", type=int, default=config.MAX_ROUNDS)
    ap.add_argument("--min-rounds", type=int, default=config.MIN_ROUNDS)
    ap.add_argument("--epsilon", type=float, default=config.EPSILON_TOTAL,
                    help="record-level DP budget per hospital for the job (0 = no DP noise)")
    ap.add_argument("--delta", type=float, default=config.DELTA)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--base-port", type=int, default=8101)
    ap.add_argument("--outage", default="", help="simulate outages, e.g. 'switzerland:4,5;va:9'")
    ap.add_argument("--no-mirror", action="store_true", help="do not update the hub mock's real-run bundle")
    ap.add_argument("--noise-seed", type=int, default=None, help="tests only: reproducible DP noise")
    args = ap.parse_args(argv)
    hospital_model = args.hospital_model or args.model

    if args.agents == "deepseek":
        try:
            llm.check_credentials()
            for m in sorted({args.model, hospital_model}):
                print(f"checking API access to {m} … {llm.preflight(m)} OK", flush=True)
        except llm.AgentError as e:
            raise SystemExit(f"\n{e}\n")

    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{args.agents}"
    run_dir = config.RUNS_DIR / run_id
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    outages = parse_outages(args.outage)
    mirror = None if args.no_mirror else config.HUB_MOCK_REAL_DIR / f"{events.JOB_ID}.json"
    print(f"run {run_id} → {run_dir}", flush=True)

    procs, urls, logs = [], {}, []
    try:
        for i, site in enumerate(config.SITES):
            port = args.base_port + i
            urls[site["id"]] = f"http://127.0.0.1:{port}"
            cmd = [sys.executable, "-m", "fedheart.hospital", "--site", site["id"], "--port", str(port),
                   "--run-dir", str(run_dir), "--agents", args.agents, "--model", hospital_model,
                   "--effort", args.hospital_effort, "--seed", str(args.seed)]
            if site["id"] in outages:
                cmd += ["--outage", outages[site["id"]]]
            if args.noise_seed is not None:
                cmd += ["--noise-seed", str(args.noise_seed + i)]
            log = open(run_dir / "logs" / f"{site['id']}.log", "w", encoding="utf-8")
            logs.append(log)
            procs.append(subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
            print(f"started hospital process {site['id']:<12} pid {procs[-1].pid}  port {port}", flush=True)

        cmd = [sys.executable, "-m", "fedheart.coordinator", "--run-dir", str(run_dir),
               "--hospitals", ",".join(f"{k}={v}" for k, v in urls.items()), "--agents", args.agents,
               "--model", args.model, "--effort", args.coordinator_effort, "--max-rounds", str(args.max_rounds),
               "--min-rounds", str(args.min_rounds), "--epsilon", str(args.epsilon), "--delta", str(args.delta),
               "--seed", str(args.seed)]
        if mirror is not None:
            cmd += ["--mirror", str(mirror)]
        clog = open(run_dir / "logs" / "coordinator.log", "w", encoding="utf-8")
        logs.append(clog)
        coord = subprocess.Popen(cmd, cwd=ROOT, stdout=clog, stderr=subprocess.STDOUT)
        procs.append(coord)
        print(f"started coordinator process     pid {coord.pid}\n", flush=True)

        printer = Printer(run_dir / "events.json")
        printer.start()
        while coord.poll() is None:
            dead = [p for p in procs[:-1] if p.poll() is not None]
            if dead:
                print(f"a hospital process exited unexpectedly (see {run_dir / 'logs'})", flush=True)
            time.sleep(1.0)
        printer.stop.set()
        printer.join()
        if coord.returncode != 0:
            tail = (run_dir / "logs" / "coordinator.log").read_text(encoding="utf-8", errors="replace")[-3000:]
            raise SystemExit(f"\ncoordinator failed (exit {coord.returncode}). Last log lines:\n{tail}")
    finally:
        for site, url in urls.items():
            try:
                post(url, "/shutdown", {}, timeout=5)
            except RemoteError:
                pass
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        for f in logs:
            f.close()

    finish(run_dir, mirror)


def finish(run_dir: pathlib.Path, mirror):
    model_path = run_dir / "model.json"
    bench = benchmark.run(model_path)
    events.write_json_atomic(run_dir / "benchmark.json", bench)
    config.MODELS_DIR.mkdir(exist_ok=True)
    shutil.copyfile(model_path, config.MODELS_DIR / "heart_model.json")

    # give the job page its reference lines (solo / centralized), clearly marked as the benchmark
    for path in [run_dir / "events.json", mirror]:
        if path is None or not path.exists():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        items = doc["items"] if "items" in doc else doc["events"]
        for e in items:
            if e["type"] == "model.published":
                e["metrics"].update(auc_centralized=bench["centralized"]["pooled_auc"],
                                    auc_solo_by_company={s: v["pooled_auc"] for s, v in bench["solo"].items()},
                                    benchmark_note=bench["note"])
        if "version" in doc and doc["version"]:
            doc["version"]["metrics"].update(auc_centralized=bench["centralized"]["pooled_auc"])
        events.write_json_atomic(path, doc)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    fin = summary["finished"]
    print("\n" + "=" * 78)
    print(f"model published after {fin['rounds_run']} rounds (checkpoint from round {fin['checkpoint_round']})")
    print(f"every hospital holds the same model: {fin['every_hospital_holds_same_model']}  sha256 "
          f"{fin['model_sha256'][:16]}…")
    print("\nPooled test set (each hospital's untouched 20%):\n")
    print(benchmark.table(bench))
    usage = summary.get("usage") or {}
    cost = sum((u or {}).get("estimated_cost_usd", 0) for u in usage.values())
    calls = sum((u or {}).get("api_calls", 0) for u in usage.values())
    if calls:
        print(f"\nDeepSeek API: {calls} calls, estimated cost ${cost:.2f} at peak rates "
              f"(see {run_dir / 'usage.json'})")
    print(f"\nsaved model: {config.MODELS_DIR / 'heart_model.json'}  (try: python predict.py --example)")
    print(f"run folder:  {run_dir}")
    if mirror is not None:
        print(f"job page:    open the hub with ?mock=1 → #/jobs/{events.JOB_ID}")


if __name__ == "__main__":
    main()
