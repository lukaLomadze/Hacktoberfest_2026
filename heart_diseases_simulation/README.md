# Heart-disease federated simulation — real agents, real data, real processes

Four real hospital cohorts train **one** heart-disease model together without any patient row
leaving its hospital. Every decision is made live by DeepSeek agents:
- which features to use and how to repair broken values;
- which model family to train;
- how to aggregate in each round;
- each hospital's local training plan;
- **when training is good enough to stop.**

Each hospital and the coordinator is its own OS process talking HTTP. They are separate
processes, not threads in a loop.

```
                       ┌──────────────────────────────┐
                       │ coordinator process (no data)│
                       │ DeepSeek agent: contract,    │
                       │  model, strategy, stop       │
                       └──────┬───────────────▲───────┘
         schema/contract/plan │               │ masked + DP-noised updates,
         global model         ▼               │ aggregate metrics only
 ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
 │ Cleveland    │ │ Budapest     │ │ Zurich&Basel │ │ VA Long Beach│   4 hospital processes,
 │ 303 patients │ │ 294 patients │ │ 123 patients │ │ 200 patients │   each reads ONLY its file,
 │DeepSeek agent│ │DeepSeek agent│ │DeepSeek agent│ │DeepSeek agent│   each has its own agent
 └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
        ▲ ─ ─ ─ pairwise Diffie-Hellman secrets (coordinator only relays public keys) ─ ─ ▲
```

## The problem

Heart-disease data lives in hospitals that may not share it. Each one alone has a small and
biased sample:
- **Zurich & Basel** has 123 patients, 93% of them diseased, and **no cholesterol at all**: every
  value is recorded as `0`.
- **VA Long Beach** is missing blood pressure for 28% of patients and heart-rate and exercise data for 27%.
- `slope`, `ca` and `thal` are missing for most patients at three of the four sites.
- Disease rates range from 36% (Budapest) to 93% (Zurich & Basel).

A model trained at one site is weak on the others, and pooling the data is not allowed.

## How it is solved (and what the agents decide)

| step | who decides | what really happens |
|---|---|---|
| key exchange | code | Each hospital makes a 2048-bit Diffie-Hellman key (RFC 3526 group 14). The coordinator relays only the public keys, and each pair of hospitals derives its own secret (fingerprints are cross-checked). |
| describe data | **each hospital's agent** | It inspects summary statistics of its own training split (never rows). It says what is broken (e.g. "a cholesterol of 0 is impossible, so it means missing"), what is mostly missing, and what is unusual about its patients. |
| feature contract | **coordinator agent** proposes, **hospital agents** accept or object | The contract covers features, encoding (numeric / binary / one-hot), missing-value handling and quality rules. Hospitals can refuse; the coordinator revises the contract or continues without them. |
| federated normalisation | code | Each hospital sends its sums, masked; the masks cancel, so the coordinator learns only the totals. Those give the federated mean/std used by everyone. |
| model choice | **coordinator agent** | Every hospital cross-validates the candidates (logistic regression, small MLP) on its own data. The agent weighs CV scores, parameter count under DP noise, and how unreliable the small sites' numbers are. |
| each round: aggregation | **coordinator agent** | `fedavg`, `fedprox` (with μ), or `trimmed_mean` (exclude a site before masking), plus a server learning rate. The arithmetic (unmask, apply) is fixed code. |
| each round: local plan | **each hospital's agent** | Whether to take part, how many local epochs, and the learning rate. It sees its own history and remaining privacy budget. |
| local training | code | DP-GD on the hospital's rows: per-patient gradient clipping, Gaussian noise at every step. The update is weighted, masked and sent. |
| **stop** | **coordinator agent** | No fixed round count (min 3, hard cap 40). The agent stops when the pooled validation loss/AUC and the worst-off hospital stop improving, weighing that against privacy spent. It then picks the best checkpoint. |
| publish | code | The chosen model is evaluated once on every hospital's untouched test split. Every hospital stores the identical model (same SHA-256), and it is saved to `models/heart_model.json`. |

### Why it works
- **Secure aggregation.** Masks are drawn uniformly at random, mod 2^64, from a cryptographic PRG
  (SHAKE-256), so one hospital's message on its own is indistinguishable from noise. Only the sum
  over the agreed participants is meaningful. A participant that drops out after committing makes
  the sum undecodable, so that round is discarded, never half-applied.
- **Differential privacy (record level).** One DP-GD step is a Gaussian mechanism with
  sensitivity 2·clip/n. Costs are tracked in zero-concentrated DP, which composes exactly over
  steps and rounds, and reported as (ε, δ). Each hospital's budget for the whole job is ε = 8,
  δ = 1e-5, split over at most 40 rounds. A hospital enforces its own budget and refuses to
  train once it is spent.
- **Federated learning.** The weighted sum of local steps is FedAvg. FedProx limits drift when
  disease rates differ a lot. Because the coordinator decides exclusions before masking,
  trimmed-mean stays compatible with masking.
- **Pooled metrics without pooled data.** Hospitals report histograms of predicted risk per
  true class. Summing them gives the exact pooled ROC AUC of all four sites, up to the bin width.

### Honest limits
- Public keys are not signed, so a malicious coordinator could swap keys (a man-in-the-middle).
  Production would authenticate them.
- The validation metrics, CV scores and summary statistics the agents exchange are aggregates
  but are not differentially private. Only the training updates are.
- The data are four 1980s cohorts, 920 patients in total. This is a demonstration, not a medical
  device.
- What the hospital agents see goes to the DeepSeek API. That is summary statistics of the
  training split, never a row.

## Run it

### 1. Install (once)
```bash
cd heart_diseases_simulation
python -m pip install -r requirements.txt
```

### 2. Your DeepSeek API key
1. Create a key at **https://platform.deepseek.com** → *API keys*, and add balance under
   *Top up* (the API is prepaid).
2. Put it in the terminal you will run from. Never paste it into code or commit it.
   - PowerShell, this window only: `$env:DEEPSEEK_API_KEY = "sk-..."`
   - PowerShell, permanently for your user: `setx DEEPSEEK_API_KEY "sk-..."`, then open a new terminal.
   - bash: `export DEEPSEEK_API_KEY=sk-...`
3. The launcher checks the key and the model with one small call (`GET /models`) before
   starting anything. A missing key, a rejected key (401), an empty balance (402) or a model
   your key is not offered stops the run with a clear message, before any money is spent.
4. Cost: every agent defaults to **`deepseek-v4-pro`** in thinking mode ($1.32 / $3.96 per
   million input / output tokens at peak hours, half off-peak; cached input $0.044). One job
   is 40 rounds or fewer; expect roughly 1–3 USD at peak rates, but this is an estimate, not a
   measurement. DeepSeek caches the repeated conversation prefix automatically, and the exact
   token usage and estimated cost are printed at the end and saved to `runs/<id>/usage.json`.
   To cut cost, give the hospitals the cheaper model: `--hospital-model deepseek-flash`.

### 3. Start the job
```bash
python run_simulation.py
```
The console streams what every agent does and decides as it happens. At the end it prints:
- the benchmark table: the federated model vs. each hospital alone vs. all data pooled, all
  scored on the same pooled test set;
- the API cost;
- where the model was saved.

Useful options:

| option | meaning |
|---|---|
| `--max-rounds 40 --min-rounds 3` | the hard cap and floor; the agent decides in between |
| `--epsilon 8` | privacy budget per hospital (`0` = no DP noise; masking stays on) |
| `--outage "switzerland:4,5;va:9"` | that hospital's process does not answer in those rounds, to show fault handling |
| `--coordinator-effort high --hospital-effort low` | DeepSeek `reasoning_effort` per agent: `none` (no thinking), `low`, `high`, `max` |
| `--agents rule` | offline rehearsal with scripted stand-ins: no key, no cost, everything labelled `rule` |

### 4. Watch it in the platform UI (#/jobs)
During and after a run, the simulation writes `privatelearn/hub/web/mock/real/job_heart.json`.
Open the hub in mock mode (for example `python -m http.server 8765 --directory privatelearn` →
`http://localhost:8765/hub/web/index.html?mock=1`), sign in as admin, and open `#/jobs/job_1`.
A switch at the top moves between:
- **Scripted demo** (`job_1`): the fixed three-hospital diabetes scenario.
- **Real agent run** (`job_heart`): your run. While it is still running the page follows it
  live; afterwards **Replay** plays it back.

### 5. Use the model
```bash
python predict.py --example
python predict.py --age 58 --sex 1 --cp 4 --trestbps 140 --chol 0 --thalach 120 --exang 1 --oldpeak 2
python predict.py --json patient.json
```
Missing values are fine: they get the same repair rules and imputation as in training. In code:
```python
from fedheart.inference import HeartModel
HeartModel.load().predict({"age": 58, "sex": 1, "cp": 4, "chol": None})
```

## What a run leaves behind (`runs/<id>/`)

| file | content |
|---|---|
| `events.json`, `job.json` | the job's event log in the platform's §6 format (what the UI plays) |
| `transcripts/<agent>.jsonl` | every DeepSeek request/response per agent, with its reasoning and tool calls |
| `summary.json` | contract, CV results, every round's report, the stop decision, traffic per hospital |
| `model.json` | the published model (also copied to `models/heart_model.json`) |
| `benchmark.json` | the offline comparison (solo / pooled / federated) |
| `sites/<site>/received_model.json` | the copy each hospital process stored, with the same hash |
| `usage.json`, `logs/` | API tokens and cost per agent, and process logs |

## Tests (no key needed)
```bash
python -m unittest discover -s tests -v
```
The suite covers:
- the numeric core: data, contract, gradient checks, masking, DP accounting, pooled AUC;
- a full five-process job with the scripted agents, including an outage;
- a full five-process job where every agent runs the real DeepSeek loop against a local fake
  of the chat-completions API. It checks the request shape (model, `reasoning_effort`, tools,
  no forced tool choice), that every agent's history is append-only, and that each answer's
  `reasoning_content` is sent back (the fake, like the real API, answers 400 otherwise).

Test runs go to a temporary folder and never touch `runs/` or `models/`.

## Code map

| path | what |
|---|---|
| `fedheart/hospital.py` | the hospital process: its endpoints, its agent and the agent's 4 tools |
| `fedheart/coordinator.py` | the coordinator process: its agent and 6 tools (contract, CV, model, round, finish) |
| `fedheart/llm.py` | the DeepSeek tool-use loop over stdlib HTTP (append-only with reasoning_content, retries, transcripts) |
| `fedheart/privacy.py` | Diffie-Hellman, fixed-point masking mod 2^64, zCDP accounting |
| `fedheart/models.py` | logistic regression and MLP as flat vectors, DP-GD |
| `fedheart/features.py` | the contract, quality rules, federated statistics, encoding |
| `fedheart/metrics.py` | AUC, log-loss, pooled metrics from histograms |
| `fedheart/events.py` | the job event log and the hub bundle |
| `run_simulation.py`, `predict.py` | the launcher and the model CLI |
