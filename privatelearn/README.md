# PrivateLearn

Agentic federated learning demo: three hospitals train **one** logistic-regression model
together without any patient row leaving a building. Agents negotiate a common feature
schema; only weight updates move. See `docs/EXECUTION-PLAN.md` for the full plan.

## Layout

| path | what |
|---|---|
| `src/model/logistic.py` | logistic-regression core (numpy only) |
| `src/shared/` | contract validators (`schemas.py`) and io helpers |
| `src/agents/`, `src/pipeline/` | hospital/coordinator agents and the run loop (T6–T9, not yet built) |
| `data/sample_trace.json` | 4-round example trace the frontend replays |
| `web/` | trace-replay player (vanilla JS, no framework) |
| `tests/test_model/` | unittest suite for the model |

## Run the model tests

```bash
python -m unittest discover -s tests/test_model -v
```

## Validate a trace against the contracts

```bash
python -c "import sys; sys.path.insert(0,'src'); from shared import schemas, io; schemas.validate_trace(io.load_json('data/sample_trace.json'))"
```

## Replay the demo

```bash
python -m http.server 8000
# open http://localhost:8000/web/index.html
```
