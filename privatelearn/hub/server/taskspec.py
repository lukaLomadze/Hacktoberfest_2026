"""TaskSpec validation and normalisation (PLATFORM-API §4.3).

`normalize(spec, available)` returns a complete TaskSpec with every default filled in,
or raises `422` naming the first bad field. Stored specs are always normalised, so the
federation, the predict route and the frontend can rely on every field being present.

Beyond types and ranges it checks that the pieces fit together, because a spec that
validates field-by-field can still be untrainable:
  - the model type must support the task type and every input type (logistic is binary
    and numeric only, image models take exactly one image, ...);
  - secure aggregation needs at least 2 participants: with one, the "masked sum" is that
    company's own update in the clear;
  - binary labels have exactly two classes and name the positive one;
  - the label may not also be an input (it would leak the answer into the features).
"""

from __future__ import annotations

from . import validate as v
from .enums import MODEL_TYPES, TASK_TYPES
from .errors import validation

INPUT_TYPES = ("number", "integer", "category", "image", "text")
AGGREGATIONS = ("adaptive", "fedavg", "fedprox", "trimmed_mean")
CLASSIFICATION = ("binary_classification", "multiclass_classification", "image_classification")
METRICS = {
    "binary_classification": ("auc", "accuracy"),
    "multiclass_classification": ("accuracy", "f1_macro"),
    "regression": ("rmse", "mae", "r2"),
    "image_classification": ("accuracy", "f1_macro"),
    "text_generation": ("perplexity",),
}
MODEL_COMPAT = {
    "logistic": {"tasks": ("binary_classification",), "inputs": ("number", "integer")},
    "mlp": {"tasks": ("binary_classification", "multiclass_classification", "regression"),
            "inputs": ("number", "integer")},
    "cnn": {"tasks": ("image_classification",), "inputs": ("image",)},
    "vit": {"tasks": ("image_classification",), "inputs": ("image",)},
    "llm_lora": {"tasks": ("text_generation",), "inputs": ("text",)},
}
SINGLE_INPUT = {"image": "image_classification", "text": "text_generation"}

DEFAULT_TRAINING = {"rounds": 4, "min_participants": 2, "aggregation": "adaptive"}
DEFAULT_EPSILON = 1.5
DEFAULT_DELTA = 1e-5


def _hyperparams(hp, path) -> dict:
    hp = {} if hp is None else v.obj(hp, path)
    if len(hp) > 50:
        raise validation(path, "at most 50 hyperparameters")
    out = {}
    for k, val in hp.items():
        f = f"{path}.{k}"
        if not isinstance(k, str) or not 1 <= len(k) <= 64:
            raise validation(f, "hyperparameter names are 1-64 characters")
        if k == "epochs":
            out[k] = v.integer(val, f, 1, 1000)
        elif k == "batch_size":
            out[k] = v.integer(val, f, 1, 100_000)
        elif k == "lr":
            out[k] = v.number(val, f, 0, 10, lo_open=True)
        elif k == "mu":
            out[k] = v.number(val, f, 0, 100)
        else:  # model-specific: any JSON value, checked to be finite
            out[k] = v.json_value(val, f)
    return out


def _label(lab, task_type, path) -> dict:
    lab = v.obj(lab, path)
    v.reject_unknown(lab, ("name", "classes", "positive"), f"{path}.")
    name = v.name_key(lab.get("name"), f"{path}.name")
    classes, positive = lab.get("classes"), lab.get("positive")
    if task_type in CLASSIFICATION:
        need = 2 if task_type == "binary_classification" else None
        classes = v.lst(classes, f"{path}.classes", min_len=2, max_len=need or 1000)
        if need and len(classes) != need:
            raise validation(f"{path}.classes", "a binary label has exactly 2 classes")
        classes = [v.string(c, f"{path}.classes[{i}]", max_len=100) for i, c in enumerate(classes)]
        if len({c.casefold() for c in classes}) != len(classes):
            raise validation(f"{path}.classes", "classes must be distinct")
        if task_type == "binary_classification":
            positive = v.string(positive, f"{path}.positive", max_len=100)
            if positive not in classes:
                raise validation(f"{path}.positive", "must be one of the classes")
        elif positive is not None:
            raise validation(f"{path}.positive", "only binary labels have a positive class")
    else:
        if classes is not None:
            raise validation(f"{path}.classes", f"{task_type} has no classes; use null")
        if positive is not None:
            raise validation(f"{path}.positive", f"{task_type} has no positive class; use null")
    return {"name": name, "classes": classes, "positive": positive}


def _input(item, path) -> dict:
    item = v.obj(item, path)
    v.reject_unknown(item, ("name", "type", "unit", "description", "required", "categories", "shape",
                            "channels"), f"{path}.")
    out = {"name": v.name_key(item.get("name"), f"{path}.name"),
           "type": v.enum(item.get("type"), f"{path}.type", INPUT_TYPES),
           "unit": v.opt_string(item.get("unit"), f"{path}.unit", max_len=40),
           "description": v.opt_string(item.get("description"), f"{path}.description", max_len=500),
           "required": True if item.get("required") is None else v.boolean(item["required"], f"{path}.required")}
    t = out["type"]
    if t == "category":
        cats = v.lst(item.get("categories"), f"{path}.categories", min_len=2, max_len=1000)
        cats = [v.string(c, f"{path}.categories[{i}]", max_len=100) for i, c in enumerate(cats)]
        if len(set(cats)) != len(cats):
            raise validation(f"{path}.categories", "categories must be distinct")
        out["categories"] = cats
    elif "categories" in item and item["categories"] is not None:
        raise validation(f"{path}.categories", "only category inputs have categories")
    if t == "image":
        shape = v.lst(item.get("shape"), f"{path}.shape", min_len=2, max_len=2)
        out["shape"] = [v.integer(s, f"{path}.shape[{i}]", 1, 4096) for i, s in enumerate(shape)]
        out["channels"] = v.integer(item.get("channels"), f"{path}.channels", 1, 4)
    else:
        for k in ("shape", "channels"):
            if item.get(k) is not None:
                raise validation(f"{path}.{k}", "only image inputs have a shape and channels")
    return out


def normalize(spec, available, path="task_spec") -> dict:
    spec = v.obj(spec, path)
    v.reject_unknown(spec, ("question", "task_type", "label", "input_schema", "model", "training",
                            "privacy", "metric"), f"{path}.")
    question = v.string(spec.get("question"), f"{path}.question", min_len=3, max_len=500)
    task_type = v.enum(spec.get("task_type"), f"{path}.task_type", TASK_TYPES)
    label = _label(spec.get("label"), task_type, f"{path}.label")

    raw_inputs = v.lst(spec.get("input_schema"), f"{path}.input_schema", min_len=1, max_len=500)
    inputs, seen = [], set()
    for i, item in enumerate(raw_inputs):
        inp = _input(item, f"{path}.input_schema[{i}]")
        if inp["name"] in seen:
            raise validation(f"{path}.input_schema[{i}].name", f"duplicate input {inp['name']!r}")
        if inp["name"] == label["name"]:
            raise validation(f"{path}.input_schema[{i}].name", "the label cannot also be an input")
        seen.add(inp["name"])
        inputs.append(inp)
    if not any(i["required"] for i in inputs):
        raise validation(f"{path}.input_schema", "at least one input must be required")

    model = v.obj(spec.get("model"), f"{path}.model")
    v.reject_unknown(model, ("type", "hyperparams"), f"{path}.model.")
    mtype = v.enum(model.get("type"), f"{path}.model.type", MODEL_TYPES)
    if mtype not in available:
        raise validation(f"{path}.model.type", f"{mtype!r} is not available yet; available: {', '.join(available)}")
    compat = MODEL_COMPAT[mtype]
    if task_type not in compat["tasks"]:
        raise validation(f"{path}.task_type", f"a {mtype} model supports {', '.join(compat['tasks'])}")
    for i, inp in enumerate(inputs):
        if inp["type"] not in compat["inputs"]:
            raise validation(f"{path}.input_schema[{i}].type",
                             f"a {mtype} model takes {' or '.join(compat['inputs'])} inputs")
        single_task = SINGLE_INPUT.get(inp["type"])
        if single_task and len(inputs) != 1:
            raise validation(f"{path}.input_schema", f"{single_task} takes exactly one {inp['type']} input")
    hyper = _hyperparams(model.get("hyperparams"), f"{path}.model.hyperparams")

    tr = {} if spec.get("training") is None else v.obj(spec["training"], f"{path}.training")
    v.reject_unknown(tr, tuple(DEFAULT_TRAINING), f"{path}.training.")
    training = {
        "rounds": v.integer(tr.get("rounds", DEFAULT_TRAINING["rounds"]), f"{path}.training.rounds", 1, 100),
        "min_participants": v.integer(tr.get("min_participants", DEFAULT_TRAINING["min_participants"]),
                                      f"{path}.training.min_participants", 1, 1000),
        "aggregation": v.enum(tr.get("aggregation", DEFAULT_TRAINING["aggregation"]),
                              f"{path}.training.aggregation", AGGREGATIONS),
    }

    pr = {} if spec.get("privacy") is None else v.obj(spec["privacy"], f"{path}.privacy")
    v.reject_unknown(pr, ("secure_aggregation", "dp"), f"{path}.privacy.")
    secure = True if pr.get("secure_aggregation") is None else v.boolean(pr["secure_aggregation"],
                                                                         f"{path}.privacy.secure_aggregation")
    dp = {} if pr.get("dp") is None else v.obj(pr["dp"], f"{path}.privacy.dp")
    v.reject_unknown(dp, ("enabled", "epsilon_per_round", "delta"), f"{path}.privacy.dp.")
    dp_on = True if dp.get("enabled") is None else v.boolean(dp["enabled"], f"{path}.privacy.dp.enabled")
    if dp_on:
        eps = v.number(dp.get("epsilon_per_round", DEFAULT_EPSILON), f"{path}.privacy.dp.epsilon_per_round",
                       0, 100, lo_open=True)
        delta = v.number(dp.get("delta", DEFAULT_DELTA), f"{path}.privacy.dp.delta", 0, 0.01, lo_open=True)
    else:
        eps, delta = None, None
    if secure and training["min_participants"] < 2:
        raise validation(f"{path}.training.min_participants",
                         "secure aggregation needs at least 2 participants (one masked update is not hidden)")

    metric = spec.get("metric")
    metric = METRICS[task_type][0] if metric is None else v.enum(metric, f"{path}.metric", METRICS[task_type])

    return {"question": question, "task_type": task_type, "label": label, "input_schema": inputs,
            "model": {"type": mtype, "hyperparams": hyper}, "training": training,
            "privacy": {"secure_aggregation": secure,
                        "dp": {"enabled": dp_on, "epsilon_per_round": eps, "delta": delta}},
            "metric": metric}
