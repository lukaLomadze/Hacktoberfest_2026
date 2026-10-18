"""Every enum of PLATFORM-API.md §2, in one place. `GET /meta` serves this module.

Keep the order of JOB_STEPS: the stepper in the UI draws them in this order.
"""

from __future__ import annotations

API_VERSION = "0.1"

ROLES = ("admin", "company")
COMPANY_STATUS = ("pending", "approved", "rejected", "suspended")
GROUP_STATUS = ("draft", "open", "training", "trained", "archived")
MEMBERSHIP_STATUS = (
    "requested", "approved", "rejected", "node_connected", "schema_proposed",
    "schema_confirmed", "ready", "training", "active", "left",
    "payment_pending",  # reserved for payments; nothing sets it yet
)
NODE_STATUS = ("offline", "online", "busy")
JOB_KIND = ("train", "finetune")
JOB_STATUS = ("queued", "running", "succeeded", "failed", "cancelled")
JOB_STEPS = ("waiting_for_nodes", "key_exchange", "schema_negotiation", "normalization",
             "training", "evaluating", "published")
PARTICIPANT_STATE = ("waiting", "online", "probing", "planning", "training", "masking", "sent",
                     "excluded", "offline", "done")
TASK_TYPES = ("binary_classification", "multiclass_classification", "regression",
              "image_classification", "text_generation")
MODEL_TYPES = ("logistic", "mlp", "cnn", "vit", "llm_lora")
AGENTS = ("schema", "planner", "trainer", "challenger", "validator", "coordinator", "architect")
MADE_BY = ("llm", "rule")
USAGE_KIND = ("enrollment", "training_round", "prediction", "contribution")

# What the engine can train today. X5's adapter registry will replace this constant.
AVAILABLE_MODEL_TYPES = ("logistic",)

# Membership states in which a company's data is wired up and it can be put in a job.
JOB_READY = ("ready", "active")
# Terminal membership states: a new join request may be made after these.
MEMBERSHIP_CLOSED = ("rejected", "left")
JOB_LIVE = ("queued", "running")

# Participant colors, stable per job (PLATFORM-API §4.5). The first three are DESIGN.md's
# hospital tokens; the rest avoid --coord, --ok and --danger, which carry other meanings.
PARTICIPANT_COLORS = ("#4d8bff", "#3fcf8e", "#ff6b6b", "#b48cff", "#33c3d6", "#ff9f43",
                      "#e86fb3", "#9bd54a")


def as_dict() -> dict:
    return {
        "role": list(ROLES),
        "company_status": list(COMPANY_STATUS),
        "group_status": list(GROUP_STATUS),
        "membership_status": list(MEMBERSHIP_STATUS),
        "node_status": list(NODE_STATUS),
        "job_kind": list(JOB_KIND),
        "job_status": list(JOB_STATUS),
        "job_step": list(JOB_STEPS),
        "participant_state": list(PARTICIPANT_STATE),
        "task_type": list(TASK_TYPES),
        "model_type": list(MODEL_TYPES),
        "agent": list(AGENTS),
        "made_by": list(MADE_BY),
        "usage_kind": list(USAGE_KIND),
    }
