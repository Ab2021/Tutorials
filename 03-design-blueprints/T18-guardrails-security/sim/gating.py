"""T18 -- gating: the control that detects nothing and carries the blast radius.

The corpus's ordering [T] is unambiguous: *"above all, gate the tool calls that can send money, email
a customer, or delete a record. That last layer has the biggest blast radius."* And the guide names
the reason it is skipped anyway: *"capability gating is the most underused defense; many teams add a
guardrail classifier and stop there"* [R].

The reason has a shape, and this module measures it. A detection rail reduces harm by **recognising
the attack**. A capability gate reduces harm by **removing the thing the attack needs**. Those are
multiplicative on the same residual risk and they have different floors:

  * a detector's floor is the residual it has on a technique nobody has seen -- never zero, because
    generalisation is not recognition;
  * a gate's floor is set by the human in the loop, because `irreversible -> require approval` hands
    the decision to someone who can be tired, busy, or looking at a queue of four hundred.

So `harm = (1 - catch) * gate_failure`, and every experiment in this topic is a question about which
of the two factors to spend the next dollar on. The answer the corpus gives -- the gate, first -- is
correct here, and the reason is not blast radius alone: it is that **the detection factor is already
within about a point of its ceiling** (`rails.ceiling_catch`), while the gate factor has an order of
magnitude left in it.

This module also owns trust tags, because capability gating is *driven* by them: the gate's input is
the trust level of the content currently in context, so a tag that goes missing in a middleware hop
silently widens the gate. Fail-open versus fail-closed is therefore a security decision wearing a
plumbing costume.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .rails import tag_survival, effective_privilege, tag_loss_false_refusal

# ------------------------------------------------------------------------------------------------
# Tool surface: risk classes, the allowlist, and what each class is
# ------------------------------------------------------------------------------------------------

READ = "read"
WRITE_LOW = "write_low"
EXTERNAL = "external"
IRREVERSIBLE = "irreversible"

RISK_ORDER = (READ, WRITE_LOW, EXTERNAL, IRREVERSIBLE)
RISK_NOTE = {
    READ: "look something up -- no state change, no third party",
    WRITE_LOW: "draft a note, update a preference -- state change, reversible, internal",
    EXTERNAL: "send a member email, call a partner API -- leaves the boundary, hard to unsend",
    IRREVERSIBLE: "disburse funds, delete a record, adjust a claim -- cannot be undone",
}

# Session trust required for the call to run UNATTENDED. Below it the gate escalates instead of
# denying, which is the difference between a gate and a wall.
MIN_TRUST = {READ: 2, WRITE_LOW: 3, EXTERNAL: 4, IRREVERSIBLE: 5}

# The four outcomes. `APPROVAL` is not a denial -- it is a hand-off, and its failure mode is human.
ALLOW = "allow"
ALLOW_LOGGED = "allow_logged"
DRY_RUN = "dry_run"
APPROVAL = "approval"
DENY = "deny"

OUTCOME_NOTE = {
    ALLOW: "runs, no friction",
    ALLOW_LOGGED: "runs, recorded for the audit trail",
    DRY_RUN: "computes the effect, writes nothing -- 'watch what the agent would have done without doing it' [T]",
    APPROVAL: "handed to a human -- 'irreversible actions like a refund or a delete require human approval' [T]",
    DENY: "rejected before execution -- 'a malformed or malicious call is rejected before it runs' [T]",
}


@dataclass(frozen=True)
class Tool:
    name: str
    risk: str
    allowlisted: bool = True
    note: str = ""


TOOLS = (
    Tool("search_corpus", READ, note="RAG lookup"),
    Tool("lookup_claim", READ, note="member record read"),
    Tool("draft_note", WRITE_LOW, note="internal note, reversible"),
    Tool("update_preference", WRITE_LOW, note="member preference, reversible"),
    Tool("send_member_email", EXTERNAL, note="leaves the boundary -- also an exfiltration channel"),
    Tool("call_partner_api", EXTERNAL, note="third-party data transfer"),
    Tool("disburse_funds", IRREVERSIBLE, note="money movement"),
    Tool("delete_record", IRREVERSIBLE, note="destruction"),
    Tool("adjust_claim", IRREVERSIBLE, note="financial state change"),
    Tool("register_new_tool", READ, allowlisted=False,
         note="NOT allowlisted: a dynamically registered tool is the allowlist's escape hatch"),
)

TOOL_BY_NAME = {t.name: t for t in TOOLS}
HARM_CAPABLE = (EXTERNAL, IRREVERSIBLE)

# The blast radius each class carries, which is what the rail is really sizing against.
BLAST_RADIUS = {
    READ: "low -- information disclosure at worst",
    WRITE_LOW: "low -- reversible internal state",
    EXTERNAL: "medium -- data leaves the boundary; unsending is a request, not a fact",
    IRREVERSIBLE: "high -- funds, records, or a member's coverage. The corpus's 'biggest blast radius' [T]",
}


# ------------------------------------------------------------------------------------------------
# The gate itself
# ------------------------------------------------------------------------------------------------

def session_trust(history) -> int:
    """Trust of a session is the MINIMUM over its history, not the trust of its current window.

    This is the edge case the corpus's pipeline framing invites you to miss: the agent read a hostile
    bulletin ten steps ago, the write tool is enabled because the bulletin has scrolled out of the
    context window, and the tag that would have blocked it is no longer in front of anyone.
    """
    from .rails import TRUST_RANK
    return min((TRUST_RANK.get(t, 0) for t in history), default=TRUST_RANK["system"])


@dataclass
class Decision:
    tool: str
    risk: str
    trust: int
    outcome: str
    reason: str

    @property
    def executed(self) -> bool:
        return self.outcome in (ALLOW, ALLOW_LOGGED)


def decide(tool: Tool, trust: int, level: str = "gated", dry_run: bool = False) -> Decision:
    """The tool-call rail. `level` selects how much of the corpus's recipe [T] is present.

      * `"none"`   -- every tool callable, no schema, no allowlist. The pre-incident state.
      * `"schema"` -- allowlist + strict argument schema. 'Validate every argument against a strict
                      schema. So a malformed or malicious call is rejected before it runs.' [T]
      * `"gated"`  -- the above, plus trust-based capability gating and human approval for
                      irreversible actions.
    """
    if not tool.allowlisted:
        return Decision(tool.name, tool.risk, trust, DENY, "not on the allowlist")
    if level == "none":
        return Decision(tool.name, tool.risk, trust, ALLOW, "no gate configured")
    if level == "schema":
        return Decision(tool.name, tool.risk, trust, ALLOW_LOGGED,
                        "schema validated, no capability check")

    need = MIN_TRUST[tool.risk]
    if trust >= need:
        return Decision(tool.name, tool.risk, trust, ALLOW_LOGGED, f"trust {trust} >= {need}")
    if dry_run:
        return Decision(tool.name, tool.risk, trust, DRY_RUN, "dry-run mode: shadow only")
    if tool.risk == IRREVERSIBLE:
        return Decision(tool.name, tool.risk, trust, APPROVAL,
                        f"trust {trust} < {need} and the action cannot be undone")
    if tool.risk == EXTERNAL:
        return Decision(tool.name, tool.risk, trust, DENY,
                        f"trust {trust} < {need}: untrusted content is in the session")
    if tool.risk == WRITE_LOW:
        return Decision(tool.name, tool.risk, trust, DRY_RUN, "reversible, but the session is dirty")
    return Decision(tool.name, tool.risk, trust, ALLOW_LOGGED, "read on a dirty session")


def gate_matrix(level: str, trust: int = 2) -> list:
    rows = []
    for t in TOOLS:
        d = decide(t, trust, level)
        rows.append({"tool": t.name, "risk": t.risk, "allowlisted": t.allowlisted,
                     "outcome": d.outcome, "reason": d.reason})
    return rows


# ------------------------------------------------------------------------------------------------
# The harm model: harm = (1 - catch) * gate_failure
# ------------------------------------------------------------------------------------------------

# A malicious instruction that reaches the model targets a tool that can do damage -- that is what
# makes it malicious. The split is over WHICH damage, and it is this split that decides how much the
# gate is worth, because each risk class escalates differently.
P_TARGET_IRREVERSIBLE = 0.45
P_TARGET_EXTERNAL = 0.55

# Allowlist + strict schema alone: the corpus's 'validate every argument against a strict schema' [T].
# A schema rejects a malformed or off-allowlist call but cannot reason about intent, so it catches
# less than half of targeted calls in this model.
SCHEMA_CATCH = 0.40

# The floor of the approval step. Not a constant -- `approver_accuracy_at_depth` below shows what
# happens to it under load, which is the point of experiment 6.
BASE_APPROVER_ACCURACY = 0.95
APPROVER_ACCURACY_DECAY_DEPTH = 60.0     # pending items at which accuracy has fallen by 1/e


def gate_failure(level: str, approver_accuracy: float = 0.90,
                 tag_lost: float = 0.0, fail_open: bool = False) -> float:
    """P(the gate lets a harm-capable call through | the attack reached the model).

    Three terms, and the third is the one nobody models:

      1. `none`   -> 1.0. There is nothing between the model and the tool.
      2. `schema` -> 1 - SCHEMA_CATCH. Real, but blind to intent.
      3. `gated`  -> the approval complement on irreversible actions, PLUS the external-tool term
                     that only exists when trust tags are lost and the pipeline fails OPEN. The
                     gate's floor is therefore a function of two unrelated design decisions: how
                     good the human reviewers are, and whether missing provenance is treated as
                     privileged or restricted.
    """
    if level == "none":
        return 1.0
    if level == "schema":
        return 1.0 - SCHEMA_CATCH
    if level == "gated":
        permit_irreversible = 1.0 - approver_accuracy
        permit_external = tag_lost if fail_open else 0.0
        return (P_TARGET_IRREVERSIBLE * permit_irreversible
                + P_TARGET_EXTERNAL * permit_external)
    raise ValueError(f"unknown gate level {level!r}")


def harm_rate(catch: float, level: str, approver_accuracy: float = 0.90,
              tag_lost: float = 0.0, fail_open: bool = False) -> float:
    """Expected harmful action per attack attempt, from the moment it reaches the model.

    Multiplicative, which is the whole design point: better detection and a tighter gate compose,
    and neither substitutes for the other. What differs is their CEILING.
    """
    return (1.0 - catch) * gate_failure(level, approver_accuracy, tag_lost, fail_open)


def lever_table(catch: float, catch_perfect: float, tag_lost: float = 0.03) -> list:
    """The four ways to spend the next dollar, priced against each other on one scale.

    `catch_perfect` is `rails.ceiling_catch` -- the best the detection stack can ever do. The row
    that reads `ceiling` is the reason this topic's answer is not "train a better classifier".
    """
    rows = [
        ("do nothing (pre-incident)", harm_rate(catch, "none")),
        ("allowlist + schema only", harm_rate(catch, "schema")),
        ("+ capability gating (fail-closed)", harm_rate(catch, "gated", 0.90, tag_lost, False)),
        ("+ capability gating (fail-open)", harm_rate(catch, "gated", 0.90, tag_lost, True)),
        ("detectors at their CEILING, no gate", harm_rate(catch_perfect, "none")),
        ("detectors at their CEILING + schema", harm_rate(catch_perfect, "schema")),
        ("detectors at CEILING + gating (closed)", harm_rate(catch_perfect, "gated", 0.90, tag_lost, False)),
        ("detectors at ZERO + gating (closed)", harm_rate(0.0, "gated", 0.90, tag_lost, False)),
    ]
    base = rows[0][1]
    return [{"option": n, "harm_per_attack": v, "reduction_vs_nothing": 1.0 - v / base}
            for n, v in rows]


# ------------------------------------------------------------------------------------------------
# The approval queue: the gate's floor is a resource, and resources run out
# ------------------------------------------------------------------------------------------------

@dataclass
class ApprovalQueue:
    """Human approval modelled as what it is: a finite service with a queue.

    The corpus requires approval for irreversible actions and does not say what happens when there
    are four hundred of them waiting. The failure mode is not refusal -- it is APPROVAL. An approver
    facing a deep queue stops reading, and the control degrades into a latency tax that the topic's
    edge-case list names outright ("human approval as a rubber stamp").
    """
    approvers: int = 2
    decisions_per_hour: float = 40.0
    horizon_hours: float = 8.0

    @property
    def capacity(self) -> float:
        return self.approvers * self.decisions_per_hour

    def utilisation(self, demand_per_hour: float) -> float:
        return demand_per_hour / self.capacity if self.capacity else float("inf")

    def queue_depth(self, demand_per_hour: float) -> float:
        rho = self.utilisation(demand_per_hour)
        if rho >= 1.0:
            return float("inf")
        return rho * rho / (1.0 - rho)     # M/M/1 mean queue length

    def latency_hours(self, demand_per_hour: float) -> float:
        rho = self.utilisation(demand_per_hour)
        if rho >= 1.0:
            return float("inf")
        return rho / (self.capacity * (1.0 - rho))


def approver_accuracy_at_depth(depth: float) -> float:
    """Accuracy falls with the number of items already waiting. Nobody reads item 400."""
    if depth == float("inf"):
        return 0.0
    return BASE_APPROVER_ACCURACY * math.exp(-depth / APPROVER_ACCURACY_DECAY_DEPTH)


def queue_profile(demand_per_hour: float, approvers: int = 2,
                  decisions_per_hour: float = 40.0) -> dict:
    q = ApprovalQueue(approvers, decisions_per_hour)
    depth = q.queue_depth(demand_per_hour)
    acc = approver_accuracy_at_depth(depth)
    rho = q.utilisation(demand_per_hour)
    return {
        "demand": demand_per_hour,
        "utilisation": rho,
        "depth": depth,
        "latency_h": q.latency_hours(demand_per_hour),
        "approver_accuracy": acc,
        "gate_failure": (P_TARGET_IRREVERSIBLE * (1.0 - acc)),
        "rubber_stamp": acc < 0.5,
    }


def sweep_queue(approvers: int = 2, decisions_per_hour: float = 40.0) -> list:
    cap = approvers * decisions_per_hour
    out = []
    for f in (0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0, 1.2):
        out.append(queue_profile(cap * f, approvers, decisions_per_hour))
    return out


def approval_surface(demand_actions_per_day: float, irreversible_share: float,
                     queue_hours: float = 8.0) -> dict:
    """The consequence the corpus's edge case points at: a control is only real if it is affordable.

    Returns the approval demand the design implies and whether an approver pool can serve it inside
    a working day -- the arithmetic that decides whether 'add an undo' is a nicety or a requirement.
    """
    daily = demand_actions_per_day * irreversible_share
    per_hour = daily / queue_hours
    q = ApprovalQueue()
    return {
        "daily_irreversible": daily,
        "per_hour": per_hour,
        "capacity_per_hour": q.capacity,
        "utilisation": q.utilisation(per_hour),
        "latency_h": q.latency_hours(per_hour),
        "servable": q.utilisation(per_hour) < 1.0,
    }


def undo_vs_approve(demand_actions_per_day: float, approvers: int = 2,
                    engineering_cost_days: float = 15.0) -> dict:
    """The two ways to answer an unaffordable approval queue.

    Buy approvers, or remove the irreversibility. The corpus does not name the second option; it is
    the one that scales, because approval demand grows with agent volume and approver headcount
    does not.
    """
    q = ApprovalQueue(approvers)
    demand_per_hour = demand_actions_per_day / q.horizon_hours
    need_approvers = math.ceil(demand_per_hour / q.decisions_per_hour) if demand_per_hour else 1
    return {
        "demand_per_hour": demand_per_hour,
        "approvers_needed": need_approvers,
        "approvers_have": approvers,
        "headcount_multiple": need_approvers / approvers,
        "engineering_days_for_undo": engineering_cost_days,
        "undo_pays_back_at_actions_per_day": (
            engineering_cost_days * 8.0 * q.decisions_per_hour / max(1.0, irreversible_share_default())
        ),
    }


def irreversible_share_default() -> float:
    return P_TARGET_IRREVERSIBLE


# ------------------------------------------------------------------------------------------------
# Trust propagation, exposed here because the gate consumes it
# ------------------------------------------------------------------------------------------------

def gate_under_tag_loss(hops: int, loss_per_hop: float, fail_open: bool,
                        catch: float, approver_accuracy: float = 0.90) -> dict:
    """The interaction the corpus's five-layer defence implies and never draws.

    The gate's external-tool term is `tag_lost` when the pipeline fails open. So a middleware that
    drops an attribute at 1% per hop does not degrade a metadata field -- it WIDENS THE GATE, and
    the effect is invisible in every detection metric because nothing was detected or missed. The
    only place it shows up is the harmful-action rate.
    """
    eff = effective_privilege(hops, loss_per_hop, fail_open)
    lost = eff["lost"]
    gfail = gate_failure("gated", approver_accuracy, lost, fail_open)
    gfail_closed = gate_failure("gated", approver_accuracy, lost, False)
    return {
        "hops": hops, "loss_per_hop": loss_per_hop, "fail_open": fail_open,
        "tag_survival": eff["survival"],
        "privileged_share": eff["privileged_share"],
        "gate_failure": gfail,
        "gate_failure_if_closed": gfail_closed,
        "harm": harm_rate(catch, "gated", approver_accuracy, lost, fail_open),
        "harm_if_closed": harm_rate(catch, "gated", approver_accuracy, lost, False),
    }
