"""
models.py — Type-safe data contracts for the Email Triage Environment.

Defines:
  - EmailTriageAction: What the agent can do each step
  - EmailTriageObservation: What the agent sees each step
  - EmailTriageState: Internal episode metadata (hidden from agent)
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────
#  BASE CLASSES (compatible with openenv-core)
#  If openenv-core is installed, you can instead import:
#    from openenv.core.env_server import Action, Observation, State
#  We define standalone Pydantic models so the code runs
#  BOTH on Colab (without openenv-core) and inside Docker.
# ──────────────────────────────────────────────────────────────

class Action(BaseModel):
    """Base action model."""
    pass


class Observation(BaseModel):
    """Base observation model — every step returns done + reward."""
    done: bool = False
    reward: Optional[float] = None


class State(BaseModel):
    """Base state model — episode-level metadata."""
    episode_id: Optional[str] = None
    step_count: int = 0


# ──────────────────────────────────────────────────────────────
#  EMAIL TRIAGE MODELS
# ──────────────────────────────────────────────────────────────

# Valid departments the agent can route emails to
VALID_DEPARTMENTS = [
    "billing",
    "technical_support",
    "sales",
    "human_resources",
    "general_inquiry",
]

# Valid priority levels
VALID_PRIORITIES = ["low", "medium", "high", "urgent"]


class EmailTriageAction(Action):
    """
    What the agent submits each step.

    Fields:
        department:  Which department to route this email to.
                     Must be one of VALID_DEPARTMENTS.
        priority:    How urgent is this email.
                     Must be one of VALID_PRIORITIES.
        reasoning:   (Optional) Short explanation of why.
    """
    department: str = Field(
        ...,
        description="Department to route email to. "
                    "One of: billing, technical_support, sales, "
                    "human_resources, general_inquiry",
    )
    priority: str = Field(
        ...,
        description="Priority level. One of: low, medium, high, urgent",
    )
    reasoning: str = Field(
        default="",
        description="Optional short reasoning for this classification",
    )


class EmailTriageObservation(Observation):
    """
    What the agent sees after each step (or on reset).

    Fields (inherited):
        done:    bool   — Is the episode over?
        reward:  float  — Reward from the last action (None on reset)

    Fields (custom):
        email_id:           Unique ID of the current email
        email_subject:      Subject line
        email_body:         Full email body text
        email_sender:       Sender address
        email_metadata:     Extra metadata (timestamps, headers, etc.)
        available_departments: List of valid department choices
        available_priorities:  List of valid priority choices
        emails_remaining:   How many emails are left in this episode
        feedback:           Feedback message from previous action
        score_so_far:       Running accuracy score (0.0 to 1.0)
    """
    email_id: str = ""
    email_subject: str = ""
    email_body: str = ""
    email_sender: str = ""
    email_metadata: Dict[str, Any] = Field(default_factory=dict)
    available_departments: List[str] = Field(
        default_factory=lambda: VALID_DEPARTMENTS.copy()
    )
    available_priorities: List[str] = Field(
        default_factory=lambda: VALID_PRIORITIES.copy()
    )
    emails_remaining: int = 0
    feedback: str = ""
    score_so_far: float = 0.0


class EmailTriageState(State):
    """
    Internal episode state — not visible to the agent.

    Fields (inherited):
        episode_id:  str — Unique episode identifier
        step_count:  int — Number of steps taken so far

    Fields (custom):
        task_difficulty:       "easy", "medium", or "hard"
        total_emails:          Total emails in this episode
        correct_count:         How many the agent got right so far
        current_email_index:   Which email we're currently showing
        ground_truth:          List of correct (department, priority) pairs
    """
    task_difficulty: str = "easy"
    total_emails: int = 0
    correct_count: int = 0
    current_email_index: int = 0
    ground_truth: List[Dict[str, str]] = Field(default_factory=list)
