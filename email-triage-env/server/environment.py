"""
server/environment.py — Core Email Triage Environment logic.

Implements the 3-method OpenEnv interface:
  - reset(difficulty)  → Start a new episode with emails at chosen difficulty
  - step(action)       → Classify the current email and move to the next one
  - state (property)   → Return episode metadata

This file is the HEART of the submission.
"""

import uuid
import copy
import sys
import os

# Add parent directory so we can import models, tasks, grader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    EmailTriageAction,
    EmailTriageObservation,
    EmailTriageState,
    VALID_DEPARTMENTS,
    VALID_PRIORITIES,
)
from tasks import get_tasks
from grader import grade_single_email, compute_final_score


class EmailTriageEnvironment:
    """
    Email Triage RL Environment.

    The agent receives one email at a time and must classify it by:
      1. Department (billing, technical_support, sales, human_resources, general_inquiry)
      2. Priority (low, medium, high, urgent)

    Episode flow:
      reset(difficulty="easy"|"medium"|"hard")
        → Returns first email as observation
      step(EmailTriageAction)
        → Grades the classification, returns next email or done=True
      state
        → Current episode metadata

    Supports concurrent sessions (each instance maintains its own state).
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        """Initialize with empty state."""
        self._state = EmailTriageState()
        self._emails = []
        self._total_reward = 0.0
        self._correct_count = 0
        self._current_index = 0

    def reset(
        self,
        seed=None,
        episode_id=None,
        difficulty="easy",
        **kwargs,
    ) -> EmailTriageObservation:
        """
        Start a new email triage episode.

        Args:
            seed:       Not used (tasks are predetermined)
            episode_id: Optional custom episode ID
            difficulty: "easy" (3 emails), "medium" (5), or "hard" (8)

        Returns:
            EmailTriageObservation with the first email to classify.
        """
        # Load emails for the chosen difficulty
        self._emails = get_tasks(difficulty)
        self._current_index = 0
        self._total_reward = 0.0
        self._correct_count = 0

        # Build ground truth list (hidden from agent)
        ground_truths = [e["ground_truth"] for e in self._emails]

        # Initialize state
        self._state = EmailTriageState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_difficulty=difficulty,
            total_emails=len(self._emails),
            correct_count=0,
            current_email_index=0,
            ground_truth=copy.deepcopy(ground_truths),
        )

        # Return the first email as observation
        first_email = self._emails[0]
        return EmailTriageObservation(
            done=False,
            reward=None,
            email_id=first_email["id"],
            email_subject=first_email["subject"],
            email_body=first_email["body"],
            email_sender=first_email["sender"],
            email_metadata=first_email.get("metadata", {}),
            available_departments=VALID_DEPARTMENTS.copy(),
            available_priorities=VALID_PRIORITIES.copy(),
            emails_remaining=len(self._emails) - 1,
            feedback="Episode started. Classify this email by department and priority.",
            score_so_far=0.0,
        )

    def step(
        self,
        action: EmailTriageAction,
        timeout_s=None,
        **kwargs,
    ) -> EmailTriageObservation:
        """
        Process the agent's classification of the current email.

        Args:
            action: EmailTriageAction with department, priority, reasoning

        Returns:
            EmailTriageObservation — grading feedback + next email (or done)
        """
        # Increment step count
        self._state.step_count += 1

        # Get ground truth for current email
        current_gt = self._state.ground_truth[self._current_index]

        # Grade the agent's classification (deterministic, rule-based)
        step_reward, feedback, details = grade_single_email(
            action_department=action.department,
            action_priority=action.priority,
            ground_truth=current_gt,
        )

        # Update running totals
        self._total_reward += step_reward
        if details["department_correct"] and details["priority_correct"]:
            self._correct_count += 1
            self._state.correct_count = self._correct_count

        # Move to next email
        self._current_index += 1
        self._state.current_email_index = self._current_index

        # Check if episode is done
        is_done = self._current_index >= len(self._emails)

        if is_done:
            # Compute final score
            final_score, summary = compute_final_score(
                correct_count=self._correct_count,
                total_count=len(self._emails),
                total_reward=self._total_reward,
            )
            return EmailTriageObservation(
                done=True,
                reward=final_score,
                email_id="",
                email_subject="",
                email_body="",
                email_sender="",
                email_metadata={},
                available_departments=VALID_DEPARTMENTS.copy(),
                available_priorities=VALID_PRIORITIES.copy(),
                emails_remaining=0,
                feedback=f"{feedback} || FINAL: {summary}",
                score_so_far=final_score,
            )
        else:
            # Return next email
            next_email = self._emails[self._current_index]
            score_so_far = (
                self._correct_count / self._current_index
                if self._current_index > 0
                else 0.0
            )
            return EmailTriageObservation(
                done=False,
                reward=step_reward,
                email_id=next_email["id"],
                email_subject=next_email["subject"],
                email_body=next_email["body"],
                email_sender=next_email["sender"],
                email_metadata=next_email.get("metadata", {}),
                available_departments=VALID_DEPARTMENTS.copy(),
                available_priorities=VALID_PRIORITIES.copy(),
                emails_remaining=len(self._emails) - self._current_index - 1,
                feedback=feedback,
                score_so_far=score_so_far,
            )

    @property
    def state(self) -> EmailTriageState:
        """Return current episode state (for debugging / metadata)."""
        return self._state
