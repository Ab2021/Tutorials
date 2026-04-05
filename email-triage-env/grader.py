"""
grader.py — Deterministic, rule-based grading for Email Triage.

The grader NEVER uses LLM judgment. All scoring is based on exact
string matching against the ground truth labels.

Scoring breakdown per email:
  - Department match:  +0.6   (most important)
  - Priority match:    +0.3   (second most important)
  - Valid inputs:      +0.1   (did the agent submit valid values?)
  - Wrong department:  +0.0
  - Wrong priority:    +0.0
  - Invalid input:     -0.05  (submitted a non-existent department/priority)

Step reward range: [-0.05, +1.0]
Final episode reward: average accuracy across all emails (0.0 to 1.0)
"""

from typing import Dict, Tuple
from models import VALID_DEPARTMENTS, VALID_PRIORITIES


def grade_single_email(
    action_department: str,
    action_priority: str,
    ground_truth: Dict[str, str],
) -> Tuple[float, str, Dict[str, bool]]:
    """
    Grade a single email classification.

    Args:
        action_department:  The department the agent chose
        action_priority:    The priority the agent chose
        ground_truth:       Dict with keys "department" and "priority"

    Returns:
        (reward, feedback_message, details_dict)
    """
    reward = 0.0
    feedback_parts = []
    details = {
        "department_correct": False,
        "priority_correct": False,
        "valid_department": False,
        "valid_priority": False,
    }

    # Normalize inputs
    dept = action_department.strip().lower()
    prio = action_priority.strip().lower()
    true_dept = ground_truth["department"].strip().lower()
    true_prio = ground_truth["priority"].strip().lower()

    # ── Check if inputs are valid ──
    if dept in VALID_DEPARTMENTS:
        details["valid_department"] = True
        reward += 0.05
    else:
        reward -= 0.05
        feedback_parts.append(
            f"Invalid department '{dept}'. "
            f"Valid options: {VALID_DEPARTMENTS}"
        )

    if prio in VALID_PRIORITIES:
        details["valid_priority"] = True
        reward += 0.05
    else:
        reward -= 0.05
        feedback_parts.append(
            f"Invalid priority '{prio}'. "
            f"Valid options: {VALID_PRIORITIES}"
        )

    # ── Check department match ──
    if dept == true_dept:
        details["department_correct"] = True
        reward += 0.6
        feedback_parts.append("✓ Department: Correct!")
    else:
        feedback_parts.append(
            f"✗ Department: Incorrect. You chose '{dept}'."
        )

    # ── Check priority match ──
    if prio == true_prio:
        details["priority_correct"] = True
        reward += 0.3
        feedback_parts.append("✓ Priority: Correct!")
    elif details["valid_priority"] and _is_adjacent_priority(prio, true_prio):
        # Partial credit for being one level off
        reward += 0.1
        feedback_parts.append(
            f"~ Priority: Close — you chose '{prio}' "
            f"(one level off)."
        )
    else:
        feedback_parts.append(
            f"✗ Priority: Incorrect. You chose '{prio}'."
        )

    feedback = " | ".join(feedback_parts)
    return reward, feedback, details


def _is_adjacent_priority(chosen: str, correct: str) -> bool:
    """Check if chosen priority is exactly one level away from correct."""
    order = VALID_PRIORITIES  # ["low", "medium", "high", "urgent"]
    if chosen not in order or correct not in order:
        return False
    return abs(order.index(chosen) - order.index(correct)) == 1


def compute_final_score(
    correct_count: int,
    total_count: int,
    total_reward: float,
) -> Tuple[float, str]:
    """
    Compute the final episode score.

    Args:
        correct_count:  Number of emails where both dept AND priority correct
        total_count:    Total emails in the episode
        total_reward:   Sum of all step rewards

    Returns:
        (final_score, summary_message)
    """
    if total_count == 0:
        return 0.0, "No emails were processed."

    accuracy = correct_count / total_count
    avg_reward = total_reward / total_count

    # Final score = weighted combination
    final_score = (0.6 * accuracy) + (0.4 * avg_reward)

    summary = (
        f"Episode complete! "
        f"Accuracy: {correct_count}/{total_count} "
        f"({accuracy:.0%}) | "
        f"Avg step reward: {avg_reward:.3f} | "
        f"Final score: {final_score:.3f}"
    )
    return final_score, summary
