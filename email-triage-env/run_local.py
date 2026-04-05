"""
run_local.py — Test the Email Triage Environment locally WITHOUT a server.

This script directly imports the environment class and runs all 3
difficulty levels, using a simple rule-based policy (no LLM needed).

Usage:
    python run_local.py

This is the fastest way to verify the environment works correctly.
"""

import sys
import os

# Ensure we can import from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.environment import EmailTriageEnvironment
from models import EmailTriageAction


def simple_rule_policy(obs: dict) -> dict:
    """
    A dead-simple rule-based policy for testing.

    Looks for keywords in the email subject + body to determine
    department and priority. NOT meant to be perfect — just enough
    to verify the environment works.
    """
    subject = obs.get("email_subject", "").lower()
    body = obs.get("email_body", "").lower()
    text = subject + " " + body

    # Department classification (keyword matching)
    if any(w in text for w in ["invoice", "billing", "charge", "refund",
                                 "payment", "subscription", "cancel"]):
        department = "billing"
    elif any(w in text for w in ["error", "bug", "crash", "broken", "500",
                                   "not working", "api", "login", "password"]):
        department = "technical_support"
    elif any(w in text for w in ["upgrade", "enterprise", "pricing", "buy",
                                   "licenses", "partnership", "sales"]):
        department = "sales"
    elif any(w in text for w in ["hr", "employee", "payroll", "hiring",
                                   "remote work", "policy", "recruiter"]):
        department = "human_resources"
    else:
        department = "general_inquiry"

    # Priority classification
    if any(w in text for w in ["urgent", "immediately", "blocking",
                                 "down", "asap"]):
        priority = "urgent"
    elif any(w in text for w in ["important", "soon", "deadline"]):
        priority = "high"
    elif any(w in text for w in ["question", "wondering", "curious"]):
        priority = "low"
    else:
        priority = "medium"

    return {"department": department, "priority": priority}


def run_single_difficulty(env, difficulty):
    """Run one episode at a given difficulty and print results."""
    print(f"\n{'='*60}")
    print(f"  DIFFICULTY: {difficulty.upper()}")
    print(f"{'='*60}")

    # Reset
    obs = env.reset(difficulty=difficulty)
    print(f"\nEpisode started: {obs.feedback}")
    print(f"Total emails: {obs.emails_remaining + 1}\n")

    step = 0
    while not obs.done:
        step += 1

        # Get observation as dict for the policy
        obs_dict = obs.model_dump()

        print(f"--- Email {step} ---")
        print(f"  ID:      {obs.email_id}")
        print(f"  Subject: {obs.email_subject}")
        print(f"  From:    {obs.email_sender}")
        print(f"  Body:    {obs.email_body[:100]}...")

        # Apply rule-based policy
        decision = simple_rule_policy(obs_dict)
        print(f"  → Classified: dept={decision['department']}, "
              f"priority={decision['priority']}")

        # Step the environment
        action = EmailTriageAction(
            department=decision["department"],
            priority=decision["priority"],
            reasoning="rule-based policy",
        )
        obs = env.step(action)

        print(f"  → Reward:   {obs.reward}")
        print(f"  → Feedback: {obs.feedback}")
        print()

    # Final results
    state = env.state
    print(f"{'='*60}")
    print(f"  EPISODE COMPLETE ({difficulty.upper()})")
    print(f"  Correct: {state.correct_count}/{state.total_emails}")
    print(f"  Final Score: {obs.score_so_far:.3f}")
    print(f"  Steps Taken: {state.step_count}")
    print(f"{'='*60}")

    return obs.score_so_far


def main():
    """Run all 3 difficulty levels."""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║    EMAIL TRIAGE ENVIRONMENT — LOCAL TEST                ║")
    print("║    Testing all 3 difficulty levels with rule-based bot  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    env = EmailTriageEnvironment()
    scores = {}

    for diff in ["easy", "medium", "hard"]:
        score = run_single_difficulty(env, diff)
        scores[diff] = score

    # Summary
    print(f"\n\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for diff, score in scores.items():
        bar = "█" * int(score * 30) + "░" * (30 - int(score * 30))
        print(f"  {diff:8s}: {bar} {score:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
