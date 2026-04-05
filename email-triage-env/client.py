"""
client.py — Client for connecting to the Email Triage Environment.

Works two ways:
1. WITH openenv-core: Extends EnvClient for WebSocket communication
2. WITHOUT openenv-core: Uses plain HTTP requests (works on Colab)

Example usage:
    # HTTP client (Colab / simple testing)
    client = EmailTriageClient("http://localhost:8000")
    obs = client.reset(difficulty="easy")
    while not obs["done"]:
        obs = client.step(department="billing", priority="high")

    # With openenv-core (full WebSocket client)
    async with EmailTriageEnv(base_url="http://localhost:8000") as env:
        result = await env.reset()
        ...
"""

import requests
from typing import Optional, Dict, Any


class EmailTriageClient:
    """
    Simple HTTP client for the Email Triage Environment.

    This client works WITHOUT openenv-core installed (perfect for Colab).
    It uses plain HTTP POST/GET requests.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Args:
            base_url: URL of the running environment server.
                      e.g. "http://localhost:8000" or
                           "https://username-email-triage.hf.space"
        """
        self.base_url = base_url.rstrip("/")

    def health(self) -> Dict[str, Any]:
        """Check if the server is running."""
        resp = requests.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def reset(
        self,
        difficulty: str = "easy",
        episode_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Start a new episode.

        Args:
            difficulty: "easy" (3 emails), "medium" (5), "hard" (8)
            episode_id: Optional custom episode ID

        Returns:
            Dict with observation fields (email_subject, email_body, etc.)
        """
        payload = {"difficulty": difficulty}
        if episode_id:
            payload["episode_id"] = episode_id

        resp = requests.post(
            f"{self.base_url}/reset",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def step(
        self,
        department: str,
        priority: str,
        reasoning: str = "",
    ) -> Dict[str, Any]:
        """
        Classify the current email.

        Args:
            department: One of [billing, technical_support, sales,
                        human_resources, general_inquiry]
            priority:   One of [low, medium, high, urgent]
            reasoning:  Optional explanation

        Returns:
            Dict with observation fields + reward + done + feedback
        """
        payload = {
            "department": department,
            "priority": priority,
            "reasoning": reasoning,
        }
        resp = requests.post(
            f"{self.base_url}/step",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def get_state(self) -> Dict[str, Any]:
        """Get current episode state."""
        resp = requests.get(f"{self.base_url}/state", timeout=10)
        resp.raise_for_status()
        return resp.json()


# ──────────────────────────────────────────────────────────────
#  OpenEnv-native client (requires openenv-core)
# ──────────────────────────────────────────────────────────────

try:
    from openenv.core.env_client import EnvClient
    from openenv.core.client_types import StepResult
    from models import (
        EmailTriageAction,
        EmailTriageObservation,
        EmailTriageState,
    )

    class EmailTriageEnv(
        EnvClient[EmailTriageAction, EmailTriageObservation, EmailTriageState]
    ):
        """
        OpenEnv-native WebSocket client for Email Triage.
        Requires openenv-core to be installed.
        """

        def _step_payload(self, action: EmailTriageAction) -> dict:
            return {
                "department": action.department,
                "priority": action.priority,
                "reasoning": action.reasoning,
            }

        def _parse_result(self, payload: dict) -> StepResult:
            obs_data = payload.get("observation", payload)
            return StepResult(
                observation=EmailTriageObservation(
                    done=payload.get("done", False),
                    reward=payload.get("reward"),
                    email_id=obs_data.get("email_id", ""),
                    email_subject=obs_data.get("email_subject", ""),
                    email_body=obs_data.get("email_body", ""),
                    email_sender=obs_data.get("email_sender", ""),
                    email_metadata=obs_data.get("email_metadata", {}),
                    available_departments=obs_data.get(
                        "available_departments", []
                    ),
                    available_priorities=obs_data.get(
                        "available_priorities", []
                    ),
                    emails_remaining=obs_data.get("emails_remaining", 0),
                    feedback=obs_data.get("feedback", ""),
                    score_so_far=obs_data.get("score_so_far", 0.0),
                ),
                reward=payload.get("reward"),
                done=payload.get("done", False),
            )

        def _parse_state(self, payload: dict) -> EmailTriageState:
            return EmailTriageState(
                episode_id=payload.get("episode_id"),
                step_count=payload.get("step_count", 0),
                task_difficulty=payload.get("task_difficulty", "easy"),
                total_emails=payload.get("total_emails", 0),
                correct_count=payload.get("correct_count", 0),
                current_email_index=payload.get("current_email_index", 0),
            )

except ImportError:
    # openenv-core not available — only HTTP client is available
    pass
