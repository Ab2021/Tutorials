"""
server/app.py — FastAPI server for the Email Triage Environment.

This module creates the HTTP/WebSocket server that exposes the environment.
It works two ways:

1. WITH openenv-core installed (production / HF Spaces):
   Uses create_fastapi_app() for full OpenEnv spec compliance.

2. WITHOUT openenv-core (Colab / local development):
   Falls back to a standalone FastAPI app with manual endpoints.

Endpoints:
  GET  /health              → {"status": "healthy"}
  POST /reset               → Start a new episode
  POST /step                → Take an action
  GET  /state               → Get episode metadata
  GET  /docs                → Interactive API docs (Swagger)
"""

import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from server.environment import EmailTriageEnvironment
from models import EmailTriageAction


# ──────────────────────────────────────────────────────────────
#  Try to use openenv-core's create_fastapi_app if available
# ──────────────────────────────────────────────────────────────

try:
    from openenv.core.env_server import create_fastapi_app
    app = create_fastapi_app(EmailTriageEnvironment)
    print("[INFO] Using openenv-core create_fastapi_app")

except ImportError:
    # ──────────────────────────────────────────────────────────
    #  Fallback: standalone FastAPI app (works on Colab)
    # ──────────────────────────────────────────────────────────
    print("[INFO] openenv-core not found — using standalone FastAPI server")

    app = FastAPI(
        title="Email Triage Environment",
        description=(
            "An OpenEnv-compatible RL environment for email classification. "
            "Classify emails by department and priority."
        ),
        version="1.0.0",
    )

    # CORS for Colab / browser access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global environment instance
    env = EmailTriageEnvironment()

    # ── Request / Response models ──

    class ResetRequest(BaseModel):
        difficulty: str = "easy"
        episode_id: Optional[str] = None

    class StepRequest(BaseModel):
        department: str
        priority: str
        reasoning: str = ""

    # ── Endpoints ──

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.post("/reset")
    async def reset(request: ResetRequest):
        """
        Start a new email triage episode.

        Body:
          - difficulty: "easy" | "medium" | "hard"
          - episode_id: (optional) custom episode ID
        """
        try:
            obs = env.reset(
                difficulty=request.difficulty,
                episode_id=request.episode_id,
            )
            return obs.model_dump()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/step")
    async def step(request: StepRequest):
        """
        Classify the current email.

        Body:
          - department: one of [billing, technical_support, sales,
                        human_resources, general_inquiry]
          - priority: one of [low, medium, high, urgent]
          - reasoning: (optional) explanation
        """
        action = EmailTriageAction(
            department=request.department,
            priority=request.priority,
            reasoning=request.reasoning,
        )
        obs = env.step(action)
        return obs.model_dump()

    @app.get("/state")
    async def get_state():
        """Get current episode state / metadata."""
        return env.state.model_dump()

    @app.get("/")
    async def root():
        """Root endpoint with environment info."""
        return {
            "name": "Email Triage Environment",
            "version": "1.0.0",
            "description": (
                "Classify emails by department and priority. "
                "An OpenEnv-compatible RL environment."
            ),
            "endpoints": {
                "health": "/health",
                "reset": "/reset (POST)",
                "step": "/step (POST)",
                "state": "/state (GET)",
                "docs": "/docs",
            },
            "valid_departments": [
                "billing",
                "technical_support",
                "sales",
                "human_resources",
                "general_inquiry",
            ],
            "valid_priorities": ["low", "medium", "high", "urgent"],
            "difficulty_levels": {
                "easy": "3 emails, obvious categories",
                "medium": "5 emails, some ambiguity",
                "hard": "8 emails, misleading subjects and edge cases",
            },
        }
