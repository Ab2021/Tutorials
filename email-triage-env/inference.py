"""
inference.py — LLM Agent Baseline for Email Triage Environment.

THIS FILE MUST LIVE AT THE REPO ROOT.

It connects to the running environment server and uses an LLM
(via OpenAI python client) to classify emails.

Required environment variables:
  API_BASE_URL  — OpenAI-compatible API endpoint
  MODEL_NAME    — Model identifier (e.g., "Qwen/Qwen2.5-7B-Instruct")
  HF_TOKEN      — Hugging Face token (used as API key)

Optional environment variables:
  ENV_URL       — Environment server URL (default: http://localhost:8000)
  DIFFICULTY    — Task difficulty: easy|medium|hard (default: easy)

Usage:
  # Set environment variables
  export API_BASE_URL=https://api-inference.huggingface.co/v1
  export MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
  export HF_TOKEN=hf_xxxxxxxxxxxxx
  export ENV_URL=http://localhost:8000
  export DIFFICULTY=easy

  # Run
  python inference.py
"""

import os
import sys
import json
import time
import requests
from openai import OpenAI


# ──────────────────────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get(
    "API_BASE_URL", "https://api-inference.huggingface.co/v1"
)
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")
DIFFICULTY = os.environ.get("DIFFICULTY", "easy")

# Initialize OpenAI client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "dummy-token-prevents-validation-error",
)

# ──────────────────────────────────────────────────────────────
#  LLM CALL
# ──────────────────────────────────────────────────────────────

def call_llm(messages: list, max_retries: int = 3) -> str:
    """
    Call the LLM via OpenAI python client.

    Args:
        messages: List of {"role": ..., "content": ...} dicts
        max_retries: Number of retries on failure

    Returns:
        The LLM's response text
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=512,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  [RETRY] LLM call failed: {e}. Retrying...")
                time.sleep(2 ** attempt)
            else:
                print(f"  [ERROR] LLM call failed after {max_retries} attempts: {e}")
                raise


def parse_llm_response(response_text: str) -> dict:
    """
    Parse the LLM's response to extract department and priority.

    Expects the LLM to return JSON like:
    {"department": "billing", "priority": "high", "reasoning": "..."}

    Falls back to keyword extraction if JSON parsing fails.
    """
    if not response_text:
        raise ValueError("Empty response text from LLM.")
        
    # Try direct JSON parsing
    try:
        # Handle markdown-wrapped JSON
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        data = json.loads(text)
        return {
            "department": data.get("department", "general_inquiry"),
            "priority": data.get("priority", "medium"),
            "reasoning": data.get("reasoning", ""),
        }
    except (json.JSONDecodeError, IndexError):
        pass

    # Fallback: keyword extraction
    text_lower = response_text.lower()

    department = "general_inquiry"
    for dept in [
        "billing",
        "technical_support",
        "sales",
        "human_resources",
        "general_inquiry",
    ]:
        if dept in text_lower:
            department = dept
            break

    priority = "medium"
    for prio in ["urgent", "high", "medium", "low"]:
        if prio in text_lower:
            priority = prio
            break

    return {
        "department": department,
        "priority": priority,
        "reasoning": response_text[:200],
    }


# ──────────────────────────────────────────────────────────────
#  SYSTEM PROMPT
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert email triage assistant. Your job is to classify incoming emails.

For each email, you must determine:
1. DEPARTMENT — which team should handle this email:
   - billing: payment issues, invoices, refunds, subscription charges
   - technical_support: bugs, errors, crashes, feature not working, API issues
   - sales: new purchases, upgrades, partnerships, pricing inquiries
   - human_resources: employee matters, hiring, payroll, policies
   - general_inquiry: everything else, spam, vague questions

2. PRIORITY — how urgent is this:
   - urgent: system down, blocking multiple users, security issue
   - high: important issue with clear impact, needs attention soon
   - medium: normal request, no immediate urgency
   - low: nice-to-have, informational, spam

IMPORTANT RULES:
- Look at the CONTENT, not just the subject line (subjects can be misleading)
- Spam emails should always be: department=general_inquiry, priority=low
- If an email mentions multiple departments, pick the PRIMARY one
- A double charge is billing (high), not technical support

Respond ONLY with a JSON object:
{"department": "...", "priority": "...", "reasoning": "brief explanation"}

No other text. Just the JSON object."""


# ──────────────────────────────────────────────────────────────
#  MAIN AGENT LOOP
# ──────────────────────────────────────────────────────────────

def run_agent():
    """Run the LLM agent against the Email Triage Environment."""

    print(f"[START]")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Difficulty: {DIFFICULTY}")
    print(f"  Environment: {ENV_URL}")
    print()

    # ── Step 1: Reset the environment ──
    try:
        reset_resp = requests.post(
            f"{ENV_URL}/reset",
            json={"difficulty": DIFFICULTY},
            timeout=30,
        )
        reset_resp.raise_for_status()
        obs = reset_resp.json()
    except Exception as e:
        print(f"[ERROR] Failed to connect to environment: {e}")
        print(f"  Make sure the server is running at {ENV_URL}")
        print(f"[END]")
        return

    step_num = 0
    total_reward = 0.0

    # ── Step 2: Process emails until done ──
    while not obs.get("done", False):
        step_num += 1

        # Build the prompt for this email
        email_context = (
            f"Email ID: {obs.get('email_id', 'N/A')}\n"
            f"From: {obs.get('email_sender', 'N/A')}\n"
            f"Subject: {obs.get('email_subject', 'N/A')}\n"
            f"Body:\n{obs.get('email_body', 'N/A')}\n"
        )

        metadata = obs.get("email_metadata", {})
        if metadata:
            email_context += f"Metadata: {json.dumps(metadata)}\n"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": email_context},
        ]

        # Call the LLM
        print(f"[STEP] Step {step_num}")
        print(f"  Email: {obs.get('email_subject', 'N/A')}")

        try:
            llm_response = call_llm(messages)
            classification = parse_llm_response(llm_response)
        except Exception as e:
            print(f"  LLM Error: {e}")
            classification = {
                "department": "general_inquiry",
                "priority": "medium",
                "reasoning": "LLM call failed, using default",
            }

        print(f"  → Dept: {classification['department']}")
        print(f"  → Priority: {classification['priority']}")
        print(f"  → Reasoning: {classification['reasoning'][:100]}")

        # Send action to environment
        try:
            step_resp = requests.post(
                f"{ENV_URL}/step",
                json=classification,
                timeout=30,
            )
            step_resp.raise_for_status()
            obs = step_resp.json()
        except Exception as e:
            print(f"  [ERROR] Step failed: {e}")
            break

        reward = obs.get("reward")
        if reward is not None:
            total_reward += reward
        feedback = obs.get("feedback", "")
        print(f"  Reward: {reward}")
        print(f"  Feedback: {feedback}")
        print()

    # ── Step 3: Final results ──
    print(f"[END]")
    print(f"  Total steps: {step_num}")
    print(f"  Final score: {obs.get('score_so_far', 'N/A')}")
    print(f"  Final feedback: {obs.get('feedback', 'N/A')}")


# ──────────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Validate required env vars
    if not HF_TOKEN:
        print(
            "WARNING: HF_TOKEN not set. LLM calls may fail.\n"
            "Set it with: export HF_TOKEN=hf_xxxxxxxxxxxxx\n"
        )

    run_agent()
