"""
run_colab.py — Complete Colab-ready script for Email Triage Environment.

Copy this entire file into a Google Colab notebook cell and run it.
No external files needed — everything is self-contained.

This script:
  1. Installs dependencies
  2. Starts the FastAPI server in the background
  3. Runs a simple rule-based agent against all 3 difficulty levels
  4. (Optional) Runs the LLM-based agent if API credentials are provided

HOW TO USE IN COLAB:
  1. Open https://colab.research.google.com
  2. Create a new notebook
  3. Paste this entire file into the first cell
  4. Click "Run" (Ctrl+Enter)
"""

# ══════════════════════════════════════════════════════════════
#  CELL 1: Install dependencies
# ══════════════════════════════════════════════════════════════

import subprocess
import sys

print("Installing dependencies...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "fastapi", "uvicorn[standard]", "pydantic", "requests", "openai"
])
print("✅ Dependencies installed!\n")


# ══════════════════════════════════════════════════════════════
#  CELL 2: Define all models inline
# ══════════════════════════════════════════════════════════════

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Action(BaseModel):
    pass

class Observation(BaseModel):
    done: bool = False
    reward: Optional[float] = None

class State(BaseModel):
    episode_id: Optional[str] = None
    step_count: int = 0

VALID_DEPARTMENTS = [
    "billing", "technical_support", "sales",
    "human_resources", "general_inquiry",
]

VALID_PRIORITIES = ["low", "medium", "high", "urgent"]


class EmailTriageAction(Action):
    department: str = Field(..., description="Department to route email to")
    priority: str = Field(..., description="Priority level")
    reasoning: str = Field(default="", description="Optional reasoning")


class EmailTriageObservation(Observation):
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
    task_difficulty: str = "easy"
    total_emails: int = 0
    correct_count: int = 0
    current_email_index: int = 0
    ground_truth: List[Dict[str, str]] = Field(default_factory=list)


print("✅ Models defined!\n")


# ══════════════════════════════════════════════════════════════
#  CELL 3: Define tasks (email datasets)
# ══════════════════════════════════════════════════════════════

def get_easy_tasks():
    return [
        {
            "id": "easy-001",
            "subject": "Billing question about my last invoice",
            "body": (
                "Hi,\n\nI received invoice #INV-2026-1042 last week and the total "
                "seems higher than expected. I was charged $249.99 but my plan is "
                "Basic at $19.99/month. Could you review and correct this?\n\n"
                "Account ID: ACC-78432.\n\nThanks, John Smith"
            ),
            "sender": "john.smith@example.com",
            "metadata": {"date": "2026-04-01T10:30:00Z", "account_id": "ACC-78432"},
            "ground_truth": {"department": "billing", "priority": "high"},
        },
        {
            "id": "easy-002",
            "subject": "Cannot login to my account — error 500",
            "body": (
                "Hello Support,\n\nI've been trying to log in for the past hour "
                "but keep getting 'Internal Server Error (500)'. Tried Chrome and "
                "Firefox. This is blocking my entire team.\n\nBest, Sarah Jones"
            ),
            "sender": "sarah.jones@bigcorp.com",
            "metadata": {"date": "2026-04-01T11:15:00Z", "account_id": "ACC-91205"},
            "ground_truth": {"department": "technical_support", "priority": "urgent"},
        },
        {
            "id": "easy-003",
            "subject": "Interested in upgrading to Enterprise plan",
            "body": (
                "Hi Sales Team,\n\nWe've been using Pro for 6 months and want to "
                "explore Enterprise for our expansion. We have 50 users growing to "
                "200 by Q3. Could you schedule a call?\n\nRegards, Mike Chen, CTO"
            ),
            "sender": "mike.chen@techflow.io",
            "metadata": {"date": "2026-04-01T14:00:00Z", "account_id": "ACC-44321"},
            "ground_truth": {"department": "sales", "priority": "medium"},
        },
    ]


def get_medium_tasks():
    return [
        {
            "id": "med-001",
            "subject": "Refund request for double charge",
            "body": (
                "Hi,\n\nI was charged twice for March — $49.99 on March 1st and "
                "again on March 3rd. Please refund the duplicate to my Visa ending "
                "in 4521.\n\nOrder IDs: ORD-88201 and ORD-88203.\n\nThanks, Amanda Lee"
            ),
            "sender": "amanda.lee@gmail.com",
            "metadata": {"date": "2026-04-02T09:00:00Z", "account_id": "ACC-55102"},
            "ground_truth": {"department": "billing", "priority": "high"},
        },
        {
            "id": "med-002",
            "subject": "Feature not working after update",
            "body": (
                "After update v3.2.1, the export-to-PDF feature is broken. Click "
                "'Export' → spinner for 30s → 'Export failed: timeout'. Worked fine "
                "before. macOS 14.3, Chrome 124. Team of 15 all affected.\n\nRaj Patel"
            ),
            "sender": "raj.patel@startup.co",
            "metadata": {"date": "2026-04-02T10:30:00Z", "account_id": "ACC-67890"},
            "ground_truth": {"department": "technical_support", "priority": "high"},
        },
        {
            "id": "med-003",
            "subject": "Question about remote work policy",
            "body": (
                "Hi HR,\n\nI'm new (started last Monday) and my offer says 'hybrid "
                "flexible' but my manager says 4 days in office. What's the official "
                "policy? Also haven't received my laptop yet.\n\nThanks, Lisa Wang"
            ),
            "sender": "lisa.wang@company-internal.com",
            "metadata": {"date": "2026-04-02T11:45:00Z", "employee_id": "EMP-2026-089"},
            "ground_truth": {"department": "human_resources", "priority": "medium"},
        },
        {
            "id": "med-004",
            "subject": "Partnership opportunity — AI integration",
            "body": (
                "I'm Head of Partnerships at DataVision AI. We built an AI analytics "
                "module that integrates via API. Several of your competitors use our "
                "solution. Available for a 30-min call this week?\n\nDr. Priya Sharma"
            ),
            "sender": "priya.sharma@datavision.ai",
            "metadata": {"date": "2026-04-02T13:00:00Z"},
            "ground_truth": {"department": "sales", "priority": "medium"},
        },
        {
            "id": "med-005",
            "subject": "How do I reset my password?",
            "body": (
                "Hi,\n\nI forgot my password and the 'Reset Password' link shows "
                "'Service Unavailable'. Can someone manually reset it? My email on "
                "file is this one.\n\nThanks, Tom Brooks"
            ),
            "sender": "tom.brooks@yahoo.com",
            "metadata": {"date": "2026-04-02T15:20:00Z", "account_id": "ACC-33210"},
            "ground_truth": {"department": "technical_support", "priority": "medium"},
        },
    ]


def get_hard_tasks():
    return [
        {
            "id": "hard-001",
            "subject": "URGENT: System completely down!!!",
            "body": (
                "Hey,\n\nJust wanted to let you know I can't find the option to "
                "change notification preferences. It's not really urgent but the "
                "subject got your attention didn't it? :)\n\nWhere do I turn off "
                "email notifications?\n\nCheers, Dave"
            ),
            "sender": "dave.wilson@funmail.com",
            "metadata": {"date": "2026-04-03T08:00:00Z", "account_id": "ACC-12345"},
            "ground_truth": {"department": "general_inquiry", "priority": "low"},
        },
        {
            "id": "hard-002",
            "subject": "Invoice question",
            "body": (
                "Hi,\n\nI'm trying to integrate your payment API but the /invoices "
                "endpoint returns a 404. Your REST API docs reference v1 but your "
                "changelog says v3. Can engineering update the docs? Our integration "
                "deadline is this Friday.\n\nCarlos Mendez, Senior Developer"
            ),
            "sender": "carlos.mendez@paytech.com",
            "metadata": {"date": "2026-04-03T09:15:00Z", "account_id": "ACC-99871"},
            "ground_truth": {"department": "technical_support", "priority": "high"},
        },
        {
            "id": "hard-003",
            "subject": "Re: Team outing next Friday",
            "body": (
                "Hi HR,\n\nThanks for organizing the outing! Quick question — my "
                "last paycheck was missing 32 overtime hours from March. I submitted "
                "them via the portal on March 28th. Can payroll check? The outing "
                "sounds fun, count me in!\n\nBest, Nina Rodriguez"
            ),
            "sender": "nina.rodriguez@company-internal.com",
            "metadata": {"date": "2026-04-03T10:00:00Z", "employee_id": "EMP-2025-412"},
            "ground_truth": {"department": "human_resources", "priority": "high"},
        },
        {
            "id": "hard-004",
            "subject": "Want to buy 500 licenses",
            "body": (
                "Hello,\n\nWe're interested in 500 Enterprise licenses. However, we "
                "need your SOC 2 Type II report and security questionnaire first. "
                "Also, are your data centers in the EU? GDPR compliance is mandatory.\n\n"
                "Helga Braun, Procurement Director, EuroMed Group"
            ),
            "sender": "helga.braun@euromed-group.eu",
            "metadata": {"date": "2026-04-03T11:30:00Z"},
            "ground_truth": {"department": "sales", "priority": "high"},
        },
        {
            "id": "hard-005",
            "subject": "Technical support needed",
            "body": (
                "Dear Support,\n\nI need to cancel my subscription immediately and "
                "get a prorated refund for the remaining 18 days. I already exported "
                "all my data.\n\nAccount: ACC-44556, Plan: Pro ($99.99/mo), "
                "Billing date: April 20th.\n\nMaria Gonzalez"
            ),
            "sender": "maria.gonzalez@outlook.com",
            "metadata": {"date": "2026-04-03T12:45:00Z", "account_id": "ACC-44556"},
            "ground_truth": {"department": "billing", "priority": "high"},
        },
        {
            "id": "hard-006",
            "subject": "Feedback on your product",
            "body": (
                "Hi,\n\nLove the product! Suggestions: 1) Dark mode 2) Mobile app "
                "crashes when uploading files > 10MB 3) Slack integration. The crash "
                "is a real problem — I use the app in the field daily.\n\nKevin Park"
            ),
            "sender": "kevin.park@fieldwork.co",
            "metadata": {"date": "2026-04-03T14:00:00Z", "account_id": "ACC-77654"},
            "ground_truth": {"department": "technical_support", "priority": "medium"},
        },
        {
            "id": "hard-007",
            "subject": "Hiring for VP of Engineering",
            "body": (
                "Hello,\n\nI'm a recruiter with a VP Engineering role at a Series C "
                "startup ($350K+ base). Couldn't find your careers page. Could you "
                "forward this to whoever handles recruitment?\n\nJake Morrison, TalentFirst"
            ),
            "sender": "jake.morrison@talentfirst.com",
            "metadata": {"date": "2026-04-03T15:15:00Z"},
            "ground_truth": {"department": "human_resources", "priority": "low"},
        },
        {
            "id": "hard-008",
            "subject": "Compliance audit — IMMEDIATE RESPONSE REQUIRED",
            "body": (
                "Dear Sir/Madam,\n\nThis is an automated notification from the "
                "International Data Compliance Authority. Respond to audit #AUD-2026-5543 "
                "within 48 hours or face penalties. Click: http://totally-legit-audit.com/"
                "verify\n\nCompliance Department"
            ),
            "sender": "audit@compliance-authority-intl.com",
            "metadata": {"date": "2026-04-03T16:30:00Z", "spam_score": 0.95},
            "ground_truth": {"department": "general_inquiry", "priority": "low"},
        },
    ]


TASK_REGISTRY = {"easy": get_easy_tasks, "medium": get_medium_tasks, "hard": get_hard_tasks}

def get_tasks(difficulty="easy"):
    if difficulty not in TASK_REGISTRY:
        raise ValueError(f"Unknown difficulty '{difficulty}'. Choose from: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[difficulty]()

print("✅ Tasks defined!\n")


# ══════════════════════════════════════════════════════════════
#  CELL 4: Define grader
# ══════════════════════════════════════════════════════════════

def grade_single_email(action_department, action_priority, ground_truth):
    reward = 0.0
    feedback_parts = []
    details = {
        "department_correct": False, "priority_correct": False,
        "valid_department": False, "valid_priority": False,
    }

    dept = action_department.strip().lower()
    prio = action_priority.strip().lower()
    true_dept = ground_truth["department"].strip().lower()
    true_prio = ground_truth["priority"].strip().lower()

    if dept in VALID_DEPARTMENTS:
        details["valid_department"] = True
        reward += 0.05
    else:
        reward -= 0.05
        feedback_parts.append(f"Invalid department '{dept}'.")

    if prio in VALID_PRIORITIES:
        details["valid_priority"] = True
        reward += 0.05
    else:
        reward -= 0.05
        feedback_parts.append(f"Invalid priority '{prio}'.")

    if dept == true_dept:
        details["department_correct"] = True
        reward += 0.6
        feedback_parts.append("✓ Department: Correct!")
    else:
        feedback_parts.append(f"✗ Department: Incorrect (chose '{dept}').")

    if prio == true_prio:
        details["priority_correct"] = True
        reward += 0.3
        feedback_parts.append("✓ Priority: Correct!")
    elif details["valid_priority"]:
        order = VALID_PRIORITIES
        if prio in order and true_prio in order:
            if abs(order.index(prio) - order.index(true_prio)) == 1:
                reward += 0.1
                feedback_parts.append(f"~ Priority: Close (chose '{prio}').")
            else:
                feedback_parts.append(f"✗ Priority: Incorrect (chose '{prio}').")
        else:
            feedback_parts.append(f"✗ Priority: Incorrect (chose '{prio}').")

    return reward, " | ".join(feedback_parts), details


def compute_final_score(correct_count, total_count, total_reward):
    if total_count == 0:
        return 0.0, "No emails processed."
    accuracy = correct_count / total_count
    avg_reward = total_reward / total_count
    final_score = (0.6 * accuracy) + (0.4 * avg_reward)
    summary = (
        f"Accuracy: {correct_count}/{total_count} ({accuracy:.0%}) | "
        f"Avg reward: {avg_reward:.3f} | Final score: {final_score:.3f}"
    )
    return final_score, summary


print("✅ Grader defined!\n")


# ══════════════════════════════════════════════════════════════
#  CELL 5: Define environment
# ══════════════════════════════════════════════════════════════

import uuid
import copy


class EmailTriageEnvironment:
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = EmailTriageState()
        self._emails = []
        self._total_reward = 0.0
        self._correct_count = 0
        self._current_index = 0

    def reset(self, seed=None, episode_id=None, difficulty="easy", **kwargs):
        self._emails = get_tasks(difficulty)
        self._current_index = 0
        self._total_reward = 0.0
        self._correct_count = 0

        ground_truths = [e["ground_truth"] for e in self._emails]
        self._state = EmailTriageState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_difficulty=difficulty,
            total_emails=len(self._emails),
            correct_count=0,
            current_email_index=0,
            ground_truth=copy.deepcopy(ground_truths),
        )

        first = self._emails[0]
        return EmailTriageObservation(
            done=False, reward=None,
            email_id=first["id"],
            email_subject=first["subject"],
            email_body=first["body"],
            email_sender=first["sender"],
            email_metadata=first.get("metadata", {}),
            emails_remaining=len(self._emails) - 1,
            feedback="Episode started. Classify this email by department and priority.",
            score_so_far=0.0,
        )

    def step(self, action, timeout_s=None, **kwargs):
        self._state.step_count += 1
        current_gt = self._state.ground_truth[self._current_index]

        step_reward, feedback, details = grade_single_email(
            action_department=action.department,
            action_priority=action.priority,
            ground_truth=current_gt,
        )

        self._total_reward += step_reward
        if details["department_correct"] and details["priority_correct"]:
            self._correct_count += 1
            self._state.correct_count = self._correct_count

        self._current_index += 1
        self._state.current_email_index = self._current_index

        is_done = self._current_index >= len(self._emails)

        if is_done:
            final_score, summary = compute_final_score(
                self._correct_count, len(self._emails), self._total_reward,
            )
            return EmailTriageObservation(
                done=True, reward=final_score,
                email_id="", email_subject="", email_body="",
                email_sender="", email_metadata={},
                emails_remaining=0,
                feedback=f"{feedback} || FINAL: {summary}",
                score_so_far=final_score,
            )
        else:
            next_email = self._emails[self._current_index]
            score_so_far = self._correct_count / self._current_index if self._current_index > 0 else 0.0
            return EmailTriageObservation(
                done=False, reward=step_reward,
                email_id=next_email["id"],
                email_subject=next_email["subject"],
                email_body=next_email["body"],
                email_sender=next_email["sender"],
                email_metadata=next_email.get("metadata", {}),
                emails_remaining=len(self._emails) - self._current_index - 1,
                feedback=feedback,
                score_so_far=score_so_far,
            )

    @property
    def state(self):
        return self._state


print("✅ Environment defined!\n")


# ══════════════════════════════════════════════════════════════
#  CELL 6: Start FastAPI server in background
# ══════════════════════════════════════════════════════════════

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import threading
import uvicorn
import time

app = FastAPI(title="Email Triage Environment", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

env = EmailTriageEnvironment()

class ResetRequest(BaseModel):
    difficulty: str = "easy"
    episode_id: Optional[str] = None

class StepRequest(BaseModel):
    department: str
    priority: str
    reasoning: str = ""

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/reset")
async def reset_endpoint(request: ResetRequest):
    try:
        obs = env.reset(difficulty=request.difficulty, episode_id=request.episode_id)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
async def step_endpoint(request: StepRequest):
    action = EmailTriageAction(
        department=request.department,
        priority=request.priority,
        reasoning=request.reasoning,
    )
    obs = env.step(action)
    return obs.model_dump()

@app.get("/state")
async def state_endpoint():
    return env.state.model_dump()

@app.get("/")
async def root():
    return {
        "name": "Email Triage Environment",
        "version": "1.0.0",
        "endpoints": {"health": "/health", "reset": "/reset", "step": "/step", "state": "/state", "docs": "/docs"},
    }


def start_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

thread = threading.Thread(target=start_server, daemon=True)
thread.start()
time.sleep(3)  # Wait for server to start

import requests as req
try:
    r = req.get("http://localhost:8000/health", timeout=5)
    print(f"✅ Server started! Health: {r.json()}")
except Exception as e:
    print(f"❌ Server failed to start: {e}")

print()


# ══════════════════════════════════════════════════════════════
#  CELL 7: Run rule-based agent (no LLM needed)
# ══════════════════════════════════════════════════════════════

import requests as req


def simple_rule_policy(obs_dict):
    """Simple keyword-based classification (no LLM needed)."""
    text = (obs_dict.get("email_subject", "") + " " +
            obs_dict.get("email_body", "")).lower()

    if any(w in text for w in ["invoice", "billing", "charge", "refund",
                                 "payment", "subscription", "cancel"]):
        dept = "billing"
    elif any(w in text for w in ["error", "bug", "crash", "broken", "500",
                                   "not working", "api", "login", "password"]):
        dept = "technical_support"
    elif any(w in text for w in ["upgrade", "enterprise", "pricing", "buy",
                                   "licenses", "partnership", "sales"]):
        dept = "sales"
    elif any(w in text for w in ["hr", "employee", "payroll", "hiring",
                                   "remote work", "policy", "recruiter"]):
        dept = "human_resources"
    else:
        dept = "general_inquiry"

    if any(w in text for w in ["urgent", "immediately", "blocking", "down", "asap"]):
        prio = "urgent"
    elif any(w in text for w in ["important", "soon", "deadline"]):
        prio = "high"
    elif any(w in text for w in ["question", "wondering", "curious"]):
        prio = "low"
    else:
        prio = "medium"

    return {"department": dept, "priority": prio}


print("═" * 60)
print("  RUNNING RULE-BASED AGENT (via HTTP API)")
print("═" * 60)

for difficulty in ["easy", "medium", "hard"]:
    print(f"\n--- {difficulty.upper()} ---")
    obs = req.post("http://localhost:8000/reset",
                    json={"difficulty": difficulty}).json()

    step = 0
    while not obs.get("done", False):
        step += 1
        decision = simple_rule_policy(obs)
        print(f"  Email {step}: '{obs['email_subject'][:40]}...' "
              f"→ {decision['department']}/{decision['priority']}")

        obs = req.post("http://localhost:8000/step", json=decision).json()
        print(f"    Reward: {obs.get('reward')} | {obs.get('feedback', '')[:60]}")

    print(f"  ✅ FINAL SCORE: {obs.get('score_so_far', 'N/A')}")

print("\n" + "═" * 60)
print("  ALL TESTS COMPLETE!")
print("═" * 60)


# ══════════════════════════════════════════════════════════════
#  CELL 8 (OPTIONAL): Run LLM-based agent
#  Uncomment and set your credentials to use an actual LLM
# ══════════════════════════════════════════════════════════════

"""
import os
os.environ["API_BASE_URL"] = "https://api-inference.huggingface.co/v1"
os.environ["MODEL_NAME"] = "Qwen/Qwen2.5-7B-Instruct"
os.environ["HF_TOKEN"] = "hf_xxxxxxxxxxxxx"  # PUT YOUR TOKEN HERE

# Then import and run inference.py:
# (You'd need to copy inference.py content here or !wget it)

SYSTEM_PROMPT = '''You are an expert email triage assistant. Classify each email.

Respond ONLY with JSON: {"department": "...", "priority": "...", "reasoning": "..."}

Departments: billing, technical_support, sales, human_resources, general_inquiry
Priorities: low, medium, high, urgent'''

import json
from openai import OpenAI

def call_llm(messages):
    client = OpenAI(
        base_url=os.environ['API_BASE_URL'],
        api_key=os.environ['HF_TOKEN'] or "dummy-token"
    )
    response = client.chat.completions.create(
        model=os.environ["MODEL_NAME"],
        messages=messages,
        temperature=0.1,
        max_tokens=256
    )
    return response.choices[0].message.content

for difficulty in ["easy", "medium", "hard"]:
    print(f"\\n--- LLM Agent on {difficulty.upper()} ---")
    obs = req.post("http://localhost:8000/reset", json={"difficulty": difficulty}).json()

    step = 0
    while not obs.get("done", False):
        step += 1
        email_text = f"From: {obs['email_sender']}\\nSubject: {obs['email_subject']}\\nBody:\\n{obs['email_body']}"
        response = call_llm([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": email_text}])

        try:
            text = response.strip()
            if text.startswith("```"): text = text.split("```")[1].replace("json", "", 1).strip()
            classification = json.loads(text)
        except:
            classification = {"department": "general_inquiry", "priority": "medium", "reasoning": "parse error"}

        print(f"  Email {step}: → {classification.get('department')}/{classification.get('priority')}")
        obs = req.post("http://localhost:8000/step", json=classification).json()
        print(f"    Reward: {obs.get('reward')} | {obs.get('feedback', '')[:60]}")

    print(f"  FINAL SCORE: {obs.get('score_so_far')}")
"""

print("\n💡 Uncomment Cell 8 and add your HF_TOKEN to run the LLM-based agent!")
