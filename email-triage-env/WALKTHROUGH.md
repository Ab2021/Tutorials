# 📧 Email Triage Environment — Complete Walkthrough & Submission Guide

> **A beginner-friendly, step-by-step guide to run, test, deploy, and submit your hackathon entry.**
> **⏰ Safe internal deadline: April 5, 2026, 11:59 PM IST**

---

## Table of Contents

1. [What Was Built](#what-was-built)
2. [Project Structure](#project-structure)
3. [Option A: Run Locally (Windows)](#option-a-run-locally-windows)
4. [Option B: Run on Google Colab](#option-b-run-on-google-colab)
5. [Deploy to Hugging Face Spaces](#deploy-to-hugging-face-spaces)
6. [Run the LLM Agent](#run-the-llm-agent-inferencepy)
7. [**Hackathon Submission — Step by Step**](#hackathon-submission--step-by-step)
8. [Troubleshooting](#troubleshooting)
9. [Key Files Explained](#key-files-explained)

---

## What Was Built

A complete **Email Triage RL Environment** that:

- Receives emails one at a time and asks an agent to classify them by **department** and **priority**
- Provides **deterministic, rule-based grading** (no LLM judgment in scoring)
- Has **3 difficulty levels** (easy/medium/hard) with progressively harder emails
- Gives **partial credit** (dense rewards at every step, not just 0 or 1 at the end)
- Runs as a **FastAPI microservice** following the OpenEnv pattern
- Works on **Google Colab**, **locally**, and **Hugging Face Spaces**

### Test Results (verified ✅)

| Difficulty | Emails | Rule-based Bot Score | Description |
|-----------|--------|---------------------|-------------|
| 🟢 Easy | 3 | **0.760** | Obvious keywords work well |
| 🟡 Medium | 5 | **0.424** | Some ambiguity lowers accuracy |
| 🔴 Hard | 8 | **0.310** | Misleading subjects trick the keyword bot |

> An LLM agent should significantly outperform the rule-based bot, especially on Hard level. That's the point — better reasoning = better score.

### Scoring Breakdown

| Action | Reward |
|--------|--------|
| Correct department | +0.60 |
| Correct priority | +0.30 |
| Valid input (per field) | +0.05 |
| Adjacent priority (one level off) | +0.10 partial credit |
| Invalid department/priority | −0.05 |
| **Final episode score** | 60% accuracy + 40% average step reward |

---

## Project Structure

```
email-triage-env/
├── 📄 README.md              ← Documentation for judges (required)
├── 📄 openenv.yaml           ← OpenEnv manifest (required)
├── 📄 pyproject.toml         ← Python package metadata
├── 📄 requirements.txt       ← All dependencies
├── 🐳 Dockerfile             ← Container for Hugging Face Spaces (required)
├── 🤖 inference.py           ← LLM agent baseline — MUST be at repo root (required)
├── 🧪 run_local.py           ← Quick test without server needed
├── 🧪 run_colab.py           ← Self-contained Colab version of everything
├── 🧪 test_api.py            ← API endpoint tests
├── 📦 models.py              ← Action / Observation / State type definitions
├── 📦 tasks.py               ← Email datasets (easy / medium / hard)
├── 📦 grader.py              ← Deterministic scoring logic
├── 📦 client.py              ← HTTP client (Colab) + WebSocket client (openenv)
└── 📂 server/
    ├── __init__.py
    ├── environment.py         ← Core environment logic (reset / step / state)
    └── app.py                 ← FastAPI server wiring
```

---

## Option A: Run Locally (Windows)

### Step 1: Open a Terminal

Press `Win + R`, type `powershell`, press Enter.

### Step 2: Navigate to the project folder

```powershell
cd U:\Githubs\codex\email-triage-env
```

### Step 3: Install dependencies

```powershell
pip install fastapi uvicorn pydantic requests openai
```

### Step 4: Test the environment directly (no server needed)

```powershell
python run_local.py
```

**Expected output:**
```
╔══════════════════════════════════════════════════════════╗
║    EMAIL TRIAGE ENVIRONMENT — LOCAL TEST                ║
║    Testing all 3 difficulty levels with rule-based bot  ║
╚══════════════════════════════════════════════════════════╝

============================================================
  DIFFICULTY: EASY
============================================================
--- Email 1 ---
  ID:      easy-001
  Subject: Billing question about my last invoice
  ...
  → Classified: dept=billing, priority=high
  → Reward:   1.0
  → Feedback: ✓ Department: Correct! | ✓ Priority: Correct!
...
  SUMMARY
============================================================
  easy    : ██████████████████████░░░░░░░░ 0.760
  medium  : ████████████░░░░░░░░░░░░░░░░░░ 0.424
  hard    : █████████░░░░░░░░░░░░░░░░░░░░░ 0.310
============================================================
```

### Step 5: Start the API server

```powershell
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

Leave this terminal running. Open a **new** terminal for the next steps.

### Step 6: Test all API endpoints

In the new terminal:

```powershell
cd U:\Githubs\codex\email-triage-env
python test_api.py
```

**Expected output:**
```
Health: {'status': 'healthy'}
Reset OK: subject='Billing question about my last invoice...'
  emails_remaining=2
Step 1: reward=1.0, done=False
  feedback: ✓ Department: Correct! | ✓ Priority: Correct!
Step 2: reward=1.0, done=False
Step 3: done=True, final_score=1.0
State: correct=3/3, steps=3
Root: Email Triage Environment v1.0.0

ALL API TESTS PASSED!
```

### Step 7: View the interactive API docs

Open your browser and go to: **http://localhost:8000/docs**

You will see a Swagger UI where you can click "Try it out" on each endpoint and test them directly in the browser.

---

## Option B: Run on Google Colab

Colab is a free cloud environment — no setup on your machine needed.

### Step 1: Open Colab

Go to [https://colab.research.google.com](https://colab.research.google.com) and click **"New notebook"**.

### Step 2: Copy the Colab script

1. Open `U:\Githubs\codex\email-triage-env\run_colab.py` in any text editor (Notepad, VS Code, etc.)
2. Press `Ctrl+A` to select all text
3. Press `Ctrl+C` to copy
4. Click inside the first cell of your Colab notebook
5. Press `Ctrl+V` to paste

### Step 3: Run the cell

Click the **▶ Play** button on the left side of the cell, or press `Ctrl+Enter`.

This single cell does everything:
- Installs all dependencies
- Defines all models, tasks, grader, and environment
- Starts a FastAPI server in the background
- Runs a rule-based agent via HTTP API against all 3 difficulty levels

**Expected output:**
```
✅ Dependencies installed!
✅ Models defined!
✅ Tasks defined!
✅ Grader defined!
✅ Environment defined!
✅ Server started! Health: {'status': 'healthy'}

══════════════════════════════════════════════════════════
  RUNNING RULE-BASED AGENT (via HTTP API)
══════════════════════════════════════════════════════════
--- easy ---
  Email 1: 'Billing question about my last invoice...' → billing/high
    Reward: 1.0 | ✓ Department: Correct! | ✓ Priority: Correct!
...
  ✅ FINAL SCORE: 0.76

--- medium ---
...
--- hard ---
...

══════════════════════════════════════════════════════════
  ALL TESTS COMPLETE!
══════════════════════════════════════════════════════════

💡 Uncomment Cell 8 and add your HF_TOKEN to run the LLM-based agent!
```

### Step 4 (Optional): Run with an LLM in Colab

Scroll to the bottom of the script and find the large commented-out block starting with `"""`. To activate it:

1. Delete the opening `"""` line
2. Delete the closing `"""` line
3. Replace `hf_xxxxxxxxxxxxx` with your actual HF token

**How to get a Hugging Face token (free):**
1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **"New token"**
3. Give it any name (e.g., `hackathon`)
4. Select **"Read"** permission
5. Click **"Generate a token"**
6. Copy the token — it starts with `hf_`

---

## Deploy to Hugging Face Spaces

Your environment must be deployed to HF Spaces for the hackathon. There are two methods.

---

### Method 1: Using `openenv push` (Simplest)

```powershell
# Install openenv CLI
pip install openenv-core

# Login to Hugging Face (paste your HF token when prompted)
huggingface-cli login

# Deploy from the project directory
cd U:\Githubs\codex\email-triage-env
openenv push --repo-id YOUR_HF_USERNAME/email-triage-env
```

Replace `YOUR_HF_USERNAME` with your actual HF username.

Your environment will be live at:
```
https://YOUR_HF_USERNAME-email-triage-env.hf.space
```

---

### Method 2: Manual Git Push (More Reliable)

Use this if `openenv push` doesn't work.

#### Sub-step 1: Create a new Space on Hugging Face

1. Go to [https://huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in the form:
   - **Owner:** Select your username
   - **Space name:** `email-triage-env` (no spaces, use hyphens)
   - **License:** MIT
   - **SDK:** Click **"Docker"** ← very important
   - **Docker template:** Select **"Blank"**
   - **Hardware:** Select **"CPU Basic (Free)"**
   - **Visibility:** Public ← required for hackathon judges
3. Click **"Create Space"**

You'll be taken to the Space page. It will say "Building" — that's expected.

#### Sub-step 2: Enable Git LFS

Git LFS handles large files. Run this once:

```powershell
git lfs install
```

#### Sub-step 3: Clone your empty Space

```powershell
# Replace YOUR_HF_USERNAME with your actual username
git clone https://huggingface.co/spaces/YOUR_HF_USERNAME/email-triage-env
```

You'll be asked for your HF credentials:
- **Username:** your HF username
- **Password:** your HF token (the `hf_xxx` one, NOT your account password)

#### Sub-step 4: Copy project files into the cloned Space

```powershell
# Navigate into the cloned space
cd email-triage-env

# Copy all project files (Windows PowerShell)
Copy-Item -Path "U:\Githubs\codex\email-triage-env\*" -Destination "." -Recurse -Force
```

> **Important:** Make sure the `Dockerfile` is at the root (not inside a subfolder).

#### Sub-step 5: Commit and push

```powershell
git add .
git commit -m "Initial deployment: Email Triage OpenEnv Environment"
git push
```

#### Sub-step 6: Wait for build

Go to your Space page on Hugging Face. You'll see a build log. Wait 2-5 minutes. When it says **"Running"**, your environment is live.

#### Sub-step 7: Verify deployment

Replace `YOUR_HF_USERNAME` in these URLs:

| What | URL |
|------|-----|
| Health check | `https://YOUR_HF_USERNAME-email-triage-env.hf.space/health` |
| Interactive docs | `https://YOUR_HF_USERNAME-email-triage-env.hf.space/docs` |
| Environment info | `https://YOUR_HF_USERNAME-email-triage-env.hf.space/` |

The health check should return:
```json
{"status": "healthy"}
```

---

## Run the LLM Agent (inference.py)

### Step 1: Make sure the server is running

Either your local server OR your HF Space URL can be used.

**Option A — local server:**
```powershell
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

**Option B — use your HF Space:**
No server startup needed. Just use the HF Space URL in the next step.

### Step 2: Set the required environment variables

**Windows PowerShell (local server):**
```powershell
$env:API_BASE_URL = "https://api-inference.huggingface.co/v1"
$env:MODEL_NAME   = "Qwen/Qwen2.5-7B-Instruct"
$env:HF_TOKEN     = "hf_xxxxxxxxxxxxx"
$env:ENV_URL      = "http://localhost:8000"
$env:DIFFICULTY   = "easy"
```

**Windows PowerShell (HF Space):**
```powershell
$env:API_BASE_URL = "https://api-inference.huggingface.co/v1"
$env:MODEL_NAME   = "Qwen/Qwen2.5-7B-Instruct"
$env:HF_TOKEN     = "hf_xxxxxxxxxxxxx"
$env:ENV_URL      = "https://YOUR_HF_USERNAME-email-triage-env.hf.space"
$env:DIFFICULTY   = "easy"
```

### Step 3: Run the agent

```powershell
cd U:\Githubs\codex\email-triage-env
python inference.py
```

**Expected output format (required by hackathon):**
```
[START]
  Model: Qwen/Qwen2.5-7B-Instruct
  Difficulty: easy

[STEP] Step 1
  Email: Billing question about my last invoice
  → Dept: billing
  → Priority: high
  → Reasoning: This email is about an overcharged invoice...
  Reward: 1.0
  Feedback: ✓ Department: Correct! | ✓ Priority: Correct!

[STEP] Step 2
  Email: Cannot login to my account — error 500
  → Dept: technical_support
  → Priority: urgent
  Reward: 1.0
  Feedback: ✓ Department: Correct! | ✓ Priority: Correct!

[STEP] Step 3
  Email: Interested in upgrading to Enterprise plan
  → Dept: sales
  → Priority: medium
  Reward: 1.0
  Feedback: ✓ Department: Correct! | ✓ Priority: Correct!

[END]
  Total steps: 3
  Final score: 1.0
```

> The `[START]`, `[STEP]`, and `[END]` markers are **required** by the hackathon spec. Our `inference.py` already outputs them correctly.

---

## Hackathon Submission — Step by Step

This is the most important section. Follow every step in order.

---

### Phase 1: Pre-Submission Verification

Before you push anything, verify everything works.

**Checklist — run these commands and confirm they pass:**

```powershell
cd U:\Githubs\codex\email-triage-env

# ✅ Test 1: Local environment test
python run_local.py
# Expected: Shows scores for easy/medium/hard. No errors.

# ✅ Test 2: Start the server (keep this running)
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000

# ✅ Test 3: In a new terminal, test all API endpoints
python test_api.py
# Expected: "ALL API TESTS PASSED!"

# ✅ Test 4: Test inference.py (set env vars first)
$env:ENV_URL = "http://localhost:8000"
$env:HF_TOKEN = "hf_xxxxxxxxxxxxx"
python inference.py
# Expected: [START] ... [STEP] ... [END] output
```

**Submission blockers — fix these before submitting:**

| If you see this | Fix |
|-----------------|-----|
| `ModuleNotFoundError` | `pip install fastapi uvicorn pydantic requests` |
| `ConnectionRefusedError` | Start the server first |
| `inference.py` crashes | Check your HF_TOKEN is set |
| `docker build` fails | Check `requirements.txt` syntax |

---

### Phase 2: Push Code to GitHub

Every submission needs a public GitHub repository.

#### Step 1: Create a GitHub repository

1. Go to [https://github.com/new](https://github.com/new)
2. Fill in:
   - **Repository name:** `email-triage-env`
   - **Description:** `Email Triage RL Environment — OpenEnv hackathon submission`
   - **Visibility:** ✅ **Public** (judges must be able to see it)
   - **Do NOT** check "Add a README file" (we already have one)
3. Click **"Create repository"**
4. Copy the repository URL (e.g., `https://github.com/YOUR_GITHUB_USERNAME/email-triage-env.git`)

#### Step 2: Initialize Git in your project

```powershell
cd U:\Githubs\codex\email-triage-env

git init
git add .
git commit -m "Initial commit: Email Triage OpenEnv Environment for Scaler Hackathon"
```

#### Step 3: Connect to GitHub and push

```powershell
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/email-triage-env.git
git branch -M main
git push -u origin main
```

You'll be asked for your GitHub credentials:
- **Username:** your GitHub username
- **Password:** your GitHub Personal Access Token (NOT your login password)

> **How to create a GitHub Personal Access Token:**
> 1. Go to [https://github.com/settings/tokens](https://github.com/settings/tokens)
> 2. Click **"Generate new token (classic)"**
> 3. Select scopes: ✅ `repo`
> 4. Click **"Generate token"**
> 5. Copy it immediately (you won't see it again)

#### Step 4: Verify GitHub push

Go to `https://github.com/YOUR_GITHUB_USERNAME/email-triage-env` and confirm all files are there:
- `inference.py` is at the root ← very important
- `Dockerfile` is at the root ← very important
- `README.md` is visible
- `server/` folder exists

---

### Phase 3: Deploy to Hugging Face Spaces

Follow the **"Method 2: Manual Git Push"** section above, or use `openenv push`. Once deployed:

#### Verify your Space is live

Open this URL in a browser:
```
https://YOUR_HF_USERNAME-email-triage-env.hf.space/health
```

It must return:
```json
{"status": "healthy"}
```

If it shows an error or "Building", wait a few minutes and try again.

#### Copy your Space URL

Save this for the next step:
```
https://YOUR_HF_USERNAME-email-triage-env.hf.space
```

---

### Phase 4: Test the Deployed Environment

Run `inference.py` against your *live* HF Space to confirm the full pipeline works end-to-end:

```powershell
$env:API_BASE_URL = "https://api-inference.huggingface.co/v1"
$env:MODEL_NAME   = "Qwen/Qwen2.5-7B-Instruct"
$env:HF_TOKEN     = "hf_xxxxxxxxxxxxx"
$env:ENV_URL      = "https://YOUR_HF_USERNAME-email-triage-env.hf.space"
$env:DIFFICULTY   = "easy"

python inference.py
```

Confirm that:
- `[START]` appears at the top
- `[STEP]` appears for each email
- `[END]` appears at the bottom with a final score
- No errors or crashes

---

### Phase 5: Fill Out the Submission Form

The hackathon requires you to submit a form with the following information. Gather everything before filling it in.

#### What you'll need to provide:

| Field | What to put | Where to find it |
|-------|------------|-----------------|
| **GitHub Repository URL** | `https://github.com/YOUR_GITHUB_USERNAME/email-triage-env` | GitHub after push |
| **HF Space URL** | `https://YOUR_HF_USERNAME-email-triage-env.hf.space` | HF Spaces after deploy |
| **Environment name** | `Email Triage Environment` | — |
| **Task description** | "Classify incoming support emails by department and priority across 3 difficulty levels with deterministic grading" | — |
| **Number of tasks** | 3 (easy, medium, hard) | — |
| **Reward type** | Dense (partial credit at each step) | — |
| **Action space** | Department (5 options) + Priority (4 options) | — |
| **Model used in inference.py** | `Qwen/Qwen2.5-7B-Instruct` | — |

#### Final submission URL

Submit at the official hackathon form link provided by Scaler. (Check the PDF or hackathon portal for the exact URL.)

---

### Phase 6: Final Verification Before Deadline

Do this 30 minutes before the deadline to catch any last-minute issues.

```powershell
# 1. Confirm GitHub repo is public and has all files
# Open: https://github.com/YOUR_GITHUB_USERNAME/email-triage-env

# 2. Confirm HF Space is running
# Open: https://YOUR_HF_USERNAME-email-triage-env.hf.space/health

# 3. Run one last full test against live HF Space
$env:ENV_URL = "https://YOUR_HF_USERNAME-email-triage-env.hf.space"
python inference.py

# 4. Confirm inference.py is at root of GitHub repo
# You should see it at: github.com/YOUR_GITHUB_USERNAME/email-triage-env/blob/main/inference.py
```

---

### Complete Submission Checklist

Copy this and check off each item:

```
ENVIRONMENT
  [ ] run_local.py runs without errors (all 3 difficulties)
  [ ] Server starts: uvicorn server.app:app --host 0.0.0.0 --port 8000
  [ ] test_api.py shows "ALL API TESTS PASSED!"
  [ ] /health endpoint returns {"status": "healthy"}
  [ ] /reset endpoint returns a valid email observation
  [ ] /step endpoint returns reward, feedback, and next email
  [ ] /state endpoint returns episode metadata

INFERENCE BASELINE
  [ ] inference.py is at the ROOT of the repository (not in a subfolder)
  [ ] inference.py reads API_BASE_URL, MODEL_NAME, HF_TOKEN from env vars
  [ ] inference.py outputs [START] at the beginning
  [ ] inference.py outputs [STEP] for each email
  [ ] inference.py outputs [END] with final score
  [ ] inference.py runs successfully with a real LLM

DEPLOYMENT
  [ ] Dockerfile is at the ROOT of the repository
  [ ] Hugging Face Space is created with Docker SDK
  [ ] HF Space is set to PUBLIC visibility
  [ ] HF Space shows "Running" (not "Building" or "Error")
  [ ] https://YOUR_HF_USERNAME-email-triage-env.hf.space/health returns healthy
  [ ] https://YOUR_HF_USERNAME-email-triage-env.hf.space/docs loads (Swagger UI)

GITHUB
  [ ] Repository is PUBLIC
  [ ] inference.py is at root: github.com/.../blob/main/inference.py
  [ ] Dockerfile is at root: github.com/.../blob/main/Dockerfile
  [ ] README.md clearly explains the environment
  [ ] openenv.yaml is present

SUBMISSION FORM
  [ ] GitHub URL submitted
  [ ] HF Space URL submitted
  [ ] Submitted before deadline (April 5, 2026, 11:59 PM IST — safe cutoff)
```

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `ModuleNotFoundError: No module named 'fastapi'` | Dependencies not installed | `pip install fastapi uvicorn pydantic requests` |
| `ConnectionRefusedError` when running `test_api.py` | Server not running | Start with `uvicorn server.app:app --port 8000` first |
| `openenv-core` not found warning | Normal behaviour | The server works without it — ignore the warning |
| `inference.py` says "HF_TOKEN not set" | Env var missing | PowerShell: `$env:HF_TOKEN = "hf_xxx"` |
| `inference.py` crashes with `401 Unauthorized` | Wrong HF token | Generate a new token at huggingface.co/settings/tokens |
| HF Space stuck on "Building" | Docker build error | Check the build logs on the Space page for error details |
| HF Space shows error after building | Port mismatch | Make sure Dockerfile uses port 7860 (already set correctly) |
| Docker build fails locally | Missing file | Verify `requirements.txt` exists and has correct content |
| `git push` asks for password repeatedly | Wrong credentials | Use a GitHub Personal Access Token as the password |
| GitHub push rejected | Wrong branch name | Run `git branch -M main` before pushing |
| LLM returns invalid JSON | Model formatting issue | The `parse_llm_response` fallback in `inference.py` handles this automatically |
| HF Inference API returns 503 | Model loading | Model cold-starts can take 20-30s. The retry logic in `inference.py` handles this |

---

## Key Files Explained (for Novices)

| File | Plain-English Purpose | Should you edit it? |
|------|----------------------|---------------------|
| `models.py` | Defines the "shape" of actions (what the agent sends) and observations (what it receives) | Only if adding new fields |
| `tasks.py` | Contains all 16 emails across 3 difficulty levels | To add/modify email examples |
| `grader.py` | Scores the agent's answer at each step — fully rule-based | To change how scoring works |
| `server/environment.py` | The "brain" — processes each action and returns rewards | Core logic changes |
| `server/app.py` | Turns the environment into an HTTP API with endpoints | Rarely needs editing |
| `client.py` | Lets Python code connect to the environment easily | Only if changing the client interface |
| `inference.py` | The LLM agent that plays the game — submit this as-is or improve the prompt | To try different LLMs or prompts |
| `run_local.py` | Runs all 3 difficulty levels locally without a server | Just run it to test |
| `run_colab.py` | Contains ALL code in one file — paste into Colab and run | Just paste and run |
| `Dockerfile` | Instructions for building the Docker container for HF Spaces | Don't touch unless you know Docker |
| `requirements.txt` | List of Python packages the server needs | Only if adding new packages |
| `openenv.yaml` | Describes the environment for the OpenEnv registry | Update after env changes |
| `README.md` | Documentation that judges read | Update with your name/details before submitting |

---

*Generated for the Scaler School of Technology OpenEnv Meta Hackathon — April 2026*
