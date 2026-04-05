# 📧 Email Triage Environment

> An OpenEnv-compatible RL environment for training and evaluating LLM agents on real-world email classification.

## What It Does

The agent receives incoming support emails one at a time and must classify each by:

| Decision | Options |
|----------|---------|
| **Department** | `billing`, `technical_support`, `sales`, `human_resources`, `general_inquiry` |
| **Priority** | `low`, `medium`, `high`, `urgent` |

The environment provides **deterministic, rule-based grading** — no LLM judgment is involved in scoring.

## Why Email Triage?

- ✅ **Real-world task** — every company needs email routing
- ✅ **Structured actions** — finite, typed choices (not open-ended text)
- ✅ **Deterministic grading** — exact string matching, fully reproducible
- ✅ **Partial credit** — dense rewards for partial correctness
- ✅ **Natural difficulty scaling** — 3 levels with progressively harder emails

## Difficulty Levels

| Level | Emails | Description |
|-------|--------|-------------|
| 🟢 Easy | 3 | Obvious keywords, clear categories |
| 🟡 Medium | 5 | Some ambiguity, more departments used |
| 🔴 Hard | 8 | Misleading subjects, spam, cross-department, edge cases |

## Quick Start

### 1. Install dependencies

```bash
pip install fastapi uvicorn pydantic requests
```

### 2. Test locally (no server needed)

```bash
python run_local.py
```

This runs a simple rule-based agent against all 3 difficulty levels.

### 3. Start the server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Interact via API

```bash
# Health check
curl http://localhost:8000/health

# Start an episode
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "easy"}'

# Classify an email
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"department": "billing", "priority": "high"}'

# Check state
curl http://localhost:8000/state
```

### 5. Run the LLM agent

```bash
export API_BASE_URL=https://api-inference.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
export HF_TOKEN=hf_xxxxxxxxxxxxx
export ENV_URL=http://localhost:8000
export DIFFICULTY=easy

python inference.py
```

## API Reference

### `POST /reset`

Start a new episode.

**Request:**
```json
{"difficulty": "easy"}
```

**Response:**
```json
{
  "done": false,
  "reward": null,
  "email_id": "easy-001",
  "email_subject": "Billing question about my last invoice",
  "email_body": "Hi, I received invoice #INV-2026-1042...",
  "email_sender": "john.smith@example.com",
  "email_metadata": {"date": "2026-04-01T10:30:00Z", "account_id": "ACC-78432"},
  "available_departments": ["billing", "technical_support", "sales", "human_resources", "general_inquiry"],
  "available_priorities": ["low", "medium", "high", "urgent"],
  "emails_remaining": 2,
  "feedback": "Episode started. Classify this email by department and priority.",
  "score_so_far": 0.0
}
```

### `POST /step`

Classify the current email.

**Request:**
```json
{
  "department": "billing",
  "priority": "high",
  "reasoning": "Invoice-related question about overcharge"
}
```

**Response:** Same structure as `/reset`, with `reward` and `feedback` populated.

### `GET /state`

Get episode metadata.

### `GET /health`

Returns `{"status": "healthy"}`.

## Reward Design

| Component | Reward |
|-----------|--------|
| Correct department | +0.6 |
| Correct priority | +0.3 |
| Valid inputs | +0.1 (0.05 each) |
| Adjacent priority (one level off) | +0.1 partial credit |
| Invalid department/priority | -0.05 each |
| **Final score** | 60% accuracy + 40% avg reward |

## Project Structure

```
email-triage-env/
├── README.md              ← You are here
├── openenv.yaml           ← OpenEnv manifest
├── pyproject.toml         ← Package metadata
├── requirements.txt       ← Dependencies
├── Dockerfile             ← Container for HF Spaces
├── inference.py           ← LLM agent baseline (must be at root)
├── run_local.py           ← Local test script (no server needed)
├── models.py              ← Action, Observation, State types
├── tasks.py               ← Email datasets (easy/medium/hard)
├── grader.py              ← Deterministic grading logic
├── client.py              ← HTTP + WebSocket client
└── server/
    ├── __init__.py
    ├── environment.py     ← Core environment logic
    └── app.py             ← FastAPI server
```

## Deployment to Hugging Face Spaces

### Option A: `openenv push` (if openenv-core installed)

```bash
pip install openenv-core
openenv push --repo-id your-username/email-triage-env
```

### Option B: Manual upload

1. Create a new Space on [huggingface.co/new-space](https://huggingface.co/new-space)
   - Select **Docker** as the SDK
   - Select **CPU Basic (Free)**
2. Clone and push:
   ```bash
   git clone https://huggingface.co/spaces/your-username/email-triage-env
   cp -r email-triage-env/* email-triage-env/.
   cd email-triage-env
   git add .
   git commit -m "Initial deployment"
   git push
   ```

Your environment is now live at:
- **API:** `https://your-username-email-triage-env.hf.space`
- **Docs:** `https://your-username-email-triage-env.hf.space/docs`
- **Health:** `https://your-username-email-triage-env.hf.space/health`

## Running on Google Colab

See the [Colab Setup Guide](colab_guide.md) or copy the code from `run_colab.py` into a new Colab notebook.

## License

MIT
