"""Quick test of the running FastAPI server endpoints."""
import requests
import json

BASE = "http://localhost:8000"

# Test health
r = requests.get(f"{BASE}/health")
print("Health:", r.json())

# Test reset (easy)
r = requests.post(f"{BASE}/reset", json={"difficulty": "easy"})
obs = r.json()
print(f"Reset OK: subject='{obs['email_subject'][:40]}...'")
print(f"  emails_remaining={obs['emails_remaining']}")

# Test step 1
r = requests.post(f"{BASE}/step", json={"department": "billing", "priority": "high"})
obs = r.json()
print(f"Step 1: reward={obs['reward']}, done={obs['done']}")
print(f"  feedback: {obs['feedback'][:80]}")

# Test step 2
r = requests.post(f"{BASE}/step", json={"department": "technical_support", "priority": "urgent"})
obs = r.json()
print(f"Step 2: reward={obs['reward']}, done={obs['done']}")
print(f"  feedback: {obs['feedback'][:80]}")

# Test step 3 (last email in easy)
r = requests.post(f"{BASE}/step", json={"department": "sales", "priority": "medium"})
obs = r.json()
print(f"Step 3: done={obs['done']}, final_score={obs['score_so_far']}")
print(f"  feedback: {obs['feedback']}")

# Test state
r = requests.get(f"{BASE}/state")
state = r.json()
print(f"State: correct={state['correct_count']}/{state['total_emails']}, steps={state['step_count']}")

# Test root info
r = requests.get(f"{BASE}/")
info = r.json()
print(f"Root: {info['name']} v{info['version']}")

print()
print("ALL API TESTS PASSED!")
