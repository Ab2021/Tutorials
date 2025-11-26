# Lab: Day 55 - Final Integration

## Goal
Run the full "DocuMind" stack.

## Step 1: The Test Script (`test_e2e.py`)

```python
import requests
import time

BASE_URL = "http://localhost:8000"

def test_flow():
    # 1. Signup/Login (Mocked for simplicity, assuming we get a token)
    token = "mock_jwt"
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Create Doc
    print("Creating Doc...")
    res = requests.post(f"{BASE_URL}/docs/", json={"title": "Test", "content": "AI is cool."}, headers=headers)
    assert res.status_code == 200
    doc_id = res.json()["id"]
    print(f"Doc Created: {doc_id}")

    # 3. Simulate Edit (via API for simplicity, usually WS)
    # In a real test, we'd use a WS client here.
    
    # 4. Wait for Indexing
    print("Waiting for AI Indexing...")
    time.sleep(5)

    # 5. Chat
    print("Asking AI...")
    res = requests.post(f"{BASE_URL}/chat", json={"doc_id": doc_id, "question": "Is AI cool?"})
    assert res.status_code == 200
    answer = res.json()["answer"]
    print(f"AI Answer: {answer}")
    
    if "Yes" in answer or "cool" in answer:
        print("✅ Test Passed!")
    else:
        print("❌ Test Failed (AI didn't understand)")

if __name__ == "__main__":
    try:
        test_flow()
    except Exception as e:
        print(f"❌ Error: {e}")
```

## Step 2: Run It
1.  Ensure all services are running (`docker-compose up`).
2.  `python test_e2e.py`.

## Challenge
Add **Load Testing**.
Use `locust` to simulate 100 users creating docs and chatting simultaneously.
Observe how the system behaves. Does Kafka lag? Does Qdrant slow down?
