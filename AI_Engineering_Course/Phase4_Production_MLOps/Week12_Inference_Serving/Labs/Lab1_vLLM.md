# Lab 1: vLLM Production Server

## Objective
Deploy a high-performance inference server using **vLLM**.
vLLM uses **PagedAttention** to achieve 24x higher throughput than Hugging Face.

## 1. Setup

```bash
pip install vllm
```

## 2. The Server (`server.py`)

We will use the OpenAI-compatible API server built into vLLM.

```bash
# Run in terminal
python -m vllm.entrypoints.openai.api_server \
    --model facebook/opt-125m \
    --port 8000
```

## 3. The Client (Load Test)

We will use `locust` to stress test the server.

```python
# locustfile.py
from locust import HttpUser, task, between

class LLMUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def generate(self):
        payload = {
            "model": "facebook/opt-125m",
            "prompt": "San Francisco is a",
            "max_tokens": 100,
            "temperature": 0.7
        }
        self.client.post("/v1/completions", json=payload)
```

## 4. Running the Lab
1.  Start vLLM server.
2.  Run `locust -f locustfile.py`.
3.  Open `localhost:8089`.
4.  Start swarming (10 users).
5.  Observe the **RPS (Requests Per Second)** and **Latency**.

## 5. Submission
Submit the Locust report screenshot.
