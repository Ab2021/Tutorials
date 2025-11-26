# Day 2: Languages for Backend in 2025

## 1. The Polyglot Backend Engineer

In 2025, the days of being just a "Java Developer" or a "Python Developer" are fading. Modern backend architectures are often **polyglot**, meaning they use different languages for different services based on the specific requirements of that service.

### 1.1 Why Polyglot?
- **Performance**: A high-frequency trading service needs C++ or Rust.
- **AI Integration**: A RAG pipeline needs Python because of the ecosystem (LangChain, PyTorch).
- **Real-time I/O**: A chat server might benefit from Node.js's event loop or Go's goroutines.
- **Enterprise Legacy**: Core banking ledgers often run on Java/Spring Boot.

### 1.2 The "Big Three" for Modern Backends
For this course, and for most startups/scaleups in 2025, the primary contenders are:
1.  **Python**: The king of AI and rapid prototyping.
2.  **Node.js (TypeScript)**: The king of I/O bound services and shared frontend/backend logic.
3.  **Go (Golang)**: The king of high-concurrency, low-latency microservices.

---

## 2. Language Deep Dive

### 2.1 Python (FastAPI)
Python is the *lingua franca* of AI. If your backend touches LLMs, embeddings, or data science, Python is almost mandatory.

*   **Pros**:
    *   **Ecosystem**: Unrivaled libraries for AI (HuggingFace, OpenAI), Data (Pandas), and Math (NumPy).
    *   **Speed of Development**: Very concise syntax. "Executable pseudocode."
    *   **Modern Async**: FastAPI + `asyncio` allows Python to handle thousands of concurrent connections, solving the old "Python is slow" bottleneck for I/O.
*   **Cons**:
    *   **Raw Performance**: Still slower than Go/Rust for CPU-bound tasks (due to the GIL, though Python 3.13+ is making strides here).
    *   **Runtime Errors**: Dynamic typing can lead to bugs at runtime (mitigated by Pydantic & MyPy).
*   **Best For**: AI Wrappers, Data Processing, CRUD APIs, Prototyping.

### 2.2 Node.js (TypeScript)
Node.js revolutionized the backend by bringing the Event Loop to the server.

*   **Pros**:
    *   **Single Language**: Share types (interfaces) between React/Next.js frontend and the backend.
    *   **I/O Performance**: Non-blocking I/O is perfect for streaming, websockets, and gateways.
    *   **Ecosystem**: NPM is the largest package registry in the world.
*   **Cons**:
    *   **CPU Bound Tasks**: Heavy computation blocks the event loop, killing the server (unless using Worker Threads).
    *   **Callback Hell**: (Mostly solved by async/await, but legacy code exists).
*   **Best For**: BFF (Backend for Frontend), Real-time Chat, API Gateways, Serverless Functions (fast cold starts).

### 2.3 Go (Golang)
Go was designed by Google specifically for networked services.

*   **Pros**:
    *   **Concurrency**: Goroutines are lightweight threads managed by the Go runtime. You can spawn millions of them.
    *   **Performance**: Compiled to machine code. Near C++ speed with garbage collection.
    *   **Simplicity**: Small standard library, strict formatting (`gofmt`), no complex inheritance.
    *   **Static Binary**: Compiles to a single binary file. Zero dependencies. Docker images are tiny (scratch containers).
*   **Cons**:
    *   **Boilerplate**: "Err != nil" checks everywhere. Verbose.
    *   **Generics**: Newer addition, ecosystem still adapting.
*   **Best For**: High-throughput Microservices, Infrastructure Tools (Docker/K8s are written in Go), gRPC services.

### 2.4 Honorable Mentions
*   **Rust**: The performance king. Memory safety without garbage collection.
    *   *Use when*: You need absolute max performance or safety (e.g., database engines, crypto). High learning curve.
*   **Java / C# (.NET)**: The enterprise standard.
    *   *Use when*: You are in a large org with existing JVM infrastructure. Spring Boot is powerful but heavy.

---

## 3. Code Comparison: A Simple HTTP Endpoint

Let's implement `GET /health` and `POST /users` in the Big Three.

### 3.1 Python (FastAPI)
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    username: str
    email: str

@app.get("/health")
async def health_check():
    return {"status": "ok", "language": "python"}

@app.post("/users")
async def create_user(user: User):
    # Logic to save user...
    return {"id": 1, "username": user.username}
```
*Note: Pydantic handles validation automatically.*

### 3.2 Node.js (Express + TypeScript)
```typescript
import express, { Request, Response } from 'express';
const app = express();
app.use(express.json());

interface User {
  username: string;
  email: string;
}

app.get('/health', (req: Request, res: Response) => {
  res.json({ status: 'ok', language: 'node' });
});

app.post('/users', (req: Request, res: Response) => {
  const user = req.body as User;
  // Manual validation needed or use Zod/Joi
  if (!user.username) return res.status(400).send("Missing username");
  
  res.json({ id: 1, username: user.username });
});

app.listen(3000);
```

### 3.3 Go (Standard Lib + Chi)
```go
package main

import (
    "encoding/json"
    "net/http"
    "github.com/go-chi/chi/v5"
)

type User struct {
    Username string `json:"username"`
    Email    string `json:"email"`
}

func main() {
    r := chi.NewRouter()

    r.Get("/health", func(w http.ResponseWriter, r *http.Request) {
        json.NewEncoder(w).Encode(map[string]string{
            "status": "ok", 
            "language": "go",
        })
    })

    r.Post("/users", func(w http.ResponseWriter, r *http.Request) {
        var user User
        if err := json.NewDecoder(r.Body).Decode(&user); err != nil {
            http.Error(w, err.Error(), 400)
            return
        }
        // Logic...
        json.NewEncoder(w).Encode(map[string]interface{}{
            "id": 1, 
            "username": user.Username,
        })
    })

    http.ListenAndServe(":3000", r)
}
```

---

## 4. Choosing the Right Tool (Decision Framework)

When starting a new service, ask these questions:

1.  **Is it AI-heavy?**
    *   Yes -> **Python**. Don't fight the ecosystem.
2.  **Is it a high-traffic, core infrastructure service?**
    *   Yes -> **Go**. The concurrency model will save you money on cloud bills.
3.  **Does the team already know TypeScript/React?**
    *   Yes -> **Node.js**. Developer velocity beats theoretical performance.
4.  **Do you need extreme safety/performance?**
    *   Yes -> **Rust**.

### 4.1 The "Modern Stack" Recommendation
For this course, we will adopt a **Hybrid Stack**:
*   **Core Business Logic / AI Services**: Python (FastAPI).
*   **High Performance / Gateway / Tools**: Go.
*   **Scripts / CLI**: Python or Go.

We will skip Node.js for the backend core to focus on the two extremes (Ease of use vs. Performance), but the concepts apply equally to Node.

---

## 5. Summary

Today we explored the toolbelt of a 2025 Backend Engineer. We learned that:
*   **Python** is indispensable for AI.
*   **Go** is the standard for cloud-native infrastructure.
*   **Node.js** bridges the gap between frontend and backend.

**Tomorrow (Day 3)**: We will dive into **HTTP, REST & Web Basics**. We'll stop using libraries for a moment and look at raw HTTP requests, headers, status codes, and what actually happens when you type a URL into a browser.
