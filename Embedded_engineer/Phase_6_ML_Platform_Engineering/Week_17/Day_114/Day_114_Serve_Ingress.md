# Day 114: The Front Door: Ingress and Routing
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 17: Ray Serve - Model Serving

---

> **🎯 Focus Area:** A raw `__call__` method is clumsy for complex APIs. **Ray Serve** integrates natively with **FastAPI**, allowing you to define routes, validation, and documentation (Swagger UI).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Wrap** a FastAPI application with `@serve.ingress`.
2.  **Implement** the "Facade Pattern" where one Ingress deployment delegates to multiple Model deployments.
3.  **Configure** `route_prefix` to structure the API URL space.
4.  **Visualize** the API schema using the auto-generated Swagger UI.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `pip install fastapi uvicorn`.

---

## 📖 Theoretical Foundation

### 1. Ingress vs Deployment
*   **Deployment:** The worker doing the ML.
*   **Ingress:** The web server handling JSON parsing and routing.
*   **Pattern:** One Ingress, Many Deployments. The Ingress holds `ServeHandle` references to the Deployments.

### 2. URL Routing
By default, Deployment `MyClass` is at `/MyClass`.
You can change it: `@serve.deployment(route_prefix="/api/inference")`.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: The Facade

#### 📁 `src/02_ingress.py`
```python
import ray
from ray import serve
from fastapi import FastAPI

app = FastAPI()

# 1. Backend Models
@serve.deployment
class TextModel:
    def __call__(self, text: str):
        return f"Processed text: {text}"

@serve.deployment
class ImageModel:
    def __call__(self, data: bytes):
        return f"Processed {len(data)} bytes of image"

# 2. Ingress (The API Gateway)
@serve.deployment
@serve.ingress(app)
class APIGateway:
    def __init__(self, text_handle, image_handle):
        self.text_handle = text_handle
        self.image_handle = image_handle

    @app.get("/text")
    async def handle_text(self, q: str):
        # Async call to backend
        # Using await allows the Gateway to handle other requests while waiting
        result = await self.text_handle.remote(q)
        return {"result": result}

    @app.post("/image")
    async def handle_image(self, data: bytes):
        result = await self.image_handle.remote(data)
        return {"result": result}

# 3. Wiring
text = TextModel.bind()
image = ImageModel.bind()
# Bind Gateway to backends
gateway = APIGateway.bind(text, image)

# 4. Run (Dev Mode)
if __name__ == "__main__":
    serve.run(gateway)
    # Now visit http://localhost:8000/docs
```

### 👨‍💻 Verification

1.  Open `http://localhost:8000/docs`.
2.  You see Swagger UI.
3.  Try `/text`.
4.  **Flow:** Browser -> Proxy -> APIGateway Replica -> TextModel Replica.

---

## 🔬 Lab Exercise: "Slow Backend"

### Task
Observe Async benefits.
1.  Add `time.sleep(1)` to `TextModel`.
2.  Hit `/text` 10 times rapidly (using `ab` or script).
3.  If `APIGateway` methods were synchronous (`def` instead of `async def`) and called `ray.get`, the Gateway would block.
4.  Because we use `await handle.remote()`, the Gateway stays free. Ray schedules the 10 text tasks on available TextModel replicas.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Validation:** FastAPI provides Pydantic validation for free.
2.  **Decoupling:** You can scale `TextModel` to 10 replicas and `APIGateway` to 2 replicas independently.
3.  **Composition:** This pattern allows you to update the `TextModel` code without changing the API contract in `APIGateway`.

### API Summary
```python
@serve.ingress(app)
```

---

**Day 114 Complete** ✅

*Next: Day 115 - Model Composition - Pipelines and DAGs.*
