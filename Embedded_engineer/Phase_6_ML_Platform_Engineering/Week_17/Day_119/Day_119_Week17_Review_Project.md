# Day 119: Week 17 Review & Project - Scalable RAG Service
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 17: Ray Serve - Model Serving

---

> **🎯 Focus Area:** We combine everything: Deployment Graphs, GPU handling, Autoscaling, and FastAPIs to build a robust **Retrieval Augmented Generation (RAG)** pipeline.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Architect** a multi-model pipeline (Embedder, VectorDB, LLM).
2.  **Deploy** specialized hardware actors (CPU for retrieval, GPU for generation).
3.  **Implement** batching for the Embedder to improve throughput.
4.  **Configure** autoscaling to handle burst traffic.

---

## 📚 Week 17 Recap

| Day | Topic | The "Ah-ha" Moment |
|-----|-------|---------------------|
| 113 | Serve Architecture | "I can call Python classes like HTTP endpoints." |
| 114 | Ingress | "FastAPI integration gives me Pydantic validation." |
| 115 | Composition | "I can wire models together in python code." |
| 116 | Autoscaling | "The service scales up when the queue fills." |
| 117 | Batching | "Latency went up slightly, but throughput went up 8x." |
| 118 | Rollouts | "I can switch 10% traffic to V2 instantly." |

---

## 🏗️ Final Project: "RayRAG"

### Use Case
User asks: "How do I fix a flat tire?"
Pipe:
1.  **Embedder:** Convert query to vector.
2.  **Retriever:** Find relevant docs in FAISS index.
3.  **Generator:** Feed docs + query to LLM to write answer.

### Step 1: Component Definition

#### 📁 `project/rag_app.py`
```python
import ray
from ray import serve
import asyncio
from typing import List

# Placeholder libraries
# import torch ...

@serve.deployment(
    autoscaling_config={"min_replicas": 1, "max_replicas": 5},
    ray_actor_options={"num_cpus": 1}
)
class Embedder:
    def __init__(self):
        print("Loading SentenceTransformer...")
        # self.model = SentenceTransformer(...)
    
    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.1)
    async def vectorize(self, texts: List[str]):
        # Mock vector logic
        return [[0.1, 0.2, 0.3] for _ in texts] # List of vectors

    async def __call__(self, text: str):
        # We hook __call__ to support direct calls or piped calls
        # But for batching, we usually call the batched method directly via handle
        return await self.vectorize(text)

@serve.deployment(ray_actor_options={"num_cpus": 2})
class Retriever:
    def __init__(self):
        print("Loading FAISS Index...")
        self.docs = {0: "Use a jack.", 1: "Call AAA."}
    
    def search(self, vector):
        # Mock Search
        return [self.docs[0], self.docs[1]]

@serve.deployment(ray_actor_options={"num_gpus": 0.5}) # Share GPU
class LLM:
    def __init__(self):
        print("Loading Llama-2...")
    
    def generate(self, prompt: str):
        return f"GPT Answer based on: {prompt}"

@serve.deployment
@serve.ingress(app)
class Ingress:
    def __init__(self, embedder, retriever, llm):
        self.embedder = embedder
        self.retriever = retriever
        self.llm = llm

    @app.post("/chat")
    async def chat(self, query: str):
        # 1. Embed (Batched)
        # Note: handle.vectorize.remote(query) works for single or list? 
        # Ray Batching expects single items in expected calls.
        vector_ref = await self.embedder.vectorize.remote(query)
        
        # 2. Retrieve (CPU)
        docs_ref = await self.retriever.search.remote(vector_ref)
        
        # 3. Generate (GPU)
        # We need to construct prompt. Can we pass refs to LLM?
        # Yes, Ray allows passing ObjectRefs! The receiving actor waits for them.
        # But string formatting requires the actual value.
        # So we await docs here.
        docs = ray.get(docs_ref) 
        prompt = f"Context: {docs}. Question: {query}"
        
        response_ref = await self.llm.generate.remote(prompt)
        return {"answer": response_ref}

# Wiring
embedder = Embedder.bind()
retriever = Retriever.bind()
llm = LLM.bind()
app = Ingress.bind(embedder, retriever, llm)
```

### Step 2: Resource Analysis
*   **Embedder:** Scalable. CPU based. Batched.
*   **Retriever:** Stateful (Index). Harder to scale (might need sharding). Here fixed at 1.
*   **LLM:** Scalable. GPU based. Expensive.

### Step 3: Deployment
1.  `serve run project.rag_app:app`.
2.  Send 100 concurrent requests.
3.  **Embedder** should scale to 5 replicas (CPU cheap).
4.  **LLM** scales if you have GPUs.

---

## 🔬 Lab Exercise: "Composition Overhead"

### Task
Measure the cost of hopping between actors.
1.  Time the full request.
2.  Time the individual inference steps (add logging).
3.  **Observation:** Ray Serve overhead is ~1-2ms per hop on local machine (shared memory). On cluster (network), it depends on CNI, but usually <5ms.
4.  If your inference is 100ms (LLM), overhead is negligible <5%.

---

## 📝 Success Criteria
1.  **Functionality:** `/chat` returns a sensible mock answer.
2.  **Scalability:** Locust load test triggers Embedder scaling.
3.  **Efficiency:** Batching in Embedder works (print batch size in logs).

---

**Week 17 Complete** ✅

*Next Phase: Phase 6C Finish - Week 18 - Processing Big Data with Ray.*
