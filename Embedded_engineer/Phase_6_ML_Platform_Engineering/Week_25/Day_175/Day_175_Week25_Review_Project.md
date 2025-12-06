# Day 175: Week 25 Review & Project - The Enterprise Brain
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 25: Large Language Model Infrastructure

---

> **🎯 Focus Area:** We have the pieces: vLLM for speed, LoRA for customization, Milvus for memory, and DSPy for control. Now we assemble **ChatCorp**, a private RAG system that runs entirely on-premise (or in your VPC).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Integrate** a Vector DB, an LLM Inference Server, and a Frontend into a cohesive application.
2.  **Implement** a Streaming Response pipeline (Token-by-token UI updates).
3.  **Enforce** Role-Based Access Control (RBAC) on the Document Retrieval layer.
4.  **Deploy** the entire stack using Docker Compose with GPU support.

---

## 📚 Week 25 Recap

| Day | Topic | The "Ah-ha" Moment |
|-----|-------|---------------------|
| 169 | LLM Hardware | "I need 2x A100s for 70B FP16, but only 1x A100 for INT4." |
| 170 | Distribution | "Tensor Parallelism splits the matrix multiplication itself." |
| 171 | vLLM | "PagedAttention stops the OOMs." |
| 172 | LoRA | "I don't need to retrain the whole model, just 0.1% of it." |
| 173 | RAG | "HNSW is the only way to search 1B vectors fast." |
| 174 | DSPy | "Compiling the prompt is better than writing it." |

---

## 🏗️ Final Project: "ChatCorp"

### Architecture
1.  **Frontend:** Streamlit.
2.  **Backend:** FastAPI (Orchestraor).
3.  **Brain:** vLLM (Serving Llama-3-8B-Instruct).
4.  **Memory:** Milvus (Vector DB).

### Step 1: The vLLM Service (Docker)

#### 📁 `project/docker-compose.yml`
```yaml
version: '3.8'
services:
  vllm:
    image: vllm/vllm-openai:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    command: >
      --model meta-llama/Meta-Llama-3-8B-Instruct
      --dtype half 
      --max-model-len 4096
    ports:
      - "8000:8000"

  milvus:
    image: milvusdb/milvus:v2.3.0
    ports: ["19530:19530"]
    # ... depends on etcd/minio ...

  # The App
  chat-ui:
    build: ./app
    ports: ["8501:8501"]
    environment:
      - VLLM_URL=http://vllm:8000/v1
      - MILVUS_HOST=milvus
```

### Step 2: The Retrieval Module (Milvus)

#### 📁 `project/app/retriever.py`
```python
from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer

# Connect
connections.connect(host="milvus", port="19530")
collection = Collection("corp_docs")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def get_context(query, k=3):
    query_vec = embedder.encode([query])
    res = collection.search(
        query_vec, "embeddings", 
        param={"metric_type": "L2", "params": {"ef": 10}}, 
        limit=k, output_fields=["text"]
    )
    return "\n\n".join([hit.entity.get("text") for hit in res[0]])
```

### Step 3: The DSPy Logic

#### 📁 `project/app/brain.py`
```python
import dspy
import os

# Connect DSPy to remote vLLM
lm = dspy.OpenAI(
    api_base=os.getenv("VLLM_URL"),
    api_key="EMPTY",
    model="meta-llama/Meta-Llama-3-8B-Instruct"
)
dspy.settings.configure(lm=lm)

class RAGSignature(dspy.Signature):
    """Answer questions based on the retrieved context."""
    context = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.OutputField()

class RAGModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(RAGSignature)
    
    def forward(self, context, question):
        return self.prog(context=context, question=question)

bot = RAGModule()

def ask(context, question):
    pred = bot(context=context, question=question)
    return pred.answer
```

### Step 4: The UI (Streamlit)

#### 📁 `project/app/main.py`
```python
import streamlit as st
from retriever import get_context
from brain import ask

st.title("ChatCorp: Enterprise RAG")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Ask about company policy..."):
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Retrieval
    with st.spinner("Searching Knowledge Base..."):
        context = get_context(prompt)
    
    # 3. Generation (DSPy)
    with st.chat_message("assistant"):
        # Note: DSPy streaming is possible but complex, using blocking for project simplicity
        # Real prod would use asyncio generator
        response = ask(context, prompt)
        st.markdown(response)
        st.caption(f"Sources: {context[:200]}...")
    
    st.session_state.messages.append({"role": "assistant", "content": response})
```

---

## 🔬 Lab Exercise: "The Stress Test"

### Task
Simulate 100 concurrent users.
1.  **Tool:** `locust`.
2.  **Scenario:** Users constantly asking "What is the vacation policy?".
3.  **Observation:**
    *   **Milvus:** Latency < 10ms (HNSW is fast).
    *   **vLLM:** Throughput increases (Continuous Batching kicks in). Latency per user stays decent.
    *   **GPU VRAM:** Stable (PagedAttention handles the KV cache).
4.  **Failure Mode:** If users send 10k token documents, vLLM might reject request (`max_model_len` exceeded).

---

## 📝 Success Criteria
1.  **Privacy:** No data leaves the container network (checked via Firewall rules).
2.  **Accuracy:** RAG answers strictly based on retrieved context (verified via Ragas in CI).
3.  **Performance:** Time to First Token (TTFT) < 200ms.

---

**Week 25 Complete** ✅
**Phase 6E In Progress**

*Next Phase: Multi-Cluster Federation - Scaling to the World.*
