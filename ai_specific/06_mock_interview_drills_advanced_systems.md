# THE MASTER AI MOCK INTERVIEW & RAPID-FIRE DRILL VAULT (PART 2: ADVANCED SYSTEMS & SYSTEM DESIGN)
> This file is a continuation of the mock drill vault, expanding the total drill content to ensure absolutely every edge case, system design scenario, and MLOps constraint is covered.

---

## PART 7: ADVANCED SYSTEM DESIGN (GENAAI PLATFORM)

### Q24: Multi-Tenant Architecture
**Interviewer:** "You built an Enterprise GenAI platform serving multiple clients (tenants). If one client asks for their data to be purged, or if we have strict data isolation requirements, how do you handle multi-tenancy at the database and application layer?"
**The Trap:** Suggesting you filter data in the API using `WHERE client_id = X`. This is highly insecure because a single bug in a single endpoint can leak one client's data to another. Or suggesting you spin up a completely new PostgreSQL instance for every client, which is operationally unscalable and incredibly expensive.
**The Perfect Answer (SOAR):**
> **Situation:** Enterprise clients require strict data isolation. We cannot risk leaking Tenant A's chat history or RAG documents to Tenant B. However, spinning up a separate database cluster for every tenant is an operational nightmare.
> **Action:** I implemented multi-tenancy using Row-Level Security (RLS) directly inside a single PostgreSQL database.
> Every row in every table is tagged with a `tenant_id`. When an API request comes in via FastAPI, the authentication middleware validates the JWT, extracts the tenant ID, and sets a session variable in the database connection. 
> PostgreSQL's native RLS policies automatically filter all `SELECT`, `UPDATE`, and `DELETE` queries at the database engine level.
> **Result:** This guarantees zero data leakage even if a developer forgets to add a `WHERE` clause in the Python code, while keeping our infrastructure footprint small and allowing us to onboard new tenants in seconds.

### Q25: Real-Time Streaming and Websockets
**Interviewer:** "LLM generation is slow. If a user asks a complex question, they might wait 15 seconds for a response, leading to a terrible UX. How did you architect the backend to solve this latency issue?"
**The Trap:** Saying "I used a faster model" or "I optimized the prompt." These don't solve the fundamental token-by-token generation bottleneck of LLMs.
**The Perfect Answer:**
> **Situation:** A pure REST API architecture forces the client to wait until the entire LLM response is fully generated before receiving the HTTP response payload. This results in unacceptable 10-15 second wait times.
> **Action:** I replaced standard REST endpoints with a WebSocket architecture for all conversational AI surfaces. 
> When the user submits a query, the FastAPI backend opens a persistent WebSocket connection. As the LLM (like GPT-4 or Llama 3) streams tokens out one by one, the backend instantly pushes those tokens down the socket to the frontend UI.
> **Result:** This drops the perceived "Time-To-First-Token" (TTFT) latency from 15 seconds to under 500 milliseconds. The user sees the AI typing immediately, completely transforming the user experience while the model is still processing.

### Q26: Managing Stateful Conversations in Kubernetes
**Interviewer:** "When using WebSockets and multi-turn conversations, how do you manage state? If a Kubernetes pod dies mid-conversation, does the user lose their entire chat history?"
**The Trap:** Storing conversation history in a Python list or dictionary in the FastAPI application memory. If the pod restarts or scales out, the memory is wiped.
**The Perfect Answer:**
> **Situation:** Stateless APIs are easy, but conversational AI requires state (memory of previous turns). We deploy on Kubernetes with auto-scaling, meaning API pods are constantly spinning up and dying. Storing state in-memory means catastrophic context loss upon pod death.
> **Action:** I abstracted all conversation memory out of the application pods and into an external, highly available Redis cluster. 
> Every user session is assigned a unique UUID. When a message is sent, the FastAPI pod retrieves the chat history from Redis, appends the new query, sends it to the LLM, and writes the new response back to Redis. I also implemented TTL (Time-To-Live) on the Redis keys to automatically purge inactive sessions and save memory.
> **Result:** Our API pods remained entirely stateless. A pod could crash mid-conversation, and the Kubernetes load balancer would instantly route the next user message to a new pod, which would seamlessly pull the history from Redis and continue the conversation without the user ever noticing.

---

## PART 8: ADVANCED MLOPS & OBSERVABILITY

### Q27: LLM Evaluation and "Drift"
**Interviewer:** "In standard ML, we monitor for data drift. But how do you monitor for drift in a Generative AI application where the outputs are unstructured text?"
**The Perfect Answer:**
> **Situation:** Standard metrics like precision and recall don't apply to generative text. If OpenAI updates the GPT-4 weights behind the API, the model's behavior might "drift" and become lazier or less helpful without throwing any explicit errors.
> **Action:** I implemented an LLM-as-a-Judge observability pipeline using LangFuse. 
> We log 100% of production traces (prompt + response + latency + cost). We run a nightly cron job that samples 5% of production traces and evaluates them using a strictly prompted evaluation model. The evaluator scores the responses on Helpfulness (1-5), Toxicity (Binary), and Faithfulness to the retrieved RAG context.
> **Result:** We built a dashboard tracking these aggregate scores over time. If the average Helpfulness score drops below 4.0 over a 48-hour window, an alert is fired to the ML engineering team, indicating behavioral drift.

### Q28: Cost Optimization at Scale
**Interviewer:** "LLM APIs are expensive. As our user base scales, our OpenAI bill is going to explode. What concrete architectural strategies did you use to optimize costs?"
**The Trap:** Saying "We switched to an open-source model." (Sometimes OSS models cost MORE to host yourself than using an API, depending on utilization).
**The Perfect Answer:**
> **Situation:** Token costs scale linearly with usage. A naive RAG application that crams 10 huge documents into the context window for every query will bankrupt a project quickly.
> **Action:** I implemented a three-tiered cost optimization strategy:
> 1. **Semantic Caching:** I deployed Redis with vector similarity search. If a user asks a question that has a 95% semantic similarity to a question asked 10 minutes ago, we return the cached response instantly. Cost = $0.
> 2. **Context Window Pruning:** In our RAG pipeline, I used a Cross-Encoder to aggressively re-rank and prune the retrieved chunks. Instead of sending 10 chunks to the LLM, we send only the Top 3 most relevant chunks, reducing prompt token costs by 70%.
> 3. **Model Routing:** I implemented a router that sends simple tasks (like intent classification or spelling correction) to cheap, fast models (like GPT-3.5 or Llama 3 8B), reserving GPT-4 exclusively for complex reasoning tasks.
> **Result:** We slashed our token expenditure by over 60% without degrading the perceived intelligence of the platform.

---

## PART 9: EXTREME RAPID-FIRE GAUNTLET (SYSTEM DESIGN)
> *Answer in exactly 1 sentence.*

**16. What is the difference between an API Gateway and a Load Balancer?**
> A Load Balancer distributes L4/L7 network traffic across multiple servers to prevent overload, whereas an API Gateway sits in front of the load balancer to handle cross-cutting application concerns like Authentication, Rate Limiting, and Request Routing.

**17. Why shouldn't you store API keys as Environment Variables in Docker containers?**
> Environment variables can be accidentally leaked in crash dumps, exposed in the Kubernetes UI, or logged by monitoring tools; you should use a secrets manager like HashiCorp Vault to mount them securely at runtime or inject them directly into memory.

**18. What is the 'Thundering Herd' problem in caching?**
> It occurs when a highly popular cached item expires, and simultaneously, hundreds of concurrent requests hit the database to regenerate the item, potentially crashing the database before the cache can be repopulated.

**19. In Kafka, what is a Consumer Group?**
> A consumer group is a set of consumers that cooperate to consume data from a topic; Kafka ensures that each partition in the topic is read by exactly one consumer within the group, allowing you to scale out processing horizontally.

**20. What is a Dead Letter Queue (DLQ)?**
> A DLQ is a specialized queue where messages that fail to process successfully (after multiple retries) are sent, ensuring that the main processing pipeline isn't blocked by a poison-pill message, and allowing engineers to inspect the failed messages later.

**21. Why is continuous batching better than static batching for LLM inference?**
> Static batching waits for the longest sequence in the batch to finish before accepting new requests (wasting GPU cycles), whereas continuous batching (via vLLM) instantly evicts finished requests and inserts new ones at the token level, maximizing GPU utilization.

**22. What is the difference between stateless and stateful authentication?**
> Stateful auth stores session data in a central database or Redis (requiring a DB lookup on every request), whereas stateless auth (like JWT) cryptographically signs the user data into the token itself, allowing the API to verify it instantly without a database hit.

---

## PART 10: ARCHITECTURAL DECISION DEFENSE

### Q29: "Why didn't you just use standard text search instead of vector search for RAG?"
**The Perfect Defense:**
> "Standard text search (BM25 or TF-IDF) relies on exact keyword matching. If a document says 'myocardial infarction' and the user searches for 'heart attack', standard search returns zero results. Vector search maps both phrases to the same coordinate in the embedding space based on their semantic meaning. However, for highly specific terms (like a serial number), vector search struggles, which is exactly why I implemented Hybrid RAG (BM25 + Vector Search) combined with Reciprocal Rank Fusion."

### Q30: "Why use PyTorch over TensorFlow?"
**The Perfect Defense:**
> "PyTorch uses a dynamic computation graph (define-by-run), making it incredibly intuitive for debugging complex, branching architectures like the multi-head EfficientNet I built for Jamnagar. TensorFlow's static graph (though improved in TF2) was historically clunkier for rapid R&D. Furthermore, the entire NLP and Generative AI ecosystem (HuggingFace, vLLM, DeepSpeed) is fundamentally PyTorch-first, meaning we get instant access to state-of-the-art models and optimizations without translation friction."

---
*(End of Vault Part 2.)*
