# 🎯 Interview Prep: Solutions Architect ML/AI @ Huge
### Candidate: Abhishek Bhardwaj | Role: Solutions Architect ML/AI | Company: Huge (Bengaluru/New York)

---

> [!IMPORTANT]
> This is a **US-based panel interview**. Expect multiple rounds covering: Technical Architecture, Agentic AI/GenAI, System Design, Client Communication, Leadership/Behavioral. Use American business English - be direct, confident, and quantify everything.

---

## 📚 Document Index

| # | Document | Topics Covered | Priority |
|---|----------|---------------|----------|
| 01 | [Agentic AI Deep Dive](./01_agentic_ai_deep_dive.md) | ReAct, LangGraph, Context Management, MCP, A2A, AgentOps | 🔥 Critical |
| 02 | [RAG & Vector Infrastructure](./02_rag_vector_infrastructure.md) | RAG variants, FAISS, Vector DBs, Semantic Caching, GraphRAG | 🔥 Critical |
| 03 | [MLOps, GCP Vertex AI & Architecture](./03_mlops_gcp_vertex_architecture.md) | Vertex AI Pipelines, Model Serving, Drift Detection, IaC | 🔥 Critical |
| 04 | [Project Deep Dives & Behavioral](./04_project_deep_dives_behavioral.md) | Chubb projects, Axtria MMM/Attribution, CVS Health, STAR stories | ⭐ High |
| 05 | [Marketing AI & Huge Domain Alignment](./05_marketing_ai_huge_alignment.md) | MMM, MTA, HDBSCAN, GenAI for Marketing, Huge clients | ⭐ High |

---

## 🗺️ Your Experience → JD Alignment Map

| Your Experience | JD Requirement | Alignment Strength |
|-----------------|---------------|-------------------|
| LangGraph Agentic AI @ Chubb | Agentic workflows, MCP, A2A | 🟢 Strong (direct) |
| RAG Fraud Detection @ Chubb | RAG, Vector Infrastructure | 🟢 Strong (direct) |
| BERT Fine-Tuning (LoRA/PEFT) | Fine Tuning | 🟢 Strong (direct) |
| MLflow, Kubeflow @ Chubb/Axtria | MLOps, Vertex Pipelines | 🟡 Good (transfer needed) |
| GCP (Vertex, BigQuery, Dataproc) | GCP, BigQuery, Vertex | 🟢 Strong (direct) |
| AWS SageMaker @ Chubb | AWS | 🟢 Strong (direct) |
| MMM + Omnichannel Attribution | Marketing domain, client work | 🟢 Strong (differentiator) |
| FAISS/Chroma @ Chubb | Vectors, Embeddings | 🟢 Strong (direct) |
| Neo4j @ Axtria | Graph DB, knowledge graphs | 🟡 Good |
| PySpark, Databricks @ Axtria | Big Data, ETL | 🟢 Strong (direct) |
| Python (9+ years) | Python | 🟢 Strong |
| Team leadership (4-5 members) | Mentoring, cross-team | 🟡 Good |
| HDBSCAN *(missing from resume)* | HDBscan (in JD) | 🔴 **Must prepare** |
| Terraform *(missing from resume)* | IaC, Terraform | 🔴 **Must prepare** |

---

## ⚡ Quick Recall: Your 10 Most Impressive Talking Points

1. **"60% reduction in analytics turnaround time"** — Agentic Data Scientist, Chubb
2. **"18% improvement in NER/classification over zero-shot baselines"** — BERT LoRA fine-tuning
3. **"70% reduction in processing time"** — CLV prediction, GCP Dataproc
4. **"10% revenue increase through optimized spend allocation"** — MMM, Axtria
5. **"AUC from 0.82 to 0.89"** — Patient Readmission, clinical NLP
6. **"75% accuracy on 20K plan sponsors vs 40K company names"** — NLP Entity Matching
7. **"~15% model accuracy improvement"** — Bayesian optimization, MMM
8. **"3 consecutive quarters of 5/5 SLA ratings"** — EXL/CVS Health offshore team
9. **"Q1 2025 STAR Award + 2 Q3 2025 awards"** — Chubb recognition
10. **"Multi-objective genetic algorithm for budget allocation"** — Unique technical differentiator

---

## 🎯 The 5 Questions That Will Define This Interview

> These are the most likely opening/core questions in a US technical panel. Nail these.

### Q1: "Walk me through your most technically complex project"
**Answer with**: Agentic Data Scientist at Chubb (LangGraph + multi-tool + RAG)
**Key elements**: Architecture, business problem, LangGraph StateGraph, context management, production deployment, metrics

### Q2: "How do you design a scalable AI system for an enterprise client?"
**Answer with**: Your fraud detection system as the example, then generalize to a framework
**Key elements**: Requirements → Architecture → Data pipeline → Model serving → Monitoring → Feedback loop

### Q3: "What's your experience with agentic AI? What are the production challenges?"
**Answer with**: LangGraph system + AgentOps (latency, context, observability, guardrails)
**Key elements**: MCP, A2A, context management, observability, cost optimization

### Q4: "How does your background align with what Huge does?"
**Answer with**: MMM + Attribution + GenAI + Marketing AI
**Key elements**: Axtria pharma marketing work → transferable to consumer brands (Nike, McDonald's), GenAI for content

### Q5: "Design [X] system from scratch"
**Answer with**: Structured approach — Requirements → Data Architecture → Model Architecture → Serving → Monitoring
**Key elements**: Always ask clarifying questions first, state your assumptions, discuss trade-offs

---

## 🚨 Key Gaps to Address Proactively

### 1. Terraform (JD mentions it, not on resume)
**Preemptive answer**: "I've worked extensively with Kubeflow Pipelines and Vertex AI, and while Terraform specifically wasn't in my toolkit at Chubb, I'm familiar with IaC principles through GitHub Actions CI/CD and Kubeflow. I've been actively upskilling on Terraform for GCP, particularly for Vertex AI infrastructure provisioning."

### 2. HDBSCAN (JD lists it prominently)
**Preemptive answer**: "I've used density-based clustering in customer segmentation work. HDBSCAN's advantage over K-means is that it doesn't require specifying k upfront and handles clusters of varying density — which is critical for marketing customer segmentation where you don't know the natural cluster count. I'd use it with UMAP for high-dimensional embeddings reduction before clustering."

### 3. Gemini/Vertex AI Agent Builder
**Preemptive answer**: "My LLM work has been primarily with GPT-4 and Claude via LangChain/LangGraph, but the architectural patterns are transferable. I've worked with GCP's Vertex AI for model training and serving, and I've been exploring Vertex AI Agent Builder as it's the natural evolution for enterprise agentic deployments on GCP."

### 4. Java (listed in JD)
**Preemptive answer**: "My primary language is Python, and I'm highly proficient there. I have exposure to Java-based environments, particularly around Spark (Java/Scala underpinning), and I'm comfortable reading and reviewing Java code. For new development, I'd advocate for Python in the ML/AI stack given the ecosystem maturity."

---

## 🎓 Key Concepts Cheat Sheet

### Agentic AI
| Concept | One-Line Explanation |
|---------|---------------------|
| ReAct | Interleaves Reasoning and Action steps to solve tasks iteratively |
| LangGraph | Framework for building stateful, multi-actor LLM applications as directed graphs |
| MCP | Anthropic's Model Context Protocol — standardized way for models to access tools/resources |
| A2A | Google's Agent-to-Agent protocol for inter-agent communication |
| HITL | Human-in-the-Loop — pausing agent for human approval at critical steps |
| Semantic Routing | Using embedding similarity to route queries to the right agent/pipeline |
| AgentOps | Observability, monitoring, and governance for production AI agents |

### RAG
| Concept | One-Line Explanation |
|---------|---------------------|
| Naive RAG | Retrieve → Generate without any refinement |
| HyDE | Generate a hypothetical answer, embed it, use that for retrieval |
| CRAG | Evaluate retrieval quality; fallback to web search if poor |
| Self-RAG | Model decides whether to retrieve at each step |
| RRF | Reciprocal Rank Fusion — combines results from multiple retrievers |
| RAGAS | Framework to evaluate RAG: faithfulness, relevancy, recall, precision |

### MLOps
| Concept | One-Line Explanation |
|---------|---------------------|
| PSI | Population Stability Index — detects input data drift |
| KS Test | Kolmogorov-Smirnov test — distribution drift detection |
| Canary | Route small % of traffic to new model before full rollout |
| Shadow Mode | Run new model in parallel without serving its output |
| KV Cache | Key-Value cache for LLM attention — speeds up inference |
| vLLM | High-throughput LLM serving with PagedAttention |

---

## 💼 Interview Day Strategy

### Opening (First 5 minutes)
- Introduce as: "Senior Data Scientist and AI Engineering Leader with 9+ years, currently at Chubb leading AI initiatives in insurance fraud detection using RAG and agentic AI workflows"
- Immediately signal alignment with Huge: "I'm particularly excited because my background bridges both the technical AI engineering side and the marketing/analytics side, which maps directly to the kind of intelligent experiences Huge builds for brands like Google, Nike, and McDonald's"

### During Technical Questions
1. **Repeat and reframe**: "Great question — let me break this into the problem, the approach, and the trade-offs..."
2. **Quantify**: Always attach metrics (latency, accuracy improvement, cost reduction, time saved)
3. **Acknowledge trade-offs**: Top architects say "It depends, here's the trade-off..."
4. **Ask clarifying questions on system design**: "Before I design this, can you tell me: is this batch or real-time? What's the expected QPS? What's the latency budget?"

### On Questions You're Uncertain About
"I've worked adjacent to that — in my case [related experience]. I'd approach it by [reasoned answer]. I'd also want to learn [specific aspect] more deeply — do you have a preferred approach at Huge?"

---

*Generated for interview at Huge India | Solutions Architect ML/AI | June 2026*
