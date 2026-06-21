# 🔍 Resume Project Drill-Down: Chubb (2024 - Present)
### Extreme Deep Dive for Huge Solutions Architect ML/AI Interview

---

> [!IMPORTANT]
> This document provides an **excruciatingly detailed** technical drill-down into your Chubb projects. The Huge interview panel (US-based) will test the *depth* of your knowledge. You must be able to defend every architectural choice, mathematical concept, and code structure presented here.

---

## 🏢 CHUBB PROJECT 1: Insurance Fraud Detection & Risk Modeling (RAG + LLMs)

**Resume Line:** *"Architected and deployed scalable ML-based fraud detection system using RAG (Retrieval-Augmented Generation) and Large Language Models (LLMs) to identify potentially fraudulent insurance claims across long-tail claim lifecycles."*

### 🔴 1. Architecture & System Design
**The Interview Question:** "Draw the architecture of your RAG-based fraud detection system. Walk me through the data flow from claim ingestion to fraud alert."

**Deep Dive Architecture:**
*   **Ingestion Layer:** Claims data (FNOL - First Notice of Loss, adjuster notes, medical records) arrives via AWS Kinesis data streams. Unstructured documents (PDFs, Word docs) are processed using AWS Textract for OCR and layout preservation.
*   **Preprocessing & Chunking:**
    *   *Challenge:* Insurance policies and medical records are highly structured; naive chunking destroys context.
    *   *Solution:* We implemented **Semantic Chunking**. Instead of fixed 512-token chunks, we used a lightweight sentence-transformer model to group adjacent sentences that have high cosine similarity, creating chunks that represent complete "thoughts" or "clauses." We added metadata to every chunk (Claim ID, Date, Document Type, Policy Section).
*   **Embedding Model:** We used `BGE-m3` or a domain-adapted `sentence-transformers/all-MiniLM-L6-v2` fine-tuned on insurance corpora using Contrastive Learning (SimCSE) to ensure terms like "whiplash" and "cervical sprain" map closely in vector space.
*   **Vector Store (FAISS / Chroma):**
    *   We started with Chroma for local prototyping.
    *   We migrated to **FAISS (Facebook AI Similarity Search)** deployed on AWS SageMaker for production. We used the `IVFFlat` index (Inverted File with Flat search) for a balance of speed and recall. The index is partitioned by Line of Business (e.g., Auto, Workers Comp) to narrow the search space.
*   **Retrieval Strategy:**
    *   *Hybrid Search:* We combined dense vector retrieval (FAISS) with sparse keyword retrieval (BM25 via Elasticsearch) using Reciprocal Rank Fusion (RRF) to ensure exact matches on specific medical codes (ICD-10) or policy numbers weren't lost.
    *   *Re-ranking:* The top 20 results from Hybrid Search are passed through a Cross-Encoder (like `ms-marco-MiniLM-L-6-v2`) to re-rank and select the top 5 most relevant chunks.
*   **Generation (LLM):** The top 5 chunks + the new claim details are injected into a strict prompt template evaluated by an LLM (GPT-4 or Claude 3.5 Sonnet) hosted on a secure VPC endpoint. The LLM outputs a structured JSON object containing a `fraud_score` (0-100) and an array of `reasoning_flags`.

### 🔴 2. Code Implementation Mental Model
**The Interview Question:** "How did you implement the retrieval pipeline in Python?"

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from elasticsearch import Elasticsearch

class FraudRetrievalPipeline:
    def __init__(self):
        self.embedding_model = SentenceTransformer('insurance-adapted-bge-m3')
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.es_client = Elasticsearch("https://vpc-es-cluster.aws.com")
        self.faiss_index = faiss.read_index("s3://models/faiss_ivfflat.index")
        
    def reciprocal_rank_fusion(self, dense_results, sparse_results, k=60):
        # Combines ranks from FAISS and ES. Score = 1 / (k + rank)
        fused_scores = {}
        for rank, doc_id in enumerate(dense_results):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        for rank, doc_id in enumerate(sparse_results):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        return sorted(fused_scores, key=fused_scores.get, reverse=True)

    def retrieve(self, claim_text, top_k=5):
        # 1. Dense Retrieval (FAISS)
        query_vector = self.embedding_model.encode([claim_text])
        _, dense_indices = self.faiss_index.search(query_vector, k=20)
        
        # 2. Sparse Retrieval (Elasticsearch / BM25)
        es_res = self.es_client.search(index="claims", query={"match": {"text": claim_text}}, size=20)
        sparse_indices = [hit['_id'] for hit in es_res['hits']['hits']]
        
        # 3. RRF Fusion
        fused_indices = self.reciprocal_rank_fusion(dense_indices[0], sparse_indices)[:15]
        
        # 4. Fetch actual text for fused indices
        documents = self.fetch_documents_by_ids(fused_indices)
        
        # 5. Cross-Encoder Re-ranking
        cross_inp = [[claim_text, doc] for doc in documents]
        rerank_scores = self.reranker.predict(cross_inp)
        
        # Sort and return Top K
        ranked_docs = [doc for _, doc in sorted(zip(rerank_scores, documents), reverse=True)]
        return ranked_docs[:top_k]
```

### 🔴 3. Cross-Examination & Trap Questions

*   **Trap Q: "Why use RAG for fraud detection? Shouldn't you just use an XGBoost model on structured claim features?"**
    *   *Defense:* XGBoost is excellent for structured data (claim amount, claimant age, time of day), and we DO use it as a baseline. However, fraud in long-tail claims (like Workers Comp) hides in the *unstructured* adjusters' notes—e.g., subtle inconsistencies in a claimant's description of pain over a 6-month period, or a doctor's notes conflicting with physical therapy records. XGBoost cannot parse this narrative nuance. RAG allows us to compare the current narrative against millions of historical fraudulent narratives and policy exclusions dynamically. We actually use an ensemble: the LLM's RAG-based `fraud_score` becomes a *feature* in the final XGBoost decision model.
*   **Trap Q: "How did you measure the accuracy of the RAG retrieval itself, separate from the LLM's output?"**
    *   *Defense:* We used **RAGAS (Retrieval Augmented Generation Assessment)** metrics. Specifically, we tracked *Context Precision* (is the retrieved context relevant to the claim?) and *Context Recall* (did we retrieve all necessary information to make a fraud determination?). We built a golden dataset of 500 historically investigated claims where the SIU (Special Investigation Unit) had highlighted the exact sentences that proved fraud. We measured whether our retrieval pipeline returned those exact sentences in the top-k results.
*   **Q: "What happens when a claim is 100 pages long? The context window will overflow."**
    *   *Defense:* We implemented a map-reduce summarization pipeline for the ingestion phase. If a medical file exceeds 10k tokens, we split it by date or provider. We use a cheaper LLM (like Claude 3 Haiku) to extract a structured summary (Diagnosis, Treatments, Anomalies) from each section. We then embed these structured summaries rather than the raw 100-page OCR output.

---

## 🏢 CHUBB PROJECT 2: BERT Fine-Tuning for Insurance Domain NLP

**Resume Line:** *"Fine-tuned domain-adapted BERT and RoBERTa models on proprietary insurance claims corpora using parameter-efficient fine-tuning (LoRA/PEFT), improving Named Entity Recognition (NER) and claim-intent classification accuracy by ~18% over zero-shot baselines."*

### 🔴 1. The Mathematics of LoRA
**The Interview Question:** "You used LoRA for fine-tuning. Explain the mathematics behind it and why it's more efficient than full fine-tuning."

**Deep Dive Answer:**
In full fine-tuning, we take a pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$ and update it with a gradient matrix $\Delta W$ of the same dimension. This requires storing optimizer states for every single parameter, which causes OOM (Out of Memory) errors on standard GPUs for large models.

LoRA (Low-Rank Adaptation) hypothesizes that the task-specific adaptation space has a low intrinsic dimension. Instead of updating the massive matrix $W_0$, we freeze $W_0$ and approximate the update $\Delta W$ using a low-rank decomposition:
$$ W = W_0 + \Delta W = W_0 + BA $$
Where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$, and the rank $r$ is much smaller than $d$ and $k$ (e.g., $r=8$ when $d=768$).

During the forward pass:
$$ h = W_0 x + BA x $$

Because $A$ and $B$ are tiny matrices, we reduce the number of trainable parameters by 99%. For our RoBERTa NER model, we applied LoRA exclusively to the Attention mechanism's Query ($W_q$) and Value ($W_v$) projection matrices. This allowed us to fine-tune on a single NVIDIA T4 GPU via AWS SageMaker, drastically cutting training costs while preventing catastrophic forgetting of the model's base English syntax.

### 🔴 2. Code Implementation Mental Model
**The Interview Question:** "Write out the configuration for your PEFT model."

```python
import torch
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

# 1. Load Base Model (frozen)
model_name = "roberta-base"
base_model = AutoModelForTokenClassification.from_pretrained(
    model_name, 
    num_labels=12 # e.g., B-CLAIMANT, I-CLAIMANT, B-DIAGNOSIS, etc.
)

# 2. Define LoRA Config
lora_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS, 
    inference_mode=False,
    r=16,                  # The rank of the update matrices
    lora_alpha=32,         # Scaling factor (usually 2x rank)
    lora_dropout=0.1,      # Regularization
    target_modules=["query", "value"], # Apply to self-attention
    modules_to_save=["classifier"]     # The final token classification head must be fully trained
)

# 3. Create PEFT Model
peft_model = get_peft_model(base_model, lora_config)
peft_model.print_trainable_parameters()
# Output: trainable params: 1,188,876 || all params: 125,834,508 || trainable%: 0.94%

# 4. Training (integrated with MLflow as per resume)
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=3e-4, # Can use a higher LR with LoRA than full FT
    per_device_train_batch_size=16,
    num_train_epochs=5,
    report_to="mlflow" # Directly logs to MLflow tracking server
)
```

### 🔴 3. Cross-Examination & Trap Questions

*   **Q: "You mentioned an 18% improvement. 18% improvement on what exact metric, and what was the baseline?"**
    *   *Defense:* The 18% improvement was on the **Macro F1-score** for the Named Entity Recognition (NER) task. The baseline was a zero-shot prompt using GPT-3.5-Turbo instructed to extract entities like `Policy_Number`, `Claimant_Name`, `Injury_Type`, and `Vehicle_VIN`. GPT-3.5 scored an F1 of around 0.68. It struggled heavily with proprietary internal jargon and formatting (e.g., misclassifying partial policy numbers as phone numbers). Our LoRA-fine-tuned RoBERTa model hit a Macro F1 of 0.86. The precision specifically jumped dramatically because the model learned the exact character distributions of our internal identifiers.
*   **Q: "If GPT-4 exists, why bother fine-tuning a small BERT model for NER?"**
    *   *Defense:* **Cost, Latency, and Privacy.** Running millions of historical and daily claims through GPT-4 for simple NER extraction would cost hundreds of thousands of dollars and introduce massive API latency. Furthermore, sending bulk PII to OpenAI was an InfoSec nightmare. By fine-tuning RoBERTa, we created a model that runs locally in our VPC, infers in milliseconds, costs pennies in compute, and outperforms zero-shot LLMs on our specific, narrow task.
*   **Q: "How did you handle the tokenization of domain-specific jargon?"**
    *   *Defense:* Standard RoBERTa tokenizers will break down insurance jargon into useless subwords. We updated the tokenizer vocabulary with our top 1,000 most frequent domain-specific terms (e.g., "subrogation", "indemnification", "CPT-code") before fine-tuning, ensuring the model treated these as single semantic units.

---

## 🏢 CHUBB PROJECT 3: Agentic AI Assistant for Analytics

**Resume Line:** *"Designed and built an Agentic AI Data Scientist capable of autonomously executing end-to-end analytical workflows... Implemented multi-step reasoning chains using LangGraph with tool-use agents (SQL executor, Python REPL, chart generator)."*

### 🔴 1. Architecture & System Design
**The Interview Question:** "Agentic workflows can easily get stuck in infinite loops or hallucinate SQL. How did you architect the LangGraph system to be robust enough for business analysts?"

**Deep Dive Architecture:**
We used **LangGraph** because it treats the agent workflow as a cyclical graph (StateGraph) rather than a linear chain (like standard LangChain). This allows for deterministic control over a non-deterministic LLM.

*   **The State (TypedDict):** The core of the graph is the state object passed between nodes. It contains `messages` (the chat history), `current_plan`, `sql_query`, `sql_result`, `python_code`, `base64_image` (for charts), and crucial for stability, an `error_trace` and `iteration_count`.
*   **Nodes (The Actors):**
    *   `Planner`: Takes the user's natural language request (e.g., "Show me the trend of auto claims in California vs Texas over the last 2 years") and generates a JSON array of steps.
    *   `SQL_Agent`: Given the plan and a injected schema map of our Snowflake/BigQuery database, generates a SQL query.
    *   `Execution_Engine`: A pure Python node (no LLM). It executes the SQL safely via SQLAlchemy. If it succeeds, it saves the result to a Pandas dataframe and puts the `df.head()` into the state. If it fails, it puts the database error (e.g., "Column 'cali' does not exist") into the `error_trace`.
    *   `Data_Scientist_Agent`: Uses a Python REPL tool to write pandas/matplotlib code against the dataframe to generate charts.
    *   `Reviewer`: Checks the final output against the original user prompt.
*   **Edges (The Logic):**
    *   The key is the **Conditional Edge** after the `Execution_Engine`.
    *   `if state["error_trace"] is not None:` route back to `SQL_Agent`. The prompt to `SQL_Agent` is now: "Your previous query failed with this error: {error_trace}. Here is the schema again. Fix the query."
    *   We implemented a strict `max_retries = 3` counter in the state. If the loop hits 3, it routes to a `Fallback` node that apologizes to the user and displays the partial work.

### 🔴 2. Cross-Examination & Trap Questions

*   **Trap Q: "You gave an LLM a Python REPL? That's a massive security vulnerability. How did you secure it?"**
    *   *Defense:* Absolute isolation. We did NOT run the Python REPL in the same environment as the application server. We used a sandboxed Docker container (via libraries like `E2B` or custom restricted Docker SDK calls). The container had no internet access, no access to environment variables, and a strict timeout of 10 seconds. We also scrubbed the LLM's generated code using AST (Abstract Syntax Tree) parsing to block any imports of `os`, `sys`, or `subprocess` before execution.
*   **Trap Q: "How did the LLM know the database schema? Did you stuff the whole schema into the prompt?"**
    *   *Defense:* No, enterprise insurance schemas have thousands of tables. We built a **Semantic Semantic Schema Router**. We generated embeddings for the *descriptions* of all our tables and columns. When the user asked about "auto claims in California," we first did a vector search against our data dictionary, retrieved only the DDL (Data Definition Language) for the `auto_claims_fct` and `location_dim` tables, and injected *only* those schemas into the `SQL_Agent`'s prompt.
*   **Q: "You claim a 60% reduction in analytics turnaround time. Break down that math."**
    *   *Defense:* Before this tool, a business stakeholder would submit an IT ticket. An analyst would pick it up (1 day), write the SQL and extract data to Excel (1 day), build pivot tables and charts (0.5 days), and draft a summary email (0.5 days). Total: 3 days. With the Agentic Assistant, the stakeholder prompts the tool. The agent takes 2-3 minutes to iterate, generate the SQL, execute it, write the Python visualization code, and generate a markdown summary. The analyst now acts as a reviewer (HITL - Human in the Loop), verifying the output in 1-2 hours before sending it off. The 60% metric is conservative; for routine queries, it's a 90% reduction.

---
*End of Chubb Deep Dive. Prepare for whiteboard execution.*
