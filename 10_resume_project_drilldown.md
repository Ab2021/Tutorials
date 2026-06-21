# 🔍 Resume Project Drill-Down: Abhishek Bhardwaj
### Deep Dive for Huge Solutions Architect ML/AI Interview

---

> [!IMPORTANT]
> This document is **anchored entirely** to your actual resume. These are the exact technical questions you will be asked about the systems you built at Chubb, Axtria, and EXL. Every metric, tool, and architecture discussed here must be memorized and ready to present on a whiteboard.

---

## 🏢 SECTION 1: CHUBB PROJECT DRILLDOWNS (2024-Present)

### 🔵 Project 1: Insurance Fraud Detection & Risk Modeling (RAG + LLMs)

**Resume Line:** *"Architected and deployed scalable ML-based fraud detection system using RAG... to identify potentially fraudulent insurance claims across long-tail claim lifecycles."*

#### 🔴 The Core Question: "Walk me through the RAG architecture for fraud detection. Why RAG and not fine-tuning?"

**Model Answer:**
We chose RAG over fine-tuning because fraud patterns and policy wordings change frequently, and we needed traceability (knowing *why* a claim was flagged based on specific policy clauses). Fine-tuning bakes knowledge into the model weights, making it a black box and hard to update without full retraining. RAG grounds the LLM in the specific, up-to-date policy documents and past claim histories relevant to the current claim.

The architecture was a multi-stage retrieval pipeline:
1. **Data Ingestion & Chunking:** We processed unstructured claim notes, medical reports, and policy documents. Because insurance documents are dense and hierarchical, we used semantic chunking (sentence-transformers) rather than fixed-size chunking to preserve context.
2. **Embedding:** We used domain-adapted sentence transformers to generate dense embeddings.
3. **Vector Store:** We evaluated Chroma (for rapid prototyping) and FAISS. We ultimately deployed with FAISS for its efficiency in handling millions of dense vectors and its seamless integration with our AWS SageMaker endpoints.
4. **Retrieval:** When a new claim comes in, we embed the claim text and perform a similarity search (k=5) against historical known-fraud claims and policy guidelines.
5. **Generation/Scoring:** The retrieved context + the current claim is passed to the LLM with a strict prompt: "Based *only* on the provided context, does this claim exhibit known fraud indicators? Explain your reasoning."

#### ⚡ Cross-Questions & Trap Questions

*   **Q: "How did you handle long insurance claim documents that exceeded context windows?"**
    *   *Answer:* We didn't stuff the whole document into the prompt. We used a Map-Reduce approach for extremely long documents (e.g., 50-page medical histories). We chunked the document, asked the LLM to summarize potential fraud indicators in each chunk (Map), and then asked the LLM to synthesize those summaries into a final risk score (Reduce). We also used re-ranking models (like Cohere Rerank) to ensure only the most relevant chunks made it into the final context window.
*   **Q: "Why FAISS over a managed vector DB like Pinecone?"**
    *   *Answer:* Data privacy and compliance. Insurance claims contain highly sensitive PII and PHI. At the time, keeping the vector index entirely within our VPC on AWS SageMaker instances using FAISS was a hard requirement from InfoSec, avoiding data egress to a third-party managed service.
*   **Q: "What was your false positive rate and how did you tune the threshold?"**
    *   *Answer:* Initially, the LLM was too aggressive, flagging unusual but legitimate claims (high recall, low precision). We tuned the threshold by requiring the LLM to output a confidence score (0-1) alongside its reasoning. We set the operational threshold high enough to ensure the Special Investigations Unit (SIU) wasn't overwhelmed with false positives, prioritizing precision over recall for the automated alert pipeline. We also implemented a feedback loop where SIU investigators rated the alerts, which we used to refine the retrieval strategy.

#### 💻 Code Snippet (Mental Model for Whiteboard)
```python
# Conceptual FAISS + LangChain Retrieval
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load domain-adapted embeddings
embeddings = HuggingFaceEmbeddings(model_name="insurance-fraud-bert")

# 2. Load FAISS index (built offline)
vector_store = FAISS.load_local("s3://chubb-models/faiss_index", embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# 3. RAG Chain
def analyze_claim_for_fraud(claim_text: str):
    # Retrieve similar historical fraudulent claims or policy rules
    context_docs = retriever.invoke(claim_text)
    context_str = "\n".join([doc.page_content for doc in context_docs])
    
    prompt = f"""
    Context (Historical Fraud Patterns & Policy):
    {context_str}
    
    Current Claim:
    {claim_text}
    
    Task: Identify if the current claim exhibits patterns similar to the context. 
    Output JSON with 'fraud_probability' (0-1) and 'reasoning'.
    """
    return llm.invoke(prompt)
```

---

### 🔵 Project 2: BERT Fine-Tuning for Insurance Domain NLP

**Resume Line:** *"Fine-tuned domain-adapted BERT and RoBERTa models... using parameter-efficient fine-tuning (LoRA/PEFT), improving Named Entity Recognition (NER) and claim-intent classification accuracy by ~18% over zero-shot baselines."*

#### 🔴 The Core Question: "Explain LoRA mathematically. Why does it work? What hyperparameters did you tune?"

**Model Answer:**
LoRA (Low-Rank Adaptation) works on the hypothesis that the updates needed to adapt a pre-trained model to a specific task have a low "intrinsic rank." Instead of updating all the weights in a dense matrix $W$ (which is computationally expensive), LoRA freezes the original weights and injects trainable rank decomposition matrices.

Mathematically, if the original weight matrix is $W_0 \in \mathbb{R}^{d \times k}$, the update is constrained by representing it with a low-rank decomposition: $W = W_0 + \Delta W = W_0 + BA$, where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and the rank $r \ll \min(d, k)$. During training, only matrices $A$ and $B$ are updated. During inference, $BA$ is added back to $W_0$, so there's zero added latency.

For the insurance NER task, I tuned the rank $r$ (we found $r=8$ or $r=16$ hit the sweet spot between efficiency and performance) and the scaling factor $\alpha$ (usually set to $2r$). I applied LoRA primarily to the query and value projection matrices ($W_q$, $W_v$) in the transformer attention blocks, as research shows they are most critical for adaptation. This approach allowed us to train on a single GPU on SageMaker, which wouldn't have been possible with full fine-tuning, while achieving that 18% jump over zero-shot performance on our proprietary insurance ontology.

#### ⚡ Cross-Questions & Trap Questions

*   **Q: "What's the difference between LoRA and full fine-tuning? When would you use full fine-tuning?"**
    *   *Answer:* Full fine-tuning updates *every* parameter in the model. It's necessary when you are fundamentally changing the model's domain understanding (e.g., pre-training RoBERTa from scratch on clinical text to create ClinicalBERT). LoRA is for adapting an already capable model to a specific downstream task (like NER or intent classification) where the underlying language structure is similar, but the specific vocabulary or output format needs tuning. Full fine-tuning risks catastrophic forgetting; LoRA mitigates this.
*   **Q: "What was your insurance-specific data augmentation strategy?"**
    *   *Answer:* To handle domain jargon and make the model robust, we used techniques like synonym replacement (using an insurance taxonomy to swap words like "automobile" with "vehicle"), random masking of non-entity words, and injecting noise into policy numbers (e.g., swapping alphanumeric characters) to ensure the NER model learned the *context* surrounding a policy number, not just memorizing specific formats.
*   **Q: "PEFT vs LoRA vs QLoRA — what's the difference?"**
    *   *Answer:* PEFT (Parameter-Efficient Fine-Tuning) is the overarching category. LoRA is a specific PEFT technique (low-rank matrices). QLoRA is Quantized LoRA — it quantizes the base model weights to 4-bit precision to save memory, and then applies standard LoRA on top. We used standard LoRA for BERT/RoBERTa as they fit easily in GPU memory; QLoRA is typically reserved for massive LLMs (like Llama-70B).

#### 💻 Code Snippet (Mental Model for Whiteboard)
```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForTokenClassification

# Base model
model = AutoModelForTokenClassification.from_pretrained("roberta-base", num_labels=NUM_INSURANCE_LABELS)

# LoRA Configuration
lora_config = LoraConfig(
    task_type="TOKEN_CLS",
    r=16,               # Rank
    lora_alpha=32,      # Scaling factor
    lora_dropout=0.1,
    target_modules=["query", "value"] # Apply to attention Q and V matrices
)

# Wrap model with PEFT
peft_model = get_peft_model(model, lora_config)
# peft_model now has ~1% of the trainable parameters of the base model
```

---

### 🔵 Project 3: Agentic AI Assistant for Analytics

**Resume Line:** *"Designed and built an Agentic AI Data Scientist capable of autonomously executing end-to-end analytical workflows... reducing analytics turnaround time by ~60%."*

#### 🔴 The Core Question: "Walk me through the StateGraph definition, nodes, and conditional edges for your Agentic Data Scientist."

**Model Answer:**
The system was orchestrated using LangGraph to create a deterministic state machine for a non-deterministic LLM. We needed a multi-step reasoning chain where the agent could plan, execute tools, evaluate results, and loop back if it encountered errors.

The architecture:
1.  **State (TypedDict):** We defined an `AgentState` containing the `messages` list (conversation history), the current `plan`, `data_context` (schema info), and `errors`.
2.  **Nodes:**
    *   `planner_node`: Analyzes the user query and generates a step-by-step execution plan.
    *   `sql_generator_node`: Translates the plan into SQL queries based on the `data_context`.
    *   `sql_executor_node`: Runs the SQL against the database (the actual tool execution).
    *   `python_repl_node`: Analyzes the SQL results or generates charts using Pandas/Matplotlib.
    *   `synthesizer_node`: Reviews all tool outputs and writes the final narrative insight.
3.  **Conditional Edges:** This is the critical part. After `sql_executor_node`, a conditional edge checks for execution errors. If an error occurs (e.g., column not found), it routes *back* to `sql_generator_node` with the error message appended to the state, allowing the LLM to self-correct. If successful, it routes to the `python_repl_node` or `synthesizer_node`.
4.  **Impact:** This system reduced analytics turnaround time by 60%. Previously, an analyst spent 3 days gathering requirements, writing SQL, exporting to Excel, making charts, and writing a PowerPoint. The agent automates the SQL, data manipulation, and chart generation in minutes, leaving the analyst to simply review and refine the final narrative.

#### ⚡ Cross-Questions & Trap Questions

*   **Q: "How did you handle context overflow in a multi-step analytical workflow?"**
    *   *Answer:* Long-running agent loops quickly exceed the context window, especially if tool outputs (like raw SQL data) are large. We implemented a message pruning strategy in the state updater. We also *never* passed raw data tables directly into the LLM context. The `sql_executor_node` saved results to a temporary CSV or Pandas DataFrame, and passed only the *summary* (column names, row count, basic stats) or the file path back to the state. The `python_repl_node` then operated on that file path.
*   **Q: "What's your iteration guard and why did you set max_steps=15?"**
    *   *Answer:* LLMs can get stuck in infinite loops (e.g., repeatedly generating the same incorrect SQL syntax). A `max_steps` recursion limit in LangGraph is mandatory for production. If the agent hits step 15, we interrupt the graph, return a graceful error to the user stating the analysis was too complex, and provide the partial trace of what was attempted.
*   **Q: "How did you connect Streamlit to the LangGraph stream?"**
    *   *Answer:* Streamlit doesn't natively support asynchronous generators well. We wrapped the LangGraph `app.stream()` call in a synchronous generator, and used Streamlit's `st.write_stream()` or empty placeholders to update the UI progressively. As the graph emitted state updates from different nodes (e.g., "Node: sql_generator completed"), we updated a status spinner in the UI so the user knew the agent was working.

#### 💻 Code Snippet (Mental Model for Whiteboard)
```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
import operator

# Define State
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    sql_query: str
    tool_error: str

# Define Nodes
def sql_generator_node(state: AgentState):
    # LLM generates SQL based on messages + schema
    pass

def sql_executor_node(state: AgentState):
    # Execute SQL. If error, populate state['tool_error']
    pass

def router(state: AgentState):
    if state.get("tool_error"):
        return "sql_generator_node" # Loop back to fix
    return "synthesizer_node"     # Proceed to final output

# Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("sql_generator_node", sql_generator_node)
workflow.add_node("sql_executor_node", sql_executor_node)
workflow.add_node("synthesizer_node", synthesizer_node)

workflow.set_entry_point("sql_generator_node")
workflow.add_edge("sql_generator_node", "sql_executor_node")
workflow.add_conditional_edges("sql_executor_node", router)
workflow.add_edge("synthesizer_node", END)

app = workflow.compile()
```

---

## 🏢 SECTION 2: AXTRIA PROJECT DRILLDOWNS (2022-2024)

### 🔵 Project 4: Advanced Marketing Mix Modeling (MMM) & Revenue Optimization

**Resume Line:** *"Developed sophisticated Marketing Mix Models... achieving ~10% increased revenue through optimized marketing spend allocation. Implemented Bayesian optimization for hyperparameter tuning, improving model accuracy by 15%."*

#### 🔴 The Core Question: "Walk me through your MMM architecture for J&J. What's the model formulation and how did the genetic algorithm work?"

**Model Answer:**
For pharma MMM (J&J Immunology), standard linear regression is insufficient due to delayed effects (adstock) and diminishing returns (saturation) of marketing spend, plus strong baseline trends (disease seasonality).

**1. The Model Formulation:**
We modeled Sales at time *t* as:
$Sales_t = Baseline_t + \sum ( \beta_i \times Saturation(Adstock(Spend_{i,t})) )$
We used ensemble methods (XGBoost/Random Forest) to capture complex non-linear interactions between channels that traditional OLS misses, using engineered features representing the adstocked and saturated spend. We improved the model's predictive accuracy (MAPE) by 15% by using Bayesian Optimization (Hyperopt/Optuna) to tune the hyperparameters of the XGBoost model and the specific adstock decay rates ($\lambda$) for each channel simultaneously, rather than grid searching.

**2. The Optimization (NSGA-II):**
The client didn't just want a model; they wanted a budget recommendation. Marketing optimization is multi-objective: you want to maximize Revenue, but you also want to minimize total Cost, and perhaps maximize market Reach.
We built a custom optimizer using **NSGA-II** (Non-dominated Sorting Genetic Algorithm).
*   **Chromosomes:** A vector of budget allocations across channels (e.g., [30% TV, 50% Rep Visits, 20% Digital]).
*   **Fitness Functions:** The MMM model predicts revenue for that allocation.
*   **Output:** Instead of one "optimal" budget, NSGA-II produces a **Pareto Front**—a set of optimal allocations where you cannot improve revenue without increasing cost. We presented this curve to J&J brand managers, allowing them to choose a point on the curve that matched their strategic risk appetite for the quarter. This optimized allocation strategy drove a measured ~10% increase in revenue for the campaigns that adopted it.

#### ⚡ Cross-Questions & Trap Questions

*   **Q: "J&J Immunology — what's the challenge of MMM in pharma vs consumer goods?"**
    *   *Answer:* In consumer goods (like McDonald's), the path to purchase is short (see ad -> buy burger). In pharma, the "consumer" is the patient, but the "decision maker" is the physician (HCP). You are modeling physician prescribing behavior based on sales rep visits (detailing), medical conferences, and direct-to-consumer (DTC) TV ads. The lag effects are much longer (months, not days), and you have strict regulatory constraints on what channels can be used.
*   **Q: "How do you explain a Pareto front to a J&J brand manager?"**
    *   *Answer:* I don't use the term "Pareto front." I show them a scatter plot curve and say: "Every dot is a different budget scenario. The line represents the maximum possible return for any given budget. You are currently here (below the line). We can move you up to the line to get more revenue for the same spend, or move you left to get the same revenue for less spend. Which direction is the priority this quarter?"
*   **Q: "What's the endogeneity problem in MMM and how did you address it?"**
    *   *Answer:* Endogeneity happens when marketing spend is correlated with the error term (e.g., you increase ad spend during flu season because you know demand will be high). The model then overestimates the effect of the ads. We addressed this by incorporating strong seasonal baselines and control variables (competitor spend, disease incidence rates) to isolate the true incremental lift of the marketing activities.

---

### 🔵 Project 5: Omnichannel Marketing Attribution (Attention Mechanisms)

**Resume Line:** *"Conducted attention mechanism experiments using scaled dot-product and multi-head self-attention to model sequential customer touchpoint interactions — improving attribution model AUC by 7% over baseline Markov chain approach."*

#### 🔴 The Core Question: "You added an attention mechanism to Markov chain attribution. Walk me through exactly what you built and why."

**Model Answer:**
Traditional Multi-Touch Attribution (MTA) often relies on Markov Chains. A Markov Chain calculates the probability of conversion by looking at the transition probabilities between states (touchpoints). However, order-1 Markov Chains assume the journey is *memoryless*—that the next step only depends on the current step. In complex patient/HCP journeys, this is false. A rep visit 3 months ago strongly influences how a physician reacts to an email today.

To capture these long-range dependencies, I modeled the customer journey as a sequence of events and applied a Transformer-style **Multi-Head Self-Attention** mechanism.
1.  **Input:** A sequence of touchpoints for a specific HCP (e.g., [Rep Visit, Webinar, Email, Display Ad]).
2.  **Embeddings:** Each touchpoint type was embedded into a dense vector, along with positional encodings to capture time.
3.  **Self-Attention:** The scaled dot-product attention allowed the model to learn which touchpoints in the past were most "attended to" (relevant) when the final conversion decision was made.
4.  **Result:** We extracted the attention weights from the final layer to serve as our attribution weights (how much credit each channel gets). This deep learning approach improved the predictive AUC of whether a journey would convert by 7% compared to the baseline Markov model, proving it was capturing sequential patterns the Markov model missed. We used SHAP values on top of the model to explain *why* certain sequences had higher conversion probabilities to the client.

#### ⚡ Cross-Questions & Trap Questions

*   **Q: "How long were the customer journeys you modelled? What was the max sequence length?"**
    *   *Answer:* In pharma, journeys are long but sparse. We looked at a 6-to-12-month lookback window. We capped the sequence length (e.g., at 50 touchpoints). For journeys shorter than 50, we used padding (with a mask so the attention mechanism ignored the padding tokens). For the rare journeys longer than 50, we truncated the oldest events.
*   **Q: "Markov chains are highly interpretable. Transformers are black boxes. How did the client react?"**
    *   *Answer:* This was the biggest hurdle. We didn't present the neural network architecture to the business. We presented the aggregated attention weights as the new "Attribution Credit" percentages. We validated it by showing that the Transformer model correctly gave more credit to high-impact, early-funnel events (like attending a major medical congress) which the Markov model was unfairly ignoring because they occurred too far back in the sequence.
*   **Q: "Shapley values vs SHAP vs Markov removal effect — how do they relate?"**
    *   *Answer:* Shapley values (game theory) calculate the marginal contribution of a channel across all possible journey combinations. SHAP (SHapley Additive exPlanations) is a machine learning approximation of Shapley values to explain feature importance in complex models (like our attention model). The Markov removal effect calculates attribution by removing a node from the chain and seeing how much the total conversion probability drops. They are different mathematical approaches trying to answer the same question: "How much incremental value did this channel provide?"

---

## 🏢 SECTION 3: EXL / CVS HEALTH PROJECT DRILLDOWNS (2016-2022)

### 🔵 Project 6: Customer Lifetime Value (CLV) Prediction at Scale

**Resume Line:** *"Developed distributed Random Forest model using PySpark to predict CLV for 2M+ prospects... Utilized GCP Dataproc for orchestrating scalable model training and scoring, reducing processing time by 70%."*

#### 🔴 The Core Question: "Walk me through the Dataproc architecture for CLV. How did you achieve a 70% reduction in processing time?"

**Model Answer:**
When I took over the CLV pipeline at CVS/Aetna, it was a legacy system running Pandas scripts on an on-premise monolithic server. Training and scoring 2 million customers with rich historical features took days, and often crashed due to Out-Of-Memory (OOM) errors.

I re-architected the pipeline to run on **GCP Dataproc** using **PySpark**.
1.  **Data Engineering:** I rewrote the feature engineering pipelines in Spark SQL and PySpark DataFrames. This allowed us to perform aggregations (e.g., total spend last 12 months, frequency of claims) in a distributed manner across worker nodes, rather than loading everything into memory.
2.  **Modeling:** I replaced the single-node scikit-learn model with Spark MLlib's distributed Random Forest. Random Forest is inherently parallelizable (each tree can be trained on a different node), making it perfect for Spark.
3.  **Infrastructure:** We used an ephemeral Dataproc cluster. We spun up a cluster with a master node and several preemptible (spot) worker nodes to save costs, submitted the PySpark job, and spun the cluster down immediately after the predictions were written back to BigQuery.
4.  **Impact:** Moving from single-node Pandas to distributed PySpark on Dataproc reduced the end-to-end processing time from nearly 3 days to under 20 hours (a 70% reduction), eliminating OOM crashes entirely and allowing us to score the entire 2M+ prospect base weekly instead of monthly.

#### ⚡ Cross-Questions & Trap Questions
*   **Q: "How do you build CLV for a health plan customer vs a retail customer?"**
    *   *Answer:* Retail CLV (like IKEA) is transaction-based (BG/NBD models) driven by purchase frequency and average order value. Insurance/Health plan CLV is contract-based. Revenue is largely fixed (monthly premiums), so CLV is driven entirely by *cost prediction* (predicting future claims utilization) and *retention probability* (churn). The feature set focuses heavily on risk scoring, chronic condition flags, and past utilization patterns, rather than just transaction recency.
*   **Q: "Why Random Forest over Gradient Boosting (GBM/XGBoost)?"**
    *   *Answer:* At the time, Spark MLlib's Random Forest was more mature and easier to tune in a distributed setting than early distributed GBM implementations. RF is less prone to overfitting and requires less hyperparameter tuning, which was beneficial for a pipeline we needed to run completely autonomously every week without manual intervention.

---

### 🔵 Project 7: Patient Readmission Risk Prediction (Deep Learning)

**Resume Line:** *"Built production ML pipeline for readmission risk prediction incorporating clinical notes; leveraged PyTorch with BERT for rich text feature extraction... improving model AUC from 0.82 to 0.89."*

#### 🔴 The Core Question: "The 7-point AUC improvement is massive for healthcare. What specifically drove that jump from 0.82 to 0.89?"

**Model Answer:**
The baseline model (AUC 0.82) relied entirely on structured claims data: age, gender, diagnosis codes (ICD-10), and procedure codes. However, claims data lacks clinical nuance. A diagnosis code tells you a patient had heart failure, but it doesn't tell you the severity, the patient's adherence to medication, or social determinants of health (e.g., "patient lacks transportation to follow-up appointments").

The 7-point jump to 0.89 came from unlocking the unstructured data: the discharge summaries and clinical notes.
1.  **Feature Extraction:** We used a clinical variant of BERT (similar to ClinicalBERT) pre-trained on MIMIC-III data. We fed the raw discharge summaries through the model to extract dense embeddings representing the clinical narrative.
2.  **Multimodal Fusion:** We didn't use BERT to predict readmission directly. We used it as a feature extractor. The BERT embeddings were concatenated with the structured features (demographics, lab values) and fed into a downstream classifier (a multi-layer perceptron in PyTorch).
3.  **The "Why":** The NLP features captured critical risk factors missing from structured claims—specifically, references to medication non-compliance, lack of social support, and complex symptom presentations—which are the true drivers of 30-day hospital readmissions.

#### ⚡ Cross-Questions & Trap Questions
*   **Q: "How did you handle label leakage? It's very common in readmission models."**
    *   *Answer:* This is critical. Label leakage occurs if you include data generated *after* the patient is discharged in the prediction features. We enforced a strict temporal cutoff. The model was only allowed to see data timestamped *prior* to or strictly *at* the moment of discharge. We audited the feature pipelines to ensure no billing codes or follow-up notes generated post-discharge leaked into the training set.

---

> [!TIP]
> **Interview Strategy for Projects:**
> When asked "Walk me through a project," use the **STAR** method, but add **T**echnology and **A**rchitecture: **S-T-A-R-T-A**.
> 1.  **S**ituation: The business problem (e.g., fraud takes too long to investigate).
> 2.  **T**ask: Your specific role.
> 3.  **A**rchitecture: The high-level design (e.g., RAG pipeline with FAISS).
> 4.  **A**ction: The technical hurdles you overcame (e.g., tuning chunking).
> 5.  **R**esult: The metric (60% reduction in time).
