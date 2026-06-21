# 🔍 Resume Project Drill-Down: Axtria (2022 - 2024)
### Extreme Deep Dive for Huge Solutions Architect ML/AI Interview

---

> [!IMPORTANT]
> Axtria is where you prove your **Marketing Science** chops. Huge has massive retail and consumer clients (Nike, McDonald's). The math you built for Pharma (J&J) translates directly to consumer attribution. The panel will drill you on endogeneity, Markov chains, and attention mechanisms. Master the equations below.

---

## 📈 AXTRIA PROJECT 1: Advanced Marketing Mix Modeling (MMM) & Revenue Optimization

**Resume Line:** *"Developed sophisticated Marketing Mix Models for Immunology and Neuroscience therapeutic areas using ensemble methods (Random Forest, XGBoost) and time series analysis, achieving ~10% increased revenue through optimized marketing spend allocation. Implemented Bayesian optimization for hyperparameter tuning... Designed custom genetic algorithm for multi-objective optimization."*

### 🔴 1. Architecture & The Mathematics of MMM
**The Interview Question:** "Pharma MMM is notoriously difficult because of long sales cycles. Write out your model formulation. How did you handle adstock and saturation?"

**Deep Dive Answer & Math:**
Standard linear regression (OLS) fails for marketing because the effect of a TV ad isn't immediate, and spending $2M doesn't yield exactly twice as much as spending $1M.

**1. The Transformation Layer (Feature Engineering):**
Before any data hits XGBoost, we transform the raw marketing spend.
*   **Adstock (Carryover Effect):** A rep visit today impacts prescribing behavior for weeks. We modeled this using a geometric decay function:
    $Adstock(X_t) = X_t + \lambda \times Adstock(X_{t-1})$
    Where $X_t$ is the spend at time $t$, and $\lambda \in [0, 1]$ is the decay rate.
*   **Saturation (Diminishing Returns):** We modeled saturation using the Hill function:
    $Saturation(x) = \frac{x^\alpha}{x^\alpha + \gamma^\alpha}$
    Where $\alpha$ controls the shape (S-curve) and $\gamma$ is the inflection point.

**2. The Ensemble Model:**
Instead of a linear model $Y = \beta X$, we used XGBoost. Why? Because marketing channels interact. A digital ad is more effective if a sales rep visited the doctor last week. XGBoost naturally captures these non-linear, multi-way interactions.

**3. Bayesian Optimization:**
Finding the right $\lambda$, $\alpha$, and $\gamma$ for every single channel, plus the XGBoost hyperparameters (`max_depth`, `learning_rate`), is an impossible grid-search space. We used **Optuna (Bayesian Optimization via Tree-structured Parzen Estimator)**. We defined an objective function (minimize out-of-sample MAPE) and let Optuna search the joint space of transformations and model hyperparameters simultaneously. This holistic optimization is what drove the 15% model accuracy improvement.

### 🔴 2. The Genetic Algorithm (NSGA-II)
**The Interview Question:** "How did you translate the predictive model into a budget recommendation that drove a 10% revenue increase?"

**Deep Dive Answer:**
Predicting sales is only half the battle; the client needs to know how to allocate next quarter's $50M budget. We framed this as a Multi-Objective Optimization problem.

1.  **Objective 1:** Maximize Total Revenue (predicted by our XGBoost model).
2.  **Objective 2:** Minimize Total Spend (or keep it within a bound).
3.  **Constraints:** Business rules (e.g., "TV spend cannot decrease by more than 20% quarter-over-quarter", "Digital spend must be at least $5M").

We used **NSGA-II (Non-dominated Sorting Genetic Algorithm II)**.
*   **Initialization:** We generate a population of random budget allocations (vectors of spend across channels).
*   **Fitness Evaluation:** We pass these budgets through the Adstock/Saturation transformations and into the XGBoost model to get predicted revenue.
*   **Selection, Crossover, Mutation:** The best allocations "breed" to create the next generation.
*   **The Output:** After 500 generations, the algorithm converges on a **Pareto Front**—a curve of optimal allocations where you cannot improve revenue without increasing cost. We presented this curve to J&J, allowing them to choose a risk/reward point.

### 🔴 3. Cross-Examination & Trap Questions

*   **Trap Q: "How did you address the endogeneity problem? If you advertise more during flu season, and sales go up, how do you know it was the ad and not the flu?"**
    *   *Defense:* Endogeneity (omitted variable bias) is the death of MMM. We addressed this by rigorously controlling for baselines. We included exogenous variables: macroeconomic indicators, competitor spend, and crucial for pharma, *disease incidence rates* (epidemiological data). By forcing the model to explain the variance using these baseline factors first, the marketing variables only captured the *incremental* lift.
*   **Trap Q: "XGBoost is a black box. How did you extract the ROI for TV vs Digital to show the client?"**
    *   *Defense:* We used **SHAP (SHapley Additive exPlanations)**. We calculated the marginal contribution of the TV feature vs the Digital feature for every single prediction, and aggregated those SHAP values globally. We also used the model to simulate marginal ROI: "If we hold all else constant and add $1 to TV, what is the change in the XGBoost output?"

---

## 📈 AXTRIA PROJECT 2: Omnichannel Marketing Attribution (Attention Mechanisms)

**Resume Line:** *"Designed and deployed a multi-touch attribution model using Markov Chains and SHAP values... Conducted attention mechanism experiments using scaled dot-product and multi-head self-attention to model sequential customer touchpoint interactions — improving attribution model AUC by 7% over baseline Markov chain approach."*

### 🔴 1. Architecture: Markov Chains vs Transformers
**The Interview Question:** "You compared Markov Chains to an Attention Mechanism for attribution. Walk me through exactly what you built. Why does Attention work better?"

**Deep Dive Answer:**
Multi-Touch Attribution (MTA) aims to assign credit to marketing touchpoints in a customer journey (e.g., Email -> Rep Visit -> Webinar -> Prescription).

**The Baseline: Markov Chains**
We modeled the journey as a directed graph. The probability of conversion is the probability of traversing the graph from Start to Conversion. We calculate the "Removal Effect"—what happens to the total conversion probability if we remove a specific channel (e.g., Webinar) from the graph? The higher the drop, the more credit that channel gets.
*   *The Flaw:* First-order Markov Chains are *memoryless*. They assume the transition from Webinar -> Conversion depends *only* on the Webinar. It ignores the fact that the Rep Visit 3 steps ago primed the doctor.

**The Innovation: Transformers / Multi-Head Self-Attention**
To capture long-range dependencies, I framed attribution as an NLP sequence classification problem.
1.  **Sequence Embedding:** The journey `[Email, Rep Visit, Webinar]` is tokenized. Each touchpoint is mapped to an embedding vector. Crucially, I added **Positional Encodings** (specifically, time-decayed encodings based on the days between touchpoints).
2.  **The Attention Layer:** We pass the sequence through a Multi-Head Self-Attention block.
    $Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$
    This allows the model to learn that a "Webinar" attends very strongly to a "Rep Visit" that occurred 2 months prior, but ignores an "Email" from yesterday.
3.  **Classification & Attribution:** The output of the Transformer goes to a Sigmoid classifier predicting Conversion (0 or 1). The 7% AUC improvement came from this predictive task. To get the actual *attribution weights*, we extracted the attention weights from the final layer. If the model heavily "attended" to the Rep Visit when making a correct positive prediction, the Rep Visit received more attribution credit.

### 🔴 2. Code Implementation Mental Model
**The Interview Question:** "How do you extract attention weights for attribution in PyTorch?"

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class JourneyAttentionAttribution(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # We use MultiheadAttention directly to easily extract weights
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        embedded = self.embedding(x)
        
        # Self-attention. We pass the embedded sequence as Query, Key, and Value
        # need_weights=True is critical for attribution!
        attn_output, attn_weights = self.attention(
            embedded, embedded, embedded, need_weights=True
        )
        
        # Pool the sequence (e.g., take the mean)
        pooled_output = torch.mean(attn_output, dim=1)
        conversion_prob = torch.sigmoid(self.classifier(pooled_output))
        
        # Return probability AND the weights for attribution analysis
        return conversion_prob, attn_weights
```

### 🔴 3. Cross-Examination & Trap Questions

*   **Trap Q: "With the death of third-party cookies and Apple's ATT (App Tracking Transparency), Multi-Touch Attribution is dead. You can't stitch the journey. How do you respond to that?"**
    *   *Defense:* This is the most important question in marketing science today. You are correct that deterministic, user-level tracking across the open web is dying. My MTA models rely heavily on **First-Party Data**. In pharma, this means CRM data (Veeva), email platforms, and logged-in HCP portals where consent is given. We don't rely on third-party cookies. Furthermore, modern architecture involves **Triangulation**: we use top-down MMM to set the macro budgets, and bottom-up MTA (on the observable, opted-in cohort of users) to optimize the tactical, day-to-day bidding, extrapolating those insights to the broader audience.
*   **Q: "How did you handle the cold-start problem? What if J&J launches a brand new TikTok channel mid-year?"**
    *   *Defense:* A pure Markov or Attention model cannot attribute credit to a channel it hasn't seen in training. We handled this using a Bayesian prior. We look at historical campaigns for similar drugs and initialize the new channel's embedding with the weights of a historically similar channel (e.g., using Instagram's weights as a prior for TikTok), allowing the model to update as new data streams in.

---

## 📈 AXTRIA PROJECT 3: Pharmaceutical Rep Communication System (GenAI POC)

**Resume Line:** *"Architected end-to-end AI system generating personalized pharma rep-to-doctor communications using GPT-4, integrating knowledge graph (Neo4j), recommendation engine, and NLP generation. Deployed... orchestrating ML workflows using Kubeflow."*

### 🔴 1. Architecture: Graph-RAG (Neo4j + GPT-4)
**The Interview Question:** "Why Neo4j? Why not just use a standard vector database for RAG?"

**Deep Dive Answer:**
Standard RAG (Vector Search) is excellent for semantic similarity ("Find me documents that talk about side effects"). It is terrible for relational logic ("Find all doctors in NY who prescribe Drug A but have not been visited by Rep B in the last 60 days, and find clinical trials relevant to their specialty").

Pharma sales is highly relational. We used **Graph-RAG**.
1.  **The Schema:** We built a property graph in Neo4j with nodes for `HCP` (Doctor), `Drug`, `Territory`, `ClinicalTrial`, and `Interaction`.
2.  **The Workflow:**
    *   The Rep logs in and selects a Doctor.
    *   We execute a Cypher query on Neo4j: "Match (d:HCP {id: X})<-[p:PRESCRIBES]-(drug), Match (d)-[i:INTERACTED_WITH]-(rep) Return..."
    *   This extracts a deterministic sub-graph: The doctor's prescribing history, their recent complaints, and new FDA label updates for those drugs.
    *   We serialize this sub-graph into a JSON/text context block.
    *   We pass this context to GPT-4 with a strict prompt: "Draft a 3-paragraph email to Dr. Smith. Reference her recent drop in Drug A prescriptions. Highlight this specific clinical trial. Do NOT make medical claims not present in the context."
3.  **Why Neo4j:** It prevents hallucination by enforcing hard relational constraints *before* the LLM sees the data. If a doctor isn't linked to a specific drug in the graph, the LLM literally cannot reference it.

### 🔴 2. Kubeflow & MLOps
**The Interview Question:** "You used Kubeflow. The JD asks for Vertex AI Pipelines. How do you bridge this?"

**Defense Script:** "Vertex AI Pipelines is quite literally a managed, serverless execution engine for Kubeflow Pipelines. I orchestrated the entire Neo4j data ingestion, graph embedding generation, and prompt evaluation jobs using Kubeflow DAGs. I wrote the containerized components, defined the artifact passing (InputPath/OutputPath), and handled the orchestration. Moving to Vertex AI Pipelines simply means replacing my open-source KFP cluster with Google's managed infrastructure and swapping the SDK to `google_cloud_pipeline_components`. The architectural mental model is identical."

---
*End of Axtria Deep Dive. Next up: EXL / CVS Health.*
