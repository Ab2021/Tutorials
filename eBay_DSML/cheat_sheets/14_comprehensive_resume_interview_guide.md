# eBay DS/ML Interview — Comprehensive Resume Deep-Dive & "Golden Answers" Mapping

**Candidate:** Abhishek Bhardwaj
**Goal:** To prepare you for the grueling "Deep Dive" and "Bar-Raiser" rounds where interviewers tear apart your resume line-by-line to test your actual depth, ownership, and ability to map past work to eBay's domain.

This guide takes your actual resume bullet points and provides the **Layer 1 to Layer 3 probing questions**, the **"Bridge to eBay"**, and the **Golden Answer Structure**.

---

## 🛡️ Experience 1: Insurance Fraud Detection (RAG + LLMs) at Chubb
> **Resume Claim:** "Architected and deployed scalable ML-based fraud detection system using RAG... to identify potentially fraudulent insurance claims... Developed Information Extraction pipelines from unstructured claims using NLP Transformers/BERT."

### The Interrogation (How eBay will probe this)
*   **Layer 1 (The Basics):** "Walk me through the architecture of your RAG fraud detection system. What vector database did you use, and how did you chunk the claims data?"
*   **Layer 2 (The Engineering):** "Insurance claims can be very long. How did you handle cases where the contextWindow exceeded the LLM's limit? How did you ensure your NLP transformers (BERT) didn't suffer from catastrophic forgetting if you fine-tuned them?"
*   **Layer 3 (The 'eBay Scale' Test):** "RAG latency is typically high (seconds). Fraud detection at eBay needs to happen in milliseconds when a user swipes a credit card. How would you adapt your RAG approach here to meet a 100ms latency budget?"

### The "Bridge to eBay" (Trust & Safety / xFraud)
At eBay, fraud is transactional (fake listings, stolen credit cards, fraud rings) rather than claims-based. You must pivot from "document analysis" to "real-time anomaly detection."

### The Golden Answer Structure
*   *Architecture:* "We decoupled the heavy NLP processing from the real-time scoring. We used BERT to asynchronously extract features (entities, sentiment, contradictions) from claims documents, storing those dense representations."
*   *The Pivot:* "If I were building this for eBay's transaction monitoring, full RAG during the checkout flow is far too slow. Instead, I would use a Lambda architecture: batch-process user/listing text via transformers offline, store those embeddings in a low-latency key-value store (like eBay's NuKV), and use a fast ensemble model (like XGBoost or a Graph Neural Net) to do the real-time blocking at checkout."
*   *Evaluation:* "We evaluated the LLM's precision using human-in-the-loop validation, treating 'false positives' as highly costly since they delayed legitimate payouts."

---

## 📊 Experience 2: Omnichannel Attribution & MMM at Axtria
> **Resume Claim:** "Developed Marketing Mix Models... achieving ~10% increased revenue. Implemented Bayesian optimization... Designed multi-touch attribution using Markov Chains and SHAP values."

### The Interrogation (How eBay will probe this)
*   **Layer 1 (The Basics):** "Explain how your Markov Chain attribution model works. Why did you choose it over Shapley values or a simple heuristic like Last-Touch attribution?"
*   **Layer 2 (The Math):** "How exactly did you isolate the causal impact of your MMM to claim a '10% increased revenue'? Did you run a synthetic control or an A/B test? How did you account for seasonality and ad-stock (decay) effects?"
*   **Layer 3 (The Trade-offs):** "You mention using both Random Forest and XGBoost for MMM, and Bayesian optimization. How much absolute lift did Bayesian optimization actually give you over a simple random search, and was the compute cost worth it?"

### The "Bridge to eBay" (Promoted Listings & Ad Tech)
eBay has a massive internal advertising platform (Promoted Listings). They desperately care about attribution (did the ad cause the sale, or would the buyer have bought it anyway?).

### The Golden Answer Structure
*   *The Math:* "I chose Markov Chains because they model the sequence of touchpoints probabilistically (removal effect), capturing the actual customer journey better than static Shapley. For hyperparameter tuning, Bayesian optimization (using Tree-structured Parzen Estimator) allowed us to converge on optimal parameters in 1/4 the steps of Grid Search, which was crucial given Databricks compute costs."
*   *Causality:* "We validated the 10% lift by running controlled geo-holdout experiments—increasing spend in test regions based on our model's recommendations while holding control regions flat, running a Difference-in-Differences analysis."
*   *The Pivot:* "At eBay, applying this to Promoted Listings requires handling extreme sparsity (most buyers click 0 or 1 ad before buying). I would adapt the Markov model to incorporate graph-based embeddings of the user's search session."

---

## 🤖 Experience 3: Agentic AI & Knowledge Graphs (POCs)
> **Resume Claim:** "Designed an Agentic AI-powered BI tool using LangChain... Architected end-to-end AI system generating personalized rep-to-doctor comms using GPT-4 and Neo4j (Knowledge Graph)."

### The Interrogation (How eBay will probe this)
*   **Layer 1 (The Basics):** "How did you construct the prompt to interact with Neo4j? Did you use Text-to-Cypher, or did you retrieve graph embeddings?"
*   **Layer 2 (Reliability):** "Agentic workflows using LangChain are notorious for hallucinating tools or getting stuck in loops. How did you build guardrails around your Agentic BI tool so it didn't output confident but mathematically incorrect SQL queries to analysts?"
*   **Layer 3 (Productionization):** "These sound like very cool POCs. What were the specific bottlenecks in taking this from a Streamlit prototype to a highly concurrent production system?"

### The "Bridge to eBay" (Mercury Platform / GenAI Shopping)
eBay is currently building exactly this: AI-powered buyer assistants targeting enthusiast buyers.

### The Golden Answer Structure
*   *The Agent Guardrails:* "To prevent SQL hallucinations in the BI tool, I didn't let the LLM execute arbitrary queries. I restricted its toolset to a pre-defined semantic layer and used Corrective RAG (CRAG) patterns—if the generated query failed syntax checks, a secondary LLM specifically trained on error correction would attempt a fix up to 3 times before graceful degradation."
*   *Neo4j Integration:* "For the Knowledge Graph, we vectorized node relationships. Instead of brittle Text-to-Cypher, we used Vector Search combined with Graph Traversal (hybrid retrieval) to ground the GPT-4 generation."
*   *The Pivot:* "eBay could use a similar Graph+LLM architecture for 'Shop the Look' or parts-compatibility (e.g., 'Does this carburetor fit my 1968 Mustang?'). A knowledge graph maps the complex categorical relationship, while the LLM provides the natural language interface."

---

## 📈 Experience 4: CLV Prediction at Scale at EXL/CVS
> **Resume Claim:** "Developed distributed Random Forest model using PySpark to predict CLV for 2M+ prospects... Utilized GCP Dataproc for orchestrating scalable model training."

### The Interrogation (How eBay will probe this)
*   **Layer 1 (The Basics):** "How did you define 'Customer Lifetime Value' in this context? Was it revenue, profit, over 1 year, 5 years?"
*   **Layer 2 (Distributed Systems):** "What challenges did you face implementing Random Forest in PySpark compared to scikit-learn? How did you handle data skewness across your Dataproc cluster executors?"
*   **Layer 3 (Feature Engineering):** "LTV is heavily influenced by recent behavior. What specific aggregated features were most predictive? How did you handle the 'survivorship bias' of long-time customers?"

### The "Bridge to eBay" (Buyer Loyalty & Growth)
eBay relies heavily on repeat buyers. LTV (Lifetime Value) models drive marketing acquisition budgets (CAC limits). 

### The Golden Answer Structure
*   *Definition:* "We defined CLV as the discounted cumulative net margin over a 3-year horizon. I used the Buy Till You Die (BTYD) framework conceptually, but applied Random Forest for non-linear feature interactions."
*   *PySpark Tuning:* "One major challenge was network shuffle overhead during tree building. I optimized this by adjusting `spark.sql.shuffle.partitions` based on our cluster size and broadcasting small lookup tables for feature engineering to avoid expensive joins."
*   *The Pivot:* "For eBay, predicting a buyer's LTV is challenging because purchasing is highly sporadic compared to pharmacy refills. I would index heavily on 'Session Depth' and 'Category Diversity' (does the buyer shop in 1 category or 5?) as early predictors of high-LTV behaviors."

---

## ⚙️ Experience 5: MLOps & Production Engineering
> **Resume Claim:** "Implemented end-to-end MLOps workflow: Airflow... PySpark... MLflow for tracking, and automated model monitoring for drift detection."

### The Interrogation (How eBay will probe this)
*   **Layer 1 (The Basics):** "Walk me through your CI/CD pipeline for ML. When you merge code to the main branch, what automatically happens?"
*   **Layer 2 (Model Drift):** "You mentioned automated drift detection. Were you measuring Data Drift or Concept Drift? What specific metrics did you use (e.g., Population Stability Index, Kullback-Leibler divergence) to trigger an alert?"
*   **Layer 3 (Retraining Strategy):** "If a model triggered a drift alert, was the retraining completely automated, or was there a human in the loop? How did you ensure the newly trained model was actually better before promoting it?"

### The "Bridge to eBay" (MLOps at Scale)
eBay updates thousands of models. They need engineers who don't just write Jupyter notebooks, but write robust, self-healing pipelines.

### The Golden Answer Structure
*   *Drift Detection:* "We primarily tracked Data Drift using Population Stability Index (PSI) on our top 10 most critical features. If PSI > 0.2, it triggered a Slack alert to the DS team."
*   *CI/CD Flow:* "Code merged to 'main' triggered GitHub Actions that ran unit tests and data expectation tests. If passed, Airflow triggered a training job on Dataproc. The resulting model artifacts were logged in MLflow. We used a 'Shadow Deployment' strategy—the new model scored traffic silently for 3 days to compare its distribution against the incumbent before we fully promoted it to production."

---

## 🌟 The Ultimate "Tell Me About Yourself" (Elevator Pitch)
*Use this at the beginning of every interview to control the narrative.*

> "I’m a Senior Data Scientist and AI/ML Engineering Lead with over 9 years of experience, specializing in driving business impact through modern machine learning.
>
> In the first phase of my career at EXL and Axtria, I focused heavily on deep predictive analytics at scale—building distributed PySpark pipelines on GCP, predicting customer lifetime value for millions of users, and optimizing millions in marketing spend using Bayesian MMM algorithms that drove measurable 10% revenue lifts. 
>
> In my current role leading AI initiatives at Chubb, I’ve transitioned into the GenAI and modern NLP space. I architect and deploy production-grade RAG pipelines, Agentic BI tools with LangChain, and Knowledge Graph-integrated LLMs to fight insurance fraud and automate complex underwriting workflows.
>
> I'm excited about eBay because the challenges you are facing—balancing a massive two-sided marketplace, optimizing search relevance, and transitioning to AI-first e-commerce—perfectly align with my dual background in large-scale classical predictive modeling and cutting-edge Generative AI architectures."
