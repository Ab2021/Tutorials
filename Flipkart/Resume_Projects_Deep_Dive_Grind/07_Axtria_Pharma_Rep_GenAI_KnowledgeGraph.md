# 🔥 PROJECT GRIND: Pharma Rep Communication System (GPT-4 + Neo4j Knowledge Graph)
### Company: Axtria | Role: Decision Science & Engineering Manager

> **Resume Bullet:** Architected end-to-end AI system generating personalized pharma rep-to-doctor communications using GPT-4, integrating knowledge graph (Neo4j), recommendation engine, and NLP generation. Deployed to production using Flask + Streamlit with CI/CD pipeline in GitHub Actions; orchestrated ML workflows using Kubeflow with MLflow experiment tracking.

---

## 🏗️ 1. PRODUCTION ARCHITECTURE

```
[Doctor Profile DB] ──────────┐
                               │
[Drug Formulary / Indication] ──┼──▶ [Neo4j Knowledge Graph]
                               │        │
[Past Interaction Logs] ───────┘        │
                                        ▼
                              ┌─────────────────────────┐
                              │ Recommendation Engine    │
                              │ (Content-based Filtering │
                              │  + Graph Traversal)      │
                              └──────────┬──────────────┘
                                         │
                         Doctor Profile + Recommended Topics
                                         │
                                         ▼
                              ┌──────────────────────────┐
                              │ GPT-4 Generation Layer   │
                              │ (Prompt = Doctor context │
                              │  + Drug data + Tone      │
                              │  + Compliance guardrails)│
                              └──────────┬───────────────┘
                                         │
                              ┌──────────▼───────────────┐
                              │ Compliance / Safety      │
                              │ Guardrail Filter         │
                              │ (Regex + LLM-as-Judge)   │
                              └──────────┬───────────────┘
                                         │
                              ┌──────────▼───────────────┐
                              │ Flask API + Streamlit UI │
                              │ (CI/CD: GitHub Actions)  │
                              └──────────────────────────┘
```

---

## 🛠️ 2. PHASE-BY-PHASE DEEP DIVE & UNUSUAL EDGE CASES

### A. Knowledge Graph Design Phase (Neo4j)
*   **What you did:** Modeled entities (Doctor, Drug, Indication, Mechanism of Action, Clinical Trial) as nodes. Relationships like `PRESCRIBES`, `TREATS`, `HAS_SIDE_EFFECT`, `COMPETES_WITH` as edges. Used Cypher queries to traverse the graph for personalized content retrieval.
*   **The "Unusual" Issue:** **"The Competing Drug Landmine."** The Knowledge Graph dutifully returned that Drug A `COMPETES_WITH` Drug B. GPT-4 then used this relationship to *name the competitor drug* in the generated message, giving free advertising to the competition. The client saw this in UAT and escalated.
*   **The Fix:** Created a `BLACKLIST_ENTITY` relationship type in Neo4j. Before passing graph context to GPT-4, a Cypher query filtered out all competitor drug entity nodes. The prompt was also augmented with a strict negative constraint: `"Do NOT mention any drug not in the provided approved_drug_list."` Added a post-generation regex + NER filter to catch any leaked competitor mentions.

### B. LLM Generation & Personalization Phase
*   **What you did:** GPT-4 generated personalized email/message drafts for pharma reps. Input: doctor specialty, past prescribing behavior, recommended topics from the graph.
*   **The "Unusual" Issue:** **"The FDA Off-Label Hallucination."** GPT-4 occasionally generated messaging that implied Drug A could be used for conditions it is NOT FDA-approved for (off-label promotion). This is a **federal regulatory violation** in the US and could result in billions in fines.
*   **The Fix:** Multi-layered compliance:
    1.  **Prompt grounding:** Only provide FDA-approved indications from the KG. Never pass unapproved indications.
    2.  **Post-hoc NLI check:** Use a fine-tuned DeBERTa model to verify: "Does the generated text entail any indication NOT in the approved list?" → Block if yes.
    3.  **Human-in-the-loop:** All generated communications were shown to a compliance officer via the Streamlit UI before being flagged as "Approved for Use." The AI never sent anything autonomously.

### C. Deployment & MLOps Phase
*   **What you did:** Flask backend + Streamlit frontend, CI/CD via GitHub Actions, Kubeflow for ML workflow orchestration, MLflow for experiment tracking.
*   **The "Unusual" Issue:** **"The Cold-Start Doctor Problem."** New doctors had zero interaction history. The recommendation engine and the KG had no prescribing edges for them. GPT-4 was generating generic, unhelpful boilerplate messages.
*   **The Fix:** Implemented a **content-based fallback using doctor specialty similarity.** If a new doctor is an Immunologist in ZIP code 90210, retrieve aggregate prescribing patterns of all Immunologists in the same region from the KG. Use that population-level profile as a "warm start" prior for the recommendation engine. This is the Beta-Binomial conjugate prior from your DMM prep applied in practice!

---

## ⚔️ 3. FLIPKART ROUND-SPECIFIC GRINDING QUESTIONS

### 🔴 DDS (System Design)
1.  **"Flipkart wants to build a Knowledge Graph connecting Products → Categories → Sellers → Warehouses → Delivery Hubs. How would you architect this for real-time query serving?"**
    *   *Ans:* Neo4j for offline analytics and complex graph traversals (e.g., find all sellers connected by shared warehouses). Redis Graph or DGraph for real-time serving (sub-10ms latency for "given Product X, find nearest warehouse with stock"). Nightly sync from Neo4j to serving layer.
2.  **"How do you handle the 'entity resolution' problem when the same doctor appears as 'Dr. J. Smith', 'John Smith MD', and 'J Smith, Physician' in your Knowledge Graph?"**
    *   *Ans:* Entity Resolution pipeline: (1) TF-IDF + cosine similarity for candidate generation (blocking step). (2) Feature engineering: Jaro-Winkler similarity, token-set ratio, shared NPI number. (3) Random Forest classifier for match/no-match on feature pairs. (4) Transitive closure to merge all matched entities to one canonical node.

### 🔵 DMM (Mathematical Modeling)
1.  **"You used a Knowledge Graph for retrieval. Explain mathematically how Graph Embeddings (TransE or Node2Vec) could replace your Cypher-based retrieval and what the mathematical trade-offs are."**
    *   *Ans:* TransE models relationships: $h + r \approx t$ (head entity vector + relation vector ≈ tail entity vector). Scored by: $f(h,r,t) = -\|h + r - t\|$. Trade-off: Cypher is exact and interpretable (query returns exact edges). TransE is approximate but can generalize to unseen relationships (e.g., infer `Drug_X TREATS Condition_Y` even if that edge doesn't exist, based on latent structure).
2.  **"Derive the Personalized PageRank algorithm you could use on the KG to rank the most important topics for a specific doctor."**

### 🟢 HO (Hands-On)
1.  **"Write a Cypher query that finds all Doctors who prescribe Drug A AND have a specialty of 'Immunology' AND have NOT been contacted in the last 90 days."**
2.  **"Write a Python function that takes a doctor profile dict and a list of approved indications, and constructs a safe GPT-4 prompt with XML-tagged guardrails."**

### 🟡 HM (Hiring Manager)
1.  **"Pharma is a highly regulated industry. How did you balance the speed of GenAI innovation with the regulatory compliance requirements?"**
    *   *STAR:* **S:** Client wanted GPT-generated messages live within 4 weeks. **T:** I had to balance speed with FDA off-label risk. **A:** I proposed a phased rollout: Week 1-2 auto-generate with 100% human review. Week 3-4 analyze error rates and only auto-approve messages with confidence > 0.95 on the NLI compliance check. **R:** Launched on time, zero compliance violations in first 6 months, 60% reduction in rep message drafting time.
