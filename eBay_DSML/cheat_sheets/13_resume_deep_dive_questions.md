# eBay DS/ML Interview — Resume Deep Dive Questionnaire
**Candidate:** Abhishek Bhardwaj
**Focus:** Bridging your specific experience (Chubb, Axtria, EXL) with eBay's technical and business requirements.

In senior DS/ML interviews (especially the deepest technical rounds and the Bar-Raiser round), interviewers will grill you on the projects listed on your resume. You must be able to defend your technical choices, explain trade-offs, and map your past work to eBay's scale.

Here are 25 highly probable, challenging questions interviewers will ask you based *directly* on your resume. 

---

## 🏗️ 1. ML System Design & Scale (Based on Resume)

**Your Experience:** Fraud Detection (RAG+LLMs), CLV at Scale (PySpark), Marketing Mix Modeling.

1. **Fraud Detection Scaling:** "You architected a fraud detection system at Chubb using RAG and LLMs. Walk me through the system architecture. If we were to adapt this for eBay to identify counterfeit luxury watches among 1.9 billion listings, what would break, and how would you redesign it for our scale?"
2. **Batch vs. Real-Time:** "You mentioned implementing both batch and real-time processing for insurance risk scoring. How did you ensure feature consistency between the real-time stream and the offline batch jobs? (Hint: They are looking for 'Feature Store' concepts like NuKV)."
3. **CLV at Scale:** "At EXL/CVS, you developed a distributed Random Forest model using PySpark to predict CLV for 2M+ prospects. Why Random Forest instead of a deep learning approach like LSTMs for customer lifetime value? How did you handle the data skewness across nodes in Databricks/Dataproc?"
4. **Omnichannel Attribution:** "You built a multi-touch attribution model using Markov Chains at Axtria. How does the Markov approach compare to Shapley value attribution? How would you adapt this model for eBay's marketing spend across Google Ads, Facebook, and Affiliates?"
5. **Entity Matching:** "You designed a string matching algorithm using TF-IDF and cosine similarity for Plan Sponsors. Sparse embeddings like TF-IDF can be slow at scale. Tell me how you optimized the Spark MLlib pipeline. Why didn't you use dense neural embeddings (like BERT) for this?"

---

## 🧠 2. Generative AI, RAG & NLP (Based on Resume)

**Your Experience:** Agentic BI, Claims Summarization, GPT-4 + Neo4j (Knowledge Graphs).

6. **Agentic Workflows:** "You built an 'Agentic BI Tool' using LangChain. Walk me through how the agent decides which tools to use. How did you handle situations where the agent got stuck in a loop or hallucinated a completely wrong SQL query?"
7. **RAG vs. Fine-tuning:** "For your insurance fraud detection, you chose RAG. At what point would you have decided to fine-tune a model instead? If eBay wanted an internal bot to answer questions about complex seller policies, would you use RAG, Fine-tuning, or both? Why?"
8. **Knowledge Graphs + GenAI:** "You integrated GPT-4 with a Neo4j Knowledge Graph for pharma rep communications. Explain how the graph database interacted with the LLM prompt. How could eBay leverage a user-item-seller knowledge graph to improve our 'Best Match' search algorithm?"
9. **Evaluating Summarization:** "You developed a Claims Summarization tool using extractive/abstractive techniques. Summarization is notoriously hard to evaluate. What metrics did you use (ROUGE, BERTScore, LLM-as-a-judge), and how did you convince the underwriters to trust the summaries?"
10. **Information Extraction:** "You used BERT-based models for information extraction from unstructured clinical notes. Clinical notes are highly messy and domain-specific. Did you train a custom head on ClinicalBERT, or start from scratch? How did you handle the token limit of BERT (512 tokens) for very long clinical documents?"

---

## 📈 3. ML Theory & Optimization (Based on Resume)

**Your Experience:** Bayesian Optimization, Ensemble Methods, Survival Analysis.

11. **Bayesian Optimization:** "You used Bayesian optimization for hyperparameter tuning, improving model accuracy by 15%. Explain how Bayesian optimization differs from Grid Search or Random Search mathematically. What surrogate model (e.g., Gaussian Process, TPE) did you use?"
12. **Survival Analysis:** "You implemented Kaplan-Meier and Cox proportional hazards models for customer attrition. How is a Cox model fundamentally different from a standard Logistic Regression churn model? How did you handle right-censored data (customers who hadn't churned yet)?"
13. **Custom Genetic Algorithm:** "You designed a custom genetic algorithm for multi-objective optimization of budget allocation. What were your fitness functions, crossover, and mutation strategies? Why a genetic algorithm instead of linear programming?"
14. **Explainability (SHAP):** "You combined SHAP values and permutation importance. Why use both? Give me an example of a time when SHAP values revealed an intuitive finding vs. a counter-intuitive finding in your clinical or insurance models."
15. **Tree Ensembles:** "You used Random Forest and XGBoost for your Marketing Mix Models. What is the fundamental difference in how they reduce error (bias vs. variance)? In what scenario in your pharma project did XGBoost outperform RF?"

---

## 💼 4. Product Sense & Business Impact (Based on Resume)

**Your Experience:** Improving patient outcomes, increasing revenue by ~10%, automating workflows.

16. **Trade-offs in Fraud Detection:** "In your Chubb fraud model, what was the business cost of a False Positive (flagging a legit claim) versus a False Negative (missing fraud)? How did you set the decision threshold based on monetary impact rather than just F1-score?"
17. **Measuring Causal Impact:** "You mention achieving a 10% increased revenue through optimized marketing spend. Correlation is not causation. How did you *prove* to stakeholders that the 10% increase was directly caused by your model and not just seasonal trends or competitor actions? (Did you use A/B testing, Marketing Lift tests, synthetic controls?)"
18. **Metric Design:** "For the Agentic BI Tool you built, what was your North Star metric? What were your guardrail metrics to ensure analysts weren't blindly trusting incorrect reports?"
19. **Deploy vs. Discard:** "Have you ever built a model, like the readmission risk deep learning model (AUC 0.89), that performed great offline but failed when tested online or was rejected by the business? What happened?"
20. **Marketplace Translation:** "A lot of your background is B2B (Healthcare, Insurance, Pharma). eBay is a massive B2C/C2C two-sided marketplace. How do you think the product metrics and user behavior will differ from what you are used to?"

---

## 🤝 5. Behavioral & Leadership (STAR Method)

**Your Experience:** Managing cross-functional teams, driving innovation POCs, delivering under SLAs.

21. **Disagreement with Stakeholders:** "At Axtria, you worked on Marketing Mix Modeling. Tell me about a time your model explicitly recommended cutting budget from a channel that the Marketing VP loved. How did you handle that conversation?"
22. **Leading a Team:** "You led a DS offshore team of 3 FTEs at EXL with a perfect 5/5 SLA rating. Tell me about a time a team member was falling behind or a critical pipeline broke just before a deadline. How did you manage it?"
23. **Taking Initiative:** "The 'Claims Crosswalk Initiative' and 'Agentic BI' look like innovation/POC projects. Tell me about a time you identified a business problem nobody else saw and built a solution from scratch without being explicitly asked."
24. **Communicating Complexity:** "Graph databases (Neo4j), Markov Chains, and Agentic workflows are highly technical. Walk me through exactly how you explained the ROI of the 'Pharma Rep GenAI' system to a non-technical pharmaceutical executive."
25. **Handling Failure:** "You have a stellar track record of awards from 2016 through 2025. Tell me about your biggest failure or a project that completely derailed. What ownership did you take, and what did you learn?"

---

## 💡 Pro-Tip for Abhishek:
Your resume is **incredibly strong in GenAI and modern ML architectures**, which aligns perfectly with eBay's current push. However, since you don't have direct e-commerce marketplace experience, **you must proactively bridge the gap.** 
Whenever you answer these questions, say: *"At Chubb, I solved this by doing X. If I were doing this at eBay for [Search / Fraud / Recommendations], I would apply the same architecture but adjust Y to account for network effects/scale."*
