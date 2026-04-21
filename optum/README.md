# 🏆 Optum Sr. AI/ML Engineer — Interview Preparation Hub
### Master Index & Quick Navigation

---

## 📁 FILE INDEX

| # | File | Purpose | When to Read |
|---|---|---|---|
| **00** | [00_MASTER_QUICK_REFERENCE.md](./00_MASTER_QUICK_REFERENCE.md) | Cheat sheets: numbers, formulas, patterns, 60-sec answers | **Interview morning — print this** |
| **01** | [01_Optum_Interview_Strategy_and_Process.md](./01_Optum_Interview_Strategy_and_Process.md) | Process overview, competitive edge, 72-hr checklist, values, code patterns | Day 1 of prep |
| **02** | [02_Domain_Round_QA_Deep_Dive.md](./02_Domain_Round_QA_Deep_Dive.md) | 12 expected Q&As with model answers (GenAI, ML, Behavioral) | Day 1-2 of prep |
| **03** | [03_System_Design_Healthcare_AI.md](./03_System_Design_Healthcare_AI.md) | 4 complete healthcare AI system designs with evaluation & RAI | Day 2 of prep |
| **04** | [04_Resume_Deep_Dive_Optum_Context.md](./04_Resume_Deep_Dive_Optum_Context.md) | Every resume project mapped to JD, with counter-Q answers | Day 2-3 of prep |
| **05** | [05_LLM_Security_Responsible_AI_Deep_Dive.md](./05_LLM_Security_Responsible_AI_Deep_Dive.md) | 6 threat categories, RAI pillars, Bedrock Guardrails, evaluation checklist | Day 2 of prep |

---

## 🎯 JD REQUIREMENTS → YOUR PROOF POINTS

| JD Requirement | Your Experience | Key Story |
|---|---|---|
| **LangChain + LangGraph** | LangChain in production (Agentic BI + RAG fraud); LangGraph studied deeply | Agentic BI Tool at Chubb |
| **RAG pipelines** | Production RAG system, RAGAS evaluated, faithfulness 0.88 | Fraud Detection at Chubb |
| **Multi-Agentic workflows** | Autonomous agent with tools, trajectory evaluation, 87% task completion | Agentic BI Tool at Chubb |
| **AWS Bedrock** | AWS Lambda + SageMaker (production); Bedrock architecture deeply studied | Cloud-native section of interviews |
| **Google Vertex AI** | Vertex AI, Dataproc, BigQuery — extensive production use | CLV model at EXL/Aetna |
| **LLM Security & Guardrails** | Faithfulness guardrails, structured output, shadow-mode HITL | Fraud system at Chubb |
| **Responsible AI** | Subgroup analysis, SHAP explainability, fairness audits | Readmission model at EXL/Aetna |
| **Classification + NLP** | BERT, ClinicalBERT, BERT fine-tuning on 50K clinical notes | Patient readmission at EXL/Aetna |
| **End-to-end ML pipelines** | MLflow + Kubeflow + Airflow + Docker/K8s — full stack | All production deployments |
| **Healthcare domain** | 6 years at EXL/CVS Health/Aetna + Chubb insurance | Native domain expert |
| **Mentor junior engineers** | Led 4-5 person DS teams; 5/5 SLA at EXL for 3 quarters | Team leadership stories |

---

## ⚡ 5-MINUTE INTERVIEW MORNING REVIEW

### Your 3 Power Stories (Pick Situation-Appropriate):

**Story A: The RAG Fraud System** *(Use for: LLMs, RAG, production AI, evaluation, guardrails)*
> "4-layer system: BERT IE → domain embeddings (+18% Precision@5) → RAG retrieval → GPT-4 structured output. RAGAS: faithfulness 0.88. Shadow mode 60 days → 78% recall / 22% FPR. Guardrails: structured output schema + faithfulness gate + PHI redaction."

**Story B: The Agentic BI Tool** *(Use for: LangChain, multi-agent, evaluation of agents)*
> "LangChain agent with 3 tools: SQL generator, Python executor, chart generator. Evaluated on 4 dimensions: task completion (87%), tool accuracy, answer correctness, safety (0 failures on adversarial inputs). Cut analyst reporting time 60%."

**Story C: Patient Readmission (BERT + Clinical ML)** *(Use for: healthcare domain, NLP, ML fundamentals)*
> "AUC 0.76 → 0.82 (XGBoost + TF-IDF) → 0.89 (ClinicalBERT embeddings, 50K fine-tune). Brier score 0.08 — well calibrated. Subgroup analysis: flagged AUC gap for age >80, addressed with group-specific calibration."

---

### The Most Likely Domain Round Question Flow:

```
1. "Tell me about yourself" → 60-second intro (see Quick Reference)
2. "Walk me through your most relevant project" → Story A (RAG Fraud)
3. "How did you evaluate it?" → 4-layer RAGAS framework
4. "How did you handle hallucinations?" → Faithfulness guardrail + structured output
5. "Have you worked with LangGraph?" → Know the distinction vs LangChain + state machine
6. "Design a [healthcare AI system]" → 8-step framework (see System Design doc)
7. "How would you handle LLM security in healthcare?" → 6 threat categories
8. "Tell me about a time you..." → Story B or C with STAR+ format
9. "Why Optum?" → Prepared answer (see Strategy doc)
10. "Questions for us?" → 5 prepared questions (see Strategy doc)
```

---

### Emergency Cheat: "I Haven't Used X" Responses

| If asked about... | Say this |
|---|---|
| **LangGraph (no prod exp)** | "I know LangChain deeply in production. LangGraph's state machine model I've studied carefully — it maps directly to the multi-agent patterns I've built manually. Here's how I'd use it for [their use case]..." |
| **AWS Bedrock specifically** | "I've deployed on SageMaker and designed around Lambda for real-time ML. Bedrock extends that — I've studied its Knowledge Bases, Agents, and Guardrails features extensively, and they align directly with the architecture decisions I'd make." |
| **Azure AI Foundry** | "My cloud experience is primarily AWS and GCP in production. Azure AI Foundry is their enterprise GenAI platform — I understand the service architecture and how it integrates with Microsoft's ecosystem. I'd ramp up quickly." |
| **A paper/technique you haven't read** | "I'm not familiar with that specific paper, but from the broader approach to [topic]... [reason from first principles]. Is that the direction they took?" |

---

## 🚀 THE WINNING MINDSET FOR DOMAIN ROUND

> **You are not a generic AI/ML candidate.**
>
> You have **built production RAG + LLM + agentic systems in healthcare/insurance contexts** — exactly what Optum is building.
>
> You have **9+ years of healthcare domain knowledge** — claims, clinical notes, HIPAA, PHI, ICD codes, prior auth — you don't need to learn the domain, you bring it.
>
> The interviewer is asking "Can you build what's in this JD?" Your answer, backed by Chubb + EXL/Aetna experience, is: **"I already have."**
>
> Lead with production. Lead with healthcare. Lead with evaluation. Lead with Responsible AI. Those four things together are your moat.

---

*Optum Interview Preparation Kit — Created April 2026*
*Good luck. You've built exactly what they need.*
