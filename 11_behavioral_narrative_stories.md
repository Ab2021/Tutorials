# 🗣️ Behavioral & Narrative Stories: Abhishek Bhardwaj
### Interview Prep for Huge Solutions Architect ML/AI

---

> [!IMPORTANT]
> This document maps your 9+ years of experience into compelling narratives. The panel wants to hear **impact, leadership, and how you handle failure/ambiguity**. Memorize the metrics. Speak with confidence and directness (US style).

---

## 🌟 SECTION 1: THE CORE BEHAVIORAL QUESTIONS (STAR METHOD)

### 1. "Tell me about a time you designed something technically complex from scratch."
**The Story: LangGraph Agentic Data Scientist at Chubb**
*   **Situation:** At Chubb, our business analysts were spending 3+ days gathering requirements, writing complex SQL, exporting to Excel, creating charts, and drafting presentations for routine analytical requests. The data science team was becoming a bottleneck for basic business intelligence.
*   **Task:** I needed to design an autonomous AI system that could handle the entire natural language -> data -> analysis -> narrative pipeline without writing code.
*   **Action:** I architected an Agentic AI Data Scientist using LangGraph and GPT-4. I designed a multi-node StateGraph (planner, SQL generator, SQL executor, Python REPL for charts, synthesizer). Crucially, I implemented conditional routing so if the SQL execution failed, the agent would route back to the generator with the error message to self-correct. I also integrated RAG over historical reports so the agent's insights were grounded in past context.
*   **Result:** We reduced analytics turnaround time by **~60%** (from days to hours/minutes). This empowered analysts to focus on strategic insights rather than data wrangling, and earned me the Q1 2025 STAR Award.

### 2. "Tell me about a time you had to influence stakeholders without authority."
**The Story: Axtria MMM — Challenging the Client's Agency**
*   **Situation:** I was leading the Marketing Mix Modeling (MMM) for J&J Immunology. Our Bayesian model showed that TV spend had the highest ROI. However, the client's media agency strongly disagreed, pushing for paid social, and the CMO was torn. I was a consultant, not the decision-maker.
*   **Task:** I had to convince the CMO and the agency to trust our model over their gut feeling and existing agency models.
*   **Action:** Instead of just arguing math, I focused on interpretability and business logic. I didn't show them the PyMC MCMC traces. I showed them the **Pareto front** from our NSGA-II optimizer. I walked the CMO through the scenarios: "If we follow the agency's plan, here is the predicted revenue and the uncertainty bounds. If we shift 10% to TV, here is the new predicted revenue." I also explained *why* the model saw TV performing well (the adstock carryover effect).
*   **Result:** The CMO agreed to a controlled geo-test based on our recommendations. The test validated our model's predictions, leading to a broader adoption of our optimized budget allocation and a **~10% increase in revenue** for that brand.

### 3. "Describe a time you failed. What did you learn?"
**The Story: The First Iteration of the RAG Fraud System**
*   **Situation:** When we first built the RAG-based fraud detection system at Chubb, we were eager to deploy.
*   **Task/Action:** We used standard fixed-size chunking (e.g., 500 tokens) for the insurance policy documents and claims histories, and a basic similarity search (k=3) before passing it to the LLM.
*   **Failure:** In staging, the false positive rate was unacceptably high, and the LLM was hallucinating coverage details. We realized the fixed chunks were cutting crucial context in half (e.g., a policy exclusion clause separated from its conditions). The LLM was making decisions on incomplete sentences.
*   **Learning/Correction:** I learned that in specialized domains (like legal/insurance), data ingestion strategy is more important than the LLM itself. I stopped development, went back to the drawing board, and implemented **semantic chunking** using sentence-transformers, and added a re-ranking step. I also forced the LLM to output a confidence score.
*   **Result:** The second iteration was successful, significantly enhancing fraud detection accuracy and early-stage identification capabilities.

### 4. "Tell me about a time you led through ambiguity."
**The Story: Building the Pharma Rep GenAI System at Axtria**
*   **Situation:** In early 2023, "GenAI" was a buzzword. Axtria leadership wanted a POC to show pharma clients how GenAI could revolutionize sales rep communications, but there were no clear requirements, just an idea.
*   **Task:** Define the use case, choose the tech stack, and build a working POC quickly to demonstrate value.
*   **Action:** I led the technical strategy. I realized a simple chatbot wasn't enough; it needed domain knowledge. I architected a system combining GPT-4 with a **Neo4j knowledge graph** (linking doctors, drugs, and indications). I built the integration using LangChain, deployed a frontend using Streamlit/Flask, and set up a CI/CD pipeline using GitHub Actions to prove it could be productionized.
*   **Result:** The POC successfully generated highly personalized rep-to-doctor emails grounded in the graph data (preventing hallucinations). It became a key demo asset for the sales team.

### 5. "Describe a time you mentored or developed someone."
**The Story: Managing the EXL/CVS Health Team**
*   **Situation:** I joined EXL as a Business Analyst and eventually grew to manage the Data Science offshore team (4-5 members) dedicated to CVS Health/Aetna.
*   **Task:** I needed to upskill my team from running SQL queries and basic regressions to building production-grade ML pipelines on GCP.
*   **Action:** I instituted weekly technical deep-dives. When we transitioned the CLV prediction model to GCP Dataproc, I didn't just give them the architecture; I paired-programmed with junior analysts to teach them PySpark and distributed computing concepts. I focused on building their confidence to push back on ambiguous client requests.
*   **Result:** The team achieved **5/5 SLA ratings for 3 consecutive quarters**. More importantly, several analysts I mentored were promoted to Senior/Lead roles within the organization.

---

## 🎯 SECTION 2: HUGE-SPECIFIC BEHAVIORAL QUESTIONS

### 6. "Why do you want to leave Chubb? Why Huge?"
**Model Answer:**
"I’ve loved my time at Chubb—I’ve had the opportunity to build cutting-edge agentic workflows and RAG systems at scale. However, my passion lies at the intersection of advanced AI and marketing science. At Axtria, I saw how AI (MMM, Attribution) directly drives revenue and transforms brands. Huge is unique because it blends world-class creative and design with heavy-hitting technology for global brands like Google and Nike. I want to build 'Intelligent Experiences'—taking the agentic architectures I’m building now and applying them to consumer growth, personalization, and brand strategy, which is exactly what Huge India is scaling up to do."

### 7. "How does your insurance/pharma experience apply to Huge's clients (Nike, McDonald's)?"
**Model Answer:**
"The domains are different, but the mathematical architectures are identical.
*   At Chubb, I built fraud risk models using RAG to score claims. At Verizon, that exact same architecture is a **churn prediction and retention modeling** system.
*   At Axtria, I built Multi-Touch Attribution for pharma reps and HCPs. For McDonald's or Nike, that is **omnichannel attribution** across TikTok, Google, and in-store purchases.
*   The Agentic AI I built for insurance analysts is the same architecture Huge needs to build an **agentic brand planner** for Nike campaigns. The math transfers perfectly; I just need to learn the new data schemas, which I excel at."

### 8. "Huge is a design agency. You're a data scientist. How do you work with creative teams?"
**Model Answer:**
"I don't believe data and creativity are opposites; data informs the canvas. At EXL, I didn't just hand over a CSV of survival analysis probabilities; I built an interactive Streamlit application so healthcare providers could *visualize* survival curves and play with risk factors. At Axtria, the GPT-4 pharma rep system was fundamentally about generating *creative, personalized copy* that was constrained by compliance rules. I view my role as a Solutions Architect to build the tools and engines that allow the creative teams to do their best work faster, and at scale."

---

## 🗣️ SECTION 3: OPENING STATEMENT & THE CAREER NARRATIVE

### The 90-Second Opening Pitch (Memorize This)
> "I’m Abhishek. I’m a Senior Data Scientist and AI Engineering Leader with over 9 years of experience. My career has really spanned three distinct phases that I think perfectly align with this Solutions Architect role at Huge.
>
> First, at EXL and CVS Health, I built the foundation. I learned how to handle massive scale—processing 2 million customers using PySpark on GCP—and I learned how to lead teams, maintaining perfect SLA ratings for consecutive quarters.
>
> Second, I moved to Axtria, where I learned the business of Marketing Science. I built complex Bayesian Marketing Mix Models and Omnichannel Attribution systems for J&J, directly driving a 10% increase in revenue.
>
> Today, at Chubb, I’m focused entirely on the frontier of GenAI. I architect and deploy Agentic AI workflows and RAG systems using LangGraph, GPT-4, and Vector DBs, recently reducing analytics turnaround time by 60%.
>
> I'm excited about Huge because it’s the convergence of all three: taking cutting-edge GenAI architecture, applying it to marketing and brand problems, and doing it at scale for clients like Google and Nike."

### The Career Narrative Thread
*   **2016-2022 (EXL): The Scale & Leadership Foundation.** (Learned to build robust data pipelines, handle Big Data, and manage teams).
*   **2022-2024 (Axtria): The Marketing Science MBA.** (Learned how to map ML to revenue, MMM, Attribution, client management).
*   **2024-Present (Chubb): The GenAI Frontier.** (Mastering LLMs, RAG, Agentic architectures).
*   **Next (Huge): The Synthesis.** (Bringing Agentic AI to global marketing brands).

---

## 💰 SECTION 4: COMPENSATION & NEGOTIATION

*   **If asked your expectations early on:** "I'm currently focused on ensuring this role is the right mutual fit. Based on my research for a Solutions Architect ML/AI role at a premium global agency's GCC in Bangalore, I'm looking for a competitive package commensurate with my 9+ years of experience leading GenAI and Marketing Science initiatives. If we decide to move forward, I'm confident we can align on the exact numbers."
*   **What to ask the recruiter (when appropriate):**
    *   "How is the compensation structured between fixed base, variable bonus, and equity/RSUs (if applicable for Huge India)?"
    *   "Does Huge India offer specific budgets for continuous learning, cloud certifications (like GCP PDE/PMLE), or attending AI conferences?"

---

## ❓ SECTION 5: QUESTIONS TO ASK THE INTERVIEWERS

*Pick 2-3 based on who you are talking to.*

**For Engineering Leadership:**
1. "The JD mentions a focus on GCP (Vertex AI) but also AWS/Azure. Is Huge India standardizing on GCP for new client builds, or is it strictly multi-cloud depending on the client's existing stack?"
2. "For your Agentic workflows, are you leaning towards orchestrators like LangGraph, or are you exploring fully managed services like Vertex AI Agent Builder?"

**For Product/Business Leadership:**
3. "How does Huge measure the ROI of 'Intelligent Experiences' for clients like Nike or McDonald's? Is it purely efficiency, or are you driving net-new revenue streams?"
4. "When onboarding a new Fortune 500 client to the AI platform, what is typically the biggest bottleneck—data quality, compliance/security, or stakeholder alignment?"

**For the Hiring Manager:**
5. "What does success look like for this Solutions Architect role in the first 90 days? What is the first major client or architectural challenge I would tackle?"
