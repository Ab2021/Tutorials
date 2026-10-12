# 01 — JD DECODE & ROLE FIT

> **TL;DR.** Brit is a Lloyd's specialty insurer running a *Smarter Underwriting* and *Operational Simplification* agenda on an Azure-first, data-modernisation platform. The Lead AI Engineer is **hands-on + people-manager**, spanning classical ML, GenAI/agentic, **and low-code** (Copilot Studio, Power Platform, Cytora). They are not hiring a researcher — they are hiring someone who turns ambiguous business problems into *production AI that a regulated insurer can defend*. The two things the panel says are missing from profiles are **service design + stakeholder integration** and **mentoring a team** — treat both as scored, not soft.

**60-second interview answer:** *"Brit's JD is a delivery role, not a research role. Three things in it tell me what they're testing. First, 'ambiguous business problem to production operation with measurable business impact' — they want an end-to-end track record with numbers. Second, the pro-code-or-low-code decision — they want pragmatic build-or-buy judgement, not tool loyalty. Third, 'embed security, privacy, data governance, model risk, explainability, human oversight' — in a Lloyd's insurer, every AI output that touches underwriting is a regulated artefact. And the panel feedback says what's missing from most candidates: service design and stakeholder work, and mentoring engineers. So I'd expect this round to test architecture, judgement, and leadership in roughly equal measure."*

---

## 1. The company — what you must know before you speak

| Fact | Why it matters in the interview |
|---|---|
| Brit is a **specialty insurer and reinsurer at Lloyd's of London**, part of **Fairfax Financial** | Lloyd's is a *market*, not a company — you sell capacity through syndicates and brokers. Vocabulary differs from retail insurance. |
| **Data modernisation is their flagship claim** — "central, trusted, cost-efficient, flexible, scalable cloud-native data platform" | They have already invested in data plumbing. The AI role is the *layer on top*. Do not propose re-plumbing. |
| **Broker APIs** — "multi-class-of-business headless API" connecting Brit's pricing and document generation to broker platforms, market hubs, enabling **bulk submission ingestion** | Their ingestion problem is *broker-driven and semi-structured*. This is where document AI and extraction live. |
| **Underwriting Transformation** — pricing + **underwriter workbench**, "engineering, upskilling in python, exploring ingestion and analytics" | They are **upskilling a traditional underwriting workforce in Python**. Mentoring is a real job, not an HR line. |
| **Proprietary ML for post-catastrophe property damage** from ultra-high-resolution imagery — "first enabler for virtual claims adjusting in the London market" | This is *their* applied-AI showpiece, and it is **computer vision**. The candidate-feedback question "run images through a model for checking anomalies" is a direct echo of this. Expect it. |
| **"Continuous learning & exploration… best-in-class technology & tooling"** | Safe to talk modern tooling (vLLM, LangGraph, etc.) as long as you translate it to a business outcome. |

**Azure-first signal:** the JD names **Microsoft Azure AI services, Copilot Studio, Power Platform/Power Apps, Cytora**. This is a Microsoft-shop insurer. Lead with Azure-native where it is genuinely the right answer; bring open-source/self-hosted only with a stated reason (cost, latency, data residency, vendor lock-in).

---

## 2. The JD teardown — every accountability turned into what they will actually probe

Read the left column as the JD, the middle as the *question they will ask*, and the right as the *evidence you must have ready*.

### 2.1 Principal accountabilities

| JD says | What they will probe | Evidence you must have loaded |
|---|---|---|
| "Design, build **and deploy** applications using statistical and ML methods, LLMs, RAG, agentic workflows and automation, while contributing **production-quality code, APIs, integrations**" | "Walk me through something you personally built and shipped." They will expect *code-level* detail, not architecture-only. | One project where you wrote the service, not just the design. Know your own repo. |
| "Lead discovery, solution design, prototyping, build, evaluation, testing, production deployment, **change adoption and benefits review**, making pragmatic **build-or-buy and pro-code or low-code decisions**" | "When would you *not* build? How do you decide between pro-code and Copilot Studio?" | A real build-vs-buy call you made, with the reasoning and the outcome. |
| "Set standards for **architecture, code review, testing, CI/CD, MLOps/LLMOps, observability, reliability, cost and performance**" | "What standards did you set as a lead? Show me a code-review or CI rule you introduced." | 2–3 concrete standards you authored (eval gate in CI, PR template, golden-set regression, cost budget alert). |
| "Embed **security, privacy, data governance, model risk, explainability, human oversight and regulatory controls** throughout the lifecycle, **documenting decisions, limitations and operational controls**" | "How do you handle PII? What does 'model risk' mean in your world? Who approves a model going live?" | The phrase **"model card / decision log / limitations register"**. Know human-in-the-loop design for regulated outputs. |
| "**Manage, coach and develop a team** across AI Solution Engineering, AI/ML Engineering, Applied AI Science and AI Data Science" | "How big was your team? How did you develop someone? How did you handle an underperformer?" | STAR stories: a mentorship that changed someone's trajectory; a performance conversation; a conflict you resolved. |
| "**Translate complex AI concepts into clear business choices**, facilitate cross-functional decisions, and communicate progress, risks, trade-offs and realised outcomes to technical and non-technical audiences" | "Explain X to a non-technical underwriter." | Practise explaining embeddings, RAG, and precision/recall in *insurance language*. |

### 2.2 Technical requirements as a self-audit

Tick each line, and for every tick write the one sentence you would say as evidence. **Any line you cannot tick in 10 seconds is a gap to close this week.**

- [ ] Classical statistics, supervised & unsupervised ML, feature engineering, experimentation, model evaluation
- [ ] Generative AI: **prompt design**, **RAG**, **embeddings / vector search**, **tool / function calling**, **orchestration**, **structured outputs**, **failure handling**
- [ ] End-to-end delivery "from ambiguous business problem to production operation" with **measurable business impact and sustained adoption**
- [ ] **Technical leadership** + leading people/projects/multidisciplinary teams
- [ ] **Regulated, data-sensitive environment** (insurance/reinsurance/London Market advantageous)
- [ ] **Low-code**: Azure AI services, Copilot Studio, Power Platform/Power Apps, Cytora or comparable
- [ ] **Translation** between scientific and business language

> Note the four bolded GenAI items — *prompt design, RAG, embeddings/vector search, tool calling, orchestration, structured outputs, failure handling*. That is a precise, checkable list. The feedback questions ("Explain RAG architecture in detail with code") map straight onto it. Files `04`, `05`, `06` are built around exactly these.

### 2.3 People leadership — the JD spends more words here than on any technical topic

Re-read the "People leadership and ways of working" block. Counted, it asks for: **building a community of practice**, **goal-setting and work allocation across teams**, **one-to-ones and performance reviews**, **mentoring via pairing and technical reviews**, **handling delivery/behavioural concerns and conflict**, **psychological safety and inclusion**, and **workforce planning, recruitment, onboarding, succession**.

That is a *people-manager* job description. Prepare stories for at least **six** of those eight. See file `12`.

---

## 3. Decoding the candidate feedback — the actual question set

The candidate feedback file lists eight questions. Here is what each is *really* testing, and which file answers it.

| # | Asked | Really testing | File |
|---|---|---|---|
| 1 | "Which components of Azure AI you worked on?" | Are you actually hands-on in *their* stack, or will you need 6 months of ramp? | 11 |
| 2 | "Explain RAG architecture **in detail with code**" | Can you go from diagram to implementation — chunking, retrieval, rerank, prompt assembly? | 05, 06 |
| 3 | "How big of a team you worked on?" | Do you operate at Lead scope (people, not just tickets)? | 12 |
| 4 | "How did you evaluate your agents?" | The single most discriminating Applied-AI question. Trajectory eval, not just output eval. | 07 |
| 5 | "How can you have **linear commit history** in Git with multiple people in a team?" | Do you know real team engineering practice, or only solo work? | 06 §4 |
| 6 | "Draw a system design diagram — images run through a model for **anomaly checking**" | Their own post-catastrophe CV use case. Vision + serving + HITL + accuracy guarantees. | 05 §6 |
| 7 | "Here's RAG code — explain it and what improvement would you offer" | Code *reading* and *critique* under time pressure. | 06 §3 |
| 8 | "Drill down into your latest project" | Depth, ownership, and production maturity. | 04 |

**Notice the pattern:** there is not a single question about training models or maths. Every question is about *engineering, evaluation, integration, and team practice*. That confirms the framing in the pasted prompt: this is an **Applied AI** interview.

---

## 4. Decoding the panel feedback — the two gaps (treat as scored)

> *"The candidates should have experience and skills with doing the below in a **Product Development environment**: (1) Broader **service design** considerations, **business integration**, and working with **stakeholders and teams**. (2) **Technical and People Mentoring** for a team of data scientists, AI engineers, or ML engineers."*

| Gap | What "good" sounds like | What "bad" sounds like |
|---|---|---|
| **Service design & business integration** | "The model was 4% of the project. The other 96% was: who owns the decision, what happens when the model is wrong, which downstream system consumes the output, how the underwriter's workflow changes, and what we measure six months after go-live." | "We built a RAG pipeline that got 92% accuracy." |
| **Mentoring** | "I ran weekly pairing sessions; I moved one engineer from notebooks to owning a production service in two quarters; here's the code-review practice I introduced and why." | "I mentored juniors." |

**Insert these two sentences into your arsenal.** They are the highest-leverage lines in the whole pack:

- **Service design:** *"I think about the AI as one component of a service. The questions that decide whether it succeeds are: who acts on the output, what is the cost of a false positive versus a false negative, which system of record does it write back to, and what does the human do differently on Monday morning?"*
- **Mentoring:** *"My mentoring rule is that nobody stays in notebooks. Within a quarter, every engineer on my team has shipped something with a monitoring dashboard and an owner's name on it."*

---

## 5. What Brit will care about that a big-tech interview would not

| Dimension | Big-tech framing | **Brit framing (use this)** |
|---|---|---|
| Success metric | accuracy, F1, benchmark | **cost of error**, combined ratio impact, underwriting time saved, submissions triaged per day |
| Latency | P99 ms | *underwriter tolerance* — a 10-second extraction is fine; a 10-second chat reply is not |
| Explainability | nice-to-have | **mandatory** — an underwriter must be able to justify a decision to a regulator |
| Autonomy | full automation | **human-in-the-loop by default** for anything that binds risk or pays a claim |
| Build vs buy | build | **buy low-code unless it's a differentiator** — Cytora/Copilot Studio exist so you don't rebuild them |
| Failure | retry silently | **fail loudly, log the decision, degrade to a human** |

---

## 6. Your positioning statement — pick the version that is true and write it out

Fill this in *tonight* and say it out loud ten times. One paragraph, ~20 seconds.

**Template:**

> *"I'm a **[N]-year** engineer who moved from **[classical ML / data science / backend]** into applied GenAI. For the last **[X]** years I've led **[team size]** engineers and scientists building **[domain]** AI — most recently **[one-line project]**, which **[measurable outcome]**. What I'd bring to Brit specifically is **[pick two: regulated-domain delivery / RAG + agentic production experience / evaluation and guardrails discipline / Azure + low-code pragmatism / building a team's engineering standards]**. And the part of the JD I'm most interested in is **[Smarter Underwriting / Operational Simplification]** because **[domain reason]**."*

**Worked example (adapt, do not copy):**

> *"I'm a 12-year engineer who moved from clinical data science into applied GenAI. For the last three years I've led a team of six engineers and scientists building document-AI and RAG systems in a regulated healthcare payer — most recently a prior-authorisation intake pipeline that cut manual review time by 40% and moved 30% of submissions to auto-adjudication. What I'd bring to Brit is regulated-domain delivery — I've had to make AI explainable to auditors and design human-in-the-loop for every decision with a cost attached — plus the evaluation discipline that comes with that. And I'm most interested in the Smarter Underwriting side, because submission triage is the same problem shape as clinical intake: semi-structured documents, high cost of a wrong extraction, and a human who must stay accountable."*

---

## 7. Questions *you* should ask them (pick 3 — it signals seniority)

1. "Where does the AI team sit relative to the underwriters — is there a shared roadmap, or does AI take requirements?"
2. "Of the Smarter Underwriting portfolio, which use case is furthest into production, and what has been the hardest part — the model, or the change adoption?"
3. "How do you currently handle model risk and sign-off for an AI-assisted underwriting decision — is there a model governance forum?"
4. "You mentioned pro-code and low-code. Where has low-code genuinely won, and where did you have to fall back to code?"
5. "For the India GCC team I'd lead — is the mandate delivery capacity for UK, or capability-building as well?"
6. "What does 'sustained adoption' look like for you six months after launch — what metric tells you it worked?"

---

## 8. Common mistakes for this specific role

| Mistake | Fix |
|---|---|
| Pitching frontier-model research or custom training | This JD is about **deploying and integrating**. Lead with orchestration, evaluation, integration. |
| Ignoring the low-code half of the JD | Name Copilot Studio / Power Platform / Cytora and give a rule for when low-code wins. |
| Using retail/tech vocabulary | Learn 25 Lloyd's terms (file 11). Say "submission", "binding", "slip", not "customer ticket". |
| Treating the two panel gaps as optional | They are the *stated* reason profiles are rejected. Budget a third of prep here. |
| Only talking architecture | The feedback asks for **code**, **commits**, and **team**. Be ready to go concrete. |

---

## 9. Cheat-sheet summary

- **Brit = Lloyd's specialty insurer, Azure-first, Fairfax-owned, "Smarter Underwriting" + "Operational Simplification".**
- **Role = hands-on Lead + people manager**, spanning classical ML + GenAI/agentic + low-code.
- **Round = 4 parts**: intro, project drill-down, system design, coding exploration.
- **Two stated gaps to close on camera:** service design/business integration; technical + people mentoring.
- **Their showpiece = computer-vision post-catastrophe claims**; expect an image-anomaly design.
- **Metric language = cost of error**, not accuracy.
- **Positioning sentence must end with a Brit-specific reason.**

## 10. Cross-links

- Next: [`02-stage1-format-60-minute-battle-plan.md`](02-stage1-format-60-minute-battle-plan.md)
- Domain vocabulary: [`11-insurance-lloyds-domain-primer.md`](11-insurance-lloyds-domain-primer.md)
- The leadership gap: [`12-people-leadership-and-stakeholder-round.md`](12-people-leadership-and-stakeholder-round.md)
- Index: [`00-README-and-7-day-plan.md`](00-README-and-7-day-plan.md)
