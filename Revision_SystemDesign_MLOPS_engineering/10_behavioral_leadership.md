# BEHAVIORAL & LEADERSHIP QUESTIONS — Lead Role Mastery
> Lead roles require proving you can LEAD, not just build. This file covers the STAR-based behavioral framework.

---

## THE STAR METHOD (Mandatory for Lead Roles)

Every behavioral answer must follow this structure:
```
S - Situation: Set the context (1-2 sentences)
T - Task:      What was your specific responsibility? (1 sentence)
A - Action:    What did YOU specifically do? (3-5 sentences — most important)
R - Result:    What was the measurable outcome? (1-2 sentences with numbers)
```

**Critical for Lead roles:** In A (Action), use "I" not "we". Show YOUR decisions, YOUR judgment calls, YOUR leadership — not team's.

---

## SECTION 1: LEADERSHIP SCENARIOS

### "Tell me about a time you led a technical project end-to-end"

> **S:** At Axtria, a pharma client needed an omnichannel attribution model to justify their $50M marketing budget allocation.
>
> **T:** I was the lead data scientist responsible for the full technical solution — from problem scoping through production delivery.
>
> **A:** I started by defining the evaluation framework FIRST — before any modeling. I worked with the client to identify the primary metric: incremental ROI per marketing channel. This anchored all technical decisions.
>
> I then ran a model selection sprint: linear regression baseline (interpretable, fast, clear coefficients), then Markov chain attribution for multi-touch journeys. I built an A/B comparison showing both models and presented tradeoffs — interpretability vs accuracy.
>
> I made the architectural decision to use Markov chain as primary with regression as a validation check. I mentored two junior analysts on the implementation and reviewed their code weekly. I designed the presentation framework for client stakeholders (non-technical CFO team).
>
> **R:** Model went live in 14 weeks. Client reallocated 20% of budget away from low-ROI channels based on our attribution. This generated an estimated 18% improvement in marketing ROI, as validated in the following quarter's actuals vs model predictions.

---

### "Tell me about a time you made a technical decision with incomplete information"

> **S:** At Chubb, we needed to decide between two approaches for NLP fraud flag extraction: fine-tuning GPT-3.5 vs using GPT-4o with RAG. We had limited time to evaluate.
>
> **T:** I had to make the architecture decision within one week, without the luxury of a full comparative study.
>
> **A:** I structured a rapid evaluation: built both prototypes in 3 days — fine-tuning on 200 labeled examples vs RAG with same 200 as knowledge base. Tested both on a 50-case held-out set annotated by our senior underwriter.
>
> I evaluated three dimensions: precision/recall of flag extraction, cost per document (fine-tuning had one-time GPU cost vs RAG's API cost-per-call), and most importantly — adaptability to new fraud patterns. I documented the tradeoffs in a decision memo with recommendation: RAG, because fraud patterns change monthly and RAG allows knowledge base updates without retraining.
>
> I presented this to the team and got sign-off within 2 days.
>
> **R:** RAG was implemented. 8 months later, this proved correct: we updated the knowledge base 3 times as new fraud schemes emerged, with no model retraining cost. If we'd fine-tuned, we'd have needed $5K+ GPU hours each time.

---

### "Tell me about a time you pushed back on a stakeholder"

> **S:** At EXL, the business team wanted to increase the patient readmission model's recall from 60% to 95% — meaning flag 95% of all eventual readmissions.
>
> **T:** I needed to explain why this would break the operational program without being dismissive of the business goal.
>
> **A:** I ran the numbers and showed what 95% recall would mean: at our 20% readmission rate, the threshold to achieve 95% recall would flag 90% of all patients for outreach. Care coordinators could handle 30 patients per week. We had 10,000 patients. At 90% flag rate = 9,000 patients needing outreach per week — 300x capacity.
>
> I then reframed the conversation: instead of "how do we maximize recall", I proposed "let's find the recall level that fits care coordinator capacity while maximizing value per outreach." I suggested targeting Recall@40 — flag the 40% highest-risk patients, which corresponds to ~70% of eventual readmissions.
>
> **R:** Business accepted the reframed objective. The 40% threshold flagged the top-risk patients and care coordinator capacity was fully utilized. The program showed 18% reduction in readmission for the high-risk tier — more impactful than a diluted program covering everyone.

---

### "Tell me about a time something you built failed in production"

> **S:** At Chubb, our real-time fraud scorer had a memory leak in the FastAPI container that caused pod crashes every 72 hours in production.
>
> **T:** I was responsible for the technical investigation and fix.
>
> **A:** I identified the issue using Kubernetes pod memory metrics (Grafana): memory grew linearly with time, peaking at ~1.8GB before OOM kill. I added memory profiling to the FastAPI app using `tracemalloc` — found that SHAP explainer objects were being re-initialized on every request and not garbage collected due to a circular reference.
>
> Fix: initialized the SHAP explainer at pod startup (like the model) rather than per-request. Tested fix in staging for 5 days — memory stabilized at 0.6GB constant.
>
> I also added a memory usage metric to our Grafana dashboard and set an alert at 1.2GB (before the 1.8GB crash point) so we'd detect recurrence early.
>
> **R:** Memory leak eliminated. Pod uptime went from 72-hour cycles to 30+ days without restart. Added the memory probe pattern to our MLOps runbook so future APIs avoid the same issue.

---

## SECTION 2: LEAD-SPECIFIC QUESTIONS

### "How do you balance technical quality with delivery timelines?"

> "I use a 70-20-10 framework:
>
> 70% of effort goes to production-ready quality — unit tests, integration tests, monitoring, documentation. Non-negotiable.
>
> 20% goes to technical excellence — SHAP explanations, calibration analysis, sophisticated feature engineering. These matter but can be phased.
>
> 10% is exploration — new methods, experimental features. Time-boxed and done only if core is complete.
>
> When timeline pressure hits, I first cut the 10% exploration, then negotiate on 20% technical excellence features, but never compromise on the 70% production quality. Deploying untested code creates more timeline risk than taking extra time for quality upfront."

---

### "How do you handle a team member who is underperforming?"

> "First: I try to understand root cause before assuming performance issue. Is it capability, clarity, or motivation?
>
> Step 1: Private conversation within the first week I notice the issue. 'I've noticed [specific observation]. What's been challenging for you? Is there anything blocking you?'
>
> Step 2: If it's clarity — I was unclear in my ask. I re-scope with specific deliverables and timelines.
>
> Step 3: If it's capability — I provide structured mentorship: pair programming, code reviews, learning resources. Set a 4-week improvement plan with specific checkpoints.
>
> Step 4: If it's motivation — I try to understand and align their interests with project needs. If misaligned at a fundamental level, I escalate to manager early rather than letting it drag.
>
> I've found that most underperformance traces back to unclear expectations or skill gaps — both solvable with clear communication and targeted support."

---

### "What's your approach to technical strategy as a lead?"

> "Three principles:
>
> One: Evaluation framework before implementation. I define how we'll know if we succeeded — business KPI + technical metrics + failure conditions — before writing any code. This prevents building the wrong thing.
>
> Two: Simplicity default with justified exceptions. Every complex technique needs to demonstrate it outperforms a simple baseline by a meaningful margin. This keeps the team shipping, maintains system maintainability.
>
> Three: Production-first thinking from day one. Every model gets a monitoring plan, a rollback plan, and a fallback mode designed before it's deployed. ML systems fail in production in ways that notebooks never reveal."

---

## SECTION 3: COMMON BEHAVIORAL QUESTION PATTERNS

| Question | What They're Testing | Key Points in Your Answer |
|----------|---------------------|--------------------------|
| "Tell me about your proudest technical achievement" | Technical depth, business impact | Numbers + technical decisions + why those choices |
| "Tell me about a failure and what you learned" | Self-awareness, growth mindset | Real failure (not fake modest one), specific lesson, changed behavior |
| "How do you handle ambiguity?" | Problem structuring, leadership | Show you define the problem before solving it |
| "Tell me about a conflict with a colleague" | Collaboration, communication | Constructive resolution, specific techniques used |
| "What's the hardest technical problem you've solved?" | Depth, creativity | Complex problem + structured approach + result |
| "Why do you want this Lead role?" | Motivation, clarity | Specific: team size, domain, what you'll build |
| "What's your management style?" | Leadership philosophy | Specific principles + examples |
| "How do you prioritize competing projects?" | Decision-making | Clear framework + stakeholder alignment approach |

---

## SECTION 4: QUESTIONS TO ASK INTERVIEWERS

Show strategic thinking. Never ask about salary, vacation, or things on the website.

**For understanding the real problem:**
- "What's the biggest ML/AI challenge your team is currently trying to solve?"
- "Where do you feel your current ML systems have the most room for improvement?"
- "What does success look like in this role after 6 months?"

**For team and culture:**
- "How does the data science team interact with product and engineering?"
- "How do you balance exploration and innovation vs production stability?"
- "What does a typical model deployment process look like here?"

**For your fit:**
- "What skills or experience would make someone most effective in this role?"
- "What are the biggest technical debt areas in the current ML infrastructure?"

**Power question (use this one):**
- "If you could wave a magic wand and fix one thing about how your current ML systems work today, what would it be?"

This question often reveals the real problem they're hiring for — and lets you pitch your exact experience as the solution.

---

## SECTION 5: COMPENSATION AND NEGOTIATION

**When asked: "What are your salary expectations?"**

> "I'm focused on finding the right fit first, and I'm confident we can work out compensation if that's the case. Could you share the range for this role?"

**If they push:**

> "Based on my research and experience level, I'm targeting [X-Y range]. But I'm flexible depending on total compensation — equity, bonus structure, and growth trajectory matter to me as well."

**Key principle:** Never give a number first. Get their range first.

---

## SECTION 6: LEAD ROLE READINESS CHECKLIST

Before going into a Lead Data Scientist interview, confirm you can answer YES to:

**Technical Leadership:**
- [ ] Can I explain how I made a critical architecture decision and defend it?
- [ ] Can I describe a model I built from scratch with business context, technical choices, and results?
- [ ] Can I explain tradeoffs between 3+ approaches to any common problem?

**People Leadership:**
- [ ] Can I describe how I mentored a junior team member with a specific example?
- [ ] Can I describe a stakeholder conflict I navigated successfully?
- [ ] Can I articulate my technical leadership philosophy in 3 principles?

**Business Acumen:**
- [ ] Can I connect every technical choice to a business outcome?
- [ ] Can I explain ROI of ML projects in business terms, not just metrics?
- [ ] Can I describe when NOT to use ML and why?

**Self-Awareness:**
- [ ] Can I name 2 real technical mistakes I've made and what I learned?
- [ ] Can I identify 2 areas where I'm still growing?
- [ ] Can I articulate why THIS company, THIS role, THIS team?

---

## SECTION 7: ADDITIONAL LEADERSHIP SCENARIOS

### "Tell me about a time you had to lead without formal authority"

> **S:** At Chubb, the MLOps team and data science team disagreed on how to version the feature pipeline. MLOps wanted a single repo; data science wanted separate repos for experiment flexibility.
>
> **T:** I had no direct authority over MLOps, but I was accountable for model quality and reproducibility.
>
> **A:** I scheduled a working session and proposed a hybrid: shared feature library in a separate repo with versioning, consumed by both teams. I built a prototype showing how this preserved experiment flexibility while giving MLOps a stable interface. I documented the contract and migration plan.
>
> **R:** Both teams agreed. We reduced integration issues by 60% and made model reproduction reliable across environments.

### "Tell me about a time you managed ambiguity"

> **S:** A stakeholder asked us to "improve fraud detection" without defining what that meant or how success would be measured.
>
> **T:** I needed to turn the vague request into a concrete project plan.
>
> **A:** I interviewed SIU managers, finance, and compliance to understand their definitions of success. I proposed three metrics: fraud referral rate, precision of referrals, and investigator time per confirmed case. I ran a baseline audit of the current system and presented a 90-day roadmap with clear milestones.
>
> **R:** The project was approved with defined success criteria. We delivered an 8-point PR-AUC improvement in the first quarter.

### "Tell me about a conflict between two team members"

> **S:** Two analysts disagreed on whether to use target encoding or one-hot encoding for a high-cardinality provider feature.
>
> **T:** I needed to resolve the conflict without micromanaging the decision.
>
> **A:** I asked each person to run a controlled experiment: train a model with each encoding on the same validation split and report PR-AUC, training time, and interpretability trade-off. We reviewed results together. Target encoding won by 3 points, so we adopted it with out-of-fold leakage protection.
>
> **R:** The conflict turned into a data-driven decision. Both team members felt heard, and we established a precedent of resolving technical debates with experiments.

### "Tell me about a time you had to present to executives"

> **S:** I presented the fraud RAG project results to the Chubb CIO and head of SIU.
>
> **T:** I had 15 minutes to explain a complex NLP system and justify continued investment.
>
> **A:** I structured the deck around business outcomes, not technical details. I opened with the 12% to 23% referral rate improvement and the projected annual fraud savings. I used one simplified architecture diagram and focused on risk, monitoring, and guardrails. I had backup slides with technical depth for questions.
>
> **R:** The project received funding for expansion. The CIO specifically noted that she appreciated the business framing over model details.

### "Tell me about a failure and what you learned"

> **S:** At Axtria, I proposed a complex multi-channel attribution model that took 12 weeks to build. When we deployed it, the client found it hard to trust because they couldn't see how the numbers were derived.
>
> **T:** I owned the project and had to rebuild trust.
>
> **A:** I added a simpler linear model as a transparent reference and built a comparison dashboard. I walked the client through three concrete examples where the complex and simple models agreed and disagreed, explaining the reasons. I also documented the model in business terms.
>
> **R:** The client accepted the complex model as the primary but used the simple model as a sanity check. I learned that interpretability must be designed in from the start, especially in regulated or conservative industries.

---

## SECTION 8: HIRING AND TEAM BUILDING

### "How do you hire data scientists?"

> "I look for three things: problem structuring, technical depth, and communication. I ask candidates to walk through a project end-to-end and probe their reasoning. I also give them an ambiguous business problem to see how they scope it before jumping to a solution. The best hires can explain trade-offs and connect their work to outcomes."

### "How do you onboard a new team member?"

> "In week one, I pair them with a buddy, give them a small, well-defined task that touches the full pipeline, and set up 30-minute daily check-ins. In the first month, I have them own a component or metric. I also ask them to document one thing that confused them so we can improve onboarding materials."

### "How do you build team culture?"

> "I build culture through rituals: weekly technical sharing, blameless post-mortems, and a code review culture that treats feedback as teaching. I also make sure wins are visible. When a model hits a milestone, I share the business impact with the team so they see the purpose behind the work."

---

## SECTION 9: CROSS-FUNCTIONAL COLLABORATION

### "How do you work with product managers?"

> "I partner with product managers to translate business problems into ML problems. I ask: what decision will this model inform? What is the cost of false positives and false negatives? What data is available? I push back when a request is not feasible or not well-defined, but I always propose alternatives."

### "How do you work with engineers?"

> "I respect engineering constraints like latency, uptime, and maintainability. I involve engineers early in design reviews and write clear specs for model contracts, input schemas, and SLOs. I don't hand off a notebook; I work with them to productionize."

### "How do you work with business stakeholders?"

> "I lead with business language and metrics. I avoid talking about algorithms until they ask. I set expectations on what ML can and cannot do. I also show uncertainty — confidence intervals, error rates, and edge cases — so stakeholders can make informed decisions."

---

## SECTION 10: LEADERSHIP ONE-LINERS

Use these to signal leadership without overstating:

- "I define success metrics before we start building."
- "I default to simple solutions and add complexity only with evidence."
- "I think about production from day one: monitoring, rollback, and fallback."
- "I resolve technical debates with experiments, not opinions."
- "I mentor by giving ownership, not just tasks."
- "I translate between business goals and technical trade-offs."
- "I believe the best ML systems are the ones the team can operate and improve."
- "I treat failures as data and build better processes from them."
