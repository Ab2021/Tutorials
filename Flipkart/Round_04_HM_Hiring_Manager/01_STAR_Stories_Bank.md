# 💼 STAR Stories Bank — HM Round
### 12 Fully Developed STAR+ Stories Mapped to Flipkart Values

> **Format:** Situation → Task → Action → Result → Reflection (+)
> **Usage:** Each story can answer MULTIPLE behavioral questions. Note the "Answers" tag.

---

## STORY 1: Insurance Fraud Detection System (Chubb)
**Answers:** "Most impactful project" | "Bold technical bet" | "Audacity" | "Built production AI"

**S:** At Chubb (Aug 2024), insurance fraud in long-tail claims (>90 days old) was costing millions in undetected fraud reserves. The traditional threshold-based system reviewed only ~30% of mature claims and caught fraud at roughly 45% precision.

**T:** I was asked to architect a new fraud detection capability that could scan unstructured claims data at scale and intervene at the early maturation phase (~7-30 days) rather than 90+ days.

**A:**
1. Evaluated three approaches: fine-tuning a classifier, rule expansion, or RAG + LLM pipeline. Selected RAG because fraud patterns evolve faster than fine-tuning cycles allow.
2. Built: BERT-based info extraction → domain fine-tuned embedding model (contrastive learning) → ChromaDB vector store → GPT-4 structured output for risk scoring.
3. Implemented dual serving: Batch for historical backfill, real-time Kafka consumer for new claims.
4. Ran in shadow mode for 60 days: compared outputs to investigator verdicts without affecting workflow.
5. Added SHAP values + source citations to address investigator trust (explainability).

**R:**
- System achieved 78% recall at 22% false positive rate (vs. 45%/N/A baseline)
- Mean time to flag: dropped from >30 days to <2 days
- Freed ~40% investigator bandwidth via automated evidence compilation
- Q1 2025 STAR Award + Q3 2025 Advanced Analytics Spot Award

**+:** I underestimated the trust-building element. The technology worked, but investigator adoption took 6 weeks longer than expected because we hadn't co-designed the UI with them. Now I involve end-users in the design phase, not just the testing phase.

---

## STORY 2: Overcoming VP Resistance with Data (J&J MMM @ Axtria)
**Answers:** "Stakeholder management" | "Influencing without authority" | "Integrity" | "Data-driven decision"

**S:** At Axtria, J&J's senior marketing VP was committed to a 40% increase in TV spend based on brand team input and agency recommendation. The Marketing Mix Model I built showed TV was saturated — marginal ROI on TV spend was <0.3x (i.e., $1 spent returning less than $0.30 in revenue).

**T:** I had to convince a senior VP to move $8M from TV to digital touchpoints — directly contradicting the agency recommendation and brand team intuition. I had no direct authority over the decision.

**A:**
1. Did NOT lead with "the model says you're wrong" — that framing creates defensiveness.
2. Instead: "Let's run a 1-quarter controlled test with 15% of contested budget. We agree on metrics upfront." This separated the conversation from opinion to evidence.
3. Built the measurement framework jointly — agreed on attribution metrics BEFORE the test.
4. During the quarter: actively shared interim results to keep VP engaged (vs. presenting a surprise reveal).
5. Post-test presentation: showed actual vs. predicted outcomes side by side.

**R:**
- Q1 test: 14% ROI improvement on digital allocation vs. model's 12% prediction
- VP approved full budget reallocation for following year
- Marketing team subsequently used the model for all allocation decisions without prompting
- ~$3.2M incremental revenue attributed to optimized allocation

**+:** The lesson wasn't about the model — it was about change management. Technical correctness is necessary but never sufficient for adoption. The experiment design was the real product, not the model.

---

## STORY 3: Building GenAI from Scratch Under Ambiguity (Pharma Comm System @ Axtria)
**Answers:** "Ambiguity" | "Customer-first" | "Bias for action" | "Building with technology before it was mainstream"

**S:** In late 2022, I received a mandate at Axtria: "Build a GenAI system that helps pharma reps." No spec, no dataset, no labeled examples, no clear success criteria. GPT-4 had just launched publicly.

**T:** Define the problem, scope the solution, and deliver a working POC with measurable business impact in 3 months — with 2 junior engineers on the team.

**A:**
1. **Week 1-2: Research before building.** Interviewed 10 pharma sales reps to understand actual pain points. Found: 80% of time was spent personalizing generic corporate email templates for each doctor. This was the problem.
2. **Week 3: Define success criteria jointly with business sponsor.** KPIs agreed: email open rate (+15% target), meeting acceptance rate (+10% target).
3. **Week 4-5: V1 prototype** — GPT-4 with doctor-specific context injected via prompt. Zero knowledge graph, zero recommendation engine. Just: prompt with doctor specialty + prescribing history → personalized email.
4. **Week 6-8: V2** — Integrated Neo4j knowledge graph for relationship-aware context; built recommendation engine for drug-doctor matching.
5. **Week 9-12:** Deployed on Flask + Kubeflow. Ran A/B test: 200 doctors split 50/50.

**R:**
- A/B test: +23% email open rate, +17% meeting acceptance rate, +12% prescription intent
- Rep satisfaction: 4.3/5 vs. 2.8/5 for generic templates
- POC approved for broader rollout
- System became the template for subsequent GenAI initiatives at Axtria

**+:** The 2-week discovery sprint saved at least 2 months of building the wrong thing. Starting with user research rather than technology is the lesson I now preach to every junior DS.

---

## STORY 4: Discovering & Fixing Model Bias (Readmission Risk @ EXL)
**Answers:** "Failure" | "Integrity" | "Customer-first" | "Doing the hard right thing"

**S:** I built a patient readmission risk model using BERT + XGBoost that achieved AUC 0.89 on validation data. I was ready to push to production and was excited about the improvement over the 0.82 baseline.

**T:** Final pre-deployment check: run subgroup analysis across patient demographics.

**A:**
1. Ran AUC by age bucket: discovered AUC for patients >80 was 0.84 vs. 0.89 overall — a 5.9 percentage point gap.
2. Made the difficult call: pulled the model from the deployment queue before any patient was affected. This was not required by protocol — it was a judgment call.
3. Root cause analysis: ClinicalBERT was pre-trained primarily on MIMIC-III data (younger adult population). Clinical note patterns for elderly patients (more comorbidities, more complex notes, different clinical language) were underrepresented.
4. Fix: Added age-specific calibration (isotonic regression per age bucket). Added age as explicit feature with monotone constraint. Expanded training data with elderly patient cohort oversampling.
5. Validated fix: age >80 AUC rose to 0.87 (1-point remaining gap, flagged as known limitation in documentation).

**R:**
- Re-deployed 3 weeks later with documented subgroup performance
- Established team-wide standard: subgroup fairness analysis is mandatory gate before any healthcare model deployment
- Zero patient harm from the original model (caught pre-deployment)

**+:** Pulling the model was uncomfortable — I'd already told stakeholders it was ready. But doing the right thing when it's inconvenient is the only kind of integrity that matters. The short-term setback was worth the long-term credibility.

---

## STORY 5: 70% Processing Time Reduction @ EXL (CLV at Scale)
**Answers:** "Scale / Technical challenge" | "Engineering mindset" | "Bias for action" | "Solve a specific hard problem"

**S:** CLV model for 2M+ insurance prospects on single-node sklearn was taking 6+ hours for batch scoring. This meant: overnight job, manual monitoring, late insights delivery, and compute costs exceeding budget.

**T:** Reduce scoring time by 50% minimum; ideally to under 2 hours. Maintain all existing model features and accuracy.

**A:**
1. Profiled the pipeline: bottleneck was feature JOIN (not model scoring) — a full outer join across 2M users and 40M claim records.
2. Migrated feature pipeline from pandas to Spark SQL on GCP Dataproc.
3. Applied specific optimizations:
   - Partitioned by state (geographic distribution for parallelism)
   - Broadcast small lookup tables (drug → category mapping, 200MB) instead of shuffle joins
   - Cached intermediate feature matrices that were reused in multiple downstream calculations
   - Auto-scaling cluster: 5 nodes for startup, scales to 15 at peak, scales back down
4. Migrated model inference to PySpark MLlib: distributed Random Forest scoring.
5. Column pruning: identified top-30 features (by importance); dropped remaining 45 → 30% serialization speedup.

**R:**
- Processing time: 6+ hours → ~1.8 hours (70% reduction)
- Compute cost: 40% cheaper than dedicated machine (auto-scaling only pays for compute used)
- Monthly batch job now completes before business day starts — insights available at 9am
- 5/5 SLA rating for 3 consecutive quarters

**+:** The 70% improvement sounds impressive, but the real lesson is: profile before optimizing. The first 30 minutes of profiling revealed the JOIN as the bottleneck — without that, I would have spent weeks optimizing the model scoring code that was already fast.

---

## STORY 6: Mentoring a Junior Engineer Who Was Struggling
**Answers:** "Leadership/mentoring" | "Team building" | "Inclusion" | "Multiplying team impact"

**S:** At EXL, a junior analyst I managed — strong in Python, excellent algorithmic instincts — was consistently over-delivering technically but under-delivering on business value. He'd spend 3 weeks building a beautifully abstracted pipeline when a simpler version would have been deployed in 1 week and created more business value.

**T:** Help him ship faster without curbing his intellectual ambition or making him feel micromanaged.

**A:**
1. **Diagnosis:** Sat with him for 1:1 check-in. Asked: "What are you optimizing for in this project?" Answer: "Building it right." Asked: "What does 'right' mean to the business stakeholder?" He paused. He'd never really thought about this.
2. **Root cause:** He'd been rewarded in previous roles for technical elegance, not for business outcomes. The incentive structure was shaping his behavior.
3. **Intervention:** Had him shadow a business stakeholder for 2 weeks — attend their meetings, understand their actual decision-making cadence. After this, he could see his over-engineering was blocking the business from getting value.
4. **Framework:** Introduced "V1/V2/V3" planning: define simplest deployable V1, list improvements for V2, full vision for V3. V1 always ships first.
5. **Code review criteria update:** Added explicit "simplicity" criterion — "A simple system that's right beats a complex system that might be right."

**R:**
- His delivery speed increased by ~40% over 2 months
- His CLV decile model became the most-used business tool in his project's history (adopted without prompting by 3 business teams)
- He was promoted 6 months later
- Now applies the V1/V2/V3 framework himself and teaches it to others

**+:** Technical skill was never the issue — it was incentives and perspective. Redirecting incentive structures is the highest-leverage intervention a leader can make. Tell-don't-show might have fixed the symptom; the root cause was perspective.

---

## STORY 7: Handling Incomplete Data Creatively (CLV Missingness @ EXL)
**Answers:** "Problem solving" | "Creativity" | "Working with constraints" | "Technical judgment"

**S:** CLV model project at EXL had 35% missing rate on key behavioral features (claims history, policy start dates). Business wanted deployment in 6 weeks. Initial instinct: drop rows with missing data → lose 35% of prospects.

**T:** Maintain model quality while handling missingness appropriately. Don't just delete data — that's a lazy solution at 2M record scale.

**A:**
1. **Root cause analysis** (2 days): 70% of missingness came from customers <6 months old — they hadn't yet generated any claim history. This was NOT random missingness; it was a deterministic pattern.
2. **Insight:** "Customer tenure" predicts both missingness AND CLV itself — newer customers tend to have lower claim counts and often represent a distinct value segment. Created a `customer_tenure` feature.
3. **Remaining 30% missingness:** Random missingness due to data ingestion gaps. Applied MICE (Multiple Imputation by Chained Equations) — creates multiple imputed datasets, averages predictions → preserves uncertainty.
4. **Validation:** Ran sensitivity analysis — trained model on complete cases vs. imputed dataset. Spearman rank correlation of CLV rankings: 0.92 (high correlation → imputation didn't distort rankings).

**R:**
- Model deployed on full 2M dataset (not 1.3M after dropping)
- `customer_tenure` became the 3rd most important feature → it was an insight hiding in the data quality problem
- Zero decision-making quality loss from imputation (validated by sensitivity analysis)

**+:** Data quality problems are often domain signals in disguise. The habit of asking "WHY is this missing?" instead of "HOW do I handle missing values?" is what separates good data scientists from great ones.

---

## QUICK REFERENCE: STORIES BY VALUE TAG

| Flipkart Value | Primary Story | Backup Story |
|---|---|---|
| **Audacity** | Story 1: RAG fraud system | Story 3: GenAI pharma system |
| **Bias for Action** | Story 5: 70% processing reduction | Story 3: GenAI MVP in 3 months |
| **Customer-First** | Story 4: Fairness model fix | Story 3: User research sprint |
| **Integrity** | Story 4: Pulled model pre-deploy | Story 2: Honest about model limitations |
| **Inclusion** | Story 6: Junior mentoring | — |
| **Impact/Scale** | Story 5: 2M record at scale | Story 1: 78% fraud catch rate |
| **Leadership** | Story 6: Team building | Story 2: Influence without authority |
| **Ambiguity** | Story 3: GenAI from scratch | Story 7: Missing data creativity |
| **Failure/Learning** | Story 4: Bias discovery | Story 2: Initial resistance failure |

---

*See companion: 03_Leadership_Scenarios.md for hypothetical leadership questions*
