# Operational Readiness Review — The Go-Live Gate

> **Purpose:** a service is not "done" when it works on your laptop. It is done when the team that gets paged at 3 a.m. can operate, debug, scale, secure and recover it. This is the checklist and the review process that gets you there.

> **Applies to:** any new service going to production, any significant re-architecture, any migration that changes the failure surface, and any AI/LLM service (where the LLM-specific gates in §6 are additional, not optional).

---

## 1. Purpose & scope

### What an operational readiness review (ORR) is

A **structured, evidence-based gate** held before a service takes production traffic. The service team presents; an independent group (SRE, platform, a peer team) asks the questions a real incident would ask. It is not a paperwork exercise — it is a **rehearsal for the worst day**.

### What it is not

| It is NOT | It IS |
|---|---|
| A code review | An operations review |
| A performance test | A check that performance is *measured and bounded* |
| A security audit | A check that security controls *exist and are wired in* |
| A launch approval by one person | A shared, evidenced judgement |
| A one-time event | Re-run on major changes |

### When to run one

| Trigger | Depth |
|---|---|
| New service, first production traffic | **Full ORR** |
| Major re-architecture | Full ORR |
| New region / new datastore | Full ORR |
| Service taking on a new critical dependency | Focused (dependencies + failure modes) |
| First AI/LLM feature in production | Full ORR **+ §6 AI gates** |
| Routine feature release | None — normal CI/CD gates |
| Emergency fix | Post-hoc ORR within 5 business days |

### Who attends

| Role | Contribution |
|---|---|
| **Service owner / tech lead** | Presents the design, owns the actions |
| **On-call engineer** | Asks "what will page me, and what do I do?" |
| **SRE / platform** | Capacity, SLOs, observability, failure modes |
| **Security** | Auth, secrets, data handling, compliance |
| **Data / DBA** | Schema, migration, backup, restore |
| **Product** | Confirms the availability target matches the business need |
| **An independent peer** | The "have you considered…" voice |

---

## 2. What "operationally ready" means

A service is ready when all seven of these are true:

```
1. OBSERVABLE   You can see what it's doing, in real time, without SSH
2. BOUNDED      You know its limits and what happens at them
3. RECOVERABLE  You can roll back, fail over and restore data
4. DOCUMENTED   A stranger on call at 3 a.m. can act
5. GUARDED      Auth, secrets, rate limits, and input validation are live
6. OWNED        A named team is accountable, with escalation paths
7. AFFORDABLE   You know the unit cost and the cost at 10×
```

> **The test to apply to every item:** *"If this fails at 3 a.m. and the person on call has never seen this service, what happens?"* If the answer is "they'd have to ask the author," the item is not ready.

---

## 3. The checklist

Every item has: **what to check · why it matters · the evidence required**. "Evidence" is what you show in the review — a link, a dashboard, a test result. *"We'll do that later"* is not evidence.

### 3.1 Architecture & failure modes

| # | Check | Why it matters | Evidence required |
|---|---|---|---|
| A1 | Component diagram exists and is current | You can't reason about failure without the topology | Link to the diagram |
| A2 | Every dependency listed with its availability and failure behaviour | Your availability ceiling is your dependencies' | Dependency table with SLAs |
| A3 | Single points of failure identified and accepted explicitly | SPOFs are how total outages happen | SPOF list + mitigation or accepted-risk sign-off |
| A4 | The **series availability product** is computed | Ten 99.9% services in series is 99% | The calculation, with the floor named |
| A5 | Failure domains are genuinely independent (AZ, rack, deploy pipeline, config) | Redundancy without independence doesn't help | AZ/region placement |
| A6 | Timeouts set on **every** outbound call | A missing timeout is a thread-pool exhaustion bug waiting | Config excerpt or code reference |
| A7 | Circuit breakers on non-critical dependencies | Stops cascading failure | Config + a test that trips it |
| A8 | Retries use exponential backoff **with jitter**, and have a budget | Retry storms turn partial outages into total ones | Config + max-retry policy |
| A9 | Load shedding / admission control defined | Better to serve 80% well than 100% badly | Threshold + behaviour |
| A10 | Graceful degradation path for every non-essential dependency | Keeps non-essentials out of the availability path | Degradation ladder table |
| A11 | Idempotency for all mutating operations | Retries and at-least-once delivery require it | Idempotency-key design |
| A12 | Backpressure handled — queues bounded, not unbounded | Unbounded queues convert overload into memory exhaustion | Queue limits |

### 3.2 Observability

| # | Check | Why it matters | Evidence required |
|---|---|---|---|
| O1 | Structured logs with correlation/trace IDs | You cannot debug a distributed request without them | Sample log line |
| O2 | RED metrics for every service: Rate, Errors, Duration | The minimum for any service | Dashboard link |
| O3 | Distributed tracing enabled across service boundaries | Finds the slow hop, not just the slow request | Trace screenshot |
| O4 | p50 / p95 / p99 latency recorded, not just averages | Averages hide the users who are suffering | Dashboard panel |
| O5 | **Dashboards exist and are linked from the runbook** | On-call must find them in seconds | Runbook link |
| O6 | Alerts are **symptom-based and actionable** | Alert fatigue is an outage cause | Alert list with thresholds |
| O7 | Every alert links to a runbook | An alert without a next step is noise | Alert → runbook mapping |
| O8 | Alert routing and escalation policy configured | The right person, at the right time | Pager config |
| O9 | Dependency health visible (DB, cache, queue, third parties) | Most outages start outside your service | Dependency dashboard |
| O10 | Business-level metric instrumented (signups, orders, tokens) | Technical health ≠ business health | The metric on a dashboard |
| O11 | Log/metric cardinality bounded | Unbounded labels cause an observability outage and a bill spike | Cardinality policy |
| O12 | Retention configured and cost-reviewed | Traces and logs are the fastest-growing cost line | Retention config |

### 3.3 SLOs & error budgets

| # | Check | Why it matters | Evidence required |
|---|---|---|---|
| S1 | SLIs defined and **measured** | An unmeasured SLO is decoration | SLI definition + query |
| S2 | SLO set, and it is stricter than any external SLA | Buffer before breach | SLO vs SLA table |
| S3 | Error budget computed and visible | The reliability/velocity trade-off must be explicit | Budget dashboard |
| S4 | Burn-rate alerts configured (fast and slow windows) | Detects both sudden and slow burns | Alert config |
| S5 | The SLO matches the business need — not aspirational | 99.99% on an internal tool wastes money | Product sign-off |
| S6 | Error-budget policy written: what happens when it's exhausted | Otherwise it's ignored at the worst moment | Written policy |

### 3.4 Capacity & performance

| # | Check | Why it matters | Evidence required |
|---|---|---|---|
| C1 | Capacity model documented: QPS, storage, bandwidth, connections | Unmodelled capacity is how you discover limits in an incident | The estimate with assumptions |
| C2 | Load tested at **2× expected peak** | Headroom is measured, not assumed | Load-test report |
| C3 | The **binding bottleneck** identified (CPU, memory, connections, IOPS, tokens/sec) | So you scale the right thing | Bottleneck analysis |
| C4 | Autoscaling configured with sane min/max and cooldowns | Prevents both thrash and outages | Autoscaling policy |
| C5 | Scaling **lag** quantified | Scaling takes minutes; spikes take seconds | Time-to-scale measurement |
| C6 | Connection pool sizes matched to DB limits | Classic outage: pools × replicas > DB max connections | Pool config + DB limit |
| C7 | Cache sized for the hot working set, with a hit-rate target | The cache *is* the design in read-heavy systems | Hit-rate target + sizing |
| C8 | Storage growth forecast with a retention/tiering policy | Unbounded growth is an unbudgeted cost | Growth curve + policy |
| C9 | Quota/limit headroom confirmed with every third party | Rate limits are the usual third-party outage | Quota + current usage |
| C10 | Cold-start behaviour understood | A cold cache after a restart can take the DB down | Cold-start plan |

### 3.5 Deployment & change management

| # | Check | Why it matters | Evidence required |
|---|---|---|---|
| D1 | CI runs tests, lint and security scanning | Bad builds shouldn't reach staging | Pipeline config |
| D2 | Deploys are automated and repeatable | Manual deploys are the top source of outages | Deploy job |
| D3 | **Rollback tested**, with a time-to-rollback measured | An untested rollback is a hope, not a plan | Rollback drill result |
| D4 | Zero-downtime deploy strategy (rolling/blue-green/canary) | Deploys shouldn't drop requests | Strategy + config |
| D5 | Canary with automated analysis on error rate/latency | Catches bad changes before full blast radius | Canary config |
| D6 | Database migrations are backward-compatible and reversible | The classic: new code + old schema = outage | Migration plan (expand/contract) |
| D7 | **Feature flags** for risky changes, with a kill switch | Decouples deploy from release | Flag list |
| D8 | Config changes go through review, not hot-edited | Config is code | Config repo link |
| D9 | Change freeze windows defined | Protects peak events | Calendar/policy |
| D10 | Deploy frequency and lead time measured | You can't improve what you don't measure | DORA metrics |

### 3.6 Security & access

| # | Check | Why it matters | Evidence required |
|---|---|---|---|
| X1 | Authentication and authorisation enforced at the edge | Every endpoint, no exceptions | Auth config |
| X2 | Service-to-service auth (mTLS or signed tokens) | Internal networks are not trusted | mTLS/token config |
| X3 | Secrets in a secret manager, never in code or env dumps | Leaked secrets are the #1 breach vector | Secret store reference |
| X4 | Secrets have rotation defined and tested | Rotation that's never tested doesn't work | Rotation schedule |
| X5 | Input validation and output encoding | Injection prevention | Validation rules |
| X6 | Rate limiting and abuse protection live | Protects both capacity and cost | Rate-limit config |
| X7 | Data classified (PII, PHI, payment) with handling rules | Determines encryption, residency, retention | Data classification doc |
| X8 | Data encrypted at rest and in transit | Baseline compliance | Encryption config |
| X9 | Audit logging on privileged actions | Required for forensics and compliance | Audit log sample |
| X10 | Dependency/vulnerability scanning in CI | Known CVEs are avoidable risk | Scan results |
| X11 | Least-privilege IAM reviewed | Blast radius control | Role definitions |
| X12 | Compliance obligations identified (SOC2, HIPAA, GDPR, EU AI Act) | Retro-fitting compliance is brutal | Compliance mapping |

### 3.7 Data, backup & disaster recovery

| # | Check | Why it matters | Evidence required |
|---|---|---|---|
| R1 | Backup schedule defined and automated | — | Backup job |
| R2 | **Restore tested, with a measured restore time** | An untested backup is not a backup | Restore drill report |
| R3 | RTO and RPO agreed with the business | Determines the whole DR design | RTO/RPO sign-off |
| R4 | Failover tested end-to-end | Failover that's never run doesn't work | Failover drill result |
| R5 | Data retention and deletion policy implemented | Legal and cost | Policy + job |
| R6 | Schema/data contracts versioned | Prevents breaking consumers | Contract registry |
| R7 | Analytics/AI index rebuild path known (rebuild vs snapshot) | Re-indexing 100M vectors is an outage if unplanned | Documented rebuild procedure |
| R8 | Cross-region replication verified where required | For DR targets beyond one region | Replication status |

### 3.8 Runbooks & documentation

| # | Check | Why it matters | Evidence required |
|---|---|---|---|
| B1 | **Runbook exists** covering start, stop, scale, rollback, common failures | A stranger must be able to act | Runbook link |
| B2 | Every alert maps to a runbook section | Removes thinking at 3 a.m. | Alert → runbook mapping |
| B3 | Architecture and dependency docs current | Stale docs mislead during incidents | Doc links |
| B4 | Known-issues list with workarounds | Tacit knowledge made explicit | Known-issues doc |
| B5 | Escalation path with names and contact routes | Prevents "who do I ask?" | Escalation matrix |
| B6 | On-call rotation with a trained primary and secondary | Single on-call is a burnout and coverage risk | Rotation schedule |
| B7 | Service catalogue entry (owner, tier, SLO, dependencies) | Enables org-wide dependency reasoning | Catalogue entry |

### 3.9 Cost

| # | Check | Why it matters | Evidence required |
|---|---|---|---|
| F1 | Unit cost computed (cost per 1,000 requests / per user / per 1M tokens) | The only comparable cost metric | Cost model |
| F2 | Cost at 10× traffic projected | Prevents a growth-driven budget surprise | Projection |
| F3 | Cost anomaly alerting configured | Bills spike faster than dashboards | Alert config |
| F4 | Known cost levers documented (cache hit rate, tiering, batching) | Cost is an engineering variable | Lever list |
| F5 | Egress and third-party API costs modelled | The two most common surprise lines | Cost breakdown |

---

## 4. Scoring & outcomes

### The four outcomes

| Outcome | Meaning |
|---|---|
| **GO** | All critical items satisfied; minor items have dated owners |
| **GO WITH CONDITIONS** | Critical items satisfied; named gaps with an owner and a date, and a documented workaround |
| **NO GO** | A critical item is unmet; launch is blocked until it is |
| **EXEMPTION** | A critical item is knowingly accepted as risk, with a named executive owner and an expiry date |

### Critical vs standard items

| Tier | Definition | Examples |
|---|---|---|
| **Critical** (must pass) | Its absence risks data loss, a security breach, or an unrecoverable outage | Restore tested · rollback tested · auth enforced · secrets managed · alerting live · SPOFs identified |
| **Standard** | Its absence degrades operability | Dashboard polish · cost model · runbook completeness |

> **The exemption rule:** any critical item can be waived, but only by a named owner, in writing, with an expiry date. Untracked exceptions are how "we'll fix it later" becomes a 3 a.m. disaster.

---

## 5. The review meeting

### Format (60–90 minutes)

| Segment | Time | Content |
|---|---|---|
| 1. Context | 5 min | What the service does, who uses it, why now |
| 2. Walkthrough | 20 min | Architecture, dependencies, SLOs, capacity model |
| 3. Failure rehearsal | 20 min | *"The cache dies at peak. What happens? Who gets paged? What do you do?"* |
| 4. Checklist | 20 min | Item-by-item, evidence-based |
| 5. Actions | 10 min | Owners, dates, tiers |

### The failure rehearsal — the highest-value 20 minutes

Ask these out loud, and require a concrete answer for each:

| Scenario | What the answer must contain |
|---|---|
| "The primary database dies at peak." | Detection, failover time, data loss window (RPO), who's paged |
| "The cache tier dies." | Whether the DB survives the 20× load; shedding thresholds |
| "A dependency is 10× slower, not down." | Timeouts, circuit breaker behaviour, degradation |
| "A bad deploy reaches 100%." | Rollback time, kill switch, who decides |
| "Traffic is 5× forecast." | Autoscaling limits, what degrades first |
| "You're paged at 3 a.m. and you've never seen this service." | The runbook's first three steps |
| **AI:** "The model provider is down." | Fallback model, queue-and-replay, degraded mode |
| **AI:** "Outputs are suddenly wrong but the service is up." | Canary evals, quality alerting, rollback of prompt/model/index |

> **If any of these produces a blank look, that is the finding.** The point of the review is to find the blanks before production does.

---

## 6. AI/LLM-specific readiness gates

These are **additional** gates for any system that calls a model. Classical gates still apply in full.

### 6.1 Model & prompt management

| # | Check | Evidence |
|---|---|---|
| M1 | Model and prompt versions **pinned**, never `latest` | Version manifest |
| M2 | Prompt templates version-controlled and reviewed like code | Prompt repo |
| M3 | Every response stored with the prompt version, model version and parameters that produced it | Provenance record |
| M4 | Rollback path for prompt and model changes (they are deploys) | Rollback procedure |
| M5 | Provider fallback configured (second model or provider) | Fallback config |
| M6 | Token limits and truncation behaviour defined per route | Route config |

### 6.2 Evaluation & quality gates

| # | Check | Evidence |
|---|---|---|
| E1 | An eval suite exists with a golden dataset | Eval report |
| E2 | Evals run in CI and **block** deploys on regression | Pipeline config |
| E3 | Quality metrics monitored in production (not just offline) | Production eval dashboard |
| E4 | A/B or shadow path for model/prompt changes | Experiment config |
| E5 | Baseline quality recorded for comparison | Baseline numbers |
| E6 | Human review path for low-confidence or high-risk outputs | Review queue |

### 6.3 Guardrails & safety

| # | Check | Evidence |
|---|---|---|
| G1 | Input guardrails (injection detection, PII redaction, topic filters) live | Guardrail config |
| G2 | Output guardrails (schema validation, safety filters, PII leak detection) live | Guardrail config |
| G3 | Prompt-injection posture tested against a red-team set | Red-team results |
| G4 | Tool-call permissions scoped to least privilege | Tool registry with scopes |
| G5 | Human-in-the-loop for irreversible or high-value actions | Approval flow |
| G6 | Agent step/token/time budgets enforced, with a hard stop | Budget config |
| G7 | Hallucination mitigation for factual paths (citations, groundedness checks) | Retrieval/eval config |

### 6.4 Retrieval & index operations

| # | Check | Evidence |
|---|---|---|
| I1 | Ingestion pipeline idempotent and re-runnable | Pipeline doc |
| I2 | Index rebuild time measured and acceptable | Rebuild drill |
| I3 | Embedding model version pinned; migration plan for changes | Version manifest |
| I4 | Retrieval quality monitored (recall@k, hit rate, no-result rate) | Retrieval dashboard |
| I5 | Freshness SLA defined and monitored (how current is the index?) | Freshness metric |
| I6 | Access control enforced **at retrieval** (a user must not retrieve documents they can't see) | ACL filter in the query path |

### 6.5 AI cost & capacity

| # | Check | Evidence |
|---|---|---|
| N1 | Cost per query computed and monitored | Cost dashboard |
| N2 | Token usage per request tracked by route | Usage metrics |
| N3 | Prompt caching / batching / model routing applied where beneficial | Optimisation report |
| N4 | Rate-limit and quota handling for provider limits | Backoff + queue config |
| N5 | GPU/inference capacity model, or provider quota headroom confirmed | Capacity model / quota |

---

## 7. Failure modes of the review itself

| Failure mode | Symptom | Fix |
|---|---|---|
| **Checklist theatre** | Boxes ticked, no evidence | Require evidence links; reject "we'll do it later" |
| **No on-call in the room** | Gaps in runbooks and alerting missed | Make the on-call a required attendee |
| **Everything is critical** | Nothing is; review becomes unwinnable | Tier items explicitly; cap critical at ~10 |
| **Rubber-stamping** | Same review every time, same answers | Rotate the reviewer; include an independent peer |
| **No follow-through** | Conditions never closed | Track actions in the same system as bugs, with dates |
| **Review after launch** | Findings that can't be acted on | Gate *before* production traffic, always |
| **Exemptions without owners** | Permanent, invisible risk | Every exemption has a name and an expiry |
| **Skipped for "small" services** | Small services are how the big outage starts | Scale the depth, don't skip the gate |
| **AI gates bolted on as an afterthought** | Safety gaps found in production | Run §6 in the same review as everything else |
| **The review is adversarial** | Teams hide problems | Frame it as rehearsal, not audit; the goal is finding blanks |

---

## 8. When to use / When NOT to use

| ✅ Use | ❌ Skip |
|---|---|
| New service entering production | Routine feature deploy behind normal CI |
| Major re-architecture | Cosmetic changes |
| New region or datastore | A service already reviewed, unchanged |
| First AI feature in production | Internal prototype with no users |
| Migration changing the failure surface | Dev/test environments |
| After a Sev-1 (post-hoc, for the fixes) | — |

### Scaling the depth

| Service tier | Review depth | Time |
|---|---|---|
| Tier 3 (internal tool) | Focused checklist — security, backup, runbook | 30 min |
| Tier 2 (business-important) | Standard ORR | 60 min |
| Tier 1 (revenue-critical) | Full ORR + failure rehearsal + DR drill | 90 min + drills |
| AI/LLM on any tier | Add §6 in full, scaled to risk | +45 min |

---

## 9. Numbers, defaults & thresholds

| Parameter | Default | Notes |
|---|---|---|
| Load test target | **2× expected peak** | Headroom must be measured |
| Rollback time target | **< 5 min** | If longer, fix the pipeline before launch |
| Restore drill | **Before launch, then quarterly** | Untested backups don't count |
| Failover drill | Before launch, then quarterly | |
| RTO/RPO | Documented and signed off | No default — business decides |
| Alert → runbook coverage | **100%** | Every alert |
| On-call | Primary + secondary | Never single |
| Error budget policy | Written before launch | Defines what happens at exhaustion |
| Post-incident review | Within **5 business days** | Blameless |
| Critical checklist items | ~10 | If everything is critical, nothing is |
| Exemption expiry | **≤ 90 days** | Forces closure |
| Deploy frequency | Daily or better for Tier 1 | DORA signal |
| Cost anomaly alert | > 20% week-over-week | |

---

## 10. Q&A

**Q1. What is an operational readiness review?**
A structured, evidence-based gate held before a service takes production traffic. The service team presents the architecture, SLOs, capacity model and failure behaviour, and an independent group asks the questions a real incident would ask. It's a rehearsal for the worst day, not a paperwork exercise.

**Q2. Why hold one at all?**
Because a service is not done when it works on your laptop. Most outage causes — no runbook, untested rollback, missing timeout, single point of failure, no restore drill — are invisible in code review and cost nothing to fix before launch, and are extremely expensive to fix mid-incident.

**Q3. What's the highest-value part of the review?**
The failure rehearsal — twenty minutes of "the cache dies at peak; what happens, who gets paged, what do you do?" If an answer produces a blank look, you've found the finding before production did.

**Q4. What makes an item critical?**
Its absence risks data loss, a security breach, or an unrecoverable outage. Tested restore, tested rollback, enforced auth, managed secrets, live alerting and identified SPOFs are the core critical set. Cap the critical list at around ten — if everything is critical, nothing is.

**Q5. Can you launch with unmet critical items?**
Yes, via an exemption: a named owner, in writing, with an expiry date and (ideally) a documented workaround. What's not acceptable is an untracked exception, because "we'll fix it later" is how a 3 a.m. disaster is scheduled.

**Q6. What's the difference between an ORR and a normal code review?**
Code review asks "is this correct?" An ORR asks "can someone operate, debug, scale, secure and recover this at 3 a.m. without the author's help?" Entirely different questions with entirely different evidence.

**Q7. What are the AI-specific gates?**
Five groups: model and prompt versioning, evaluation and quality gating in CI, input/output guardrails, retrieval and index operations, and AI cost/capacity. The one that most often catches real problems is eval-gated deployment — a prompt change is a deploy and needs the same rollback discipline.

**Q8. What's the most commonly missed item?**
Tested restore. Teams configure backups and never restore them; the first restore attempt happens during a real incident with real time pressure. It's the single cheapest high-value drill.

**Q9. How does an ORR change for an AI service?**
The classical gates apply unchanged, plus §6. The new failure modes are: the model provider is down (needs a fallback), outputs are wrong but the service is up (needs production quality monitoring), and a prompt or index change silently degrades quality (needs eval gates and version pinning). "Is it up?" is no longer the same question as "is it working?"

**Q10. *Pushback:* "This feels like bureaucracy that slows us down."**
It's cheaper than the alternative. The items that matter — a timeout, a rollback, a runbook, a restore test — take hours before launch and days during an incident. And the depth scales: thirty minutes for an internal tool, ninety plus drills for a revenue-critical one.

**Q11. *Pushback:* "We're a small team; we don't have an SRE."**
The review needs a peer, not an SRE department. A person from another team asking "what pages you at 3 a.m., and what does the runbook say?" gets you most of the value. The role is the independent question, not the title.

**Q12. When do you re-run it?**
On new services, major re-architecture, new regions or datastores, first AI features, migrations that change the failure surface, and post-hoc after a Sev-1. Not on routine feature releases — those go through normal CI/CD gates.

**Q13. How do you stop it becoming rubber-stamping?**
Rotate the reviewer, require evidence rather than assertions for each item, and keep the failure rehearsal scenario-based so the questions can't be pre-scripted. If the same review returns the same answers with no findings, the reviewer isn't asking hard enough.

**Q14. What if a dependency has no published SLA?**
Treat unknown as "assume the worst" and design a degradation path. If you can't state a dependency's availability, your own availability ceiling is unknown — which is itself a finding.

**Q15. How does the review handle a service already in production that fails the checklist?**
Same process, run as a gap analysis: score the items, tier the gaps, assign owners and dates, and treat any data-loss or security gap as the immediate priority regardless of the others.

---

## 11. Cheat sheet

**The seven readiness properties:** Observable · Bounded · Recoverable · Documented · Guarded · Owned · Affordable.

**The 3 a.m. test:** *"If this fails at 3 a.m. and the on-call has never seen this service, what happens?"* If the answer involves asking the author, it isn't ready.

**Checklist groups:** Architecture & failure modes (12) · Observability (12) · SLOs (6) · Capacity (10) · Deployment (10) · Security (12) · Data & DR (8) · Runbooks (7) · Cost (5) · **AI/LLM (25 additional)**

**Critical items (cap ~10):** tested restore · tested rollback · auth enforced · secrets managed · alerting live with runbooks · SPOFs identified · timeouts everywhere · capacity model · on-call staffed · error-budget policy

**Outcomes:** GO · GO WITH CONDITIONS · NO GO · EXEMPTION (named owner + expiry)

**The failure rehearsal questions:** DB dies · cache dies · dependency 10× slower · bad deploy at 100% · traffic 5× forecast · 3 a.m. page with no context · **model provider down** · **outputs wrong but service up**

**Key numbers:** load test at 2× peak · rollback < 5 min · restore drill before launch and quarterly · 100% alert→runbook coverage · primary + secondary on-call · PIR within 5 days · exemptions expire in ≤ 90 days

**AI additions:** model/prompt pinned · evals gate deploys · input + output guardrails · ACL enforced at retrieval · step/token budgets · cost per query monitored · provider fallback configured

---

## 12. Cross-links

- Next: [Observability & monitoring](02-observability-and-monitoring.md) · [SLOs, error budgets & incident response](03-slos-error-budgets-and-incident-response.md)
- Companion playbooks: [Deployment, release & CI/CD](04-deployment-release-and-cicd.md) · [Capacity planning & autoscaling](05-capacity-planning-and-autoscaling.md) · [Cost & FinOps](06-cost-and-finops.md) · [Reliability patterns](07-reliability-patterns.md) · [Security, guardrails & compliance](08-security-guardrails-and-compliance.md) · [DR, failover & backup](09-dr-failover-and-backup.md) · [LLMOps for AI systems](10-llmops-for-ai-systems.md)
- Foundations this rests on: [Availability numbers & SLOs](../HLD/01-foundations/06-availability-numbers-and-slos.md) · [Six goals & trade-offs](../HLD/01-foundations/02-six-goals-and-tradeoffs.md)
- Question bank: [LLMOps & operationalisation questions](../Interview-Questions/06-llmops-and-operationalisation-questions.md) · [Cheat sheet](../Cheat-Sheets/06-operationalisation-cheatsheet.md)
