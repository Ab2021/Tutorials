# 12 — PEOPLE LEADERSHIP & STAKEHOLDER ROUND

> **TL;DR.** The panel feedback names this as a reason profiles are rejected: *"broader service design considerations, business integration, and working with stakeholders and teams"* plus *"technical and people mentoring for a team of data scientists, AI engineers, or ML engineers."* At Lead level this is **not a soft add-on** — the JD spends more words on people leadership than on any single technical topic. This file gives you the team narrative, six mentoring stories, the conflict/performance scripts, the standards you claim authorship of, and the "service not model" framing that answers the business-integration gap.

**60-second interview answer:** *"I lead [N] people across [disciplines]. My job splits three ways: they have clear goals and one owner per outcome; nobody stays in notebooks — everyone ships something with a dashboard and an owner; and I protect a fortnightly forum where the team reviews each other's work, because that's where standards actually propagate. On the business side, I treat the model as roughly 10% of the problem — the other 90% is who acts on the output, what happens when it's wrong, and which system it writes back to."*

---

## 1. The gap, restated precisely

The panel wrote that candidates lack, **in a product development environment**:

1. **Broader service design considerations** — not just the model, the whole service.
2. **Business integration** — how the AI lands in a real workflow and a real system of record.
3. **Working with stakeholders and teams** — cross-functional, UK/India, business + tech.
4. **Technical and people mentoring** — developing engineers and scientists.

Those four are your scoring rubric for this file. Everything below maps to one of them.

---

## 2. The team narrative — know this cold

They asked this directly in the last interview: *"How big of a team you worked on?"* The weak answer is a number. The Lead answer is a **structure**.

| Element | Your content | Why it matters |
|---|---|---|
| Size and disciplines | "[N] people: [x] engineers, [y] scientists, [z] analysts" | Proves span |
| What they owned | "Each had one named production service / workstream" | Proves clarity of purpose (JD language) |
| Your span of control | "I owned [delivery + standards + hiring], [person] owned [X]" | Precision — don't claim it all |
| How work was allocated | "One owner per outcome; I allocated by capability, not availability" | Proves "allocate work across both teams" |
| How you ran the rhythm | "Weekly 1:1s, fortnightly technical review, monthly capability plan" | Proves operating cadence |
| How you grew capability | "A skills matrix; each person has a named stretch area per quarter" | Proves "individual capability and development plans" |

**Fill this in for your real team. If you have not managed people directly**, say so honestly and pivot to *technical leadership*: "I haven't had line-management, but I've led a multidisciplinary squad of N, run code review, owned hiring loops, and mentored two people to promotion — here's how."

> **Never bluff line management.** It is checked in the next round or in references. Frame technical leadership as real leadership; then show you understand what line management adds.

---

## 3. Six mentoring stories to prepare (60 seconds each)

The JD names the mechanisms: **pairing, technical reviews, stretch opportunities**. Use those exact words.

| # | Story | Mechanism the JD names | What "good" sounds like |
|---|---|---|---|
| 1 | **Data scientist → production engineer** | pairing | "She wrote great notebooks but had never shipped a service. I paired with her two hours a week for a quarter on containerisation and CI. She now owns the extraction service." |
| 2 | **Technical review that changed behaviour** | technical reviews | "I introduced a review checklist — every PR needs a test, a metric, and a rollback. Within two months the review conversation shifted from style to design." |
| 3 | **Stretch opportunity** | stretch | "I gave him the eval harness — bigger than his level, with me as a backstop. He presented it to the business himself." |
| 4 | **The underperformer** | "address delivery or behavioural concerns constructively" | "I raised it in the first week with specific examples, agreed three measurable expectations and a six-week review. Two of three improved; the third we moved to a role that fit." |
| 5 | **Building the community** | "build and connect the AI community… reusable practices" | "I started a fortnightly internal AI forum — 20 minutes, one person presents work in progress. It became where reusable components actually got adopted." |
| 6 | **Capability plan** | "individual capability and development plans" | "I keep a skills matrix across the four disciplines; each person has one stretch area per quarter and one thing they'd teach the team." |

**Anti-pattern to avoid:** "I mentored juniors and helped them grow." That sentence is the *reason* the panel wrote the feedback. Never say it. Replace with a named person, a mechanism, and an outcome.

---

## 4. Conflict and difficult conversations — the scripts

### 4.1 "Tell me about a conflict you resolved."

> **Headline:** *"Two teams wanted different things and it was costing us a quarter. I got them to a decision in a week by moving the argument from 'which design' to 'which criterion'."*
> **Then:** who the parties were, what the real disagreement was underneath (usually ownership or risk, not technology), the mechanism you used (a written one-pager with both options and named trade-offs; a decision owner agreed up front), and the outcome. **End with the principle:** *"I've found the fastest way out of a technical disagreement is to make the decision criteria explicit before anyone defends a solution."*

### 4.2 "How do you handle an underperforming team member?"

> *"Early, specific, and documented. I don't let it drift, because drift is unfair to the person and to the team. I'd open with the observed behaviour and its impact, not a personality judgement: 'the last three deliverables slipped and the team had to absorb it'. Then agree two or three measurable expectations and a short review window. Most of the time that's enough, because the person usually knows. If it isn't, I escalate into the formal process — but by then the expectation is already written down, so it's fair to everyone."*

### 4.3 "How do you give feedback?"

> *"Same-day, evidence-based, and about the work not the person. In 1:1s I hold a slot for it so it's never a surprise. And I do it for good work too — recognition is cheaper than it looks and the JD asks for it."*

### 4.4 "How do you foster psychological safety / inclusion?"

> *"Two mechanisms, mainly. First, I run the review forum so that work-in-progress is normal — if only finished work is shared, nobody asks for help. Second, I make the silent voices explicit: in every meeting I ask two people I haven't heard from by name. It sounds mechanical, but it changes who speaks."*

---

## 5. Standards you author (the Lead differentiator)

The JD asks you to "set standards for architecture, code review, testing, CI/CD, MLOps/LLMOps, observability, reliability, cost and performance". Have **three** ready with a *consequence*:

| Standard | The rule | What it stopped |
|---|---|---|
| **Eval gate in CI** | No prompt/retrieval/model change merges without passing the golden set | Silent quality regressions |
| **Every service has a metric and an owner** | A dashboard and a named owner, or it doesn't ship | Orphaned systems nobody dares change |
| **Cost budget per unit** | Every AI path has a per-unit cost target and an alert at 80% | Runaway token spend found on the invoice |
| **Trace everything** | Every interaction logged: tokens, latency, retrieved chunks, model version, cost | "It's slower today" being unanswerable |
| **Structured outputs + server-side validation** | The LLM proposes, the validator disposes | Malformed output reaching the workflow |
| **Rollback in one command** | Every deploy has a tested rollback | Long outages |

**Line to memorise:** *"Standards only count if they have a consequence. If nothing breaks when you violate it, it's a suggestion."*

---

## 6. Service design & business integration — the framing that closes gap #1 and #2

This is the highest-value paragraph in the pack. Say a version of it at least once in the hour.

> *"I think about the model as maybe 10% of the work. The questions that decide whether an AI service succeeds are: **who acts on the output**, **which system of record it writes back to**, **what happens when it's wrong**, **how the human's day changes**, and **what we measure six months after go-live**. On the last project the model was the easy part — the hard parts were getting the output format the downstream workflow could consume, agreeing who owns a wrong extraction, and getting people to stop double-checking the model after the first month, which is the real signal of trust."*

### 6.1 The service-design checklist (ask these in any design you're given)

| Question | Why |
|---|---|
| Who consumes the output, and in what system? | Integration, not the model, is what fails |
| What is the cost of a false positive vs a false negative? | Decides the metric (see file `07`) |
| What does the human do differently on Monday? | Adoption |
| What is the fallback when the AI is unavailable? | Reliability — regulated services need a manual path |
| What is logged, for how long, and who can audit it? | Model risk / FCA / PRA |
| What is the unit cost, and who pays it? | Commercial viability |
| Who signs off that it's fit for use? | Governance |

### 6.2 Stakeholder translation — the three audiences

| Audience | They care about | Say this |
|---|---|---|
| **Underwriter** | "Will this slow me down, and can I trust it?" | "It proposes and shows its source. You approve or override — you stay the decision-maker." |
| **Product/business owner** | "What's the value and the risk?" | "It removes X minutes per submission on the top-quartile volume, with a human gate on anything below our confidence threshold." |
| **Risk/compliance** | "Can we defend this to a regulator?" | "Every output is traceable to a source, every decision is logged, and the model's limitations are documented in a card." |

**Drill:** explain your project to each of the three audiences in one sentence each. If one version leaks jargon into a non-technical audience, rewrite it.

---

## 7. "Working with stakeholders and teams" — the cross-site story

Brit has **UK and India** teams (AI Enablement, AI UW Labs). Expect a question about distributed collaboration.

> *"I've worked across sites — the pattern that works for me is: one owner per outcome regardless of site, written decisions rather than verbal ones, and a deliberate overlap window each day. Hand-offs fail when they rely on someone remembering a conversation. I also make sure the India side isn't just a build shop — the people closest to the data should propose, not just implement."*

---

## 8. Questions they will ask in this round (20)

**People**
1. How big was your team? What did you own?
2. How do you set goals and allocate work across a mixed team?
3. How do you develop an engineer who wants to go from science into engineering?
4. Tell me about an underperformer.
5. Tell me about a conflict.
6. How do you give difficult feedback?
7. How do you run a 1:1?
8. How do you build a community of practice?
9. How do you hire? What do you look for?
10. How would you handle two people who can't work together?

**Business/stakeholders**
11. How do you decide what to build, when the business asks for everything?
12. How do you say no to a stakeholder?
13. How do you measure whether an AI project succeeded?
14. Tell me about a project you stopped — and why.
15. How do you explain a model's limitations to a non-technical stakeholder?
16. How do you get people to actually adopt the AI?
17. How do you handle a stakeholder who doesn't trust the model?

**Leadership judgement**
18. What standards have you set?
19. How do you balance delivery, experimentation, production support and development (the JD's exact list)?
20. What's your approach to the first 90 days?

**Answers to #19 and #12, since they're the least obvious:**

> #19: *"I protect four buckets: roughly 70% delivery, 15% production support, 10% exploration, 5% capability development — not as a rigid split but as something I can point to when a week has gone 100% support. If support creeps above its share for two weeks running, that's a signal to fix reliability, not to hire."*

> #12: *"By giving them a choice of when, not whether. I'd say: 'we can do A now, or B now and A next quarter — which is worth more to you?' Saying no outright makes you a gatekeeper; making them choose makes them a partner. If it's genuinely non-negotiable, I say clearly what it would displace."*

---

## 9. Common mistakes in this part

| Mistake | Fix |
|---|---|
| Bluffing line management | Be precise about span; frame technical leadership honestly |
| "I mentored juniors" | Named person + mechanism + outcome |
| Only talking model quality | Name the workflow, the system of record, the human gate |
| Vague conflict story | Make the *mechanism* explicit (criteria before solutions) |
| No mention of the UK/India dimension | Have the distributed-collab answer ready |
| Treating safety/inclusion as a cliché | Give the two concrete mechanisms from §4.4 |

---

## 10. Cheat-sheet summary

- **The panel gap is the rubric:** service design · business integration · stakeholders · mentoring.
- **Team narrative = structure, not a number**: disciplines, one owner per outcome, cadence, capability plan.
- **Six mentoring stories**, using the JD's own words: pairing, technical reviews, stretch, concerns, community, capability plan.
- **"The model is 10% of the work."** Then name the workflow, the system of record, the human gate, the metric.
- **Standards need consequences.** Eval gate, dashboard+owner, cost budget, traces, validation, rollback.
- **Three audiences, three translations**: underwriter, business, risk/compliance.

## 11. Cross-links

- Where to use these lines in the hour: [`02-stage1-format-60-minute-battle-plan.md`](02-stage1-format-60-minute-battle-plan.md)
- Story bank (stories #5–#12): [`03-introduction-and-story-bank.md`](03-introduction-and-story-bank.md) §2
- Metrics that connect AI to business: [`07-agent-evaluation-and-guardrails.md`](07-agent-evaluation-and-guardrails.md)
- Regulated-environment vocabulary: [`11-insurance-lloyds-domain-primer.md`](11-insurance-lloyds-domain-primer.md)
