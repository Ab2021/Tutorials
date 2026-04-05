# eBay DS/ML — Behavioral Interview Prep (STAR Method)

> eBay uses a "Bar-Raiser" system. At least one interviewer from outside
> the team evaluates you. Behavioral rounds are NOT a formality.

---

## 🎯 The STAR Framework

```
S — SITUATION:  Set the context (1-2 sentences)
T — TASK:       What was YOUR specific responsibility?
A — ACTION:     What did YOU do? (most detail here)
R — RESULT:     Quantified outcome + what you learned
```

**Rules:**
- Say "I", not "we"
- Be specific with numbers/metrics
- Keep total answer under 3 minutes
- Always end with what you *learned*
- Never badmouth colleagues/managers

---

## 📋 12 Must-Prepare Stories (with Templates)

### Story 1: Impact Project

**Q:** *"Tell me about the most impactful DS project you've worked on."*

```
TEMPLATE:
S: "At [company], our [product/team] was facing [problem: metric declining,
    new opportunity, customer complaint]."

T: "I was responsible for [building model / analyzing data / designing experiment]
    to [solve specific problem]."

A: "First, I [explored data / talked to stakeholders / defined metrics].
    Then I [built model / designed A/B test / created pipeline].
    Key technical decisions:
    - I chose [model/approach] because [trade-off reasoning]
    - I handled [challenge] by [specific solution]
    - I collaborated with [eng/PM] to [deploy/iterate]"

R: "The result was [+X% metric, saved $Y, reduced Z by N%].
    I learned that [key insight about DS impact]."
```

### Story 2: Messy Data

**Q:** *"Tell me about a time you worked with messy or incomplete data."*

```
TEMPLATE:
S: "I joined a project where the data for [use case] had [specific issues:
    missing values, inconsistent formats, duplicate records, no ground truth]."

T: "I needed to create a reliable dataset for [model/analysis] despite
    data quality issues."

A: "I took a systematic approach:
    1. Audited data sources — found [X% missing, Y% duplicated]
    2. Built data quality dashboard to track issues over time
    3. For missing values: used [imputation strategy + justification]
    4. For duplicates: created deduplication logic based on [fuzzy matching]
    5. Communicated data limitations to stakeholders before modeling"

R: "Final model achieved [metric] despite data challenges.
    More importantly, I established a data quality monitoring process
    that reduced future data issues by [X%]."
```

### Story 3: Stakeholder Conflict

**Q:** *"Describe a time your analysis contradicted what stakeholders expected."*

```
TEMPLATE:
S: "A PM/exec believed [their assumption about a feature/metric]
    based on [intuition/anecdotal evidence]."

T: "My analysis showed [different conclusion], and I needed to
    communicate this without damaging the relationship."

A: "Instead of saying 'you're wrong,' I:
    1. Validated my analysis with a colleague to ensure correctness
    2. Scheduled a 1-on-1 (not a group setting) to present findings
    3. Showed the data visually — let the data tell the story
    4. Proposed a compromise: [A/B test / phased rollout / additional analysis]
    5. Acknowledged their perspective and the uncertainty in my analysis"

R: "The stakeholder agreed to [compromise]. Results confirmed
    [my / their / a modified] hypothesis. We shipped [outcome].
    I learned that presenting data with empathy builds trust."
```

### Story 4: Technical Decision Under Uncertainty

**Q:** *"Tell me about a time you made a technical decision with limited information."*

```
TEMPLATE:
S: "We had [tight deadline / limited data / unclear requirements]
    for [project/model]."

T: "I needed to choose between [Option A] and [Option B]
    without full information."

A: "I structured my decision:
    1. Listed known facts vs assumptions for each option
    2. Identified the reversibility — was this a one-way door?
    3. Chose [option] because [it was more reversible / simpler / faster]
    4. Set clear checkpoints to evaluate and pivot if needed
    5. Documented my reasoning so the team could follow the logic"

R: "The decision turned out to be [correct / partially correct].
    At checkpoint 2, I [continued / adjusted]. Final outcome: [metric].
    Lesson: bias toward reversible decisions when uncertain."
```

### Story 5: Cross-Functional Collaboration

**Q:** *"Tell me about working with engineers/PMs to deploy a model."*

```
Highlight:
- How you translated model requirements into engineering specs
- How you handled disagreements on latency/accuracy trade-offs
- How you ensured the model worked in production (monitoring)
- How you iterated based on production feedback
```

### Story 6: Failure / Mistake

**Q:** *"Tell me about a project that failed or where you made a significant mistake."*

```
KEY: Show self-awareness, NOT excuses.

Good pattern:
- "I made [specific technical error: data leakage, wrong metric, bad assumption]"
- "The impact was [delayed launch, wrong recommendation to stakeholders]"
- "I caught it [how and when]"
- "I fixed it by [actions]"
- "I prevented recurrence by [process change, checklist, code review]"
- "What I learned: [genuine insight]"

BAD answer: "The project failed because the PM changed requirements."
GOOD answer: "I should have pushed for a written agreement on success metrics
              before starting. That's now part of my project kickoff process."
```

### Story 7: Prioritization

**Q:** *"How do you prioritize competing requests from multiple teams?"*

```
Framework:
1. Assess impact: Which request moves the most important metric?
2. Assess urgency: Is there a time-sensitive deadline or business event?
3. Assess effort: Quick wins vs. multi-week projects
4. Communicate timeline to all stakeholders (no silent deprioritization)
5. Block dedicated "deep work" time for high-impact projects

Example: "I used an Impact × Effort matrix to prioritize. High impact + low
effort first. For my highest-priority project, I blocked 4-hour focus windows
and set expectations with the other team for a 2-week delivery."
```

### Story 8: Learning New Technology/Domain

**Q:** *"Tell me about a time you had to learn a new technology or domain quickly."*

```
Highlight:
- What the new domain/tech was
- Your learning approach (structured: docs → tutorials → hands-on project)
- How quickly you became productive
- How you applied it to create business value
```

### Story 9: Ownership & Initiative

**Q:** *"Give an example of when you went beyond your job description."*

```
Highlight:
- You noticed a gap/opportunity no one was addressing
- You proposed a solution proactively
- You got buy-in from your manager/team
- You delivered measurable results
- This became a standard practice/tool
```

### Story 10: Communication to Non-Technical Audience

**Q:** *"How do you explain complex technical concepts to non-technical stakeholders?"*

```
Framework:
- Start with the business impact, not the technical details
- Use analogies from everyday life
- Use visuals (charts > tables > equations)
- Provide a clear recommendation with confidence level
- Offer 2-3 options with trade-offs

Example: "Instead of explaining NDCG to the VP, I showed a side-by-side
comparison of search results: 'Here's what users see today (irrelevant
items at top) vs what they'd see with our new model (relevant items at top).
This change would increase purchase rate by 5%.'"
```

### Story 11: Why eBay?

**Q:** *"Why do you want to work at eBay?"*

```
FRAMEWORK (be authentic, but hit these points):

1. SCALE: "eBay operates at a scale that's truly fascinating —
   130+ million active buyers, billions of listings. The ML
   challenges are unique and impactful."

2. TWO-SIDED MARKETPLACE: "The two-sided marketplace creates
   interesting data science problems that don't exist in
   traditional e-commerce — balancing buyer and seller needs,
   handling network effects, building trust systems."

3. AI-FIRST TRANSFORMATION: "I'm excited about eBay's
   investment in GenAI — eBayCoder, Mercury, agentic shopping.
   The Bengaluru GCC is at the center of this transformation."

4. PERSONAL FIT: [Connect to your specific experience]
   "My work on [recommendation/NLP/fraud] directly applies to
   the problems eBay is solving."
```

### Story 12: Career Goals

**Q:** *"Where do you see yourself in 3-5 years?"*

```
GOOD ANSWER:
"In the next 3-5 years, I want to grow into a senior DS leader
who drives end-to-end product impact through ML. Specifically:
- Year 1-2: Master eBay's data landscape, ship impactful models
- Year 2-3: Lead a workstream, mentor junior team members
- Year 3-5: Influence product strategy through data-driven decisions

I'm drawn to eBay because the breadth of problems — from search
to fraud to GenAI — gives me the opportunity to grow across
multiple domains while building deep marketplace expertise."
```

---

## 🏷️ Quick-Reference: Question → Story Mapping

| Question Type | Which Story to Use |
|---|---|
| "Most impactful project" | Story 1 |
| "Dealt with bad data" | Story 2 |
| "Disagreed with someone" | Story 3 |
| "Decision under uncertainty" | Story 4 |
| "Worked with eng/PM" | Story 5 |
| "Made a mistake / failed" | Story 6 |
| "How do you prioritize" | Story 7 |
| "Learned something new fast" | Story 8 |
| "Went above and beyond" | Story 9 |
| "Explained to non-tech" | Story 10 |
| "Why eBay / this role" | Story 11 |
| "Where in 5 years" | Story 12 |

---

## ⚠️ Behavioral Anti-Patterns (AVOID These)

| ❌ Don't | ✅ Do Instead |
|---|---|
| "We built a model that..." | "**I** designed the feature engineering pipeline..." |
| "The PM was unreasonable" | "The PM had a different perspective driven by business pressures" |
| "I can't think of a failure" | Have 2-3 genuine failure stories ready |
| Vague: "I worked on improving metrics" | Specific: "I improved search CTR by 12% by adding BERT embeddings" |
| Long-winded (>5 min answer) | Tight: 2-3 minutes per STAR story |
| Generic: "I'm a team player" | Specific example of HOW you collaborated |
