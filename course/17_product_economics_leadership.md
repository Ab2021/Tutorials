# Module 17 — Product, Economics, and Leadership

> **Program:** Biomedical Data Science for Industry (2026)
> **Time:** 24 hours (9 lecture, 5 case clinic, 10 lab)
> **Track fit:** Common core; required for product and leadership roles
> **Prerequisites:** Modules 00, 05, 14
> **Tools:** Value-case spreadsheet, stakeholder map, budget-impact model, pitch template
> **Industry question:** "Who pays, who adopts, what changes, and why is this worth the money and risk?"

---

## 1. Why industry cares

A technically excellent model can fail because:

- No one will pay for it.
- It does not fit the workflow.
- The buyer is not the user.
- The evidence does not satisfy the payer.
- The integration cost exceeds the benefit.
- The organization cannot operationalize it.
- The model creates liability without a contract.

Data scientists are often the last to learn this. This module teaches the commercial and organizational literacy to avoid it.

## 2. Learning outcomes

By the end of this module, you can:

- Map stakeholders, users, buyers, and blockers.
- Build a value proposition tied to a measurable outcome.
- Quantify cost avoidance, revenue, capacity, and quality impact.
- Explain cost-effectiveness and budget impact.
- Navigate reimbursement and procurement.
- Design an adoption and change-management plan.
- Choose a business model.
- Prioritize a portfolio under uncertainty.
- Communicate with executives, clinicians, and regulators.
- Lead a cross-functional data product team.

## 3. Mental model: value = outcome × adoption − cost − risk

```
Clinical/operational problem
        │
        ├── User: who does the work?
        ├── Buyer: who pays?
        ├── Beneficiary: who benefits?
        ├── Blocker: who can say no?
        └── Workflow: what changes?
                │
                ▼
        Value proposition
                │
        ├── Clinical outcome
        ├── Operational capacity
        ├── Financial return
        └── Strategic option value
                │
                ▼
        Evidence → Adoption → Contract → Scale
```

## 4. Product discovery

### 4.1 Questions

1. What job is the user trying to do?
2. What is painful, frequent, and measurable?
3. What do they do today?
4. What would they stop doing?
5. Who controls the budget?
6. What evidence would make them switch?
7. What would make them say no?

### 4.2 Stakeholder map

| Stakeholder | Role | Incentive | Risk |
|---|---|---|---|
| Patient | Beneficiary | Better care, access | Privacy, harm |
| Clinician | User | Time, trust, liability | Alert fatigue, autonomy |
| Department chief | Budget holder | Throughput, quality | Cost, disruption |
| CIO/CTO | Technical owner | Integration, security | Vendor lock-in |
| Compliance | Gatekeeper | Legal risk | Audit failure |
| Payer | Funder | Cost, quality | Unnecessary spend |
| Procurement | Buyer | Value, terms | Liability |
| Regulator | Approver | Safety, effectiveness | Insufficient evidence |

## 5. Value quantification

### 5.1 Value categories

| Category | Example | Measurement |
|---|---|---|
| Cost avoidance | Avoided admissions | Admissions × cost |
| Revenue | Additional procedures | Volume × margin |
| Capacity | Reduced length of stay | Bed-days freed |
| Quality | Reduced complications | Rate × cost |
| Access | Reduced wait time | Third-next-available |
| Experience | Less documentation | Minutes saved |
| Strategic | Market position | Option value |

### 5.2 ROI template

```
Annual benefit =
  (baseline events − intervention events) × value per event
  + capacity value
  + quality incentive value

Annual cost =
  license + implementation + integration + training
  + clinical review time + monitoring + maintenance

ROI = (annual benefit − annual cost) / annual cost
```

**Rule:** include the cost of false positives and clinician review, not just the model.

## 6. Health economics

### 6.1 Cost-effectiveness

- **Cost per QALY:** incremental cost divided by incremental quality-adjusted life years.
- **ICER:** incremental cost-effectiveness ratio.
- **Willingness-to-pay threshold:** context-specific.
- **Perspective:** payer, health system, societal.
- **Time horizon:** long enough to capture costs and benefits.
- **Discounting:** future costs and benefits.

### 6.2 Budget impact

- Annual cost to the payer.
- Eligible population.
- Uptake.
- Offset savings.
- Net budget impact.

**Rule:** cost-effectiveness and budget impact answer different questions. A cost-effective technology can still blow a budget.

## 7. Reimbursement and procurement

### 7.1 US pathways

- CPT codes for procedures.
- CMS payment rules.
- Coverage decisions.
- Prior authorization.
- Value-based contracts.

### 7.2 Evidence for payers

- Clinical utility.
- Cost-effectiveness.
- Budget impact.
- Real-world outcomes.
- Equity.

### 7.3 Procurement

- Security review.
- Integration requirements.
- Liability and indemnification.
- References.
- Total cost of ownership.
- Exit terms.

## 8. Adoption

### 8.1 Adoption drivers

- Clear workflow fit.
- Trusted evidence.
- Clinical champion.
- Minimal added work.
- Visible benefit.
- Training and support.
- Payment alignment.

### 8.2 Change management

- Identify champions and blockers.
- Co-design with users.
- Pilot with feedback.
- Measure and share results.
- Iterate.
- Scale.

### 8.3 Adoption metrics

- Active users.
- Use rate at eligible encounters.
- Time saved.
- Override rate.
- Alert burden.
- Outcome change.
- Retention.

## 9. Business models

| Model | Description | Best for | Risk |
|---|---|---|---|
| SaaS subscription | Recurring fee | Health systems, payers | Long sales cycles |
| Per-use | Fee per prediction/scan | Diagnostics, imaging | Volume uncertainty |
| Enterprise license | Site license | Large systems | Procurement complexity |
| Device sale | Hardware + software | MedTech | Capital cost |
| Pharma partnership | Co-development | Biomarkers, trials | Long timelines |
| Value-based | Payment tied to outcomes | Care management | Attribution |

## 10. Leadership

### 10.1 Cross-functional team

- Clinical lead.
- Product manager.
- Data scientists.
- Engineers.
- Regulatory/quality.
- Security/privacy.
- Health economist.
- Commercial.

### 10.2 Communication

- To clinicians: patient impact, workflow, uncertainty.
- To executives: value, risk, timeline, options.
- To regulators: intended use, evidence, controls.
- To engineers: requirements, interfaces, failure modes.
- To patients: benefits, limitations, privacy.

### 10.3 Prioritization

Use a portfolio view:

- Impact.
- Probability of success.
- Time to value.
- Cost.
- Strategic fit.
- Risk.

## 11. Industry case studies

### 11.1 IDx-DR

IDx-DR combined a narrow clinical claim, a prospective trial, and a primary-care workflow. Reimbursement and adoption required more than FDA authorization.

**Lesson:** regulatory clearance is necessary but not sufficient.

Source: FDA De Novo DEN180001. https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180001.pdf

### 11.2 Remote patient monitoring

Remote monitoring can reduce admissions in selected heart-failure populations, but adoption depends on reimbursement, workflow, and adherence.

**Lesson:** the business model and care pathway determine value.

Source: Cochrane review, 2015. https://doi.org/10.1002/14651858.CD007228.pub3

### 11.3 AI documentation

Ambient documentation tools have a clear value proposition: clinician time. Evidence on note quality and safety is still maturing.

**Lesson:** time savings are measurable, but quality and liability matter.

Source: AMA documentation burden resources. https://www.ama-assn.org/

### 11.4 Sepsis alerts

Sepsis alerts can improve care when paired with a response team, but false alerts create cost and fatigue.

**Lesson:** value depends on the full intervention, not the model.

Source: Wong, A. et al., *JAMA Internal Medicine*, 2021. https://doi.org/10.1001/jamainternmed.2021.2626

## 12. Labs

### Lab 17.1 — Stakeholder and value map

**Objective:** Map stakeholders and value for a product.

**Steps:**

1. Identify user, buyer, beneficiary, blocker.
2. Map incentives and risks.
3. Quantify baseline and benefit.
4. Identify adoption barriers.
5. Write a one-page value proposition.

**Deliverable:** value map.

**Acceptance criteria:** buyer ≠ user considered; blockers named.

### Lab 17.2 — ROI and budget impact

**Objective:** Build a 3-year value case.

**Steps:**

1. Estimate eligible population.
2. Estimate effect size.
3. Estimate uptake.
4. Estimate costs.
5. Run sensitivity analysis.
6. Present net impact.

**Deliverable:** spreadsheet and memo.

**Acceptance criteria:** assumptions; sensitivity; total cost.

### Lab 17.3 — Cost-effectiveness

**Objective:** Calculate an ICER for a screening program.

**Steps:**

1. Define comparators.
2. Estimate costs and QALYs.
3. Discount future effects.
4. Calculate ICER.
5. Compare with threshold.
6. Discuss uncertainty.

**Deliverable:** model and memo.

**Acceptance criteria:** perspective; time horizon; sensitivity.

### Lab 17.4 — Adoption plan

**Objective:** Design a 12-month adoption plan.

**Steps:**

1. Identify champions.
2. Pilot site.
3. Training.
4. Feedback loop.
5. Metrics.
6. Scale criteria.

**Deliverable:** plan.

**Acceptance criteria:** workflow; ownership; metrics.

## 13. Assignment: product value case

Write a 2,000-word value case:

1. Problem and user.
2. Stakeholder map.
3. Value proposition.
4. Evidence.
5. ROI and budget impact.
6. Reimbursement path.
7. Adoption plan.
8. Risks.
9. 3-year roadmap.
10. Recommendation.

## 14. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Problem | 20% | Technology looking for use | Clear user and workflow | Quantified pain |
| Value | 25% | Generic savings | ROI with assumptions | Sensitivity and payer perspective |
| Adoption | 20% | Ignores | Plan and champions | Change management and metrics |
| Economics | 20% | No model | ROI/budget impact | Cost-effectiveness and uncertainty |
| Communication | 15% | Technical | Clear | Executive-ready |

## 15. Regulatory, ethical, and commercial lens

- **Reimbursement:** coverage is separate from approval.
- **Procurement:** security and liability are gatekeepers.
- **Equity:** adoption can widen disparities if access is unequal.
- **Transparency:** value claims must be evidence-based.
- **Commercial:** price to value, not to cost.

## 16. Failure modes and red flags

- No buyer.
- No workflow change.
- ROI without cost of review.
- Ignoring procurement.
- Assuming approval equals adoption.
- Pilot purgatory.
- No champion.
- No measurement plan.
- Overpromising.
- Underestimating maintenance.

## 17. Interview and discussion questions

1. **Who is the buyer vs the user?**
   Hint: clinician uses, administrator pays.
2. **How do you build an ROI case?**
   Hint: baseline, effect, value, costs, sensitivity.
3. **What is an ICER?**
   Hint: incremental cost per QALY.
4. **Why can a cost-effective product fail?**
   Hint: budget impact, adoption, workflow.
5. **How do you get a hospital to adopt?**
   Hint: champion, evidence, integration, workflow, payment.
6. **What is pilot purgatory?**
   Hint: endless pilots without scale.
7. **How do you price a product?**
   Hint: value, alternatives, procurement, cost.
8. **What metrics matter for adoption?**
   Hint: use rate, time saved, override, outcomes.
9. **How do you prioritize a portfolio?**
   Hint: impact, probability, time, cost, strategy.
10. **How do you communicate risk to an executive?**
    Hint: scenarios, probabilities, downside, mitigation.
11. **What is value-based care?**
    Hint: payment tied to quality and cost.
12. **How do you measure success after launch?**
    Hint: adoption, outcomes, cost, equity, safety.

## 18. Key terms

| Term | Definition |
|---|---|
| ROI | Return on investment |
| QALY | Quality-adjusted life year |
| ICER | Incremental cost-effectiveness ratio |
| Budget impact | Net cost to a payer |
| HTA | Health technology assessment |
| CPT | Current Procedural Terminology |
| Value-based care | Payment tied to outcomes |
| Procurement | Purchasing process |
| Total cost of ownership | Full lifecycle cost |
| Adoption | Sustained use in workflow |
| Pilot purgatory | Endless pilot without scale |

## 19. Sources and further reading

- ICER: https://icer.org/
- NICE: https://www.nice.org.uk/
- CMS: https://www.cms.gov/
- FDA, *CDS guidance*. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software
- AMA documentation burden: https://www.ama-assn.org/
- Cochrane review, telemonitoring: https://doi.org/10.1002/14651858.CD007228.pub3
- Porter, M. E., *Redefining Health Care*.
- Christensen, C. M. et al., *The Innovator's Prescription*.

> **Next:** [Module 18](18_quality_evaluation_reporting.md) ties the entire lifecycle together with quality, evaluation, and reporting standards.
