# Day 1: Insurance Analytics Career Landscape - Interview Questions & Case Studies

## PART 1: INTERVIEW QUESTIONS

### Question 1: Career Path Understanding
**Q: "Explain the difference between a Data Scientist and an Actuary in insurance. When would you use one over the other?"**

**Model Answer:**
"The key differences lie in credentials, regulatory responsibility, and focus areas. An Actuary holds professional credentials (ASA/FSA from SOA or ACAS/FCAS from CAS) and is legally authorized to sign off on reserves and rate filings. They focus on pricing, reserving, and capital modeling using established methods like GLMs and Chain-Ladder. A Data Scientist typically has a Master's in DS or related field, uses ML techniques (GBMs, neural networks), and focuses on fraud detection, customer analytics, and optimization problems that don't require regulatory sign-off.

You'd use an Actuary when: (1) Regulatory filing is required, (2) Traditional methods are sufficient, (3) Interpretability is critical. You'd use a Data Scientist when: (1) No regulatory constraint, (2) Complex non-linear patterns exist, (3) Large datasets are available, (4) Speed of iteration matters.

The trend is toward hybrid roles - 'Actuarial Data Scientists' who combine both skill sets."

**Follow-ups:**
1. "Can a Data Scientist sign off on reserves?" (No, requires actuarial credentials)
2. "Are actuaries learning ML?" (Yes, SOA Exam PA now covers predictive analytics)

---

### Question 2: Business Acumen
**Q: "A P&C insurer has a combined ratio of 105%. What does this mean and what would you recommend?"**

**Model Answer:**
"A combined ratio of 105% means the insurer is losing 5 cents on every dollar of premium from underwriting. The formula is (Losses + Expenses) / Premium. Above 100% indicates underwriting losses.

My recommendations would be:
1. **Immediate**: Analyze by segment (state, product, channel) to identify unprofitable pockets
2. **Short-term**: File for rate increases where actuarially justified and regulatorily allowed
3. **Medium-term**: Tighten underwriting standards, non-renew bottom 10% of policies
4. **Long-term**: Improve claims management, reduce expenses, consider product redesign

I'd also check if investment income compensates (operating ratio = combined ratio - investment income ratio). Some insurers accept 102-103% combined ratios if investment returns are strong."

---

### Question 3: Technical Foundation
**Q: "Walk me through how you would price a new auto insurance product from scratch."**

**Model Answer:**
"I'd follow a structured approach:

**Phase 1: Data Collection (Weeks 1-2)**
- Gather internal data: policies, claims, exposures
- Obtain external data: competitor rates, industry loss costs (ISO)
- Identify rating variables: age, vehicle, territory, coverage

**Phase 2: Exploratory Analysis (Weeks 3-4)**
- Calculate current loss ratios by segment
- Identify trends in frequency and severity
- Check for missing data and outliers

**Phase 3: Modeling (Weeks 5-8)**
- Build separate GLMs for frequency (Poisson) and severity (Gamma)
- Select variables using AIC/BIC and business judgment
- Validate on hold-out data (temporal split)

**Phase 4: Rate Making (Weeks 9-10)**
- Combine frequency × severity = pure premium
- Add expense loadings and profit margin
- Create rate tables (relativities by age, territory, etc.)

**Phase 5: Implementation (Weeks 11-12)**
- Document methodology for regulatory filing
- Build rating engine in production system
- Set up monitoring dashboards

Key decisions: GLM vs. GBM (I'd start with GLM for regulatory acceptance), granularity of rating variables, expense allocation method."

---

### Question 4: Regulatory Awareness
**Q: "Your model uses credit score and improves accuracy by 8%. A regulator questions its use due to potential bias. How do you respond?"**

**Model Answer:**
"I would take this seriously and conduct a thorough analysis:

**Step 1: Acknowledge & Analyze**
- Perform disparate impact analysis by protected class
- Calculate the ratio of average premium impact across groups
- Test if credit score is predictive within each demographic group

**Step 2: Quantify Trade-offs**
- Removing credit score loses 8% accuracy
- This could lead to adverse selection (good risks leave, bad risks stay)
- Estimate the financial impact: if loss ratio increases 2%, that's $X million

**Step 3: Propose Solutions**
- Option A: Remove credit score entirely (safest regulatory path)
- Option B: Reduce weight by 50%, add fairness constraints
- Option C: Replace with alternative variables (payment history, prior insurance)
- Option D: Defend with evidence of business necessity

**My Recommendation**: Option B (hybrid approach)
- Reduces disparate impact while retaining some predictive power
- Shows good faith effort to address concerns
- Implement ongoing monitoring for fairness metrics

**Documentation**: I'd document all analysis and decisions for regulatory review and model governance."

---

### Question 5: Problem Solving
**Q: "You have 3 years of claims data, but there was a system change 18 months ago that affected claim coding. How do you handle this in reserving?"**

**Model Answer:**
"This is a structural break problem. I'd approach it systematically:

**Step 1: Quantify the Impact**
- Compare claim coding before and after the change
- Identify which fields changed and how
- Estimate the magnitude of distortion

**Step 2: Evaluate Options**
**Option A: Use Only Post-Change Data (18 months)**
- Pro: Clean, consistent data
- Con: Limited history, less credibility
- Use when: Change is fundamental and cannot be adjusted

**Option B: Adjust Pre-Change Data**
- Pro: Full 3 years of data
- Con: Adjustment introduces error
- Method: Create mapping from old codes to new codes
- Use when: Mapping is straightforward

**Option C: Segment Analysis**
- Pro: Avoids mixing incompatible data
- Con: More complex
- Method: Run separate triangles for pre/post periods, blend with credibility
- Use when: Partial overlap exists

**My Recommendation**: Option C for reserving
- Build two triangles: pre-change (older AYs) and post-change (recent AYs)
- Use pre-change data for mature years, post-change for immature
- Apply credibility weighting in the transition period

**Validation**: Compare results across all three methods to assess sensitivity."

---

## PART 2: CASE STUDIES

### Case Study 1: Career Decision Framework

**Background:**
Sarah is a recent graduate with a BS in Mathematics. She has three job offers:

1. **Actuarial Analyst at Regional P&C Insurer**
   - Salary: $62K
   - Exam support: Full (study time + materials + bonuses)
   - Work: Pricing and reserving for commercial lines
   - Location: Midwest city (low cost of living)

2. **Data Scientist at InsurTech Startup**
   - Salary: $95K + equity
   - Exam support: None
   - Work: Building ML models for fraud detection
   - Location: San Francisco (high cost of living)

3. **Actuarial Consultant at Big 4 Firm**
   - Salary: $70K
   - Exam support: Partial (materials only, limited study time)
   - Work: Diverse projects across life and P&C
   - Location: New York (high cost of living)

**Your Task:**
Help Sarah make a decision by analyzing each option across multiple dimensions.

**Solution:**

**Dimension 1: Financial Analysis (5-Year Projection)**

| Year | Option 1 (P&C Insurer) | Option 2 (Startup) | Option 3 (Consulting) |
|------|------------------------|--------------------|-----------------------|
| 1 | $62K | $95K | $70K |
| 2 | $68K (1 exam) | $105K | $77K |
| 3 | $76K (2 exams) | $115K or $0 (if startup fails) | $88K (1 exam) |
| 4 | $88K (3 exams) | $130K | $100K (2 exams) |
| 5 | $105K (ASA/ACAS) | $150K | $120K (3 exams) |

**Adjusted for Cost of Living:**
- Midwest: 100% of salary
- SF: 60% of salary (due to 1.67x cost of living)
- NYC: 70% of salary (due to 1.43x cost of living)

**Real Income (Year 5):**
- Option 1: $105K × 100% = $105K
- Option 2: $150K × 60% = $90K (if startup survives)
- Option 3: $120K × 70% = $84K

**Dimension 2: Career Capital**

**Option 1 Strengths:**
- Clear path to credentials (ACAS/FCAS)
- Deep expertise in pricing/reserving
- Job security (actuaries always needed)

**Option 2 Strengths:**
- Cutting-edge ML experience
- Equity upside (could be worth $500K+ if startup succeeds)
- Fast-paced learning environment

**Option 3 Strengths:**
- Broad exposure (life, P&C, health)
- Big 4 brand name (prestigious)
- Consulting skills (client management, presentations)

**Dimension 3: Risk Assessment**

**Option 1 Risks:**
- Slower salary growth initially
- Regional insurer may have limited upside
- 5-7 years to credential

**Option 2 Risks:**
- 60% of startups fail within 3 years
- No actuarial credentials (harder to pivot back)
- Equity may be worthless

**Option 3 Risks:**
- Long hours (60-70/week common in consulting)
- Limited exam study time (may slow credential progress)
- High cost of living

**Recommendation:**

**If Sarah values stability and long-term earning potential: Option 1**
- Best exam support
- Clearest path to $200K+ (as FCAS)
- Lowest risk

**If Sarah is risk-tolerant and wants to maximize upside: Option 2**
- Highest immediate salary
- Equity could be life-changing
- Best for learning cutting-edge skills

**If Sarah wants optionality and prestige: Option 3**
- Broad experience keeps options open
- Big 4 brand opens doors
- Can pivot to industry or stay in consulting

**My Personal Recommendation: Option 1 with a twist**
- Take the P&C insurer role
- Pass 3-4 exams in first 3 years
- Learn Python/ML on the side
- After 3 years, pivot to a hybrid role (Actuarial Data Scientist) at a larger insurer or InsurTech
- This gives you the best of both worlds: credentials + modern skills

---

### Case Study 2: Unprofitable Book Turnaround

**Background:**
You're hired as VP of Pricing at a mid-sized auto insurer. The company has been losing money for 3 years:

**Financial Summary:**
- 2021: Combined Ratio 107%, Premium $400M
- 2022: Combined Ratio 109%, Premium $450M
- 2023: Combined Ratio 112%, Premium $500M

**Additional Data:**
- Market share growing (good sign?)
- Expense ratio stable at 30%
- Loss ratio increasing: 77% → 79% → 82%
- Regulatory cap on rate increases: 7% per year
- Competitor A (similar size): Combined Ratio 98%

**Your Task:**
1. Diagnose the root cause
2. Develop a turnaround plan
3. Quantify expected results

**Solution:**

**Step 1: Diagnostic Deep Dive**

**Hypothesis 1: Inadequate Pricing**
- Current loss ratio: 82%
- Target loss ratio: 70% (to achieve 100% combined ratio)
- Required rate increase: 82/70 - 1 = 17%
- Regulatory cap: 7%
- **Conclusion**: We're 10 points short of where we need to be

**Hypothesis 2: Adverse Selection**
- Growing market share while losing money suggests we're winning on price
- Likely attracting high-risk customers from competitors
- **Test**: Compare new business vs. renewal loss ratios
  - If new business LR > renewal LR, confirms adverse selection

**Hypothesis 3: Claims Inflation**
- Auto repair costs: +12% per year (supply chain, labor)
- Medical costs: +8% per year
- Our trend assumption in pricing: 5% per year
- **Conclusion**: We're underestimating inflation by 3-7 points

**Hypothesis 4: Geographic Concentration**
- **Test**: Break down combined ratio by state
- **Finding**: 
  - State A: CR 95% (profitable)
  - State B: CR 125% (disaster)
- **Root Cause in State B**: High litigation rates, severe weather

**Step 2: Turnaround Plan**

**Immediate Actions (Months 1-3):**
1. **File Maximum Rate Increase**: +7% across the board
2. **Tighten Underwriting**: 
   - Increase credit score threshold
   - Non-renew bottom 5% of policies by loss ratio
   - Impact: Reduce premium by 3%, improve LR by 2 points

3. **Geographic Rebalancing**:
   - Stop writing new business in State B
   - Non-renew unprofitable segments in State B
   - Impact: Reduce premium by 5%, improve LR by 3 points

**Short-Term Actions (Months 4-9):**
4. **Improve Claims Management**:
   - Implement AI fraud detection (save 1% of losses)
   - Accelerate subrogation (recover 0.5% of losses)
   - Impact: Improve LR by 1.5 points

5. **Product Redesign**:
   - Introduce higher deductible options ($1,000 → $2,500)
   - Offer telematics discount (attract better risks)
   - Impact: Improve LR by 1 point

**Medium-Term Actions (Months 10-18):**
6. **Expense Reduction**:
   - Digitize customer service (reduce call center costs by 15%)
   - Renegotiate vendor contracts
   - Impact: Reduce expense ratio by 2 points

7. **Advanced Analytics**:
   - Build GBM pricing model (improve risk selection)
   - Implement dynamic pricing (adjust rates monthly)
   - Impact: Improve LR by 2 points

**Step 3: Quantified Projections**

**2024 Projection (With Plan):**
- Premium: $500M × 1.07 (rate) × 0.92 (volume reduction) = $492M
- Loss Ratio: 82% - 2% (UW) - 3% (geo) - 1.5% (claims) - 1% (product) - 2% (analytics) = 72.5%
- Expense Ratio: 30% - 2% (efficiency) = 28%
- **Combined Ratio: 100.5%** (break-even)

**2025 Projection (Full Effect):**
- Premium: $492M × 1.07 = $526M
- Loss Ratio: 70%
- Expense Ratio: 27%
- **Combined Ratio: 97%** (profitable!)
- **Underwriting Profit: $526M × 3% = $15.8M**

**Investment Required:**
- Technology (fraud detection, telematics platform): $3M
- Severance (staff reductions): $2M
- Total: $5M

**ROI: $15.8M / $5M = 316% in Year 2**

**Step 4: Risk Mitigation**

**Risk 1: Regulatory Pushback on Non-Renewals**
- Mitigation: Phase non-renewals over 18 months, document actuarial justification

**Risk 2: Competitor Response**
- If competitors also raise rates, we're fine
- If competitors hold rates, we may lose more volume
- Mitigation: Monitor competitor rates weekly, adjust strategy

**Risk 3: Execution Challenges**
- Mitigation: Hire experienced turnaround consultant, weekly steering committee

---

## PART 3: INDUSTRY APPLICATIONS

### Application 1: Progressive's Snapshot Program

**Challenge:**
Progressive wanted to price based on actual driving behavior, not just demographics.

**Solution:**
Launched Snapshot in 2008 - a telematics device that tracks:
- Miles driven
- Time of day
- Hard braking events

**Results:**
- 10M+ enrolled customers
- Average discount: 10-15% for safe drivers
- Improved loss ratio by 3-5 points on Snapshot policies
- Competitive advantage (first-mover in telematics)

**Lessons:**
1. Innovation can differentiate in a commoditized market
2. Customers will share data for discounts
3. Telematics data improves risk selection

---

### Application 2: Lemonade's AI Claims Processing

**Challenge:**
Traditional claims processing is slow (weeks) and expensive ($100-200 per claim).

**Solution:**
Lemonade built an AI bot (Jim) that:
- Processes simple claims in 3 seconds
- Uses computer vision to verify damage
- Detects fraud with ML models

**Results:**
- 30% of claims paid instantly
- Claims expense reduced by 60%
- Customer satisfaction (NPS) increased by 40 points

**Lessons:**
1. AI can dramatically reduce costs in high-volume processes
2. Speed improves customer experience
3. Fraud detection must be embedded, not bolted on

---

## PART 4: INTERVIEW PREPARATION TIPS

### Research the Company
- Check recent news (acquisitions, new products, regulatory issues)
- Understand their lines of business (life vs. P&C vs. health)
- Review their annual report (combined ratio, growth rate, strategy)

### Prepare Your Stories (STAR Format)
**Example:**
- **Situation**: "In my internship at XYZ Insurance..."
- **Task**: "I was asked to build a pricing model for a new product..."
- **Action**: "I gathered data, built a GLM, validated on hold-out data..."
- **Result**: "The model improved accuracy by 12%, leading to a 3-point improvement in loss ratio, saving $2M annually."

### Common Mistakes to Avoid
- ❌ Being too theoretical (ground answers in practical examples)
- ❌ Not asking questions (shows lack of curiosity)
- ❌ Badmouthing previous employers
- ❌ Claiming to know everything (admit when you don't know)

### Questions to Ask the Interviewer
1. "What does success look like in this role in the first 6 months?"
2. "How does the company balance traditional actuarial methods with modern ML?"
3. "What's the biggest challenge the team is facing right now?"
4. "How does the company support exam progress / professional development?"

---

*Document Version: 1.0*
*Lines: 500+*
