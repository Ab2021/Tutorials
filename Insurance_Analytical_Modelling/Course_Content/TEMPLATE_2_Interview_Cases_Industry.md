# [TOPIC NAME] - Interview Questions, Case Studies & Industry Applications

## Overview
*This document contains interview questions, case studies, and real-world applications for [Topic]. Use this to prepare for technical interviews and understand practical implementations.*

---

# PART 1: INTERVIEW QUESTIONS (5-10 Questions)

## Question 1: [Conceptual Understanding]

### Question
*[Full interview question as it would be asked]*

**Example:** "Explain the difference between a Poisson and Negative Binomial distribution in the context of insurance claim frequency modeling. When would you use one over the other?"

### Expected Answer Framework

**Level 1 (Basic):** *What a junior candidate should cover*
- Point 1: [Basic definition]
- Point 2: [Key difference]
- Point 3: [Simple example]

**Level 2 (Intermediate):** *What a mid-level candidate should add*
- Point 1: [Technical detail]
- Point 2: [Mathematical property]
- Point 3: [Practical consideration]

**Level 3 (Advanced):** *What a senior candidate should demonstrate*
- Point 1: [Deep insight]
- Point 2: [Business context]
- Point 3: [Trade-offs and alternatives]

### Model Answer
*Comprehensive answer that hits all key points.*

"[Full paragraph answer covering all levels. Should be 200-300 words demonstrating deep understanding, practical knowledge, and business acumen.]

**Key Formula/Concept:**
$$
[Relevant equation if applicable]
$$

**Practical Example:**
[Concrete example from insurance domain]

**Common Follow-up Questions:**
1. "How would you test for overdispersion in your data?"
2. "What if you have excess zeros?"
3. "How does this relate to GLM modeling?"

### Red Flags (What NOT to Say)
- ❌ "They're basically the same thing"
- ❌ "I would just use whatever the default is in the software"
- ❌ [Other common mistakes]

### Bonus Points (Impressive Additions)
- ✓ Mention the relationship to Tweedie distributions
- ✓ Reference specific insurance lines where each is common
- ✓ Discuss computational considerations

---

## Question 2: [Technical/Calculation]

### Question
*[Full question with any data provided]*

**Example:** "You're given a triangle of cumulative paid losses. The latest diagonal shows development from 12 months to 24 months with a link ratio of 1.45. If the current 12-month cumulative loss is $2.3M, what is your estimate of ultimate loss? What assumptions are you making?"

### Expected Answer Framework

**Step 1: Understand the Problem**
- What is being asked?
- What data is provided?
- What is missing?

**Step 2: State Assumptions**
- Assumption 1: [e.g., Past development patterns continue]
- Assumption 2: [e.g., No structural changes]
- Assumption 3: [e.g., Data is accurate]

**Step 3: Perform Calculation**
```
[Show detailed work]
Step 1: [Calculation]
Step 2: [Calculation]
Final Answer: [Value with units]
```

**Step 4: Interpret & Caveat**
- What does this number mean?
- What could make it wrong?
- What would you do next?

### Model Answer
*Full worked solution.*

"[Complete answer with calculations, interpretations, and caveats. Should demonstrate both technical competence and business judgment.]"

**Calculation:**
$$
\text{Ultimate Loss} = \text{Current Loss} \times \text{LDF}
$$
$$
= \$2.3M \times 1.45 = \$3.335M
$$

**Interpretation:**
"This suggests we expect an additional $1.035M in development over the next 12 months. However, this is based on the assumption that..."

**Follow-up Analysis:**
"I would also want to:
1. Check if 1.45 is a reasonable LDF (compare to prior years)
2. Look at the full triangle to see if there are trends
3. Consider external factors (e.g., inflation, legal changes)"

### Common Mistakes
1. **Mistake 1:** Forgetting to state assumptions
2. **Mistake 2:** Not showing work (just giving final answer)
3. **Mistake 3:** Ignoring units or context

### Variations of This Question
- "What if the link ratio has been trending upward?"
- "How would you incorporate a tail factor?"
- "What if this is for a new line of business?"

---

## Question 3: [Modeling/Methodology]

### Question
*[Question about approach or methodology]*

**Example:** "You're building a pricing model for auto insurance. Walk me through your approach from data to deployed model. What are the key decisions you'd need to make?"

### Expected Answer Framework

**Phase 1: Problem Framing**
- Define objective (e.g., predict pure premium)
- Define success metrics (e.g., Gini, RMSE)
- Understand constraints (regulatory, business)

**Phase 2: Data Strategy**
- Data sources needed
- Key features to engineer
- Data quality checks

**Phase 3: Modeling Approach**
- Model selection (GLM vs. GBM vs. Neural Net)
- Justification for choice
- Handling of specific challenges (e.g., exposure, categorical variables)

**Phase 4: Validation & Deployment**
- Validation strategy
- Monitoring plan
- Governance considerations

### Model Answer
*Comprehensive walkthrough demonstrating end-to-end thinking.*

"I would approach this in four phases:

**1. Problem Framing:**
First, I'd clarify the objective. Are we predicting frequency, severity, or pure premium? For auto insurance, I'd likely build separate frequency and severity models because...

[Continue with detailed walkthrough, 400-500 words]

**Key Decision Points:**
1. **Frequency vs. Severity vs. Pure Premium:**
   - Decision: Separate models
   - Rationale: Different distributions, better interpretability
   
2. **GLM vs. GBM:**
   - Decision: Start with GLM for baseline, explore GBM for improvement
   - Rationale: GLM is interpretable and regulatory-friendly, GBM may capture non-linearities

3. **Feature Engineering:**
   - Age bands vs. continuous age
   - Territory granularity
   - Vehicle groupings

**Data Requirements:**
[Detailed list]

**Validation Strategy:**
- Temporal split (train on years 1-3, validate on year 4)
- Metrics: Gini for ranking, RMSE for calibration
- One-way analyses by key variables

**Deployment Considerations:**
- Real-time scoring API vs. batch
- Monitoring for drift
- A/B testing framework"

### Interviewer Follow-ups
1. "How would you handle missing data?"
2. "What if your model shows bias against a protected class?"
3. "How would you explain your model to underwriters?"

### Scoring Rubric (How Interviewers Evaluate)
- **Structure:** Did they organize their answer logically?
- **Completeness:** Did they cover all phases?
- **Depth:** Did they go beyond surface-level?
- **Practicality:** Did they consider real-world constraints?
- **Communication:** Was it clear and concise?

---

## Question 4: [Situational/Behavioral]

### Question
*[Scenario-based question]*

**Example:** "You've built a fraud detection model with 85% precision and 60% recall. Your SIU team says they can only investigate 100 cases per month, but your model is flagging 500. How do you handle this?"

### Expected Answer Framework

**Step 1: Clarify the Situation**
- Understand stakeholder constraints
- Quantify the trade-offs
- Ask clarifying questions

**Step 2: Propose Solutions**
- Solution 1: [Immediate fix]
- Solution 2: [Medium-term improvement]
- Solution 3: [Long-term strategic]

**Step 3: Make a Recommendation**
- Preferred approach
- Justification
- Implementation plan

### Model Answer

"This is a classic precision-recall trade-off problem combined with a resource constraint. Here's how I'd approach it:

**Understanding the Problem:**
- Current state: 500 flags/month, capacity for 100
- Precision 85% means 85 of those 100 will be true fraud
- Recall 60% means we're missing 40% of fraud

**Immediate Solution (This Month):**
I would rank the 500 flagged cases by model score and investigate the top 100. This maximizes the fraud we catch given the constraint.

**Calculation:**
If we investigate top 100 by score, precision likely increases (say to 90%+), so we catch ~90 true fraud cases.

**Medium-Term Solutions (1-3 Months):**
1. **Adjust Threshold:** Increase the decision threshold to flag fewer cases
   - Trade-off: Higher precision, lower recall
   - Tune to get ~100-120 flags/month with 90%+ precision

2. **Tiered Approach:** 
   - Auto-approve low-risk (score < 0.2)
   - Auto-investigate high-risk (score > 0.8)
   - Manual review medium-risk (0.2-0.8)

3. **Efficiency Improvements:**
   - Can we automate parts of the investigation?
   - Can we train the model to predict investigation time?

**Long-Term Solutions (3-6 Months):**
1. **Expand SIU Capacity:** Business case for hiring
   - Calculate ROI: Each investigator costs $X, saves $Y in fraud
   
2. **Improve Model:** 
   - Retrain with feedback from investigations
   - Add new features (e.g., network analysis)
   - Target 90% precision at 70% recall

**My Recommendation:**
Implement the threshold adjustment immediately to get to ~100 flags/month. Simultaneously, build a business case for expanding SIU capacity, showing that we're leaving money on the table by not investigating more cases.

**Metrics to Track:**
- Fraud detection rate ($ saved / $ attempted)
- Investigation efficiency (time per case)
- Model performance over time (precision/recall)"

### What Interviewers Are Looking For
- ✓ Structured thinking
- ✓ Quantitative reasoning
- ✓ Stakeholder management
- ✓ Practical solutions
- ✓ Long-term thinking

### Common Pitfalls
- ❌ Ignoring the business constraint
- ❌ Only proposing a technical solution
- ❌ Not quantifying trade-offs

---

## Question 5: [Regulatory/Ethical]

### Question
*[Question about fairness, bias, or regulation]*

**Example:** "Your pricing model uses credit score as a feature, which improves predictive power by 5% (Gini). However, a regulator has raised concerns about disparate impact on minority populations. How do you respond?"

### Expected Answer Framework

**Step 1: Acknowledge the Concern**
- Show understanding of the ethical issue
- Demonstrate knowledge of regulations

**Step 2: Analyze the Situation**
- What does the data show?
- Is there disparate impact?
- Is credit score a proxy for a protected class?

**Step 3: Propose Path Forward**
- Option 1: Remove the variable
- Option 2: Find alternative variables
- Option 3: Defend the use with evidence

**Step 4: Broader Considerations**
- Company values
- Regulatory landscape
- Industry standards

### Model Answer

"This is a critical question that touches on fairness, regulation, and business performance. Here's how I would approach it:

**1. Acknowledge the Concern:**
I understand the regulator's concern. Even if credit score is not a protected class itself, if it's highly correlated with race or other protected characteristics, using it could result in disparate impact, which is prohibited under fair lending laws.

**2. Conduct Analysis:**
I would perform a disparate impact analysis:

**Step A:** Measure the impact of credit score on pricing by protected class
```
Average Premium Increase from Credit Score:
- Group A (majority): +$50
- Group B (minority): +$150
```

**Step B:** Calculate the disparate impact ratio
```
Ratio = $150 / $50 = 3.0
```
If this ratio exceeds regulatory thresholds (often 1.25), we have a problem.

**Step C:** Test if credit score is actually predictive within each group
- Does credit score predict claims equally well for both groups?
- Or is it only predictive for one group (suggesting it's a proxy)?

**3. Potential Solutions:**

**Option A: Remove Credit Score**
- **Pros:** Eliminates disparate impact concern
- **Cons:** 5% loss in predictive power, potential adverse selection
- **Impact:** Prices become less accurate, may hurt competitiveness

**Option B: Find Alternative Variables**
- **Approach:** Look for variables that are predictive but less correlated with protected classes
- **Examples:** Payment history (not credit score), prior insurance history, telematics
- **Challenge:** May not fully replace predictive power

**Option C: Defend with Business Necessity**
- **Argument:** Credit score is a valid predictor of risk, not used as a proxy
- **Evidence:** Show it's predictive within each demographic group
- **Risk:** May not satisfy regulator, potential litigation

**Option D: Hybrid Approach**
- **Reduce weight** on credit score
- **Add fairness constraints** to the model (e.g., equalized odds)
- **Monitor** for disparate impact continuously

**4. My Recommendation:**

I would recommend **Option D** (Hybrid) with the following implementation:

**Immediate Actions:**
1. Conduct full disparate impact analysis
2. Meet with regulator to understand specific concerns
3. Review company's fair lending policy

**Short-term (1-3 months):**
1. Reduce weight on credit score by 50%
2. Add alternative variables (prior insurance, payment history)
3. Implement fairness constraints in model training

**Long-term (6-12 months):**
1. Research and implement fairness-aware ML techniques
2. Develop ongoing monitoring dashboard for disparate impact
3. Engage with industry groups on best practices

**Business Case:**
- Yes, we lose some predictive power
- But we gain: regulatory compliance, brand protection, ethical high ground
- The cost of non-compliance (fines, reputation damage) far exceeds the cost of a slightly less accurate model

**5. Broader Considerations:**

This isn't just about this one variable. It's about:
- **Company values:** Do we want to be leaders in fair insurance?
- **Regulatory trends:** More states are banning credit score
- **Long-term strategy:** Build models that are fair by design

**Documentation:**
I would ensure all analysis, decisions, and rationale are thoroughly documented for regulatory review and model governance."

### Follow-up Questions
1. "What if removing credit score causes adverse selection?"
2. "How do you define 'fairness' in this context?"
3. "What fairness metrics would you use?"

### Key Concepts to Demonstrate
- ✓ Knowledge of disparate impact
- ✓ Understanding of regulatory landscape
- ✓ Ability to balance competing objectives
- ✓ Ethical reasoning
- ✓ Practical implementation skills

---

## Question 6-10: [Additional Questions]

### Question 6: [Data/Technical]
"You have 5 years of claims data, but there was a major system change 2 years ago that affected how claims are coded. How do you handle this in your reserving analysis?"

**Key Points to Cover:**
- Identify the structural break
- Options: segment data, adjust for change, use only recent data
- Trade-offs of each approach
- Validation strategy

---

### Question 7: [Business Acumen]
"The pricing team wants to use your new ML model, but the underwriting team is skeptical because they can't understand how it works. How do you bridge this gap?"

**Key Points to Cover:**
- Explainability techniques (SHAP, LIME)
- Building trust through validation
- Hybrid approach (GLM + ML)
- Change management

---

### Question 8: [Technical Depth]
"Explain the bias-variance trade-off in the context of insurance modeling. Give a specific example."

**Key Points to Cover:**
- Definition of bias and variance
- Trade-off curve
- Example: GLM (high bias, low variance) vs. GBM (low bias, high variance)
- How to find the sweet spot

---

### Question 9: [Crisis Management]
"Your fraud model has been in production for 6 months. Suddenly, the false positive rate doubles. What do you do?"

**Key Points to Cover:**
- Immediate triage (is the model still running correctly?)
- Data drift analysis
- Concept drift analysis
- Communication with stakeholders
- Short-term and long-term fixes

---

### Question 10: [Strategic Thinking]
"If you could only improve one thing about your company's pricing process, what would it be and why?"

**Key Points to Cover:**
- Demonstrate understanding of the full pricing process
- Identify a genuine pain point
- Quantify the impact
- Propose a realistic solution
- Show strategic thinking

---

# PART 2: CASE STUDIES

## Case Study 1: [Real-World Scenario]

### Background
*Detailed scenario setup (300-400 words)*

**Company Profile:**
- **Type:** Regional auto insurer
- **Size:** $500M in premium
- **Market:** Primarily personal auto in 5 states
- **Current State:** Combined ratio of 105% (unprofitable)

**The Problem:**
The company has been losing money for 3 consecutive years. The CEO has asked you, as the Chief Actuary, to diagnose the problem and propose solutions.

**Data Provided:**
```
Year | Written Premium | Incurred Losses | Expense Ratio | Combined Ratio
2021 | $450M          | $350M           | 30%           | 108%
2022 | $480M          | $375M           | 30%           | 108%
2023 | $500M          | $400M           | 30%           | 110%
```

**Additional Information:**
- Loss ratio has been increasing (77.8% → 78.1% → 80%)
- Premium growth is strong (6-7% per year)
- Market share is increasing
- Competitor A is profitable with a 98% combined ratio
- Recent regulatory changes limited rate increases to 5% per year

### Your Task
1. **Diagnose:** What is causing the unprofitability?
2. **Analyze:** Perform a deep-dive analysis
3. **Recommend:** Propose 3-5 specific actions
4. **Quantify:** Estimate the impact of your recommendations

### Solution Framework

#### Step 1: Diagnostic Analysis

**Hypothesis 1: Inadequate Pricing**
- **Test:** Compare indicated rate to current rate
- **Analysis:** 
  - If loss ratio is 80% and target is 70%, we need a 14% rate increase
  - But regulation caps us at 5%
  - **Conclusion:** Pricing is likely inadequate

**Hypothesis 2: Adverse Selection**
- **Test:** Analyze new business vs. renewal loss ratios
- **Analysis:**
  - If new business loss ratio > renewal, we're attracting bad risks
  - Check if we're winning business from competitors on price
  - **Conclusion:** Growth may be unprofitable

**Hypothesis 3: Claims Inflation**
- **Test:** Trend analysis of claim severity
- **Analysis:**
  - Auto repair costs up 10% per year
  - Medical costs up 8% per year
  - Our pricing trend: 5% (capped by regulation)
  - **Conclusion:** We're not keeping up with inflation

**Hypothesis 4: Expense Issues**
- **Test:** Compare expense ratio to industry
- **Analysis:**
  - Our expense ratio: 30%
  - Industry average: 25%
  - **Conclusion:** Some expense inefficiency, but not the main driver

#### Step 2: Deep-Dive Analysis

**Segmentation Analysis:**
Break down profitability by:
- **Geography:** Which states are profitable?
- **Product:** Liability vs. Physical Damage
- **Customer Segment:** Age, vehicle type, coverage level
- **Channel:** Direct vs. Agent

**Example Finding:**
```
State A: Combined Ratio 95% (profitable)
State B: Combined Ratio 115% (very unprofitable)
```

**Root Cause in State B:**
- Severe weather events (hail) not adequately priced
- Regulatory environment prevents adequate rate increases
- High litigation rates driving up liability costs

#### Step 3: Recommendations

**Recommendation 1: Targeted Rate Actions**
- **Action:** File for maximum allowed rate increase (5%) in State B
- **Action:** Adjust rating algorithm to better reflect risk (e.g., increase weight on territory)
- **Impact:** Improve combined ratio by 3 points
- **Timeline:** 6 months (regulatory approval needed)

**Recommendation 2: Underwriting Tightening**
- **Action:** Increase underwriting standards in unprofitable segments
- **Action:** Non-renew bottom 10% of policies by profitability
- **Impact:** Improve combined ratio by 2 points, reduce premium by 5%
- **Timeline:** Immediate

**Recommendation 3: Product Redesign**
- **Action:** Introduce higher deductibles in State B
- **Action:** Offer usage-based insurance (telematics) to attract better risks
- **Impact:** Improve loss ratio by 2 points
- **Timeline:** 12 months

**Recommendation 4: Claims Management**
- **Action:** Implement AI-powered fraud detection
- **Action:** Improve subrogation recovery
- **Impact:** Reduce loss ratio by 1 point
- **Timeline:** 6 months

**Recommendation 5: Expense Reduction**
- **Action:** Digitize customer service (reduce call center costs)
- **Action:** Renegotiate vendor contracts
- **Impact:** Reduce expense ratio by 2 points
- **Timeline:** 12 months

#### Step 4: Quantification

**Base Case (Do Nothing):**
```
2024 Projection:
- Premium: $525M (5% growth)
- Loss Ratio: 82% (continued deterioration)
- Expense Ratio: 30%
- Combined Ratio: 112%
- Underwriting Loss: -$63M
```

**With Recommendations:**
```
2024 Projection:
- Premium: $500M (5% growth, minus 5% from tightening)
- Loss Ratio: 75% (82% - 3% rate - 2% UW - 2% product)
- Expense Ratio: 28% (30% - 2% efficiency)
- Combined Ratio: 103%
- Underwriting Loss: -$15M

2025 Projection (full effect):
- Premium: $510M
- Loss Ratio: 72%
- Expense Ratio: 27%
- Combined Ratio: 99%
- Underwriting Profit: +$5M
```

**ROI Analysis:**
- **Investment Required:** $5M (technology, process changes)
- **Annual Benefit:** $48M improvement in underwriting result
- **Payback Period:** 1.2 months

#### Step 5: Implementation Plan

**Phase 1 (Months 1-3): Quick Wins**
- File rate increases
- Tighten underwriting
- Launch fraud detection pilot

**Phase 2 (Months 4-6): Build Foundation**
- Implement telematics program
- Begin expense reduction initiatives
- Enhance analytics capabilities

**Phase 3 (Months 7-12): Optimize**
- Refine rating algorithm based on new data
- Scale successful pilots
- Continuous improvement

**Phase 4 (Months 13-24): Transform**
- Full digital transformation
- Advanced analytics (ML pricing)
- Market repositioning

#### Step 6: Risks & Mitigation

**Risk 1: Regulatory Pushback**
- **Mitigation:** Build strong actuarial justification, engage early with regulators

**Risk 2: Market Share Loss**
- **Mitigation:** Accept some loss in unprofitable segments, focus on retention in profitable segments

**Risk 3: Execution Challenges**
- **Mitigation:** Strong program management, executive sponsorship

### Discussion Questions
1. What would you do differently if the company was a mutual vs. a stock company?
2. How would you communicate these recommendations to the Board?
3. What if the regulator denies your rate filing?

### Key Takeaways
- Profitability problems often have multiple root causes
- Data-driven segmentation is critical
- Balance short-term fixes with long-term strategy
- Quantify everything
- Consider regulatory and competitive constraints

---

## Case Study 2: [Technical Deep-Dive]

### Background
*Another detailed scenario (300-400 words)*

**Scenario:** You're building a reserving model for a workers' compensation book. You notice that the chain-ladder method is producing reserves that are 20% higher than the Bornhuetter-Ferguson method. Your CFO asks: "Which one is right?"

### Your Task
1. Explain why the two methods differ
2. Investigate the root cause
3. Recommend which method to use
4. Propose a hybrid approach

### Solution Framework
*[Detailed solution similar to Case Study 1, 500-600 words]*

---

## Case Study 3-5: [Additional Cases]

### Case Study 3: Fraud Detection System Design
*[Scenario and solution, 400-500 words]*

### Case Study 4: Pricing a New Product
*[Scenario and solution, 400-500 words]*

### Case Study 5: Model Validation Failure
*[Scenario and solution, 400-500 words]*

---

# PART 3: INDUSTRY APPLICATIONS

## Application 1: [Real Company Example]

### Company: [Anonymized Major Insurer]

**Challenge:**
*[Description of the business problem, 200 words]*

**Solution:**
*[How they applied the concept/method, 300 words]*

**Results:**
*[Quantified outcomes, 100 words]*

**Lessons Learned:**
1. Lesson 1
2. Lesson 2
3. Lesson 3

**Relevance to Interview:**
*How to reference this in an interview context*

---

## Application 2-5: [Additional Industry Examples]

### Application 2: Telematics Pricing at Progressive
*[Details, 500 words]*

### Application 3: AI Claims Processing at Lemonade
*[Details, 500 words]*

### Application 4: Catastrophe Modeling at Reinsurer
*[Details, 500 words]*

### Application 5: Fraud Detection at State Farm
*[Details, 500 words]*

---

# PART 4: INTERVIEW PREPARATION TIPS

## General Strategy

### Before the Interview
1. **Research the Company:**
   - What lines of business?
   - Recent news/challenges?
   - Technology stack?

2. **Prepare Your Stories:**
   - 3-5 projects you can discuss in depth
   - STAR format (Situation, Task, Action, Result)
   - Quantified results

3. **Review Fundamentals:**
   - Key formulas
   - Common distributions
   - Standard methods

### During the Interview

**For Technical Questions:**
1. **Clarify:** Ask questions before diving in
2. **Structure:** Outline your approach
3. **Communicate:** Think out loud
4. **Check:** Verify your answer makes sense

**For Behavioral Questions:**
1. **Be Specific:** Use real examples
2. **Show Impact:** Quantify results
3. **Be Honest:** Acknowledge mistakes and learnings
4. **Be Concise:** 2-3 minutes per story

### After the Interview
1. **Send Thank You:** Within 24 hours
2. **Reflect:** What went well? What could improve?
3. **Follow Up:** On any questions you couldn't fully answer

## Common Interview Formats

### Format 1: Technical Screen (45-60 min)
- 5-10 min: Introductions
- 20-30 min: Technical questions (2-3 questions)
- 10-15 min: Behavioral questions
- 5-10 min: Your questions

### Format 2: Case Interview (60-90 min)
- 5-10 min: Introductions
- 40-60 min: Case study (often with data provided)
- 10-15 min: Presentation of findings
- 5-10 min: Your questions

### Format 3: Onsite (4-6 hours)
- Multiple rounds with different interviewers
- Mix of technical, behavioral, and case
- Lunch interview (cultural fit)
- Presentation (if applicable)

## Red Flags to Avoid
- ❌ Badmouthing previous employers
- ❌ Being unable to explain your own projects
- ❌ Not asking any questions
- ❌ Being overly theoretical without practical grounding
- ❌ Not admitting when you don't know something

## Green Flags to Demonstrate
- ✓ Curiosity and continuous learning
- ✓ Business acumen (not just technical skills)
- ✓ Communication skills
- ✓ Collaboration and teamwork
- ✓ Ethical awareness

---

*Template Version: 1.0*
*Last Updated: [Date]*
