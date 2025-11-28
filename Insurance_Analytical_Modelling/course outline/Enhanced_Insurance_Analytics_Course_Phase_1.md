# Comprehensive Insurance Analytics, Actuarial Science & ML Course (180 Days)

**Integrated with SOA & CAS Syllabus Themes**
*Covers: Pricing · Reserving · Severity · Litigation & Claims · Fraud · Attribution · Premiums · Capital · Reinsurance · LTV · Customer & Marketing Analytics · Personalization · Revenue Cycle · Full Actuarial Math & Data Science*

---

## Phase 0 – Orientation & Roadmap (Days 1–3)

### Day 1 – Course Overview & Career/Interview Roadmap
*   **Topic:** The Insurance Ecosystem & Role Mapping
*   **Content:**
    *   **Ecosystem:** Life (Mortality/Longevity), Non-life/P&C (Property, Casualty, Specialty), Health, Reinsurance, InsurTech.
    *   **Roles:**
        *   **Actuary (SOA/CAS):** Focus on solvency, reserving, regulatory pricing.
        *   **Pricing Actuary/Analyst:** Rate making, GLMs, competitive analysis.
        *   **Data Scientist:** ML for fraud, pricing (non-regulatory), marketing, claims automation.
        *   **Product Manager:** Product design, P&L ownership.
    *   **Syllabus Mapping:**
        *   **SOA:** Exams P, FM, FAM, SRM (Statistics for Risk Modeling), PA (Predictive Analytics).
        *   **CAS:** Exams MAS-I, MAS-II (Modern Actuarial Statistics), Exam 5 (Ratemaking/Reserving).
*   **Interview Pointers:**
    *   "Walk me through a time you explained a complex model to a non-technical stakeholder." (Crucial for actuaries/DS).
    *   "What is the difference between a Data Scientist and an Actuary in insurance?" (Answer: Regulatory responsibility vs. predictive freedom).
*   **Challenges:** Bridging the gap between traditional actuarial methods (GLMs, tables) and modern ML (GBMs, Neural Nets).
*   **Data Requirements:** Understanding the difference between Policy Admin Systems (PAS) data and Claims Management Systems (CMS) data.

### Day 2 – Fundamentals of Risk & Insurance Contracts
*   **Topic:** Core Principles & Contract Structure
*   **Content:**
    *   **Risk Pooling:** Law of Large Numbers (LLN), Central Limit Theorem (CLT) application.
    *   **Principles:** Insurable interest, indemnity (no profit from loss), subrogation (rights transfer), utmost good faith.
    *   **Contract Anatomy:** Declarations (Who/What), Insuring Agreement (The Promise), Exclusions (What's not covered), Conditions (Rules), Endorsements (Changes).
    *   **Product Types:** Term vs. Whole Life, Annuities, Auto (Liability/Physical Damage), Workers' Comp, Cyber.
*   **Interview Pointers:**
    *   "Why is the Law of Large Numbers fundamental to insurance?"
    *   "Explain 'Adverse Selection' and 'Moral Hazard' with examples."
*   **Tricky Parts:** Understanding "Claims-Made" vs. "Occurrence" policies (critical for reserving).
*   **Data Requirements:** Policy limits, deductibles, effective dates, coverage codes.

### Day 3 – Regulatory & Professional Landscape
*   **Topic:** Regulation, Solvency & Ethics
*   **Content:**
    *   **Regulation:** State-based (US - NAIC), Solvency II (EU), IFRS 17 (Global Accounting).
    *   **Capital:** RBC (Risk-Based Capital) formulas.
    *   **Professionalism:** ASOPs (Actuarial Standards of Practice) in US.
    *   **Model Governance:** SR 11-7 (Fed guidance on model risk), validation frameworks.
*   **Interview Pointers:**
    *   "How does regulation impact pricing innovation?" (e.g., fairness, disallowed variables).
    *   "What is IFRS 17?" (High-level: market-consistent valuation of insurance contracts).
*   **Challenges:** Balancing innovation (e.g., using credit score or telematics) with regulatory constraints (fairness/bias).

---

## Phase 1 – Quantitative Foundations & Actuarial Math Core (Days 4–30)

### Day 4 – Probability Refresher (SOA Exam P / CAS MAS-I)
*   **Topic:** Probability Theory for Insurance
*   **Content:**
    *   Random variables (Discrete vs. Continuous).
    *   Expectation (Mean), Variance, Skewness, Kurtosis (Fat tails are key!).
    *   Conditional Probability & Bayes' Theorem (Updating beliefs with new info).
    *   Moment Generating Functions (MGFs).
*   **Interview Pointers:**
    *   "Explain the difference between independent and mutually exclusive events."
    *   "Calculate the expected value of a policy with a deductible."
*   **Tricky Parts:** Conditional expectation $E[X|X>d]$ (Expected payment given a claim exceeds deductible).
*   **Data Requirements:** Synthetic datasets of loss amounts to practice calculating moments.

### Day 5 – Common Distributions in Insurance
*   **Topic:** Frequency & Severity Distributions
*   **Content:**
    *   **Frequency (Count):** Poisson (Mean=Var), Negative Binomial (Mean < Var, overdispersion), Binomial.
    *   **Severity (Amount):** Gamma, Lognormal (Skewed), Pareto (Heavy tail/Catastrophes), Weibull.
    *   **Mixture Models:** Zero-Inflated Poisson (ZIP) for excess zeros.
*   **Interview Pointers:**
    *   "Why use Negative Binomial instead of Poisson?" (Answer: To handle overdispersion/variance > mean).
    *   "Which distribution fits large liability claims best?" (Answer: Pareto).
*   **Challenges:** Fitting tails of distributions where data is sparse.

### Day 6 – Aggregate Loss Models (SOA FAM-S)
*   **Topic:** Collective Risk Model
*   **Content:**
    *   Model: $S = \sum_{i=1}^{N} X_i$ where $N$ is frequency, $X$ is severity.
    *   Compound Distributions: Compound Poisson-Gamma (Tweedie).
    *   Panjer Recursion: Numerical method to compute PDF of $S$.
    *   Normal & Log-Normal Approximations for $S$.
*   **Interview Pointers:**
    *   "How do you model total loss for a portfolio?"
    *   "What is the Tweedie distribution and why is it popular in GLMs?" (Models pure premium directly).
*   **Tricky Parts:** Convolutions of distributions.
*   **Data Requirements:** Aggregated claims data (total loss per period).

### Day 7 – Time Value of Money & Interest Theory (SOA FM)
*   **Topic:** Financial Mathematics
*   **Content:**
    *   Accumulation functions $a(t)$, effective rates $i$, discount factors $v = 1/(1+i)$.
    *   Annuities: Immediate $a_{\overline{n|}}$, Due $\ddot{a}_{\overline{n|}}$, Perpetuities.
    *   Loan Amortization.
*   **Interview Pointers:**
    *   "Calculate the present value of a stream of premium payments."
    *   "Difference between effective and nominal interest rates."
*   **Challenges:** Variable interest rate environments.

### Day 8 – Survival Models Basics (SOA FAM-L / ALTAM)
*   **Topic:** Survival Analysis Fundamentals
*   **Content:**
    *   Survival function $S(x)$, CDF $F(x)$, PDF $f(x)$.
    *   Force of mortality (Hazard rate) $\mu_x = -S'(x)/S(x)$.
    *   Complete vs. Curtate future lifetime ($T_x$ vs $K_x$).
*   **Interview Pointers:**
    *   "What is the force of mortality?" (Instantaneous risk of death).
    *   "Explain the relationship between survival function and hazard rate."
*   **Data Requirements:** Mortality tables (e.g., CSO tables).

### Day 9 – Life Tables & Mortality Laws
*   **Topic:** Life Table Construction
*   **Content:**
    *   Notation: $_tp_x$ (prob of surviving t years), $_jq_x$ (prob of dying).
    *   Select vs. Ultimate tables (Impact of underwriting wears off).
    *   Laws: Gompertz, Makeham.
*   **Interview Pointers:**
    *   "What is a 'Select' period in a mortality table?" (Period after underwriting where mortality is lower).
*   **Tricky Parts:** Interpolating fractional ages (UDD - Uniform Distribution of Deaths assumption).

### Day 10 – Life Insurance Cash-Flow Building Blocks
*   **Topic:** Insurance Benefits Valuation
*   **Content:**
    *   Whole Life ($A_x$), Term ($A_{x:\overline{n|}}^1$), Endowment ($A_{x:\overline{n|}}$).
    *   Actuarial Present Value (APV) calculations.
    *   Relationships between insurances and annuities ($A_x = 1 - d \ddot{a}_x$).
*   **Interview Pointers:**
    *   "Derive the relationship between a whole life insurance and a whole life annuity."
*   **Challenges:** Integrating stochastic interest rates.

### Day 11 – Net Premium Calculations
*   **Topic:** Benefit Premiums
*   **Content:**
    *   Equivalence Principle: APV(Premiums) = APV(Benefits).
    *   Net Level Premium reserves.
    *   Fully continuous vs. fully discrete premiums.
*   **Interview Pointers:**
    *   "How do you calculate the net premium for a 20-year term policy?"
*   **Data Requirements:** Mortality rates ($q_x$), interest rate ($i$).

### Day 12 – Gross Premium & Expense Loadings
*   **Topic:** Real-world Pricing
*   **Content:**
    *   Loadings: Per policy, per premium %, per claim.
    *   Gross Premium = (PV Benefits + PV Expenses) / PV Annuity.
    *   Profit margins.
*   **Interview Pointers:**
    *   "What are the components of a Gross Premium?"
*   **Challenges:** Allocating fixed vs. variable expenses accurately.

### Day 13 – Life Policy Values & Reserves (SOA FAM-L)
*   **Topic:** Reserving Mechanics
*   **Content:**
    *   Prospective Reserve (Future Out - Future In).
    *   Retrospective Reserve (Past In - Past Out).
    *   Thiele's Differential Equation (Continuous reserves).
*   **Interview Pointers:**
    *   "Why do reserves increase over time for a whole life policy?"
*   **Tricky Parts:** Negative reserves (and why they are usually set to zero).

### Day 14 – Multiple Life & Multiple Decrement Models (SOA ALTAM)
*   **Topic:** Complex States
*   **Content:**
    *   Joint Life ($T_{xy} = \min(T_x, T_y)$) vs. Last Survivor ($T_{\overline{xy}} = \max(T_x, T_y)$).
    *   Multiple Decrements: Death, Withdrawal (Lapse), Disability.
    *   Associated single decrement tables.
*   **Interview Pointers:**
    *   "How do you price a 'First-to-Die' policy?"
    *   "Explain competing risks in the context of life insurance."
*   **Data Requirements:** Lapse rates, disability incidence rates.

### Day 15 – Profit Testing & Cash-flow Projection
*   **Topic:** Profitability Analysis
*   **Content:**
    *   Profit Vector, Profit Signature.
    *   Net Present Value (NPV), Internal Rate of Return (IRR).
    *   Profit Margin (NPV / PV Premiums).
    *   Breakeven year.
*   **Interview Pointers:**
    *   "What is a profit signature?"
    *   "How does a lapse assumption affect profitability?" (Lapses early usually hurt due to acquisition costs).
*   **Challenges:** Sensitivity testing (what if interest rates drop 1%?).

### Day 16 – Non-life Basics: Perils, Coverages & Products
*   **Topic:** P&C Product Knowledge
*   **Content:**
    *   **Lines:** Personal Auto, Homeowners, Commercial General Liability (CGL), Workers Comp, Professional Liability.
    *   **Concepts:** Third-party vs. First-party coverage.
    *   **Perils:** Fire, Theft, Collision, Wind/Hail.
*   **Interview Pointers:**
    *   "Difference between 'Claims-Made' and 'Occurrence' policies." (Claims-made: covered if claim made during policy period. Occurrence: covered if event happened during policy period).
*   **Data Requirements:** Policy form data, coverage limits.

### Day 17 – Underwriting & Risk Classification
*   **Topic:** Selection & Segmentation
*   **Content:**
    *   Underwriting cycle (Hard vs. Soft market).
    *   Risk Classification criteria (Homogeneity, Reliability, Practicality).
    *   Adverse Selection spiral.
*   **Interview Pointers:**
    *   "Why is credit score used in auto insurance pricing?" (Correlation with loss propensity).
*   **Challenges:** Regulatory bans on certain rating factors (e.g., gender in EU).

### Day 18 – Exposure & Premium Base Concepts
*   **Topic:** Measuring Risk Exposure
*   **Content:**
    *   **Exposure Bases:** Car-years (Auto), Payroll (Workers Comp), Sales (Liability), Insured Value (Property).
    *   **Premium Types:** Written (booked), Earned (revenue recognized), In-force.
    *   **Calendar Year vs. Policy Year vs. Accident Year.**
*   **Interview Pointers:**
    *   "Calculate Earned Premium for a policy written on July 1st." (Using 365ths method).
*   **Tricky Parts:** Unearned Premium Reserve (UPR) calculation.

### Day 19 – Frequency–Severity Framework in Non-life
*   **Topic:** The Actuarial Control Cycle
*   **Content:**
    *   Pure Premium = Frequency × Severity.
    *   Loss Ratio = Losses / Premium.
    *   Combined Ratio = (Losses + Expenses) / Premium.
*   **Interview Pointers:**
    *   "If Frequency is down 5% and Severity is up 7%, what is the impact on Pure Premium?" ($0.95 \times 1.07 \approx 1.0165$, up 1.65%).
*   **Data Requirements:** Claim counts, claim amounts, exposure units.

### Day 20 – Loss Distributions in Practice
*   **Topic:** Fitting Data
*   **Content:**
    *   **EDA:** Histograms, Q-Q plots, Mean Residual Life plots.
    *   **Fitting:** Maximum Likelihood Estimation (MLE).
    *   **Goodness of Fit:** Kolmogorov-Smirnov, Anderson-Darling, AIC/BIC.
*   **Interview Pointers:**
    *   "How do you handle a dataset with a lot of small claims and a few massive ones?" (Splitting frequency/severity, using splicing).
*   **Challenges:** Right censoring (policy limits) and left truncation (deductibles).

### Day 21 – Credibility Theory (CAS MAS-II / Exam 5)
*   **Topic:** Blending Experience
*   **Content:**
    *   Concept: $Z \times (\text{Own Experience}) + (1-Z) \times (\text{Manual/Prior})$.
    *   **Limited Fluctuation Credibility:** "Full credibility" standards.
    *   **Bühlmann Credibility:** Least squares approach ($Z = N / (N+K)$).
    *   **Bühlmann-Straub:** Varying exposure.
*   **Interview Pointers:**
    *   "Explain Credibility to a non-actuary." (Weighting your history vs. the group average based on how much history you have).
*   **Tricky Parts:** Calculating the 'K' parameter in Bühlmann.

### Day 22 – Experience Rating & Bonus-Malus Systems
*   **Topic:** Individual Pricing
*   **Content:**
    *   **Experience Rating:** Modifying manual rate based on past claims (common in Workers Comp - E-Mod).
    *   **Bonus-Malus:** Class transitions based on claim-free years (common in Euro Auto).
    *   Stationary distributions of Markov Chains.
*   **Interview Pointers:**
    *   "Design a simple Bonus-Malus system."
*   **Data Requirements:** Historical claim counts per policyholder.

### Day 23 – Reinsurance Basics
*   **Topic:** Risk Transfer
*   **Content:**
    *   **Types:** Treaty vs. Facultative.
    *   **Structures:**
        *   **Proportional:** Quota Share (QS), Surplus Share.
        *   **Non-Proportional:** Excess of Loss (XoL), Stop Loss.
    *   **Pricing:** Experience rating vs. Exposure rating (Pareto curves).
*   **Interview Pointers:**
    *   "Why would an insurer buy Quota Share vs. XoL?" (QS for capital relief/volume, XoL for severity protection).

### Day 24 – Risk Measures & Capital Concepts
*   **Topic:** Quantifying Tail Risk
*   **Content:**
    *   **VaR (Value at Risk):** Percentile of loss distribution.
    *   **TVaR (Tail VaR) / CTE:** Average loss given loss > VaR (Coherent risk measure).
    *   **Economic Capital:** Capital needed to survive 1-in-200 year event.
*   **Interview Pointers:**
    *   "Why is TVaR preferred over VaR?" (Subadditivity - diversification benefit holds).

### Day 25 – Asset–Liability Management (ALM) Intro
*   **Topic:** Matching Assets to Liabilities
*   **Content:**
    *   **Duration:** Macaulay, Modified. Sensitivity to interest rates.
    *   **Convexity:** Second derivative.
    *   **Immunization:** Redington's immunization (matching duration, convexity >).
    *   **Segmentation:** Life (Long duration assets) vs. P&C (Liquidity focus).
*   **Interview Pointers:**
    *   "What happens to a life insurer if interest rates fall?" (Liabilities increase more than assets if duration gap exists).

### Day 26 – Intro to Triangles & Claims Development
*   **Topic:** Reserving Data Structure
*   **Content:**
    *   **The Triangle:** Accident Year (Rows) vs. Development Age (Columns).
    *   **Types:** Paid Loss, Incurred Loss (Paid + Case Reserves), Claim Counts.
    *   **Link Ratios:** Age-to-Age factors (ATAFs).
*   **Interview Pointers:**
    *   "What is a 'Link Ratio'?"
    *   "Why do we use triangles?" (To track how cohorts of claims mature over time).
*   **Data Requirements:** Transactional claims data aggregated to (AY, Dev) coordinates.

### Day 27 – Review & Problem Session (Foundations)
*   **Topic:** Consolidation
*   **Content:**
    *   Review of Probability, Interest Theory, Life Contingencies, and Loss Models.
    *   Practice problems from SOA Exam P/FM/FAM and CAS MAS-I.
*   **Activity:** Solve 5 difficult problems covering mixed concepts (e.g., probability of ruin).

### Days 28–30 – Mini Project 0: Simple Life & Non-Life Models
*   **Topic:** Practical Implementation (Excel/Python)
*   **Project 1 (Life):** Build a term life pricing model in Excel.
    *   Inputs: Mortality table, interest rate, expense loading.
    *   Output: Level premium.
*   **Project 2 (Non-Life):** Fit a Gamma distribution to a severity dataset in Python (`scipy.stats`).
    *   Calculate Mean, VaR(99%).
*   **Deliverable:** A short report explaining the premium derived and the tail risk estimated.
*   **Interview Hook:** "I built a ground-up pricing model using CSO tables and demonstrated sensitivity to interest rate changes."

