# Phase 2 – Classical Pricing & Reserving (CAS/SOA GI Focus) (Days 31–70)

**Theme:** The Actuary's Toolkit – From GLMs to Triangles

---

### Days 31–33 – Generalized Linear Models (GLMs) Theory (CAS MAS-I)
*   **Topic:** The Industry Standard for Pricing
*   **Content:**
    *   **Structure:** $g(E[Y]) = X\beta$. Link function $g(\cdot)$, Linear Predictor $X\beta$.
    *   **Exponential Family:** Normal, Poisson, Gamma, Inverse Gaussian.
    *   **Assumptions:** Independence, constant variance (or variance function $V(\mu)$).
*   **Interview Pointers:**
    *   "Why do we use a Log link function for frequency?" (Ensures predictions are positive; multiplicative structure).
    *   "What is the 'Offset' in a GLM?" (Accounting for exposure, e.g., $\ln(\text{Exposure})$).
*   **Tricky Parts:** Interpreting coefficients as relativities ($e^\beta$).
*   **Data Requirements:** Policy-level data with earned exposure and claim counts.

### Days 34–36 – Frequency GLMs
*   **Topic:** Modeling Claim Counts
*   **Content:**
    *   **Poisson GLM:** Variance = Mean. Often under-dispersed for real data.
    *   **Negative Binomial:** Variance = $\mu + \phi \mu^2$. Handles overdispersion.
    *   **Zero-Inflated Models:** For excess zeros.
*   **Interview Pointers:**
    *   "How do you test for overdispersion?" (Compare variance to mean; Chi-square test).
*   **Challenges:** Sparse classes (e.g., young drivers with high-performance cars).

### Days 37–39 – Severity GLMs
*   **Topic:** Modeling Claim Amounts
*   **Content:**
    *   **Gamma GLM:** Constant coefficient of variation. Good for attrition.
    *   **Inverse Gaussian:** Heavier tail than Gamma.
    *   **Tweedie:** Models Pure Premium directly (Compound Poisson-Gamma).
*   **Interview Pointers:**
    *   "Why not use a Normal distribution for severity?" (Severity is skewed and positive; Normal is symmetric and allows negatives).
*   **Tricky Parts:** Modeling large losses (capping/excess removal) before fitting GLM.

### Day 40 – Pure Premium & Rate Indication
*   **Topic:** Putting it Together
*   **Content:**
    *   Pure Premium = Freq Model $\times$ Sev Model.
    *   **Rate Making:** Indicated Rate = (Pure Premium + Fixed Expenses) / (1 - Var Expense % - Profit %).
    *   **Relativities:** Base Rate $\times$ Age Factor $\times$ Area Factor.
*   **Interview Pointers:**
    *   "Walk me through a rate indication."
*   **Data Requirements:** Current rate level (on-level premium).

### Day 41 – Model Diagnostics & Validation
*   **Topic:** Is the Model Good?
*   **Content:**
    *   **Residuals:** Deviance residuals, Pearson residuals.
    *   **Tests:** AIC/BIC (Penalizing complexity), Gini Index (Lorenz curve for lift).
    *   **Consistency:** One-way plots (Actual vs. Expected by variable).
*   **Interview Pointers:**
    *   "Explain the Lift Chart." (How much better does the model segment risk compared to random/current rating?).
*   **Challenges:** Overfitting to noise in small segments.

### Day 42 – Regulatory & Fairness Constraints
*   **Topic:** Ethics in Pricing
*   **Content:**
    *   **Disparate Impact:** Unintentional bias against protected classes.
    *   **Proxy Variables:** Is 'Credit Score' a proxy for race?
    *   **CAS/SOA Ethics:** ASOP 12 (Risk Classification).
*   **Interview Pointers:**
    *   "How do you ensure your model isn't discriminatory?" (Testing for disparate impact, removing sensitive variables).

### Days 43–45 – Chain-Ladder Method (CAS Exam 5)
*   **Topic:** Deterministic Reserving
*   **Content:**
    *   **The Triangle:** Cumulative Paid/Incurred.
    *   **LDFs:** Link Ratios (Age-to-Age factors).
    *   **CDF:** Cumulative Development Factor (Product of LDFs).
    *   **Ultimate Loss:** Current $\times$ CDF.
*   **Interview Pointers:**
    *   "What is the fundamental assumption of the Chain Ladder method?" (Past development patterns predict future development).
*   **Tricky Parts:** Selecting tail factors (beyond the triangle).

### Days 46–47 – Bornhuetter–Ferguson (BF) Method
*   **Topic:** Blending Stability and Responsiveness
*   **Content:**
    *   **Concept:** Ultimate = Paid + (Expected Unpaid).
    *   **Formula:** $L_{BF} = L_{Paid} + (1 - 1/CDF) \times L_{Prior}$.
    *   **Usage:** Good for immature years where Chain Ladder is volatile.
*   **Interview Pointers:**
    *   "When would you use BF over Chain Ladder?" (New lines of business, recent accident years).
*   **Data Requirements:** A priori loss ratio assumption.

### Days 48–49 – Stochastic Reserving: Mack & Bootstrap (CAS MAS-II)
*   **Topic:** Quantifying Reserve Uncertainty
*   **Content:**
    *   **Mack's Model:** Analytic variance of Chain Ladder.
    *   **Bootstrapping:** Resampling residuals to generate a distribution of reserves.
    *   **Output:** Reserve ranges (e.g., 75th percentile).
*   **Interview Pointers:**
    *   "Why is a point estimate for reserves insufficient?" (Management needs to know the range of outcomes for capital planning).
*   **Challenges:** Heteroscedasticity in residuals.

### Days 50–51 – Advanced Reserving: Berquist-Sherman
*   **Topic:** Adjusting for Changing Conditions
*   **Content:**
    *   Adjusting for changes in **Claim Settlement Rates** (Speed-up/Slow-down).
    *   Adjusting for changes in **Case Reserve Adequacy** (Strengthening/Weakening).
*   **Interview Pointers:**
    *   "If claims are settling faster, what happens to the Chain Ladder projection?" (It over-projects ultimate losses).
*   **Tricky Parts:** Detrending the diagonal.

### Days 52–53 – Litigation & Large Loss Modelling
*   **Topic:** The Volatile Tail
*   **Content:**
    *   **Social Inflation:** Rising claim costs due to jury verdicts/legal climate.
    *   **Modelling:** Separating 'Attritional' vs. 'Large/Litigated' claims.
    *   **EVA:** Extreme Value Analysis for severity.
*   **Interview Pointers:**
    *   "How do you reserve for a potential class-action lawsuit?" (Scenario analysis, not standard triangles).
*   **Data Requirements:** Flags for 'In Suit', legal expense payments.

### Days 54–55 – Reinsurance & Reserving Interactions
*   **Topic:** Net vs. Gross
*   **Content:**
    *   Constructing Gross, Ceded, and Net triangles.
    *   Impact of XoL on development patterns (Ceded develops later).
    *   **Risk Transfer Testing:** 10-10 rule (10% chance of 10% loss).
*   **Interview Pointers:**
    *   "Why does Net development usually lag Gross development?" (Reinsurance kicks in for large, late-developing claims).

### Day 56 – Data Architecture for Pricing & Reserving
*   **Topic:** The Plumbing
*   **Content:**
    *   **Data Marts:** Policy, Claims, Billing.
    *   **Granularity:** Transactional vs. Aggregated.
    *   **As-of Dates:** Recreating the world as it looked in the past.
*   **Interview Pointers:**
    *   "Describe a robust data pipeline for reserving."
*   **Challenges:** Handling retroactive policy changes.

### Days 57–58 – Non-life Pricing Case Study (Auto)
*   **Project:** End-to-End Pricing
*   **Task:**
    1.  Clean data (remove duplicates, cap losses).
    2.  Fit Frequency (Poisson) and Severity (Gamma) GLMs.
    3.  Select variables (Age, Vehicle Type, Territory).
    4.  Calculate Indicated Rate Change.
*   **Deliverable:** Rate filing memorandum.
*   **Interview Hook:** "I conducted a relativistic pricing analysis for an auto book and identified a 5% rate need."

### Days 59–60 – Non-life Reserving Case Study (Workers’ Comp)
*   **Project:** Reserve Review
*   **Task:**
    1.  Build triangles (Paid, Incurred).
    2.  Select LDFs (exclude outliers).
    3.  Apply CL and BF methods.
    4.  Select Ultimate Loss for each Accident Year.
*   **Deliverable:** Schedule P (Part 1) style exhibit.
*   **Interview Hook:** "I performed a reserve review using multiple methods and justified my selection of the BF method for recent years."

### Days 61–63 – Life Pricing & Reserving Case Study
*   **Project:** Term Life Product
*   **Task:**
    1.  Set assumptions (Mortality 100% of CSO, Lapse 5%, Interest 3%).
    2.  Calculate Net and Gross Premiums.
    3.  Project Statutory Reserves over 20 years.
    4.  Calculate Profit Margin.
*   **Deliverable:** Excel pricing tool.

### Days 64–66 – Capital & Solvency Intro Case Study
*   **Topic:** Capital Modeling
*   **Task:**
    *   Calculate RBC (Risk Based Capital) for a hypothetical company.
    *   Components: Asset Risk, Insurance Risk, Interest Rate Risk.
*   **Interview Pointers:**
    *   "What is the RBC ratio?" (Total Adjusted Capital / Authorized Control Level RBC).

### Days 67–68 – Cross-LOB Aggregation & Diversification
*   **Topic:** Portfolio View
*   **Content:**
    *   Correlation Matrices (Copulas).
    *   Diversification Benefit: Sum(VaR) > VaR(Sum).
    *   Allocating Capital: Euler Allocation.
*   **Interview Pointers:**
    *   "Why is capital allocation important for pricing?" (Target ROE depends on allocated capital).

### Days 69–70 – Phase 2 Review & Interview Drill
*   **Activity:** Mock Interview
*   **Questions:**
    *   "Explain the difference between IBNR and Case Reserves."
    *   "How do you handle a change in claims handling speed in your reserve analysis?"
    *   "What are the pros and cons of GLMs?"
    *   "Derive the formula for the BF method."

