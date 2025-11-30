# Comprehensive Insurance Analytics, Actuarial Science & ML Course (180 Days)

**Integrated with SOA & CAS-style Syllabus Themes**  
*Covers: Pricing · Reserving · Severity · Litigation & Claims · Fraud · Attribution · Premiums · Capital · Reinsurance · LTV · Customer & Marketing Analytics · Personalization · Revenue Cycle · Full Actuarial Math & Data Science*

---

## Phase 0 – Orientation & Roadmap (Days 1–3)

### Day 1 – Course Overview & Career/Interview Roadmap
- Big picture: Insurance ecosystem (Life, Non-life/P&C, Health, Specialty, Reinsurance, InsurTech)
- Typical roles: Actuary (SOA/CAS path), Pricing/Reserving Actuary, Data Scientist (Insurance), Product & Growth Analytics
- Mapping to SOA/CAS syllabi:
  - SOA: Prelims (P/Exam PA style concepts), FAM, ALTAM/LTAM, SRM, PA, IFRM/ERM
  - CAS: Exams MAS-I/II, 5–9 (GI pricing, reserving & capital modelling)
- Key modeling families in this course: pricing, severity, frequency, fraud, litigation risk, LTV, marketing, attribution, reserving, capital
- **Interview focus:** "Walk me through a pricing project"; how to translate business problem → model
- **Challenges:** Overlap between actuarial & DS, clarity on roles
- **Data requirements:** List of canonical datasets (policy, exposure, claims, customer, marketing, external)

### Day 2 – Fundamentals of Risk & Insurance Contracts
- Risk pooling, law of large numbers, risk transfer
- Principles of insurance: insurable interest, indemnity, utmost good faith, subrogation, contribution, proximate cause
- Insurance contract structure: declarations, insuring agreement, exclusions, conditions, endorsements
- Product types: term/whole life, annuities, health, auto, property, liability, workers’ comp, reinsurance
- **Interview focus:** Explain why LLN is key to insurance; difference between life vs non-life risk
- **Challenges:** Mapping qualitative policy language → quantitative model boundaries
- **Data requirements:** Coverage definitions, limits, deductibles, reinstatements, policy form metadata

### Day 3 – Regulatory & Professional Landscape (SOA/CAS Context)
- Overview of solvency regimes (Solvency II, RBC, IFRS 17/GAAP themes at high level)
- Professional standards, code of conduct, model governance
- SOA & CAS exam structure; where pricing/reserving/fraud/LTV topics live
- **Interview focus:** How regulation constrains modelling choices
- **Challenges:** Communicating uncertainty; stakeholders expecting point estimates
- **Data requirements:** Regulatory reporting data, statutory triangles, financial statements

---

## Phase 1 – Quantitative Foundations & Actuarial Math Core (Days 4–30)

### Day 4 – Probability Refresher (SOA P-style)
- Random variables, distributions, expectation, variance, covariance
- Joint/conditional distributions, Bayes theorem
- **Interview focus:** Give an example of conditional probability in an insurance context
- **Challenges:** Making probability intuitive to business stakeholders
- **Data requirements:** Synthetic distribution examples; simple claim-count datasets

### Day 5 – Common Distributions in Insurance
- Poisson, Binomial, Negative Binomial (frequency)
- Exponential, Gamma, Lognormal, Pareto (severity)
- Mixture distributions and overdispersion
- **Interview focus:** Why Gamma for severity and Poisson for frequency?
- **Challenges:** Heavy tails vs data sparsity
- **Data requirements:** Claim amount datasets with full tails (capped & uncapped)

### Day 6 – Aggregate Loss Models (SOA “Loss Models” theme)
- Collective risk model: N (frequency), X (severity), S = Σ X_i
- Panjer recursion (conceptual), normal approx, compound distributions
- **Interview focus:** Intuition of frequency–severity decomposition
- **Challenges:** Estimating both components with limited data
- **Data requirements:** Policy-level frequency & severity, exposure units

### Day 7 – Time Value of Money & Interest Theory (CM1/FAM)
- Simple vs compound interest, discounting; present value, accumulation
- Annuities-immediate & -due; perpetuities
- **Interview focus:** Why discount reserves? Impact of interest on pricing
- **Challenges:** Choosing discount curves in volatile markets
- **Data requirements:** Yield curves, historical interest rates

### Day 8 – Survival Models Basics (ALTAM/LTAM style)
- Lifetime random variable T, survival function S(t), cdf, pdf
- Hazard/force of mortality μ(t); relations between S, μ, f
- **Interview focus:** Difference between pdf & hazard; real-world interpretation
- **Challenges:** Censoring, truncation in survival data
- **Data requirements:** Life duration, policy duration, lapse/claim time stamps

### Day 9 – Life Tables & Mortality Laws
- Life table notation: l_x, d_x, q_x, p_x, e_x
- Simple laws: De Moivre, constant force, Gompertz/Makeham
- **Interview focus:** Describe how you would build a mortality table from raw data
- **Challenges:** Small volumes, portfolio selection effects
- **Data requirements:** Large life portfolios by age, gender, underwriting class

### Day 10 – Life Insurance Cash-Flow Building Blocks
- Benefits & premiums as random variables
- Actuarial PV notation: A_x, ä_x, ä_x:
- Discrete vs continuous, level vs varying payments
- **Interview focus:** How do you compute EPV of a term assurance?
- **Challenges:** Multiple decrements (lapse, death, disability)
- **Data requirements:** Product definitions, mortality assumptions, lapse assumptions

### Day 11 – Net Premium Calculations (SOA CM1)
- Net single premium vs net level premium
- Equivalence principle; allocating benefits & expenses over time
- **Interview focus:** Net vs gross premium; why net not used directly in market
- **Challenges:** Circularity when assumptions depend on premium
- **Data requirements:** Expense data, commission structures, mortality, interest

### Day 12 – Gross Premium & Expense Loadings
- Types of expenses: acquisition, maintenance, claim, overhead
- Premium rating with expense loadings & profit margins
- **Interview focus:** How to incorporate expense analysis into pricing
- **Challenges:** Allocating overhead in multi-product firms
- **Data requirements:** Expense studies, cost allocations, policy counts

### Day 13 – Life Policy Values & Reserves
- Prospective vs retrospective reserves
- Relation between premiums, benefits, reserves
- **Interview focus:** Prospective vs retrospective reserve explanation
- **Challenges:** Changing assumptions mid-duration
- **Data requirements:** Policy-level in-force data, assumptions over time

### Day 14 – Multiple Life & Multiple Decrement Models
- Joint-life & last-survivor benefits
- Multiple decrements: death, surrender, disability; decrement tables
- **Interview focus:** Example of multiple decrement in real product
- **Challenges:** Dependent decrements; limited data per cause
- **Data requirements:** Cause-of-termination codes, multi-status datasets

### Day 15 – Profit Testing & Cash-flow Projection
- Cash-flow projection frameworks
- IRR, NPV, statutory vs economic profits
- **Interview focus:** How would you assess if a product is profitable?
- **Challenges:** Scenario & sensitivity design
- **Data requirements:** Detailed projected cash flows, stress scenarios

### Day 16 – Non-life Basics: Perils, Coverages & Products
- Product taxonomy: personal vs commercial lines
- Auto, property, GL, workers’ comp, marine, specialty
- **Interview focus:** Explain occurrence vs claims-made policies
- **Challenges:** Mapping product language to data fields
- **Data requirements:** Product hierarchy, coverage feature catalogs

### Day 17 – Underwriting & Risk Classification
- Underwriting guidelines; risk selection & tiers
- Rating variables: driver, vehicle, property, industry, geography
- **Interview focus:** Distinguish rating vs underwriting factor
- **Challenges:** Regulatory limits on rating variables (e.g. use of credit)
- **Data requirements:** Underwriting rules, risk scores, inspection reports

### Day 18 – Exposure & Premium Base Concepts
- Exposure measures: car-years, house-years, payroll, sales
- Written vs earned vs in-force premium
- **Interview focus:** Earned vs written premium explanation
- **Challenges:** Accurately tracking exposure for mid-term changes
- **Data requirements:** Policy start/end, endorsements, cancellations

### Day 19 – Frequency–Severity Framework in Non-life
- Frequency λ, severity Y, pure premium = λ * E[Y]
- Deductibles, limits, coinsurance, inflation
- **Interview focus:** Why frequency–severity vs direct loss modeling?
- **Challenges:** Accurate treatment of deductibles/limits in data
- **Data requirements:** Claim-level payments with original/unlimited estimates

### Day 20 – Loss Distributions in Practice
- Choosing severity distribution (lognormal, Gamma, Pareto)
- Parameter estimation (MLE, method of moments)
- **Interview focus:** How to test goodness of fit for severity?
- **Challenges:** Tail fitting with few large losses
- **Data requirements:** Uncensored large-loss dataset, inflation indices

### Day 21 – Credibility Theory (CAS/SOA advanced)
- Limited fluctuation vs Bühlmann credibility
- Application in experience rating & blending manual & experience
- **Interview focus:** Intuition behind credibility factors
- **Challenges:** Heterogeneity vs limited data per risk/unit
- **Data requirements:** Multi-year experience per risk/class

### Day 22 – Experience Rating & Bonus-Malus Systems
- Bonus-malus in auto; NCD (no-claim discounts)
- Experience-rated workers’ comp
- **Interview focus:** Design of bonus-malus; fairness vs incentives
- **Challenges:** Adverse selection; gaming behavior
- **Data requirements:** Multi-year claim history by policy

### Day 23 – Reinsurance Basics
- Proportional vs non-proportional
- QS, surplus, XoL, stop-loss, cat covers
- **Interview focus:** Why buy reinsurance? Examples
- **Challenges:** Pricing treaties with limited cat history
- **Data requirements:** Large-loss & cat-event data, exposure aggregates

### Day 24 – Risk Measures & Capital Concepts (ERM flavour)
- VaR, TVaR, expected shortfall
- Economic vs regulatory capital
- **Interview focus:** Limitations of VaR; why TVaR
- **Challenges:** Modeling tail dependencies
- **Data requirements:** Long historical time series or sim outputs

### Day 25 – Asset–Liability Management (Intro)
- ALM for life & long-tail non-life
- Duration, convexity, cash-flow matching
- **Interview focus:** ALM example in life insurance
- **Challenges:** Interest rate model choice vs complexity
- **Data requirements:** Asset portfolio data; liability cash flows

### Day 26 – Intro to Triangles & Claims Development
- Accident/underwriting year vs development year
- Paid vs incurred vs reported
- **Interview focus:** Why triangles for reserving?
- **Challenges:** Changing reserving practices over time
- **Data requirements:** Historical triangle construction-ready data

### Day 27 – Review & Problem Session (Foundations)
- Mixed exercises on life & non-life concepts
- Short case studies linking to SOA/CAS exam-style problems
- **Interview focus:** Behavioral: explaining complex math simply
- **Challenges:** Time-boxed explanations
- **Data requirements:** Practice problem sets

### Days 28–30 – Mini Project 0: Simple Life & Non-Life Models
- Build simple life pricing spreadsheet or notebook
- Build basic pure-premium non-life model
- Document assumptions & limitations
- **Interview focus:** Be able to narrate this as a project
- **Challenges:** Balancing detail vs explainability
- **Data requirements:** Synthetic life mortality, non-life frequency & severity data

---

## Phase 2 – Classical Pricing & Reserving (CAS/SOA GI Focus) (Days 31–70)

### Days 31–33 – Generalized Linear Models (GLMs) Theory
- GLM structure: link, linear predictor, exponential family
- Canonical links; Poisson, NB, Gamma
- **Interview focus:** Why log link for count data?
- **Challenges:** GLM assumptions vs real data
- **Data requirements:** Policy-level dataset with claim counts & exposures

### Days 34–36 – Frequency GLMs
- Poisson vs NB; overdispersion testing
- Exposure offsets; rating variables
- **Interview focus:** Interpreting a coefficient in log-link model
- **Challenges:** High-cardinality categorical features
- **Data requirements:** Rich rating-variable dataset (territory, class, etc.)

### Days 37–39 – Severity GLMs
- Choice of severity family; log-link Gamma
- Left-truncation/right-censoring (deductibles/limits)
- **Interview focus:** Handling deductibles in severity modelling
- **Challenges:** Truncation & censoring design
- **Data requirements:** Paid-loss data with policy limits & deductibles

### Day 40 – Pure Premium & Rate Indication from GLMs
- Combined frequency–severity → pure premium; loadings
- Rate relativity tables, tariff construction
- **Interview focus:** Build a simple rate table from GLM output
- **Challenges:** Monotonicity & smoothing of relativities
- **Data requirements:** Model predictions by risk cell

### Day 41 – Model Diagnostics & Validation (GLMs)
- Residuals, deviance, AIC/BIC, calibration
- Variable selection, interaction terms
- **Interview focus:** Overfitting GLMs & how to detect
- **Challenges:** Noise vs signal in sparse variables
- **Data requirements:** Train/validation/test splits, out-of-time data

### Day 42 – Regulatory & Fairness Constraints in Rating
- Prohibited variables (e.g. gender, race, some credit use)
- Proxy discrimination; fairness-aware modelling
- **Interview focus:** Ethical considerations in pricing
- **Challenges:** Removing bias vs losing predictive power
- **Data requirements:** Demographic & outcome data for fairness diagnostics

### Days 43–45 – Chain-Ladder Method (Deterministic Reserving)
- Cumulative triangles; age-to-age factors
- Chain-ladder for paid & incurred; tail factors
- **Interview focus:** Explain Chain-Ladder in non-technical language
- **Challenges:** Short triangles, structural breaks
- **Data requirements:** Clean development triangles by AY/AY/LOB

### Days 46–47 – Bornhuetter–Ferguson & Other Deterministic Methods
- BF method: prior expectations + development pattern
- Expected loss ratio method; frequency–severity triangle methods
- **Interview focus:** CL vs BF; when to use which
- **Challenges:** Selecting appropriate prior in BF
- **Data requirements:** Premium & expected loss ratio assumptions

### Days 48–49 – Stochastic Reserving: Mack & Bootstrap
- Mack model assumptions & variance estimation
- Bootstrap methods: residual resampling, predictive distribution
- **Interview focus:** Why stochastic reserving is needed
- **Challenges:** Interpreting wide uncertainty bands
- **Data requirements:** Sufficient history for reliable variance estimates

### Days 50–51 – Advanced Reserving: Occurrence–Development Models
- Joint modelling of occurrence & development (pricing + reserving link)
- Exposure-based reserving, frequency/severity triangles
- **Interview focus:** How pricing & reserving can be misaligned
- **Challenges:** Integrated models demand richer data
- **Data requirements:** Claim-level data with occurrence, report & payment dates

### Days 52–53 – Litigation & Large Loss Modelling
- Claim severity inflation, social inflation
- Identifying potential litigated claims; severity tail modelling
- **Interview focus:** Modelling impact of litigation on reserves
- **Challenges:** Very sparse, heavy-tailed data
- **Data requirements:** Flag for litigated claims, settlement amounts & durations

### Days 54–55 – Reinsurance & Reserving Interactions
- Gross vs net triangles; ceded & recoverable accounting
- Impact of reinsurance on reserve volatility
- **Interview focus:** How XoL changes tail risk
- **Challenges:** Complex attachment/aggregation terms in data
- **Data requirements:** Treaty terms, reinstatements, ceded loss data

### Day 56 – Data Architecture for Pricing & Reserving
- Data flows: policy admin → data warehouse → pricing/reserving marts
- Keys & joins: policy, claim, exposure, calendar dimensions
- **Interview focus:** Data challenges in GI projects
- **Challenges:** Duplicates, misaligned keys, inconsistent coding
- **Data requirements:** ERDs, documentation of source systems

### Days 57–58 – Non-life Pricing Case Study (Auto)
- End-to-end: data → EDA → GLM → rates
- Drafting a pricing memo to underwriters/management
- **Interview focus:** Case-style: price a new segment
- **Challenges:** Communicating uncertainty & assumptions
- **Data requirements:** Full auto portfolio dataset

### Days 59–60 – Non-life Reserving Case Study (Workers’ Comp)
- Build triangles, run CL/BF/Mack/Bootstrap
- Reserve range, scenarios; impact on income
- **Interview focus:** Defend selected reserve vs range
- **Challenges:** Ad-hoc manual adjustments vs model
- **Data requirements:** Historical workers’ comp triangles

### Days 61–63 – Life Pricing & Reserving Case Study
- Design a term product; price; compute reserves
- Profit-testing under multiple scenarios
- **Interview focus:** Explain life pricing vs GI pricing differences
- **Challenges:** Multi-decrement & dynamic lapses
- **Data requirements:** Mortality curves, lapse profiles, expenses

### Days 64–66 – Capital & Solvency Intro Case Study
- Simple capital model for a line of business
- Reserve, premium, and catastrophe risk
- **Interview focus:** Capital vs reserves distinction
- **Challenges:** Limited cat-history; parameter risk
- **Data requirements:** Loss distributions, scenario catalogue

### Days 67–68 – Cross-LOB Aggregation & Diversification
- Correlations across LOBs; portfolio VaR/TVaR
- Reinsurance optimization basics
- **Interview focus:** Diversification benefit explanation
- **Challenges:** Estimating correlations under stress
- **Data requirements:** Multi-LOB historical results

### Days 69–70 – Phase 2 Review & Interview Drill
- Rapid-fire GLM, reserving, pricing questions
- Whiteboard derivations & scenario questions
- **Challenges:** Explaining complex tech to non-actuaries
- **Data requirements:** Curated interview Q&A bank

---

## Phase 3 – Machine Learning for Insurance (Days 71–120)

### Days 71–72 – ML Foundations & Insurance-specific Considerations
- Supervised vs unsupervised, train/val/test, CV
- Evaluation metrics: RMSE, MAE, AUC, PR, log-loss
- **Interview focus:** Why AUC may be misleading in imbalanced fraud
- **Challenges:** Data leakage; temporal splits
- **Data requirements:** Time-stamped features & outcomes

### Days 73–75 – Tree-based Models for Frequency & Severity
- Decision trees, Random Forests, Gradient Boosting
- Implementing exposure offsets, sample weights
- **Interview focus:** GLM vs GBM for pricing
- **Challenges:** Interpretability, monotonicity
- **Data requirements:** Rich features; careful pre-processing

### Days 76–78 – Neural Nets & Interpretable Deep Pricing Models
- Embeddings for high-cardinality cats (zip, agent, etc.)
- Monotone constraints & partial dependence
- **Interview focus:** How to keep NNs interpretable in pricing
- **Challenges:** Overfitting & regulatory acceptance
- **Data requirements:** Large datasets; regularization strategies

### Days 79–81 – Fraud Detection & Anomaly Detection (Classification)
- Binary classification with severe imbalance
- Oversampling, undersampling, cost-sensitive learning
- Unsupervised anomaly detection (Isolation Forest, autoencoders)
- **Interview focus:** Precision vs recall trade-offs in fraud
- **Challenges:** Label noise, concept drift (fraudsters adapt)
- **Data requirements:** Fraud label history, case outcomes, features

### Days 82–84 – Litigation & High-Severity Claim Modelling (ML)
- Classification: will a claim litigate?
- Regression/quantile/EVT for high severity
- **Interview focus:** Handling small-sample high-severity events
- **Challenges:** Extreme imbalance & heavy tails
- **Data requirements:** Annotated litigated claims; lawyer involvement flags

### Days 85–87 – Claim Lifecycle & Severity Development Modelling
- Claim-level trajectory modelling (paid vs case reserve)
- Event-history, survival models for time to close
- **Interview focus:** Use of survival models on claims
- **Challenges:** Multiple events (reopenings), competing risks
- **Data requirements:** Detailed claim transaction history

### Days 88–90 – Customer Churn & Renewal Modelling
- Binary classification & survival for lapse/renewal
- Features: price changes, claims, service interactions
- **Interview focus:** Key drivers of churn in insurance
- **Challenges:** Price vs non-price factors; data fragmentation
- **Data requirements:** Multi-year policy & interaction data

### Days 91–93 – Customer Lifetime Value (LTV/CLV) Modelling
- Analytical CLV models vs ML-based (stacked regression)
- Incorporating claim risk, cross-sell, discounting
- **Interview focus:** How CLV differs from premium
- **Challenges:** Long horizons & structural shifts
- **Data requirements:** Full lifetime behavioral data by customer

### Days 94–96 – Customer Segmentation (Clustering)
- K-means, hierarchical, GMM, density-based
- Segment interpretation: risk, profitability, behavior
- **Interview focus:** Defining actionable segments
- **Challenges:** Choosing K & business interpretability
- **Data requirements:** Clean feature set for segmentation

### Days 97–99 – Recommendation Systems & Personalization
- Content-based & collaborative filtering for insurance
- Next-best-offer, next-best-action in renewal journeys
- **Interview focus:** Recommenders in a low-frequency purchase domain
- **Challenges:** Sparse interactions; cold start
- **Data requirements:** Interaction logs, product matrices, profile data

### Days 100–102 – Marketing Mix Modelling & Attribution
- MMM basics: regression/time-series of spend vs outcomes
- Multi-touch attribution, channel contribution
- **Interview focus:** MMM vs MTA; pros/cons
- **Challenges:** Collinearity in channels; lag effects
- **Data requirements:** Channel-wise spend, impressions, conversions over time

### Days 103–105 – Uplift Modelling for Retention & Campaigns
- Treatment vs control; uplift vs response
- Qini curves; targeting high-uplift customers
- **Interview focus:** Why uplift > propensity models for campaigns
- **Challenges:** Biased treatment assignment, small control groups
- **Data requirements:** Controlled campaign data with treatment flags

### Days 106–108 – Advanced Loss & Tail Modelling (EVT + ML)
- Peaks-over-threshold, GPD; tail index estimation
- Combining EVT with ML-predicted baseline
- **Interview focus:** Modelling cat/tail risk with limited data
- **Challenges:** Threshold selection; parameter uncertainty
- **Data requirements:** Historical large-loss data, cat-event catalogues

### Days 109–111 – Integrated Pricing + Reserving + Capital ML Use Cases
- Linking occurrence & development models with rating
- Simulating P&L, capital needs under ML-based risk
- **Interview focus:** How ML can break silos between pricing & reserving
- **Challenges:** Complexity, governance, data lineage
- **Data requirements:** Unified data model across functions

### Days 112–114 – Data Engineering & Pipelines for Insurance ML
- Data ingestion, cleaning, feature engineering at scale
- Batch vs real-time; feature stores; MLOps basics
- **Interview focus:** Data pipeline challenges in insurance
- **Challenges:** Legacy systems; changing source schemas
- **Data requirements:** Access to raw source schemas & feeds

### Days 115–117 – Model Governance, Monitoring & Drift in Insurance
- Performance monitoring, calibration, stability
- Drift detection, re-training strategies
- **Interview focus:** How to monitor a pricing/fraud model in production
- **Challenges:** Data privacy, logging limitations
- **Data requirements:** Production logs & monitoring metrics

### Days 118–120 – Phase 3 Capstone Planning & Design
- Choose a major ML project (fraud, pricing, LTV, litigation, churn, recommendations)
- Problem framing, success metrics, data plan
- **Interview focus:** Structuring an end-to-end project story
- **Challenges:** Scope vs feasibility in limited time
- **Data requirements:** Consolidated dataset for selected problem

---

## Phase 4 – Customer, Marketing & Revenue Cycle Analytics (Days 121–155)

### Days 121–123 – Acquisition Funnel & Lead Scoring in Insurance
- Defining funnel stages: impression → click → quote → bind → onboard
- Lead scoring features; online + offline channels
- **Interview focus:** Acquisition funnel KPI definitions
- **Challenges:** Partial visibility across channels
- **Data requirements:** Web/app logs, CRM, quote/policy data

### Days 124–126 – Channel & Campaign Optimization
- Comparing agents, brokers, aggregators, direct online
- Campaign-level performance; CPA, CPL, ROI
- **Interview focus:** Channel mix recommendation scenario
- **Challenges:** Attribution across overlapping campaigns
- **Data requirements:** Channel identifiers, campaign metadata, spend

### Days 127–129 – Pricing Elasticity & Demand Modelling
- Elasticity estimation from quote-to-bind data
- Price sensitivity by segment; demand models
- **Interview focus:** Balancing risk-based vs demand-based pricing
- **Challenges:** Confounding (e.g. discounts, underwriting)
- **Data requirements:** Quote, offered price, competitor proxies, bind outcomes

### Days 130–132 – End-to-end CLV + Acquisition Cost Optimization
- CAC vs LTV; payback period
- Portfolio-level CLV modelling
- **Interview focus:** When is high CAC acceptable?
- **Challenges:** Estimating long-term CLV reliably
- **Data requirements:** Acquisition cost data per channel/customer

### Days 133–135 – Renewal Journey & Personalization
- Renewal touchpoints, pricing, messaging
- Next-best-action for renewal (discount, outreach, product change)
- **Interview focus:** Renewal strategy to reduce churn
- **Challenges:** Regulatory limits on individualized offers
- **Data requirements:** Renewal communication history, responses

### Days 136–138 – Cross-sell & Upsell Strategy in Insurance
- Basket analysis; product affinities (auto+home, life+health)
- Recommendation & propensity for cross-sell
- **Interview focus:** Designing cross-sell campaign post-claim
- **Challenges:** Not over-selling; compliance constraints
- **Data requirements:** Multi-product holdings per customer

### Days 139–141 – Revenue Cycle & Cash-flow Analytics
- Premium receivables, payment behavior, lapses
- Claim payments, recoveries, commission & expense timing
- **Interview focus:** Working capital vs profitability in insurance
- **Challenges:** Multiple time scales (monthly, annual, claim lifetime)
- **Data requirements:** Billing, collections, claims, commission data

### Days 142–144 – Customer 360 View & Data Integration
- Customer-level entity resolution (ID stitching)
- Combining policy, claims, marketing, service, external data
- **Interview focus:** Building a customer 360 in a legacy environment
- **Challenges:** Identity resolution errors; privacy restrictions
- **Data requirements:** Identifiers across systems; PII governance

### Days 145–147 – Experience Management & NPS/CSAT Modelling
- Linking customer experience scores to churn & CLV
- Text analytics on complaints/feedback (NLP intro)
- **Interview focus:** Role of CX in actuarial/analytics decisions
- **Challenges:** Unstructured data; sentiment model bias
- **Data requirements:** Survey + free-text feedback linked to policies

### Days 148–150 – Integrated Growth & Risk Strategy Case Study
- Design strategy for a line of business: acquisition + pricing + retention + reinsurance
- Build high-level financial projections
- **Interview focus:** Strategic case with multiple levers
- **Challenges:** Reconciling risk appetite vs growth targets
- **Data requirements:** Combined model outputs from earlier phases

### Days 151–155 – Phase 4 Mini Capstone
- Choose a marketing/growth/revenue problem & build solution
- Deliver: deck + tech appendix + metrics plan
- **Interview focus:** Product/growth analytics story
- **Challenges:** Proper framing and scoping
- **Data requirements:** Subset of full course dataset

---

## Phase 5 – Advanced Topics, Projects & Interview Mastery (Days 156–180)

### Days 156–158 – Extreme Value Theory & Catastrophe Modelling
- EVT recap; cat modelling concepts (hazard, vulnerability, financial modules)
- Secondary uncertainty; use in reinsurance & capital
- **Interview focus:** How do you price cat XoL with limited history?
- **Challenges:** Black-box vendor models; correlation across perils
- **Data requirements:** Cat risk vendor outputs, exposure data by location

### Days 159–161 – Bayesian & Hierarchical Models in Insurance
- Bayesian GLMs; hierarchical structures (territory, agent, LOB)
- Credibility as special case of hierarchical modelling
- **Interview focus:** Benefits of Bayesian approaches
- **Challenges:** Computation; explaining priors to stakeholders
- **Data requirements:** Multi-level data structure (policy, region, firm)

### Days 162–164 – Litigation & Fraud Deep Dive (Process + ML)
- Operational fraud & SIU workflows
- Combined rules + ML + investigator feedback loop
- **Interview focus:** Integrating fraud model into claims workflow
- **Challenges:** Model adoption; false positive costs
- **Data requirements:** Fraud flags, investigation outcomes, claim metadata

### Days 165–167 – Model Risk Management & Documentation
- Model inventories, validation, challenge model structure
- Documentation structure: purpose, data, methods, testing, limitations
- **Interview focus:** How to handle model failure in production
- **Challenges:** Keeping docs updated; change management
- **Data requirements:** Versioned model artifacts & reports

### Days 168–170 – SOA/CAS-aligned Problem Sets & Exam-style Practice
- Non-life pricing & reserving exam-style questions (CAS 5/7/8-type)
- Life & ERM-style questions (SOA FAM/ERM-type)
- **Interview focus:** Whiteboard problem solving under time pressure
- **Challenges:** Translating theory to numeric answers quickly
- **Data requirements:** Curated problem banks

### Days 171–173 – Full Capstone Project Build (Tech Focus)
- Implement production-like pipeline for chosen use case:
  - Pricing, reserving, fraud, LTV, litigation, marketing, or combined
- Include: data ingestion, prep, modelling, validation, monitoring plan
- **Interview focus:** Be able to present this as your flagship project
- **Challenges:** End-to-end integration & narrative
- **Data requirements:** Full integrated dataset

### Days 174–176 – Capstone Project: Business Story & Presentation
- Turn capstone into executive-level narrative
- Include risk vs reward, assumptions, limitations, roadmap
- **Interview focus:** VP-level storytelling; trade-off discussions
- **Challenges:** Avoid over-technical explanations
- **Data requirements:** Visualizations & key summary tables

### Days 177–178 – Interview Deep Dive: Role-specific Question Banks
- Pricing Actuary & Analyst roles (SOA/CAS focus)
- Data Scientist in Insurance roles
- Product/Growth Analytics in InsurTech
- **Interview focus:** Behavioural + case + technical blends
- **Challenges:** Tailoring same story to different roles
- **Data requirements:** Personalized Q&A lists based on profile

### Days 179–180 – Final Review, Gaps & Personal Roadmap
- Identify weak areas in theory, coding, business storytelling
- Define next 3–6 month learning & project plan
- **Interview focus:** Articulating your roadmap in interviews
- **Challenges:** Being honest about gaps while showing growth mindset
- **Data requirements:** Self-evaluation notes, mentor feedback

---

## Global Threads (To be Woven Across Days)

For **every modelling topic**, ensure you explicitly consider:

1. **Data Requirements & Risks**
   - Source systems & fields needed
   - Keys & joins; time granularity
   - Common quality issues

2. **Modelling Tricky Parts**
   - Censoring, truncation, heavy tails
   - Imbalanced classes, label noise
   - Non-stationarity, regime shifts

3. **Business & Regulatory Context**
   - How decisions affect customers, capital, solvency
   - Fairness & ethics; regulatory constraints

4. **Interview Hooks**
   - Keep 2–3 “stories” per theme (pricing, reserving, fraud, LTV, marketing, litigation, capital)
   - Prepare both technical explanation & plain-English summary

5. **Documentation & Governance**
   - Always think: how would I document this model?
   - What validation tests are required in a regulated environment?

This 180-day curriculum is designed so a learner finishing it will:
- Understand and apply **core actuarial mathematics** (SOA/CAS-aligned)
- Build and critique **pricing, reserving, capital, fraud, LTV, marketing & recommendation models**
- Design **data architectures & pipelines** for insurance ML
- Communicate effectively with **underwriters, actuaries, data scientists, and executives**
- Be well-prepared for **actuarial-style exams** and **industry interviews** in insurance analytics, actuarial data science, and InsurTech roles.

