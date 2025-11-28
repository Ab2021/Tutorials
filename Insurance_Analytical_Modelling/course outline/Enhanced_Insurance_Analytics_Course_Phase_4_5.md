# Phase 4 – Customer, Marketing & Revenue Cycle Analytics (Days 121–155)

**Theme:** Growth & Profitability – The Business Side

---

### Days 121–123 – Acquisition Funnel & Lead Scoring
*   **Topic:** Getting Customers
*   **Content:**
    *   **Funnel:** Impressions -> Clicks -> Quotes -> Binds.
    *   **Lead Scoring:** Predicting conversion probability ($P(\text{Bind}|\text{Quote})$).
    *   **Dynamic Questions:** Asking the right questions to qualify leads.
*   **Interview Pointers:**
    *   "How do you optimize the quote flow?" (Remove friction/questions that don't add predictive value).
*   **Data Requirements:** Web analytics (Google Analytics) joined with Quote data.

### Days 124–126 – Channel & Campaign Optimization
*   **Topic:** Marketing ROI
*   **Content:**
    *   **Channels:** Direct (Web), Aggregators (Price Comparison Websites), Agents, Brokers.
    *   **Metrics:** CPA (Cost Per Acquisition), ROAS (Return on Ad Spend).
    *   **Optimization:** Allocating budget to efficient channels.
*   **Interview Pointers:**
    *   "Why might an aggregator customer be riskier than a direct customer?" (Price sensitivity often correlates with risk).

### Days 127–129 – Pricing Elasticity & Demand Modelling
*   **Topic:** Price Optimization
*   **Content:**
    *   **Elasticity:** $\epsilon = \frac{\%\Delta Q}{\%\Delta P}$.
    *   **Demand Models:** Logistic regression for conversion as a function of Price.
    *   **Optimization:** Maximize Profit = $(Price - Cost) \times Demand(Price)$.
*   **Interview Pointers:**
    *   "What is the 'Winner's Curse' in insurance?" (If you underprice, you get all the bad risks).
*   **Tricky Parts:** Endogeneity (Price depends on Risk).

### Days 130–132 – End-to-end CLV + CAC Optimization
*   **Topic:** The Economics of Growth
*   **Content:**
    *   **Payback Period:** Time to recover CAC.
    *   **Value-Based Bidding:** Bidding more for leads with high predicted LTV.
*   **Interview Pointers:**
    *   "Should you always maximize retention?" (No, not for unprofitable customers).

### Days 133–135 – Renewal Journey & Personalization
*   **Topic:** Keeping Customers
*   **Content:**
    *   **Renewal Pricing:** Capping increases to prevent shock lapses.
    *   **Interventions:** Proactive outreach for at-risk customers.
*   **Challenges:** Regulatory bans on "Price Optimization" (charging loyal customers more).

### Days 136–138 – Cross-sell & Upsell Strategy
*   **Topic:** Share of Wallet
*   **Content:**
    *   **Market Basket Analysis:** Association rules (Apriori).
    *   **Propensity Models:** $P(\text{Buy Home}|\text{Has Auto})$.
    *   **Bundling:** Discounts for multi-line.
*   **Interview Pointers:**
    *   "How does bundling affect retention?" (Increases switching costs -> higher retention).

### Days 139–141 – Revenue Cycle & Cash-flow Analytics
*   **Topic:** Money In, Money Out
*   **Content:**
    *   **Premium Receivables:** Predicting late payments/cancellations.
    *   **Subrogation/Salvage:** Recovering money from third parties.
*   **Data Requirements:** Billing system data.

### Days 142–144 – Customer 360 View & Data Integration
*   **Topic:** The Single Truth
*   **Content:**
    *   **Entity Resolution:** Linking "John Smith" and "J. Smith".
    *   **Master Data Management (MDM):** Golden records.
*   **Challenges:** Legacy systems with different keys.

### Days 145–147 – Experience Management (NPS/CSAT)
*   **Topic:** Soft Metrics
*   **Content:**
    *   **NPS:** Net Promoter Score.
    *   **Text Analytics:** Sentiment analysis on call transcripts.
    *   **Linkage Analysis:** Does high NPS correlate with lower churn?
*   **Interview Pointers:**
    *   "How do you operationalize NPS?" (Trigger alerts for detractors).

### Days 148–150 – Integrated Growth & Risk Strategy Case Study
*   **Project:** The CEO's Dilemma
*   **Task:** Balance growth (lower prices) vs. profitability (higher prices).
    *   Build a simulation model.
*   **Deliverable:** Strategy Deck.

### Days 151–155 – Phase 4 Mini Capstone
*   **Project:** Marketing Optimization
*   **Task:** Optimize a marketing budget of $1M across 3 channels to maximize CLV.

---

# Phase 5 – Advanced Topics, Projects & Interview Mastery (Days 156–180)

**Theme:** Mastery & Career Launch

---

### Days 156–158 – Extreme Value Theory & Catastrophe Modelling
*   **Topic:** The Big One
*   **Content:**
    *   **Cat Models:** Event Generation -> Intensity -> Vulnerability -> Financial.
    *   **Vendors:** RMS, AIR (Verisk), CoreLogic.
*   **Interview Pointers:**
    *   "How does a Cat Model differ from a GLM?" (Physics-based simulation vs. Statistical history).

### Days 159–161 – Bayesian & Hierarchical Models
*   **Topic:** Borrowing Strength
*   **Content:**
    *   **Hierarchical GLMs:** Random effects for Territory/Agent.
    *   **MCMC:** Stan/PyMC3.
*   **Interview Pointers:**
    *   "Why use Bayesian methods?" (Better uncertainty quantification, incorporates priors).

### Days 162–164 – Litigation & Fraud Deep Dive (Process + ML)
*   **Topic:** Operationalizing ML
*   **Content:**
    *   **SIU (Special Investigation Unit):** The human-in-the-loop.
    *   **Triage:** Auto-pay vs. Adjuster vs. SIU.

### Days 165–167 – Model Risk Management & Documentation
*   **Topic:** Governance
*   **Content:**
    *   **SR 11-7:** The bible of model risk.
    *   **Validation:** Conceptual soundness, outcome analysis, ongoing monitoring.
*   **Interview Pointers:**
    *   "What belongs in a model document?" (Assumptions, Limitations, Data lineage, Testing results).

### Days 168–170 – SOA/CAS-aligned Problem Sets
*   **Activity:** Exam Drill
*   **Content:**
    *   Solve 20 difficult problems from CAS Exam 5/8 and SOA Exam PA/FAM.

### Days 171–173 – Full Capstone Project Build (Tech Focus)
*   **Project:** The Portfolio
*   **Task:** Build a production-ready end-to-end system.
    *   **Example:** Real-time pricing engine with fraud check.
    *   **Tech Stack:** Python, Docker, API (FastAPI), Streamlit (Frontend).
*   **Deliverable:** GitHub Repo with README.

### Days 174–176 – Capstone Project: Business Story
*   **Project:** The Pitch
*   **Task:** Present the Capstone to a non-technical audience.
    *   Focus on Business Value, ROI, and Risks.

### Days 177–178 – Interview Deep Dive: Role-specific
*   **Activity:** Mock Interviews
*   **Roles:**
    *   **Pricing Actuary:** Focus on GLMs, Regulation, Commercial awareness.
    *   **Data Scientist:** Focus on GBMs, NLP, Engineering, Deployment.
    *   **Reserving Actuary:** Focus on Triangles, Uncertainty, Financial Reporting.

### Days 179–180 – Final Review & Roadmap
*   **Activity:** Launch
*   **Task:**
    *   Review all notes.
    *   Polish Resume/LinkedIn.
    *   Apply to 5 jobs.
*   **Closing:** Continuous Learning (Papers, Conferences).

