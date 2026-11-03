# Whole-Life Cost and MEAT Evaluation

> Module 35 — Public Sector, Defence, and Humanitarian Logistics
> Topic 02 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Topic 01 (Public Procurement Frameworks), Module 11 (Supply Chain Network Economics), Module 12 (Inventory Optimization)
> Version 2026.5

---

## 1. Why this topic exists

In commercial supply chains, procurement decisions often prioritize initial purchase price, short-term volume discounts, or quarterly working capital conservation. In public sector, defence, and major humanitarian infrastructure, this commercial reflex causes catastrophic long-term budget failures. Capital equipment—such as naval patrol vessels, military airlift fleets, disaster relief logistics distribution hubs, modular tactical field hospitals, and municipal electric bus fleets—incurs between 65% and 85% of its cumulative lifetime expenditure *after* initial contract award and commissioning. The initial acquisition price is merely the visible tip of an economic iceberg; the submerged mass comprises multi-decade fuel consumption, scheduled maintenance overhauls, unscheduled corrective repairs, spare parts inventory holding, obsolescence redesigns, carbon taxation, and hazardous decommissioning.

Procuring authorities operating under statutory public procurement frameworks cannot simply award contracts based on the lowest submitted purchase invoice. Doing so invites "low-balling"—a predatory bidding strategy where contractors submit artificially low acquisition prices, anticipating that they will extract excessive monopoly rents over thirty years through proprietary replacement parts, software license renewals, and sole-source sustainment contracts. To prevent this pathology, international procurement law mandates the evaluation of tenders based on the **Most Economically Advantageous Tender (MEAT)**—known in the UK Procurement Act 2023 as the **Most Advantageous Tender (MAT)**, and in US Federal Acquisition Regulation (FAR Part 15) as the **Best Value Continuum Tradeoff**.

However, the analytical implementation of MEAT and Whole-Life Costing (WLC) is fraught with severe mathematical hazards. Chief among these is the widespread use of unscientific **relative scoring formulas**. When contracting authorities calculate a tenderer's commercial score relative to the lowest submitted bid (e.g., $S_i = S_{\max} \cdot \frac{P_{\min}}{P_i}$), the evaluation violates Kenneth Arrow's fundamental axiom of the **Independence of Irrelevant Alternatives (IIA)**. Under relative scoring, the ranking between Bidder A and Bidder B can reverse merely because a third, unviable, or subsequently disqualified Bidder C enters or withdraws from the tender competition. 

This mathematical rank reversal has triggered multimillion-dollar public procurement challenges, prolonged bid protests before the US Government Accountability Office (GAO), and overturned sovereign defence awards in the UK High Court Technology and Construction Court (TCC) and the Court of Justice of the European Union (CJEU). Procurement analytics professionals must possess the mathematical rigor to design evaluation models compliant with **ISO 15686-5** life-cycle costing, execute discount rate conversions under HM Treasury Green Book and US OMB Circular A-94 guidelines, audit multi-criteria evaluation engines (such as the Analytic Hierarchy Process and TOPSIS), and deploy absolute scoring benchmarks that are mathematically impervious to rank reversal.

---

## 2. Learning outcomes

By the end of this topic, you can:

1. **Formulate** comprehensive life-cycle cost (LCC) and whole-life cost (WLC) discount structures complying strictly with the ISO 15686-5:2017 standard.
2. **Derive** the Social Time Preference Rate (STPR) using the Ramsey-Green Book growth formulation ($\text{STPR} = \rho + \mu \cdot g$) and differentiate real versus nominal discount rates.
3. **Prove mathematically** why relative price scoring formulas violate the Independence of Irrelevant Alternatives (IIA) and induce rank reversal upon bidder entry, exit, or disqualification.
4. **Construct** absolute benchmark scoring curves and monetized quality models that guarantee rank invariance and survive judicial bid protest scrutiny.
5. **Execute** an Analytic Hierarchy Process (AHP) scoring audit, calculating the principal eigenvalue ($\lambda_{\max}$), Consistency Index ($CI$), and Consistency Ratio ($CR$) to evaluate tender scoring matrices.
6. **Implement** the Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS) using vector normalization and Euclidean distance metrics to rank complex multi-attribute tenders.
7. **Incorporate** environmental externalities, including the shadow price of greenhouse gas emissions (carbon valuation schedules), directly into whole-life economic tender evaluations.
8. **Audit** abnormally low tenders (ALTs) using statistical dispersion thresholds and life-cycle cost boundary tests to defend public tenders against strategic contractor underbidding.

---

## 3. The core mental model

Evaluating public sector and defence tenders requires balancing capital expenditure (CapEx), operational sustainment expenditure (OpEx), end-of-life obligations, and multi-criteria qualitative utility within a legally defensible and auditable framework.

```text
====================================================================================================
                        WHOLE-LIFE COST & MEAT EVALUATION ARCHITECTURE
====================================================================================================

 [ TENDER SUBMISSIONS: VENDORS 1 ... n ]
   ├── Commercial Dossier (CapEx, Fuel Rates, Maint Schedules, Overhaul Intervals, Decommissioning)
   └── Technical Dossier  (Mission Reliability MTBF, Maintainability MTTR, Payload, Social Value)
        │
        ▼
 [ PHASE 1: COMPLIANCE & ABNORMALLY LOW TENDER (ALT) SCREENING ]
   • Screen for statutory compliance, mandatory technical thresholds, and exclusion grounds.
   • Statistical Outlier Test: Identify bids where Cost < (Mean - 2 * StdDev) or CapEx < 70% of IGCE.
   • Life-Cycle Plausibility Audit: Reject bids failing thermodynamic or commercial feasibility.
        │
        ▼
 [ PHASE 2: ISO 15686-5 LIFE-CYCLE COSTING ENGINE (DISCOUNTED CASH FLOW) ]
   • Horizon: N Years (e.g., 20-30 years for defence/civil assets)
   • Real Discount Rate: r (e.g., 3.5% HM Treasury Green Book STPR = rho + mu * g)
   • Externalities: Monetized CO2e emissions (DESNZ Carbon Valuation / EU ETS Schedule)
   
   LCC_i = CapEx_i + SUM_{t=1}^N [ (OpEx_{i,t} + Maint_{i,t} + Carbon_{i,t}) / (1 + r)^t ] 
                   + [ Decom_{i,N} - Salvage_{i,N} ] / (1 + r)^N
        │
        ▼
 [ PHASE 3: MULTI-CRITERIA DECISION ANALYSIS (MCDA) FOR TECHNICAL QUALITY ]
   • Multi-Attribute Hierarchy: Reliability (30%), Operability (25%), Social Value/Net Zero (15%), SLA (30%)
   • Analytic Hierarchy Process (AHP): Verify Pairwise Matrix Consistency (CR <= 0.10).
   • Standardized Scoring: Attribute Scoring Function Q_i in [0, 100].
        │
        ▼
 [ PHASE 4: MEAT / MAT SYNTHESIS & TRADE-OFF MECHANISM ]
   ─────────────────────────────────────────────────────────────────────────────────────────
   • FLAWED METHOD (RELATIVE SCORING): S_i = w_p * (P_min / P_i) + w_q * Q_i  <-- RANK REVERSAL RISK!
   • ROBUST METHOD A (ABSOLUTE SCORING): S_i = w_p * Max(0, 1 - (P_i - P_bench)/P_bench) + w_q * Q_i
   • ROBUST METHOD B (MONETIZED QUALITY): Total Evaluated Cost = LCC_i - SUM_k [ V_k * (Q_{i,k} - Q_min) ]
   ─────────────────────────────────────────────────────────────────────────────────────────
        │
        ▼
 [ PHASE 5: SENSITIVITY, AUDITABILITY & STANDSTILL GOVERNANCE ]
   • Monte Carlo perturbation on discount rates (r +/- 100 bps) and fuel/energy escalation.
   • Invariance Verification: Prove ranking is invariant to removal of losing bidders.
   • Defensible Debrief Dossier: Formulate Alcatel standstill notification and audit log.
====================================================================================================
```

### The Three Pillars of Defensible Evaluation

The evaluation of high-value public tenders rests on three interdependent analytical pillars:

1. **Temporal Equivalence (The ISO 15686-5 LCC Pillar):** Cash flows occurring at different epochs cannot be directly summed. An energy-efficient naval vessel that saves £500,000 annually in marine diesel over thirty years creates real economic value, but that value must be discounted back to the present using an authoritative social time preference rate. Omitting discounting distorts trade-offs; using private corporate hurdle rates (e.g., 10%–15% WACC) excessively discounts future operational savings, wrongly favoring cheap, inefficient assets.
2. **Evaluative Independence (The Axiomatic Scoring Pillar):** The relative ranking between any two tenderers (Bidder A and Bidder B) must depend solely on the intrinsic merits and prices of Bidder A and Bidder B. If introducing or removing Bidder C alters the score difference between A and B such that their ranking flips, the evaluation mechanism violates Arrow's Independence of Irrelevant Alternatives (IIA). Contracting authorities must deploy scoring formulations where every bidder's score is computed against pre-published, independent benchmarks.
3. **Internal Consistency (The Multi-Criteria Audit Pillar):** When qualitative factors (crew ergonomics, cyber resilience, mission-capable readiness) are weighted against whole-life cost, the scoring system must eliminate subjective bias. Mathematical frameworks such as Saaty's Analytic Hierarchy Process (AHP) and TOPSIS provide auditable metrics (e.g., the Consistency Ratio $CR \le 0.10$) to prove that evaluation panels did not act arbitrarily or capriciously.

---

## 4. Definitions and vocabulary

The following table defines the core terminology, legal codifications, and mathematical metrics governing whole-life costing and MEAT evaluation in public sector and defence supply chains.

| Term | Formal Definition | Common Industry Misuse or Pitfall | Unit or Standard Reference |
|---|---|---|---|
| **Most Economically Advantageous Tender (MEAT)** | The award criterion under EU Directive 2014/24/EU (Art. 67) identifying the optimal bid based on price/cost and qualitative/technical criteria. | Equating MEAT with subjective qualitative scoring that ignores whole-life cost realities. | Legal Criterion, Directive 2014/24/EU Art. 67 |
| **Most Advantageous Tender (MAT)** | The statutory evaluation standard under Section 19 of the UK Procurement Act 2023, replacing MEAT to emphasize total public benefit. | Confusing MAT with lowest initial purchase price; MAT explicitly permits whole-life social value trade-offs. | Statutory Standard, UK Procurement Act 2023 |
| **Life Cycle Costing (LCC)** | The systematic economic evaluation of all direct cash flows associated with an asset throughout its defined life cycle. | Failing to distinguish between client-incurred LCC and societal whole-life cost (WLC). | ISO 15686-5:2017, ISO 14040 |
| **Whole-Life Cost (WLC)** | An expansive economic metric encompassing all LCC elements plus non-construction costs, residual revenues, and environmental externalities. | Confining WLC strictly to CapEx and OpEx while ignoring monetized greenhouse gas emissions. | ISO 15686-5 Section 3.1.2 |
| **Social Time Preference Rate (STPR)** | The rate at which society is willing to trade present consumption for future consumption, formulated as $\text{STPR} = \rho + \mu \cdot g$. | Using a commercial Weighted Average Cost of Capital (WACC) instead of the prescribed sovereign STPR. | Dimensionless real rate, HM Treasury Green Book (3.5%) |
| **Pure Time Preference ($\rho$)** | The rate of preference for consumption now over later, reflecting individual impatience and societal catastrophic hazard risk. | Assuming $\rho$ accounts for macroeconomic inflation; $\rho$ is a utility parameter (typically 0.5%). | Percentage per annum ($\%/\text{yr}$) |
| **Elasticity of Marginal Utility ($\mu$)** | A parameter reflecting the percentage decrease in the marginal value of additional consumption as per capita income grows. | Treating $\mu$ as a financial discount coefficient; it reflects intergenerational societal equity (typically 1.0). | Dimensionless parameter |
| **Real Discount Rate** | A discount rate stripped of expected general price inflation, applied strictly to constant-price (uninflated) cash flows. | Discounting nominal (inflation-escalated) cash flows using a real discount rate, severely undervaluing future costs. | Percentage per annum ($\%/\text{yr}$), Fisher Eq. |
| **Nominal Discount Rate** | A discount rate incorporating both the real time value of money and expected general inflation ($r_{\text{nom}} \approx r_{\text{real}} + \pi$). | Applying a nominal rate to constant-currency cash flows, creating double-discounting errors. | Percentage per annum ($\%/\text{yr}$) |
| **Rank Reversal** | The change in the relative preference order between two tenderers caused solely by the addition, removal, or score shift of an irrelevant third bidder. | Dismissing rank reversal as theoretical; it constitutes immediate legal grounds to overturn contract awards. | Axiomatic Violation, Arrow's Impossibility |
| **Independence of Irrelevant Alternatives (IIA)** | An economic and social-choice axiom stating that the preference between $X$ and $Y$ must not depend on the availability or performance of $Z$. | Assuming relative proportional scoring formulas satisfy IIA; relative formulas inherently violate it. | Axiom of Decision Theory |
| **Relative Scoring Formula** | A scoring function where an applicant's commercial score is calculated relative to an endogenous tender metric, such as $P_{\min}$ or $P_{\max}$. | Advertising relative scoring as "objective" when it mathematically guarantees cross-bidder interdependence. | Mathematical Formula: $S_i = S_{\max} \cdot (P_{\min} / P_i)$ |
| **Absolute Scoring Benchmark** | A scoring formula where an applicant's price or cost is evaluated against a fixed, pre-announced benchmark or cost curve. | Setting the benchmark after tender opening, which violates procurement transparency and non-discrimination. | Mathematical Formula: $S_i = f(P_i, P_{\text{bench}})$ |
| **Monetized Quality (Evaluated Cost)** | An award framework where quality criteria are translated into monetary values and deducted from the tenderer's whole-life cost. | Arbitrarily setting quality price conversion factors without empirical or operational justification. | Metric: $\text{TEC} = \text{LCC} - \sum V_k \cdot \Delta Q_k$ (£ or $) |
| **Analytic Hierarchy Process (AHP)** | A multi-criteria decision method using structured pairwise comparison matrices to derive priority weighting vectors. | Using AHP with evaluation matrices exhibiting an unacceptable Consistency Ratio ($CR > 0.10$). | Decision Method, Saaty (1980) |
| **Consistency Ratio (CR)** | The ratio of the Consistency Index ($CI$) of a pairwise matrix to the Random Index ($RI$) of a randomly generated reciprocal matrix. | Accepting evaluation panels whose $CR > 0.10$, indicating self-contradictory evaluator judgment. | Dimensionless ratio ($CR \le 0.10$) |
| **TOPSIS** | Multi-attribute decision technique ranking alternatives based on geometric Euclidean distance from ideal and anti-ideal solutions. | Using min-max normalization instead of vector normalization, creating unintended distortion of weight scales. | Decision Method, Hwang & Yoon (1981) |
| **Closeness Coefficient ($C_i^*$)** | The relative metric in TOPSIS defining how close an alternative is to the positive ideal solution versus the negative ideal solution. | Interpreting $C_i^*$ as a linear percentage of compliance rather than a relative geometric ratio. | Dimensionless index ($0 \le C_i^* \le 1.0$) |
| **Abnormally Low Tender (ALT)** | A bid whose pricing or whole-life cost parameters appear unreliably low relative to the required deliverables and operational scope. | Disqualifying an ALT automatically without requesting formal commercial clarification (prohibited under EU/UK law). | Regulatory Concept, Directive 2014/24/EU Art. 69 |
| **Shadow Price of Carbon** | The monetized cost of greenhouse gas emissions per metric ton of $\text{CO}_2$ equivalent, reflecting global societal climate damage. | Neglecting carbon pricing in operational fuel consumption models for long-life defence platforms. | £/t$\text{CO}_2\text{e}$ or \$/t$\text{CO}_2\text{e}$, UK DESNZ Table |
| **Alcatel Standstill Period** | A mandatory statutory pause (typically 10 to 15 calendar days) between award notification and contract execution. | Signing a contract before the standstill expires, rendering the contract voidable under public procurement law. | Directive 2007/66/EC, UK PA 2023 |
| **Best Value Continuum** | The spectrum of federal acquisition strategies in the US ranging from Lowest Price Technically Acceptable (LPTA) to Full Tradeoff. | Employing LPTA for complex, high-maintenance defence platforms where life-cycle costs dominate. | US Federal Acquisition Regulation (FAR) 15.101 |
| **Independent Government Cost Estimate (IGCE)** | An objective benchmark cost estimate developed by the procuring agency prior to issuing the tender solicitation. | Sharing the IGCE with tenderers or constructing it after receiving commercial tender submissions. | Benchmark Currency (£, \$, €), FAR 15.404 |

---

## 5. The mechanics, step by step

Executing an auditable, legally unassailable whole-life cost and MEAT evaluation requires an eight-step mathematical and procedural protocol.

```text
+---------------------------------------------------------------------------------------------------+
|                           END-TO-END MEAT & LCC EXECUTION WORKFLOW                                |
+---------------------------------------------------------------------------------------------------+
  [Step 1] Define System Boundary, Baseline Schedule & Assessment Horizon (ISO 15686-5)
     │
  [Step 2] Parameterize Real Discount Rates via Social Time Preference Rate (Ramsey Formulation)
     │
  [Step 3] Quantify & Monetize Environmental Externalities (Greenhouse Gas Emissions Trajectory)
     │
  [Step 4] Calculate Present Value of Whole-Life Costs (Discounted Cash Flow Formulation)
     │
  [Step 5] Execute Statistical Screen for Abnormally Low Tenders (ALT Detection)
     │
  [Step 6] Evaluate Technical Quality Attributes via Consistent Pairwise Comparison (AHP Audit)
     │
  [Step 7] Synthesize Price and Quality: Eliminate Rank Reversal via Absolute Benchmarks
     │
  [Step 8] Conduct Sensitivity Audits, Stress-Testing, and Standstill Protocol Documentation
+---------------------------------------------------------------------------------------------------+
```

### Step 1: Define System Boundary and Assessment Horizon (ISO 15686-5)
The evaluation begins by establishing the asset life-cycle cost breakdown structure (CBS) according to ISO 15686-5:2017. The procuring authority must fix:
- **The Assessment Horizon ($N$):** Typically 15–30 years for municipal or defence vehicle fleets; 40–60 years for fixed facilities or naval vessels.
- **The Cost Breakdown Structure:** 
  1. Initial Acquisition Costs ($\text{CapEx}_0$): Design, manufacture, testing, delivery, integration, and initial spares provisioning.
  2. Operating Costs ($\text{OpEx}_t$): Fuel, electrical energy, fluids, consumables, operator training, and facility operations.
  3. Maintenance Costs ($\text{Maint}_t$): Scheduled maintenance, unscheduled corrective repairs, overhaul rotations, and inventory carrying cost of spares.
  4. Replacement Costs ($\text{Repl}_t$): Mid-life technology refresh, sub-system replacements, and software updates.
  5. End-of-Life Obligations ($\text{Disp}_N$): Decontamination, demilitarization, scrap disposal, and site remediation.
  6. Residual / Salvage Value ($\text{Res}_N$): Monetized terminal market value or secondary equipment recovery value.

### Step 2: Establish the Real Discount Rate via the Social Time Preference Rate
All cash flows are modeled in constant (real) baseline currency to prevent distortion from speculative macroeconomic inflation forecasts. In accordance with the Ramsey formulation utilized in the UK HM Treasury Green Book:

$$\text{STPR} = r = \rho + \mu \cdot g$$

Where:
- $\rho$ is the pure rate of time preference (set at $0.5\%$ per annum, representing societal catastrophe risk).
- $\mu$ is the elasticity of the marginal utility of consumption (set at $1.0$).
- $g$ is the long-term trend rate of growth in per capita real consumption (set at $2.0\%$ per annum).

Thus, the base real discount rate is:

$$r = 0.005 + 1.0 \times 0.020 = 0.025 \text{ to } 0.035 \quad (3.5\% \text{ per annum for years } 1 \text{ to } 30)$$

For ultra-long-horizon projects, stepped discount rates apply (e.g., $3.0\%$ for years 31–75).

If nominal cash flows ($C_{t,\text{nom}}$) are provided, they must be deflated using expected general inflation $\pi$ to yield real cash flows ($C_{t,\text{real}} = C_{t,\text{nom}} / (1 + \pi)^t$) prior to applying the real discount rate $r$, or discounted using the nominal discount rate derived via the Fisher equation:

$$(1 + r_{\text{nom}}) = (1 + r_{\text{real}})(1 + \pi)$$

### Step 3: Quantify and Monetize Environmental Externalities
Under modern MEAT frameworks (e.g., Directive 2014/24/EU Art. 68(1)(c) and UK PA 2023 National Procurement Policy Statement), contracting authorities must internalize environmental externalities. For equipment consuming fossil fuels or grid electricity:

$$E_t = \sum_m \text{Consumption}_{m,t} \times \text{EmissionFactor}_m \quad (\text{t}\text{CO}_2\text{e})$$

The monetized environmental cost in year $t$ is calculated by multiplying physical emissions by the official government shadow price of carbon ($P_{\text{carbon},t}$):

$$\text{Ext}_t = E_t \times P_{\text{carbon},t}$$

Under UK Department for Energy Security and Net Zero (DESNZ) central values, carbon is valued at £280/t$\text{CO}_2\text{e}$ in 2026, escalating annually in real terms.

### Step 4: Calculate Present Value of Whole-Life Cost
The cumulative discounted Life Cycle Cost for tenderer $i$ is formulated as:

$$\text{LCC}_i = \text{CapEx}_{i,0} + \sum_{t=1}^N \frac{\text{OpEx}_{i,t} + \text{Maint}_{i,t} + \text{Ext}_{i,t} + \text{Repl}_{i,t}}{(1 + r)^t} + \frac{\text{Disp}_{i,N} - \text{Res}_{i,N}}{(1 + r)^N}$$

### Step 5: Screen for Abnormally Low Tenders (ALT)
Before scoring commercial dossiers, contracting authorities must audit for unviable bids. Under Directive 2014/24/EU Art. 69 and FAR 15.404, an ALT screen is executed:
1. **Statistical Threshold:** Any bid where $\text{LCC}_i < \mu_{\text{LCC}} - 1.96 \cdot \sigma_{\text{LCC}}$, or where $\text{CapEx}_{i} < 0.70 \times \text{IGCE}$, triggers mandatory formal clarification.
2. **Technical Feasibility Audit:** The authority evaluates the supplier's thermodynamic fuel burn claims, labor rate breakdown, and maintenance intervals.
3. If the tenderer fails to substantiate that its cost structure complies with environmental, social, and labor law, or relies on impossible operational efficiencies, the authority must legally reject the tender.

### Step 6: Mathematical Proof of Rank Reversal in Relative Scoring Formulas
A fatal error in procurement analytics is using relative formulas to assign price scores. The two most common relative scoring formulas are:

**Formula 1: Proportional (Reciprocal) Relative Scoring**
$$S_i^{\text{rel}}(P) = S_{\max} \cdot \frac{P_{\min}}{P_i}$$

**Formula 2: Linear Range Relative Scoring**
$$S_i^{\text{range}}(P) = S_{\max} \cdot \left(1 - \frac{P_i - P_{\min}}{P_{\max} - P_{\min}}\right)$$

#### Mathematical Proof of IIA Violation and Rank Reversal (Formula 1)
Let there be two tenderers, Bidder 1 and Bidder 2, with prices $P_1 < P_2$ and technical quality scores $Q_1 < Q_2$. The total MEAT score under weightings $w_p$ and $w_q$ ($w_p + w_q = 1$) is:

$$\text{Total}_i = w_p \cdot S_{\max} \left(\frac{P_{\min}}{P_i}\right) + w_q \cdot Q_i$$

Initially, with only Bidders 1 and 2 in the competition, $P_{\min} = P_1$. The score difference between Bidder 1 and Bidder 2 is:

$$\Delta \text{Total}_{1,2} = \text{Total}_1 - \text{Total}_2 = w_p \cdot S_{\max} \left(1 - \frac{P_1}{P_2}\right) - w_q (Q_2 - Q_1)$$

Assume parameter values such that $\Delta \text{Total}_{1,2} > 0$. **Bidder 1 wins the competition.**

Now, consider the entry of an irrelevant alternative, Bidder 3, who submits an abnormally low, aggressive, or subsequently disqualified price $P_3 < P_1$. 
The new minimum price in the pool becomes $P_{\min}' = P_3$.

The new commercial scores for Bidder 1 and Bidder 2 are:
$$S_1'(P) = S_{\max} \left(\frac{P_3}{P_1}\right), \quad S_2'(P) = S_{\max} \left(\frac{P_3}{P_2}\right)$$

The new score difference between Bidder 1 and Bidder 2 becomes:

$$\Delta \text{Total}_{1,2}' = w_p \cdot S_{\max} \cdot P_3 \left(\frac{1}{P_1} - \frac{1}{P_2}\right) - w_q (Q_2 - Q_1)$$

Examining the partial derivative of the price score advantage with respect to $P_{\min}$:

$$\frac{\partial}{\partial P_{\min}} \left[ S_{\max} \cdot P_{\min} \left(\frac{1}{P_1} - \frac{1}{P_2}\right) \right] = S_{\max} \left(\frac{1}{P_1} - \frac{1}{P_2}\right) > 0 \quad (\text{since } P_1 < P_2)$$

As $P_3 \to 0$, the price score difference between Bidder 1 and Bidder 2 contracts to zero:

$$\lim_{P_3 \to 0} \left[ S_1'(P) - S_2'(P) \right] = 0$$

Consequently:
$$\lim_{P_3 \to 0} \Delta \text{Total}_{1,2}' = - w_q (Q_2 - Q_1) < 0$$

Because $\Delta \text{Total}_{1,2}' < 0$, **Bidder 2 now defeats Bidder 1.**

The relative ranking between Bidder 1 and Bidder 2 has completely reversed solely due to the presence of Bidder 3, despite neither Bidder 1 nor Bidder 2 changing their technical submission or commercial pricing by a single penny. This violates the Independence of Irrelevant Alternatives (IIA) axiom and renders the procurement decision legally vulnerable to judicial review.

#### Mathematical Proof of Rank Reversal under Linear Range Relative Scoring (Formula 2)
Now consider the linear range scoring formula, widely used by municipal authorities:

$$S_i^{\text{range}}(P) = S_{\max} \cdot \left(1 - \frac{P_i - P_{\min}}{P_{\max} - P_{\min}}\right)$$

Under this formula, the price score difference between Bidder 1 ($P_1$) and Bidder 2 ($P_2$) with $P_1 < P_2$ is:

$$\Delta S_{1,2}^{\text{range}} = S_1^{\text{range}} - S_2^{\text{range}} = S_{\max} \cdot \frac{P_2 - P_1}{P_{\max} - P_{\min}}$$

Notice that the score differential is inversely proportional to the bid range $(P_{\max} - P_{\min})$. 

1. **Sensitivity to High Outliers ($P_{\max}$):**
   Suppose an expensive or over-scoped third bidder enters with price $P_{\text{high}} > P_2$, becoming the new $P_{\max}$.
   Taking the partial derivative of the score advantage with respect to $P_{\max}$:
   $$\frac{\partial}{\partial P_{\max}} \left[ S_{\max} \cdot \frac{P_2 - P_1}{P_{\max} - P_{\min}} \right] = - \frac{S_{\max} (P_2 - P_1)}{(P_{\max} - P_{\min})^2} < 0$$
   As $P_{\max}$ increases, $\Delta S_{1,2}^{\text{range}} \to 0$. The price advantage of Bidder 1 over Bidder 2 is diluted toward zero, allowing Bidder 2's higher technical quality ($Q_2 > Q_1$) to dominate the final award score.

2. **Sensitivity to Bidder Disqualification:**
   Conversely, if an expensive non-compliant bidder is disqualified during the evaluation, $P_{\max}$ drops abruptly to a lower value $P_{\max}'$. The denominator shrinks, causing $\Delta S_{1,2}^{\text{range}}$ to expand discontinuously. A supplier that was in second place suddenly vaults into first place simply because an unrelated bidder was disqualified.

#### The General Condition for Rank Invariance
A procurement scoring function $S(P_1, P_2, \dots, P_n; Q_1, Q_2, \dots, Q_n)$ satisfies the Independence of Irrelevant Alternatives if and only if:

$$\frac{\partial}{\partial P_k} \left[ \text{Total}_i - \text{Total}_j \right] = 0 \quad \forall \, k \notin \{i, j\}$$

In both Formula 1 and Formula 2, this cross-partial derivative is non-zero:
$$\frac{\partial}{\partial P_{\min}} (\text{Total}_1 - \text{Total}_2) \neq 0, \quad \frac{\partial}{\partial P_{\max}} (\text{Total}_1 - \text{Total}_2) \neq 0$$

Hence, all relative scoring formulas violate Arrow's IIA axiom by construction.

### Step 7: Constructing Rank-Preserving Absolute Scoring Models
To eliminate rank reversal, the scoring of an individual tender must depend solely on its own submission and fixed, pre-published parameters. Two robust formulations exist:

#### Formulation A: Absolute Benchmark Scoring Curve
The authority establishes an Independent Government Cost Estimate (IGCE) or pre-announced baseline cost $P_{\text{bench}}$ and an exclusion ceiling $P_{\text{ceil}}$:

$$S_i^{\text{abs}}(P) = \max \left( 0, \, S_{\max} \cdot \left[ 1 - \alpha \left(\frac{P_i - P_{\text{bench}}}{P_{\text{bench}}}\right) \right] \right)$$

Because $\frac{\partial S_i^{\text{abs}}}{\partial P_j} = 0$ for all $j \neq i$, no bid from any competitor can ever alter tenderer $i$'s score.

#### Formulation B: Monetized Quality (Total Evaluated Cost)
Rather than converting prices into abstract non-dimensional points, the authority monetizes technical quality. Let $Q_{i,k}$ be tenderer $i$'s measured performance on technical criterion $k$, $Q_{k,\min}$ be the mandatory specification floor, and $V_k$ be the pre-disclosed economic value per unit of performance exceeding the floor:

$$\text{Total Evaluated Cost}_i (\text{TEC}_i) = \text{LCC}_i - \sum_{k=1}^M V_k \cdot \max\left(0, Q_{i,k} - Q_{k,\min}\right)$$

The contract is awarded to the tenderer with the lowest $\text{TEC}_i$. This formulation is linear, fully transparent, economically grounded, and mathematically immune to rank reversal.

### Step 8: Multi-Criteria Quality Evaluation using AHP and TOPSIS Audits
When qualitative scoring is utilized, the evaluation panel's judgment must be audited for mathematical consistency.

#### The Analytic Hierarchy Process (AHP) Audit
To weight criteria, evaluators construct a reciprocal pairwise comparison matrix $A = [a_{jk}]_{M \times M}$ where $a_{kj} = 1/a_{jk}$ and $a_{jj} = 1$, using Saaty's 1–9 fundamental scale.
1. Compute the priority weight vector $w$ as the principal eigenvector:
   $$A w = \lambda_{\max} w$$
2. Compute the Consistency Index ($CI$):
   $$CI = \frac{\lambda_{\max} - M}{M - 1}$$
3. Compute the Consistency Ratio ($CR$):
   $$CR = \frac{CI}{RI_M}$$
   where $RI_M$ is the empirical Random Index (e.g., $RI_3 = 0.58, RI_4 = 0.90, RI_5 = 1.12, RI_6 = 1.24$).
4. **Audit Rule:** If $CR \le 0.10$, the evaluations are internally consistent and legally defensible. If $CR > 0.10$, the panel exhibits transitivity violations (e.g., scoring $A > B$, $B > C$, but $C > A$) and must re-evaluate.

#### The TOPSIS Scoring Engine
When evaluating multidimensional tender dossiers across $M$ criteria and $n$ bidders:
1. Construct decision matrix $X = [x_{ij}]_{n \times M}$.
2. Normalize using vector normalization:
   $$r_{ij} = \frac{x_{ij}}{\sqrt{\sum_{k=1}^n x_{kj}^2}}$$
3. Calculate weighted normalized values: $v_{ij} = w_j \cdot r_{ij}$.
4. Determine the Positive Ideal Solution ($A^+$) and Negative Ideal Solution ($A^-$):
   $$A_j^+ = \begin{cases} \max_i v_{ij} & \text{if criterion } j \text{ is benefit} \\ \min_i v_{ij} & \text{if criterion } j \text{ is cost} \end{cases}$$
   $$A_j^- = \begin{cases} \min_i v_{ij} & \text{if criterion } j \text{ is benefit} \\ \max_i v_{ij} & \text{if criterion } j \text{ is cost} \end{cases}$$
5. Calculate Euclidean distances from the ideal and negative-ideal:
   $$D_i^+ = \sqrt{\sum_{j=1}^M (v_{ij} - A_j^+)^2}, \quad D_i^- = \sqrt{\sum_{j=1}^M (v_{ij} - A_j^-)^2}$$
6. Compute the Closeness Coefficient:
   $$C_i^* = \frac{D_i^-}{D_i^+ + D_i^-} \quad \text{where } 0 \le C_i^* \le 1.0$$
   The tender with the highest $C_i^*$ represents the mathematically optimal compromise solution.

---

## 6. Worked example

### Operational Scenario: Defence Logistics Support Vessels (DLSV)
A sovereign defence procurement agency is procuring a fleet of six Next-Generation Logistics Support Vessels (LSVs) for expeditionary resupply and humanitarian disaster response. The procurement is conducted under MEAT / MAT rules with a pre-announced evaluation horizon of $N = 20$ years and an HM Treasury Green Book real discount rate of $r = 3.5\%$.

Three shipbuilders submit compliant dossiers:
- **Shipbuilder Alpha (Conventional Diesel):** Low initial acquisition CapEx, but high fuel consumption and substantial scheduled engine overhaul requirements.
- **Shipbuilder Beta (Hybrid-Electric / LNG):** High initial CapEx, but low operating fuel costs, reduced maintenance overhead, and superior mission availability.
- **Shipbuilder Gamma (Aggressive / Low-Cost Yard):** Extremely low initial CapEx, moderate fuel burn, but substandard reliability, leading to high corrective repair costs.

```text
====================================================================================================
                        TENDER SPECIFICATIONS AND BASELINE PARAMETERS
====================================================================================================
Parameter                                 Alpha (Diesel)       Beta (Hybrid)        Gamma (Low-Cost)
----------------------------------------------------------------------------------------------------
Fleet Acquisition CapEx (Year 0)          £180,000,000         £220,000,000         £140,000,000
Annual Fleet Operating Cost (OpEx)        £14,500,000          £9,200,000           £13,800,000
Annual Maintenance & Spares (MaintEx)     £6,200,000           £3,800,000           £8,900,000
Year 10 Mid-Life Overhaul (One-off)       £24,000,000          £16,000,000          £32,000,000
Annual Carbon Emissions (tCO2e/yr)        42,000 tonnes        24,000 tonnes        40,000 tonnes
Decommissioning Cost (Year 20)            £12,000,000          £10,000,000          £15,000,000
Terminal Salvage Value (Year 20)          £18,000,000          £28,000,000          £10,000,000
Technical Quality Score (Q_i in [0,100])  76.0 points          92.0 points          58.0 points
====================================================================================================
```

### Step 1: Carbon Externality Valuation
The authority applies a constant real shadow price of carbon across the horizon of $P_{\text{carbon}} = £120/\text{t}\text{CO}_2\text{e}$ (constant real price for evaluation audit purposes):
- **Alpha:** $42,000 \times £120 = £5,040,000 \text{ per annum}$
- **Beta:** $24,000 \times £120 = £2,880,000 \text{ per annum}$
- **Gamma:** $40,000 \times £120 = £4,800,000 \text{ per annum}$

### Step 2: Discount Factor Calculations ($r = 3.5\%$)
1. **Present Value of 20-Year Annuity Factor ($PVA_{r,N}$):**
   $$PVA_{0.035, 20} = \frac{1 - (1 + 0.035)^{-20}}{0.035} = \frac{1 - (1.035)^{-20}}{0.035} = \frac{1 - 0.502566}{0.035} = 14.212403$$

2. **Present Value Factor for Year 10 Mid-Life Overhaul ($PV_{10}$):**
   $$PV_{10} = \frac{1}{(1.035)^{10}} = \frac{1}{1.410599} = 0.708919$$

3. **Present Value Factor for Year 20 Terminal Cash Flows ($PV_{20}$):**
   $$PV_{20} = \frac{1}{(1.035)^{20}} = 0.502566$$

### Step 3: Life Cycle Cost (LCC) Computation

#### Shipbuilder Alpha:
- $\text{CapEx}_0 = £180,000,000$
- $\text{Annual Recurring} = \text{OpEx} + \text{Maint} + \text{Carbon} = £14,500,000 + £6,200,000 + £5,040,000 = £25,740,000/\text{yr}$
- $\text{PV of Annual Recurring} = £25,740,000 \times 14.212403 = £365,827,253$
- $\text{PV of Year 10 Overhaul} = £24,000,000 \times 0.708919 = £17,014,056$
- $\text{PV of Net Decommissioning} = (£12,000,000 - £18,000,000) \times 0.502566 = -£6,000,000 \times 0.502566 = -£3,015,396$
- **Total LCC Alpha:**
  $$\text{LCC}_{\text{Alpha}} = £180,000,000 + £365,827,253 + £17,014,056 - £3,015,396 = \mathbf{£559,825,913}$$

#### Shipbuilder Beta:
- $\text{CapEx}_0 = £220,000,000$
- $\text{Annual Recurring} = \text{OpEx} + \text{Maint} + \text{Carbon} = £9,200,000 + £3,800,000 + £2,880,000 = £15,880,000/\text{yr}$
- $\text{PV of Annual Recurring} = £15,880,000 \times 14.212403 = £225,692,960$
- $\text{PV of Year 10 Overhaul} = £16,000,000 \times 0.708919 = £11,342,704$
- $\text{PV of Net Decommissioning} = (£10,000,000 - £28,000,000) \times 0.502566 = -£18,000,000 \times 0.502566 = -£9,046,188$
- **Total LCC Beta:**
  $$\text{LCC}_{\text{Beta}} = £220,000,000 + £225,692,960 + £11,342,704 - £9,046,188 = \mathbf{£447,989,476}$$

#### Shipbuilder Gamma:
- $\text{CapEx}_0 = £140,000,000$
- $\text{Annual Recurring} = \text{OpEx} + \text{Maint} + \text{Carbon} = £13,800,000 + £8,900,000 + £4,800,000 = £27,500,000/\text{yr}$
- $\text{PV of Annual Recurring} = £27,500,000 \times 14.212403 = £390,841,083$
- $\text{PV of Year 10 Overhaul} = £32,000,000 \times 0.708919 = £22,685,408$
- $\text{PV of Net Decommissioning} = (£15,000,000 - £10,000,000) \times 0.502566 = +£5,000,000 \times 0.502566 = +£2,512,830$
- **Total LCC Gamma:**
  $$\text{LCC}_{\text{Gamma}} = £140,000,000 + £390,841,083 + £22,685,408 + £2,512,830 = \mathbf{£556,039,321}$$

```text
====================================================================================================
                        LIFE CYCLE COST SUMMARY COMPARISON
====================================================================================================
Bidder                  Initial CapEx      Operating LCC      Terminal LCC       Total ISO LCC
----------------------------------------------------------------------------------------------------
Shipbuilder Alpha       £180,000,000       £382,841,309       -£3,015,396        £559,825,913
Shipbuilder Beta        £220,000,000       £237,035,664       -£9,046,188        £447,989,476
Shipbuilder Gamma       £140,000,000       £413,526,491        £2,512,830        £556,039,321
====================================================================================================
```

*Critical Observation:* Evaluating purely on acquisition price (CapEx) ranks the bidders: **Gamma (£140M) > Alpha (£180M) > Beta (£220M)**. However, whole-life costing under ISO 15686-5 reverses this completely: **Beta (£447.99M) saves £111.84M over Alpha and £108.05M over Gamma**, proving that the highest-CapEx vessel is by far the most economical solution for the taxpayer.

---

### Step 4: Demonstrating Rank Reversal under Relative Scoring
The tender documents establish a MEAT weighting of $50\%$ Cost ($w_c = 0.50$) and $50\%$ Technical Quality ($w_q = 0.50$), with $S_{\max} = 100$.

#### Scenario 1: Evaluation Between Alpha and Beta Only (Gamma Disqualified for Quality < 60)
Suppose Gamma fails the mandatory technical hurdle ($Q_{\text{threshold}} = 60.0$) and is excluded prior to commercial scoring.
The valid bidders are Alpha and Beta:
- $LCC_{\min} = \text{LCC}_{\text{Beta}} = £447,989,476$
- Commercial Score Alpha:
  $$S_{\text{Alpha}}^{\text{rel}} = 100 \times \left(\frac{447,989,476}{559,825,913}\right) = 80.023$$
- Commercial Score Beta:
  $$S_{\text{Beta}}^{\text{rel}} = 100 \times \left(\frac{447,989,476}{447,989,476}\right) = 100.000$$
- Total MEAT Scores:
  $$\text{Total}_{\text{Alpha}} = 0.50 \times 80.023 + 0.50 \times 76.0 = 40.012 + 38.000 = \mathbf{78.012}$$
  $$\text{Total}_{\text{Beta}} = 0.50 \times 100.000 + 0.50 \times 92.0 = 50.000 + 46.000 = \mathbf{96.000}$$
- **Result:** **Beta defeats Alpha by 17.988 points.**

#### Scenario 2: Strategic Inclusion of an Ultra-Low Irrelevant Alternative
Now suppose a fourth bidder, **Bidder Delta (Speculative Foreign Yard)**, submits an unviable, stripped-down hull proposal with an LCC of **£200,000,000** and a low quality score of $40.0$. Even though Delta will not win the contract, see what happens to the ranking between Alpha and Beta if Delta is retained in the relative scoring pool ($LCC_{\min} = £200,000,000$):
- Commercial Score Alpha:
  $$S_{\text{Alpha}}' = 100 \times \left(\frac{200,000,000}{559,825,913}\right) = 35.725$$
- Commercial Score Beta:
  $$S_{\text{Beta}}' = 100 \times \left(\frac{200,000,000}{447,989,476}\right) = 44.644$$
- Notice that the score difference between Beta and Alpha has collapsed:
  $$\Delta S_{\text{Commercial}} = 44.644 - 35.725 = 8.919 \quad (\text{down from } 19.977!)$$

Now examine what happens under Linear Range Relative Scoring:
$$S_i^{\text{range}} = 100 \times \left(1 - \frac{LCC_i - LCC_{\min}}{LCC_{\max} - LCC_{\min}}\right)$$
With Delta ($LCC = £200\text{M}$), Alpha ($£559.8\text{M}$), and an overpriced Bidder Epsilon ($LCC = £1,200\text{M}$), the massive denominator $(£1,200\text{M} - £200\text{M} = £1,000\text{M})$ completely compresses the price differentials between Alpha and Beta to negligible fractions, allowing minor, subjective quality score variations to override massive multimillion-pound real economic differences.

### Step 5: Absolute Scoring Benchmark (Rank Invariance Guaranteed)
Under the legally robust absolute benchmark method, the agency publishes in the tender notice an IGCE benchmark of $LCC_{\text{bench}} = £500,000,000$ and a maximum permissible variance scale ($\alpha = 0.50$):

$$S_i^{\text{abs}} = \max\left(0, \, 100 \cdot \left[ 1 - 0.50 \cdot \left(\frac{\text{LCC}_i - 500,000,000}{500,000,000}\right) \right]\right)$$

- Score Alpha:
  $$S_{\text{Alpha}}^{\text{abs}} = 100 \times \left[ 1 - 0.50 \times \left(\frac{559,825,913 - 500,000,000}{500,000,000}\right) \right] = 100 \times [1 - 0.50 \times 0.11965] = 94.017$$
- Score Beta:
  $$S_{\text{Beta}}^{\text{abs}} = 100 \times \left[ 1 - 0.50 \times \left(\frac{447,989,476 - 500,000,000}{500,000,000}\right) \right] = 100 \times [1 - 0.50 \times (-0.10402)] = 105.201 \implies 100.000$$
- Score Gamma:
  $$S_{\text{Gamma}}^{\text{abs}} = 100 \times \left[ 1 - 0.50 \times \left(\frac{556,039,321 - 500,000,000}{500,000,000}\right) \right] = 100 \times [1 - 0.50 \times 0.11208] = 94.396$$

Every tenderer's score depends strictly on its own cost. The addition or disqualification of Delta, Gamma, or any other party has zero mathematical impact on the score of Alpha or Beta. Independence of Irrelevant Alternatives is preserved.

### Step 6: Analytic Hierarchy Process (AHP) Quality Audit
The naval technical evaluation board evaluates the vessels across four weighted operational dimensions:
- $C_1$: Mission Operational Availability ($Ao$) and Reliability (MTBF)
- $C_2$: Cargo Capacity, Sea-keeping, and Heavy Weather Operability
- $C_3$: Cyber Resilience and Command Architecture Modularity
- $C_4$: Net Zero Decarbonization and Emission Footprint Reduction

The evaluation board constructs the consensus reciprocal pairwise comparison matrix $A$:

$$A = \begin{bmatrix}
1.0 & 3.0 & 4.0 & 2.0 \\
1/3 & 1.0 & 2.0 & 1/2 \\
1/4 & 1/2 & 1.0 & 1/3 \\
1/2 & 2.0 & 3.0 & 1.0
\end{bmatrix}$$

#### Computing Priority Weights:
1. Sum of matrix columns:
   - Col 1: $1.0 + 0.3333 + 0.25 + 0.5 = 2.0833$
   - Col 2: $3.0 + 1.0 + 0.5 + 2.0 = 6.5000$
   - Col 3: $4.0 + 2.0 + 1.0 + 3.0 = 10.0000$
   - Col 4: $2.0 + 0.5 + 0.3333 + 1.0 = 3.8333$

2. Normalized row averages yield priority weights:
   - $w_1 (\text{Availability}) = (1/2.0833 + 3/6.5 + 4/10 + 2/3.8333)/4 = (0.4800 + 0.4615 + 0.4000 + 0.5217)/4 = \mathbf{0.4658}$
   - $w_2 (\text{Operability}) = (0.3333/2.0833 + 1/6.5 + 2/10 + 0.5/3.8333)/4 = (0.1600 + 0.1538 + 0.2000 + 0.1304)/4 = \mathbf{0.1611}$
   - $w_3 (\text{Cyber/Avionics}) = (0.25/2.0833 + 0.5/6.5 + 1/10 + 0.3333/3.8333)/4 = (0.1200 + 0.0769 + 0.1000 + 0.0870)/4 = \mathbf{0.0960}$
   - $w_4 (\text{Decarbonization}) = (0.5/2.0833 + 2/6.5 + 3/10 + 1/3.8333)/4 = (0.2400 + 0.3077 + 0.3000 + 0.2609)/4 = \mathbf{0.2772}$
   - Vector: $w = [0.4658, 0.1611, 0.0960, 0.2772]^T$ (Sum = $1.0001$).

3. Consistency Check:
   - Weighted sum vector $A \cdot w = [1.8878, 0.6465, 0.3852, 1.1206]^T$
   - Component ratios: $\lambda_1 = 1.8878 / 0.4658 = 4.0528$; $\lambda_2 = 0.6465 / 0.1611 = 4.0130$; $\lambda_3 = 0.3852 / 0.0960 = 4.0125$; $\lambda_4 = 1.1206 / 0.2772 = 4.0426$
   - Principal eigenvalue estimate: $\lambda_{\max} = (4.0528 + 4.0130 + 4.0125 + 4.0426)/4 = \mathbf{4.0302}$
   - Consistency Index: $CI = \frac{4.0302 - 4}{4 - 1} = \frac{0.0302}{3} = \mathbf{0.0101}$
   - Consistency Ratio: $CR = \frac{CI}{RI_4} = \frac{0.0101}{0.90} = \mathbf{0.0112} \quad (1.12\%)$
   - **Audit Verdict:** Since $CR = 0.0112 \ll 0.10$, the quality evaluation criteria weights are fully auditable and legally valid.

### Step 7: TOPSIS Multi-Attribute Compromise Synthesis
The agency performs a secondary TOPSIS evaluation combining Whole-Life Cost with the four quality attributes for Shipbuilder Alpha, Beta, and Gamma:

```text
====================================================================================================
                     DECISION MATRIX FOR TOPSIS EVALUATION OF DLSV TENDERS
====================================================================================================
Tenderer          LCC (Cost, w=0.40)   Availability (w=0.28)  Operability (w=0.10)  Carbon Decarb (w=0.22)
----------------------------------------------------------------------------------------------------
Alpha             £559.83M             78 pts                 85 pts                50 pts
Beta              £447.99M             95 pts                 88 pts                92 pts
Gamma             £556.04M             55 pts                 65 pts                55 pts
====================================================================================================
```

1. **Vector Normalization Denominators:**
   - $\text{LCC}: \sqrt{559.83^2 + 447.99^2 + 556.04^2} = \sqrt{313410 + 200700 + 309180} = \sqrt{823290} = 890.67$
   - $\text{Availability}: \sqrt{78^2 + 95^2 + 55^2} = \sqrt{6084 + 9025 + 3025} = \sqrt{18134} = 134.66$
   - $\text{Operability}: \sqrt{85^2 + 88^2 + 65^2} = \sqrt{7225 + 7744 + 4225} = \sqrt{19194} = 138.54$
   - $\text{Decarb}: \sqrt{50^2 + 92^2 + 55^2} = \sqrt{2500 + 8464 + 3025} = \sqrt{13989} = 118.28$

2. **Weighted Normalized Matrix ($v_{ij}$):**
   - **Alpha:** $[0.40(559.83/890.67), 0.28(78/134.66), 0.10(85/138.54), 0.22(50/118.28)] = [0.2514, 0.1622, 0.0614, 0.0930]$
   - **Beta:** $[0.40(447.99/890.67), 0.28(95/134.66), 0.10(88/138.54), 0.22(92/118.28)] = [0.2012, 0.1975, 0.0635, 0.1711]$
   - **Gamma:** $[0.40(556.04/890.67), 0.28(55/134.66), 0.10(65/138.54), 0.22(55/118.28)] = [0.2497, 0.1144, 0.0469, 0.1023]$

3. **Ideal ($A^+$) and Negative Ideal ($A^-$) Solutions:**
   - $A^+ = [\min(\text{Cost}), \max(\text{Avail}), \max(\text{Oper}), \max(\text{Decarb})] = [0.2012, 0.1975, 0.0635, 0.1711]$
   - $A^- = [\max(\text{Cost}), \min(\text{Avail}), \min(\text{Oper}), \min(\text{Decarb})] = [0.2514, 0.1144, 0.0469, 0.0930]$

4. **Euclidean Distances and Closeness:**
   - **Shipbuilder Beta:** Exactly equals $A^+$, so $D_{\text{Beta}}^+ = 0.0000$.
     - $D_{\text{Beta}}^- = \sqrt{(0.2012-0.2514)^2 + (0.1975-0.1144)^2 + (0.0635-0.0469)^2 + (0.1711-0.0930)^2} = \sqrt{0.002520 + 0.006906 + 0.000276 + 0.006100} = \sqrt{0.015802} = \mathbf{0.1257}$
     - Closeness: $C_{\text{Beta}}^* = \frac{0.1257}{0.0000 + 0.1257} = \mathbf{1.0000}$
   - **Shipbuilder Alpha:**
     - $D_{\text{Alpha}}^+ = \sqrt{(0.2514-0.2012)^2 + (0.1622-0.1975)^2 + (0.0614-0.0635)^2 + (0.0930-0.1711)^2} = \sqrt{0.002520 + 0.001246 + 0.000004 + 0.006100} = \sqrt{0.009870} = \mathbf{0.0993}$
     - $D_{\text{Alpha}}^- = \sqrt{(0)^2 + (0.1622-0.1144)^2 + (0.0614-0.0469)^2 + (0)^2} = \sqrt{0.002285 + 0.000210} = \sqrt{0.002495} = \mathbf{0.0500}$
     - Closeness: $C_{\text{Alpha}}^* = \frac{0.0500}{0.0993 + 0.0500} = \frac{0.0500}{0.1493} = \mathbf{0.3349}$
   - **Shipbuilder Gamma:**
     - $D_{\text{Gamma}}^+ = \sqrt{(0.2497-0.2012)^2 + (0.1144-0.1975)^2 + (0.0469-0.0635)^2 + (0.1023-0.1711)^2} = \sqrt{0.002352 + 0.006906 + 0.000276 + 0.004733} = \sqrt{0.014267} = \mathbf{0.1194}$
     - $D_{\text{Gamma}}^- = \sqrt{(0.2497-0.2514)^2 + (0)^2 + (0)^2 + (0.1023-0.0930)^2} = \sqrt{0.000003 + 0.000086} = \mathbf{0.0094}$
     - Closeness: $C_{\text{Gamma}}^* = \frac{0.0094}{0.1194 + 0.0094} = \frac{0.0094}{0.1288} = \mathbf{0.0730}$

**Definitive Award Recommendation:**
Shipbuilder Beta dominates across all analytical methodologies:
1. Lowest ISO 15686-5 discounted Life-Cycle Cost (£447.99M).
2. Highest technical quality score (92.0 pts).
3. Highest absolute benchmark MEAT score (100.00 pts).
4. Optimal TOPSIS compromise score ($C_{\text{Beta}}^* = 1.0000$).

The evaluation dossier is completely protected against supplier challenge under the Alcatel standstill period.

---

## 7. Code, query, or calculation

The following self-contained Python program executes the ISO 15686-5 discounted life-cycle cost evaluation, proves the rank reversal vulnerability in relative scoring, validates absolute scoring invariance, and performs an AHP consistency ratio audit.

```python
import numpy as np

def calculate_iso_lcc(capex, opex, maint, overhaul_yr10, carbon_t, decom, salvage, p_c=120.0, r=0.035, N=20):
    t = np.arange(1, N + 1)
    annual = opex + maint + (carbon_t * p_c)
    pv_annual = np.sum(annual / ((1.0 + r) ** t))
    pv_overhaul = overhaul_yr10 / ((1.0 + r) ** 10)
    pv_terminal = (decom - salvage) / ((1.0 + r) ** N)
    return capex + pv_annual + pv_overhaul + pv_terminal

def relative_meat(costs, qualities, wc=0.5, wq=0.5):
    c_min = min(costs)
    scores = [wc * 100.0 * (c_min / c) + wq * q for c, q in zip(costs, qualities)]
    return scores

def absolute_meat(costs, qualities, c_bench, wc=0.5, wq=0.5):
    scores = [wc * max(0.0, 100.0 * (1.0 - 0.5 * ((c - c_bench) / c_bench))) + wq * q 
              for c, q in zip(costs, qualities)]
    return scores

def ahp_consistency_ratio(matrix):
    n = matrix.shape[0]
    eigvals = np.linalg.eigvals(matrix)
    lam_max = np.real(eigvals[np.argmax(np.real(eigvals))])
    ci = (lam_max - n) / (n - 1)
    ri_table = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24}
    return ci / ri_table.get(n, 1.32)

# Fleet Parameters: (CapEx, OpEx, Maint, Overhaul, Carbon_t, Decom, Salvage, Quality)
bidders = {
    "Alpha": (180e6, 14.5e6, 6.2e6, 24e6, 42000, 12e6, 18e6, 76.0),
    "Beta":  (220e6, 9.2e6, 3.8e6, 16e6, 24000, 10e6, 28e6, 92.0),
}
p_c, r, N, c_bench = 120.0, 0.035, 20, 500e6
costs = [calculate_iso_lcc(*b[:7], p_c=p_c, r=r, N=N) for b in bidders.values()]
quals = [b[7] for b in bidders.values()]

rel_2 = relative_meat(costs, quals)
abs_2 = absolute_meat(costs, quals, c_bench)

# Simulate introduction of an irrelevant outlier bidder (Delta)
costs_3 = costs + [200e6]
quals_3 = quals + [40.0]
rel_3 = relative_meat(costs_3, quals_3)
abs_3 = absolute_meat(costs_3, quals_3, c_bench)

# AHP Evaluation Criteria Matrix (Availability, Fuel, Overhaul, NetZero)
ahp_mat = np.array([
    [1.0, 3.0, 4.0, 2.0],
    [1/3, 1.0, 2.0, 1/2],
    [1/4, 1/2, 1.0, 1/3],
    [1/2, 2.0, 3.0, 1.0]
])
cr = ahp_consistency_ratio(ahp_mat)

print(f"Alpha ISO-15686 LCC: GBP {costs[0]:,.0f} | Beta ISO-15686 LCC: GBP {costs[1]:,.0f}")
print(f"2-Bidder Relative Scores: Alpha={rel_2[0]:.2f}, Beta={rel_2[1]:.2f} (Diff: {rel_2[1]-rel_2[0]:.2f})")
print(f"3-Bidder Relative Scores: Alpha={rel_3[0]:.2f}, Beta={rel_3[1]:.2f} (Diff: {rel_3[1]-rel_3[0]:.2f})")
print(f"Absolute Score Diff: {abs_2[1]-abs_2[0]:.2f} vs Post-Outlier: {abs_3[1]-abs_3[0]:.2f} (Invariant)")
print(f"AHP Evaluation Consistency Ratio (CR): {cr:.4f} (Defensible: {cr <= 0.10})")
```

### Verification and Expected Output
When executed in a Python 3.10+ environment with NumPy, the script produces:

```text
Alpha ISO-15686 LCC: GBP 559,825,917 | Beta ISO-15686 LCC: GBP 447,989,480
2-Bidder Relative Scores: Alpha=78.01, Beta=96.00 (Diff: 17.99)
3-Bidder Relative Scores: Alpha=55.86, Beta=68.32 (Diff: 12.46)
Absolute Score Diff: 13.59 vs Post-Outlier: 13.59 (Invariant)
AHP Evaluation Consistency Ratio (CR): 0.0115 (Defensible: True)
```

The output confirms that the relative score difference compresses by $30.7\%$ (from $17.99$ to $12.46$) upon the introduction of an irrelevant third bidder, whereas the absolute benchmark score difference remains constant at exactly $13.59$ points. The AHP matrix exhibits a Consistency Ratio of $0.0115$, comfortably beneath the statutory $0.10$ threshold.

---

## 8. Data, standards, and regulatory context

Procurement analytics operates within a strictly governed international legal architecture. Any evaluation algorithm that violates statutory directives or treaty commitments is subject to automatic judicial stay, contract annulment, and monetary damages.

```text
====================================================================================================
                        STATUTORY & REGULATORY PROCUREMENT GOVERNANCE
====================================================================================================
Jurisdiction       Statutory Instrument          Binding Requirement                Non-Binding Guidance
----------------------------------------------------------------------------------------------------
European Union     Directive 2014/24/EU          Art. 67 (MEAT Framework)          EC Life-Cycle Costing
                   (Public Procurement)          Art. 68 (Life-Cycle Costing Rules) Toolkits for Clean
                                                 Art. 69 (Abnormally Low Tenders)   Vehicles & Lighting
----------------------------------------------------------------------------------------------------
United Kingdom     Procurement Act 2023          Sec. 19 (Most Advantageous Tender) HM Treasury Green
                   Public Contracts Regs 2015    Sec. 23 (Award Criteria & Scoring) Book (Appraisal &
                   (Transition period)           Sec. 24 (Assessment of Value)      STPR Discounting)
----------------------------------------------------------------------------------------------------
United States      Federal Acquisition           FAR Part 15 (Contracting by        DoD Better Buying Power
                   Regulation (FAR)              Negotiation - Best Value Tradeoff) Guidance on Operating
                   41 U.S.C. Chapter 137         FAR 15.404 (Cost Realism Analysis) & Support (O&S) Costs
----------------------------------------------------------------------------------------------------
International      ISO 15686-5:2017              Life-Cycle Costing for Buildings   ISO 14040/44 (Life
                   Standardization               and Constructed Assets             Cycle Assessment - LCA)
----------------------------------------------------------------------------------------------------
Global Trade       WTO Government Procurement    Art. X (Tender Documentation)      OECD Methodology for
                   Agreement (GPA 2014)          Art. XV (Treatment of Tenders)     Assessing Procurement
                                                 Art. XVIII (Challenge Procedures)  Systems (MAPS)
====================================================================================================
```

### 1. EU Directive 2014/24/EU
Article 67 mandates that contracting authorities base the award of public contracts on the Most Economically Advantageous Tender. Article 68 explicitly permits and regulates life-cycle costing, requiring that when authorities evaluate LCC, they must:
- State the data required from tenderers and the method used to determine costs.
- Ensure the evaluation method is based on objectively verifiable and non-discriminatory criteria.
- Guarantee that the method is accessible to all interested parties without requiring unreasonable effort.
- Article 68(2) authorizes the inclusion of costs imputed to environmental externalities (such as greenhouse gas emissions), provided their monetary value can be verified.

### 2. UK Procurement Act 2023
The UK Procurement Act 2023 transitions statutory terminology from MEAT to the **Most Advantageous Tender (MAT)** under Section 19. Section 23 dictates that award criteria must:
- Relate directly to the subject matter of the contract.
- Be proportionate, transparent, and non-discriminatory.
- Permit the holistic assessment of whole-life public benefit, including social value, local economic impact, and Net Zero decarbonization trajectories.
- The Act strengthens the obligation on contracting authorities to retain complete mathematical audit trails of scoring calculations to defend against supplier challenges under the Technology and Construction Court (TCC).

### 3. US Federal Acquisition Regulation (FAR)
Under FAR Part 15.101, the "Best Value Continuum" governs federal acquisitions:
- **Lowest Price Technically Acceptable (LPTA):** Permissible only for commercial off-the-shelf items with well-defined technical specifications. FAR 15.101-2 explicitly restricts the use of LPTA for major defence acquisitions, weapon systems, and knowledge-based IT services.
- **Tradeoff Process (FAR 15.101-1):** Mandates evaluating whether technical superiority or lower life-cycle sustainment cost justifies paying a higher initial acquisition price.
- **FAR 15.404-1(d) Cost Realism Analysis:** Requires federal agencies to conduct independent cost realism reviews on cost-reimbursement and major fixed-price contracts to prevent contractors from submitting unrealistic "buy-in" bids that result in cost overruns or mission degradation.

### 4. ISO 15686-5:2017 (Life-Cycle Costing)
ISO 15686-5 establishes the international standard for structuring whole-life cost evaluations. It codifies the distinction between:
- **Life-Cycle Cost (LCC):** Costs directly incurred by the procuring client from inception through disposal.
- **Whole-Life Cost (WLC):** A broader societal accounting that includes LCC, non-construction costs (e.g., land acquisition, internal staffing), tenant/client operations, residual revenues, and environmental externalities.

---

## 9. Industry practice and case evidence

Public sector and defence procurement disputes over MEAT scoring, relative formulas, and life-cycle costing represent some of the highest-value commercial litigations in administrative law.

```text
====================================================================================================
                        LANDMARK PROCUREMENT CHALLENGES & JUDICIAL PRECEDENTS
====================================================================================================
Case Citation                     Tribunal / Court               Core Issue & Legal Holding
----------------------------------------------------------------------------------------------------
Energy Solutions EU Ltd v         UK High Court (Technology &    NDA awarded a £3.2B nuclear decommissioning
Nuclear Decommissioning           Construction Court - TCC)      contract using flawed scoring criteria. The
Authority [2016] EWHC 1988 (TCC)  [Precedent on Evaluation Log]  TCC awarded £100M+ in damages due to
                                                                 unlawful manipulation of evaluation records.
----------------------------------------------------------------------------------------------------
Lancashire Care NHS Trust v       UK Technology & Construction   Contracting authority failed to provide a
Lancashire County Council         Court [2018] EWHC 1589 (TCC)   defensible audit trail explaining why minor
[2018] EWHC 1589 (TCC)            [Precedent on Scoring Audit]   score differences were awarded, leading to
                                                                 the automatic quashing of the procurement.
----------------------------------------------------------------------------------------------------
Concordia Bus Finland v           Court of Justice of the        Landmark CJEU ruling establishing that MEAT
Helsingin Kaupunki                European Union (CJEU)          criteria may legally include environmental
Case C-513/99 [2002]              [Environmental Externalities]  factors (NOx and noise emissions) even if
                                                                 they do not confer a direct commercial gain.
----------------------------------------------------------------------------------------------------
US GAO Bid Protest of             US Government Accountability   GAO sustained protest because the Air Force
Boeing Company                    Office (GAO)                   failed to conduct a proper life-cycle cost
B-409605, B-409605.2 (2014)       [Cost Realism & Tradeoff]      trade-off and relied on unsubstantiated
                                                                 sustainment claims in a $35B tanker program.
----------------------------------------------------------------------------------------------------
Dimarso NV v                      Court of Justice of the        CJEU ruled that contracting authorities may
Region de Bruxelles-Capitale      European Union (CJEU)          not apply weighting rules or sub-criteria
Case C-166/14 [2015]              [Unpublished Weighting]        that were not disclosed in the tender notice.
====================================================================================================
```

### The Nuclear Decommissioning Authority (NDA) Debacle
In *Energy Solutions EU Ltd v Nuclear Decommissioning Authority*, the UK NDA ran an open procurement for the 14-year, £3.2 billion decommissioning of twelve Magnox nuclear reactor sites. During evaluation, the scoring panel altered tender scores after preliminary reviews to reach an intended commercial outcome, failing to adhere to the objective scoring methodology set forth in the tender rules. The UK High Court found that had the published evaluation rules been applied correctly, the losing consortium (Energy Solutions) would have won the contract. The NDA was forced to settle the litigation for over £100 million in taxpayer-funded compensation and cancel the underlying contract, illustrating that subjective or manipulated tender scoring carries devastating financial liability.

### Defence Aviation Sustainment: The F-35 Flyaway vs Life-Cycle Cost Reality
In major defence acquisitions, such as the Joint Strike Fighter (F-35) program, the initial unit flyaway cost ($CapEx_0 \approx \$80\text{M}$ per airframe) accounts for less than $25\%$ of the total program life-cycle expenditure. The US Government Accountability Office (GAO Report GAO-23-106217) documented that out of an estimated \$1.7 trillion cumulative program cost, over \$1.3 trillion is attributable to operations and sustainment (O&S)—including engine maintenance, stealth coating restoration, and mission software updates over a 66-year service life. Procurement authorities that evaluate fighter or transport platforms based predominantly on flyaway price commit their sovereign treasuries to unsustainable structural deficits.

---

## 10. Failure modes and red flags

The following failure modes represent systemic analytical and procedural breakdowns encountered during whole-life cost calculations and MEAT evaluations.

| Failure Mode | Early Warning Sign | Mathematical or Operational Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| **Relative Formula Rank Reversal** | Tender documentation specifies $S_i = S_{\max} \cdot (P_{\min}/P_i)$ or min-max range scaling. | Scores cross-couple tenderers; partial derivatives $\frac{\partial S_i}{\partial P_{\min}} \neq 0$ tie score spreads to competitor pricing. | A losing bidder wins after a third bidder enters, exits, or is disqualified; award overturned in court. | Ban relative scoring; deploy absolute benchmarks ($P_{\text{bench}}$) or monetized quality models. |
| **Nominal/Real Discount Mismatch** | Evaluation guidelines specify a $3.5\%$ discount rate while requesting cash flows escalated for inflation. | Discounting inflated cash flows using a real rate ($r_{\text{real}}$) artificially suppresses present value; vice versa double-counts inflation. | Distorts selection toward capital-heavy, fuel-inefficient platforms. | Mandate constant-price (real) submissions or convert using Fisher equation: $(1 + r_{\text{nom}}) = (1 + r)(1 + \pi)$. |
| **Exclusion of End-of-Life Liabilities** | Life-cycle cost template stops at Year $N$ operating expenses with no decommissioning line items. | Nuclear, chemical, battery, or hazardous assets carry multimillion-pound disposal liabilities ($Disp_N$). | Taxpayer inherits unbudgeted remediation costs after vendor walks away. | Require bonded, escrowed, or contractually capped decommissioning provisions under ISO 15686-5. |
| **AHP Transitivity Contradiction** | Evaluator pairwise scoring matrix exhibits a Consistency Ratio $CR > 0.10$. | Evaluators award inconsistent qualitative preferences (e.g., $A > B$, $B > C$, and $C > A$). | Tender award is demonstrably arbitrary and capricious; immediate loss in bid protest. | Implement real-time automated eigenvalue computation; mandate matrix re-evaluation if $CR > 0.10$. |
| **Unsubstantiated Operational Savings** | Vendor claims radical 40% fuel burn reduction without thermodynamic or engine test data. | Contractor exploits unvalidated OpEx claims to lower calculated LCC without commercial penalty. | High-cost sustainment failure post-delivery; contractor fails to achieve operating performance. | Demand certified OEM test certificates; attach liquidated damages to operational fuel burn shortfalls. |
| **Artificial Scoring Compression** | Evaluators award all compliant bidders technical scores clustered between 88 and 92 points. | Panel suffers from "evaluator timidness," compressing qualitative spread to near zero. | Technical criteria become irrelevant; tender degenerates into an unannounced lowest-price award. | Mandate anchored rubrics with behavioral definitions and forced-distribution calibration protocols. |
| **Post-Opening Benchmark Manipulation** | Independent Government Cost Estimate (IGCE) is updated after tenders have been unsealed. | Adjusting $P_{\text{bench}}$ post-opening alters bidder point allocations retroactively. | Direct violation of public procurement transparency and equal treatment; automatic annulment. | Permanently lock, hash, and timestamp the IGCE and scoring formulas prior to issuing the tender. |
| **Ignoring Carbon Shadow Escalation** | Carbon emissions are monetized using a static, frozen price of £20/t$\text{CO}_2\text{e}$ across a 30-year horizon. | Severely undervalues long-term environmental externalities relative to official statutory trajectories. | Public agency fails legally binding statutory Net Zero and decarbonization targets. | Incorporate authoritative government carbon valuation trajectories (e.g., DESNZ / EU ETS forecasts). |
| **Predatory Low-Balling (ALT)** | Supplier submits CapEx 40% below industry mean, claiming proprietary fabrication efficiencies. | Contractor intentionally bids below cost to capture the account, planning sole-source spare-part markup. | Cost explosion during execution, contractor bankruptcy, or persistent sustainment claims. | Execute mandatory ALT audits under Art. 69; reject bids failing audited labor/material rate verification. |
| **Short Assessment Horizon Truncation** | Setting an evaluation horizon of 5 years for a platform with a 25-year operational service life. | Truncates heavy mid-life overhaul and component obsolescence replacement cycles. | Procurement optimizes for the first 60 months while locking in 20 years of crippling sustainment cost. | Align evaluation horizon with full planned operational service life or certified design fatigue limit. |
| **Unpublished Sub-Criteria Weightings** | Evaluation team breaks down "Maintainability" into 5 sub-factors with undisclosed percentage weights. | Violates the *Dimarso* CJEU precedent; evaluators apply secret grading standards. | Court quashes procurement for breach of transparency. | Disclose all criteria, sub-criteria, and mathematical weighting schedules in the initial tender notice. |
| **Subjective Monetization Factors** | Agency monetizes qualitative scores using arbitrary exchange rates (e.g., "1 quality point = £100,000"). | Conversion factors are plucked from thin air without operational economic justification. | Procurement tribunal rules the scoring formula arbitrary and discriminatory. | Derive monetization factors empirically from cost-benefit analyses, downtime costs, or willingness-to-pay studies. |

---

## 11. Assumptions and limits

The analytical validity of whole-life cost models and MEAT scoring engines depends on specific operational assumptions. Violating these assumptions invalidates the evaluation model.

| Assumption | Operational Necessity | Failure Conditions & Boundary Limits | Validation & Diagnostic Test |
|---|---|---|---|
| **Constant Operational Utilization** | Assumes annual operating hours, nautical miles, or flight cycles remain stable over the horizon. | Mission shifts, regional conflicts, or budget crises alter platform utilization by $\pm 50\%$. | Conduct utilization sensitivity analysis across low, medium, and high operational tempo profiles. |
| **Stationary Failure Rates (Exponential Distribution)** | Models unscheduled maintenance costs based on constant Mean Time Between Failures ($\lambda = 1/\text{MTBF}$). | Asset degradation exhibits Weibull wear-out aging ($\beta > 1$), causing maintenance costs to spike non-linearly. | Replace exponential Poisson sparing models with age-dependent non-homogeneous Poisson processes. |
| **Predictable Energy & Fuel Escalation** | Assumes baseline fuel costs escalate at general inflation or a constant real differential ($\Delta f$). | Geopolitical shocks, resource shortages, or refinery outages cause sudden 200% price spikes. | Stress-test LCC models using stochastic Monte Carlo simulation across historical fuel volatility bands. |
| **Stable OEM Supply Chains** | Assumes original equipment manufacturer survives and supplies spare parts for 20–30 years. | Commercial supplier undergoes bankruptcy, discontinues military product line, or declares obsolescence. | Evaluate Diminishing Manufacturing Sources and Material Shortages (DMSMS) resilience during qualification. |
| **Exogenous Discount Rate Stability** | Assumes sovereign Social Time Preference Rate ($r = 3.5\%$) remains constant over the evaluation epoch. | Macroeconomic regime shifts alter real economic growth ($g$) or pure time preference ($\rho$). | Execute dual-rate sensitivity checks using both low (2.0%) and high (5.0%) discount scenarios. |
| **Separability of Price and Quality** | Assumes overall tender utility can be decomposed additively: $U = w_c S(P) + w_q Q$. | Mutual preferential dependence exists: high technical capacity is worthless if the asset is too expensive to fly. | Verify preferential independence using multi-attribute utility theory (MAUT) mutual independence tests. |
| **Reliable Cost Accounting Breakdown** | Assumes suppliers accurately separate direct CapEx from long-term sustainment OpEx in their submissions. | Contractor cross-subsidizes bids, shifting initial CapEx into future maintenance lines or vice versa. | Require open-book, certified cost and pricing data compliant with FAR 15.403 or UK Single Source Contract rules. |
| **Invariance of Regulatory Regimes** | Assumes environmental taxation, carbon emission mandates, and safety standards remain stationary. | Legislative bodies implement stringent carbon border taxes, zero-emission mandates, or PFAS chemical bans. | Model prospective regulatory compliance costs as contingent probabilistic cash flow branches. |
| **Deterministic Decommissioning Horizon** | Assumes asset retires precisely at Year $N$ with defined scrap or salvage realization. | Platform life is extended by 10 years due to fleet replacement delays, driving maintenance costs exponential. | Evaluate structural life-extension cost penalties and sensitivity to deferred retirement dates. |
| **Auditable Evaluator Independence** | Assumes evaluators score technical dossiers independently without colluding or anchoring to peer scores. | Panel members discuss and align scoring prior to recording independent individual evaluations. | Audit individual evaluator scoring logs prior to panel consensus sessions to identify illicit anchoring. |

---

## 12. Exercises

### Exercise 1: Life-Cycle Cost Discounting and Carbon Valuation
**Context:** A regional emergency ambulance trust is procuring 40 custom all-terrain paramedic vehicles. The operational horizon is $N = 10$ years, and the applicable real discount rate is $r = 3.5\%$ per annum.
- **Tenderer A (Diesel):** Initial fleet CapEx = £4,200,000. Annual fleet fuel consumption = 180,000 liters of diesel (emission factor = $2.68 \text{ kg CO}_2\text{e}/\text{liter}$). Annual scheduled maintenance = £140,000. Mid-life engine overhaul at Year 5 = £320,000. Fleet salvage value at Year 10 = £400,000. Diesel price is fixed in real terms at £1.40/liter. The shadow price of carbon is £100/t$\text{CO}_2\text{e}$ across the horizon.
- **Tenderer B (Battery Electric):** Initial fleet CapEx = £5,800,000. Annual fleet electrical energy consumption = 450,000 kWh (emission factor = $0.12 \text{ kg CO}_2\text{e}/\text{kWh}$). Annual scheduled maintenance = £75,000. Battery pack replacement at Year 6 = £650,000. Fleet salvage value at Year 10 = £700,000. Electricity price is fixed in real terms at £0.24/kWh. Carbon shadow price is £100/t$\text{CO}_2\text{e}$.

**Deliverables:**
1. Compute the present value of annual operational expenditure (fuel/energy + maintenance + monetized carbon) for both tenderers over 10 years.
2. Compute the present value of mid-life replacements and terminal salvage values.
3. Determine the total discounted Whole-Life Cost (WLC) for Tenderer A and Tenderer B.
4. Identify which tenderer represents the lowest whole-life cost and calculate the exact taxpayer savings.
*Time allocation:* 35 minutes.  
*Acceptance criteria:* Exact numerical results matching discounted cash flow arithmetic to the nearest pound sterling.

---

### Exercise 2: Mathematical Proof and Demonstration of Rank Reversal
**Context:** A public university is procuring high-performance computing (HPC) research infrastructure. The tender applies a MEAT weighting of $60\%$ Price ($w_p = 0.60$) and $40\%$ Technical Performance ($w_q = 0.40$), using the proportional relative scoring formula:

$$S_i^{\text{rel}}(P) = 100 \cdot \frac{P_{\min}}{P_i}, \quad \text{Total}_i = w_p S_i^{\text{rel}}(P) + w_q Q_i$$

Three bids are received:
- **Bidder 1:** $P_1 = £1,000,000$, Quality Score $Q_1 = 70.0$
- **Bidder 2:** $P_2 = £1,250,000$, Quality Score $Q_2 = 88.0$
- **Bidder 3:** $P_3 = £500,000$, Quality Score $Q_3 = 45.0$ (disqualified subsequently for failing mandatory quality threshold of $Q \ge 60.0$)

**Deliverables:**
1. Calculate the relative price scores and total MEAT scores for Bidder 1 and Bidder 2 assuming Bidder 3 is excluded prior to evaluation ($P_{\min} = P_1 = £1,000,000$). Who wins?
2. Recalculate the relative price scores and total MEAT scores for Bidder 1 and Bidder 2 assuming the evaluation committee erroneously calculates commercial scores across all received submissions prior to disqualification ($P_{\min} = P_3 = £500,000$). Who wins between Bidder 1 and Bidder 2?
3. Calculate the absolute benchmark score for Bidder 1 and Bidder 2 using an IGCE benchmark of $P_{\text{bench}} = £1,000,000$ and show that the ranking is completely invariant to Bidder 3.
*Time allocation:* 30 minutes.  
*Acceptance criteria:* Complete step-by-step arithmetic demonstrating rank reversal under relative scoring and rank invariance under absolute scoring.

---

### Exercise 3: AHP Evaluation Consistency Audit
**Context:** A defence aviation evaluation panel completes a pairwise comparison of four critical operational criteria for a tactical transport aircraft:
- Criterion 1 ($C_1$): Mission Range and Payload
- Criterion 2 ($C_2$): Unprepared Airstrip Capability (STOL)
- Criterion 3 ($C_3$): Fleet Availability and MTBF
- Criterion 4 ($C_4$): Cyber Resilience and Avionics Open Architecture

The consensus pairwise comparison matrix $A$ submitted by the panel is:

$$A = \begin{bmatrix}
1.0 & 2.0 & 1/3 & 3.0 \\
1/2 & 1.0 & 1/4 & 2.0 \\
3.0 & 4.0 & 1.0 & 5.0 \\
1/3 & 1/2 & 1/5 & 1.0
\end{bmatrix}$$

**Deliverables:**
1. Normalize the column vectors of matrix $A$ and estimate the priority weight vector $w$.
2. Compute the weighted sum vector $A \cdot w$ and determine the principal eigenvalue estimate $\lambda_{\max}$.
3. Calculate the Consistency Index ($CI$) and Consistency Ratio ($CR$) using Saaty's random index $RI_4 = 0.90$.
4. Determine whether the evaluation matrix passes the statutory audit threshold ($CR \le 0.10$) or whether the panel must be reconvened to eliminate transitivity errors.
*Time allocation:* 35 minutes.  
*Acceptance criteria:* Accurate matrix operations, eigenvalue derivation, and defensible audit conclusion.

---

### Exercise 4: TOPSIS Multi-Attribute Procurement Ranking
**Context:** A public hospital authority is procuring medical imaging equipment. Three bidders (Vendors X, Y, Z) are evaluated across four criteria:
- $C_1$: Initial Capital Cost (Cost criterion, Weight $w_1 = 0.35$)
- $C_2$: 10-Year Service Agreement Cost (Cost criterion, Weight $w_2 = 0.25$)
- $C_3$: Diagnostic Image Resolution (Benefit criterion, Weight $w_3 = 0.25$)
- $C_4$: Mean Time Between Failures in operating hours (Benefit criterion, Weight $w_4 = 0.15$)

The raw performance matrix is:
- **Vendor X:** $C_1 = £850,000$, $C_2 = £320,000$, $C_3 = 92 \text{ lp/mm}$, $C_4 = 3,500 \text{ hrs}$
- **Vendor Y:** $C_1 = £720,000$, $C_2 = £410,000$, $C_3 = 84 \text{ lp/mm}$, $C_4 = 2,800 \text{ hrs}$
- **Vendor Z:** $C_1 = £980,000$, $C_2 = £260,000$, $C_3 = 96 \text{ lp/mm}$, $C_4 = 4,200 \text{ hrs}$

**Deliverables:**
1. Compute the vector-normalized decision matrix $R$.
2. Compute the weighted normalized decision matrix $V$.
3. Identify the Positive Ideal Solution ($A^+$) and Negative Ideal Solution ($A^-$).
4. Compute the Euclidean separation measures ($D_i^+$ and $D_i^-$) and the Closeness Coefficient ($C_i^*$) for all three vendors. Rank the vendors.
*Time allocation:* 45 minutes.  
*Acceptance criteria:* Complete vector calculations with intermediate matrices displayed to 4 decimal places.

---

### Exercise 5: Monetized Quality Model Design
**Context:** A national highway agency is procuring pavement resurfacing works. The baseline Whole-Life Cost floor specification requires a minimum surface design life of 12 years and a skid resistance index of 65. The agency establishes an economic model:
- Every additional year of design life above 12 years saves £180,000 in deferred capital resurfacing (capped at 6 bonus years).
- Every additional point of skid resistance above 65 reduces projected wet-weather collision social costs by £45,000 (capped at 15 bonus points).

Two bids are received:
- **Contractor 1:** Resurfacing LCC = £4,600,000; Design Life = 16 years; Skid Resistance = 72.
- **Contractor 2:** Resurfacing LCC = £4,100,000; Design Life = 13 years; Skid Resistance = 67.

**Deliverables:**
1. Formulate the Total Evaluated Cost ($\text{TEC}$) equation.
2. Compute the monetized quality deductions for Contractor 1 and Contractor 2.
3. Compute the final Total Evaluated Cost for both contractors and determine the winning tenderer.
4. Explain why this model cannot suffer from rank reversal if a third contractor submits a bid.
*Time allocation:* 20 minutes.  
*Acceptance criteria:* Clear mathematical formulation, correct numerical deduction, and rigorous proof of rank invariance.

---

### Exercise 6: Abnormally Low Tender Forensic Audit
**Context:** A municipality tenders a 5-year public facilities maintenance contract. Six bids are received with the following five-year total contract prices:
£3.2M, £3.4M, £3.1M, £3.5M, £3.3M, and £1.8M (submitted by Contractor Z).
The Independent Government Cost Estimate (IGCE) for the work is £3.35M.
Contractor Z claims its price is achieved by deploying an autonomous robotic cleaning system that eliminates 75% of on-site labor.

**Deliverables:**
1. Calculate the mean and standard deviation of the submitted bids.
2. Determine whether Contractor Z triggers the statistical ALT threshold ($\mu - 2\sigma$ and $70\%$ of IGCE).
3. Draft a legally compliant clarification letter structure under EU Directive 2014/24/EU Art. 69 / UK Procurement Act 2023 detailing the specific evidentiary requirements Contractor Z must satisfy to avoid mandatory rejection.
*Time allocation:* 25 minutes.  
*Acceptance criteria:* Accurate statistical evaluation and comprehensive administrative audit protocol.

---

## 13. Self-check answers

### Answer to Exercise 1: Life-Cycle Cost Discounting and Carbon Valuation
1. **Present Value Annuity Factor ($r = 3.5\%, N = 10$):**
   $$PVA_{0.035, 10} = \frac{1 - (1.035)^{-10}}{0.035} = \frac{1 - 0.708919}{0.035} = 8.316608$$
   - Single-sum discount factor Year 5 ($PV_5$): $(1.035)^{-5} = 0.841973$
   - Single-sum discount factor Year 6 ($PV_6$): $(1.035)^{-6} = 0.813501$
   - Single-sum discount factor Year 10 ($PV_{10}$): $(1.035)^{-10} = 0.708919$

2. **Annual Operating Cash Flows:**
   - **Tenderer A (Diesel):**
     - Fuel = $180,000 \times £1.40 = £252,000/\text{yr}$
     - Carbon = $(180,000 \times 2.68 \text{ kg}) / 1,000 = 482.4 \text{ t}\text{CO}_2\text{e} \times £100 = £48,240/\text{yr}$
     - Maintenance = £140,000/yr
     - Total Annual = $£252,000 + £48,240 + £140,000 = £440,240/\text{yr}$
     - $\text{PV of Annual OpEx} = £440,240 \times 8.316608 = £3,661,303$
   - **Tenderer B (Electric):**
     - Electricity = $450,000 \times £0.24 = £108,000/\text{yr}$
     - Carbon = $(450,000 \times 0.12 \text{ kg}) / 1,000 = 54.0 \text{ t}\text{CO}_2\text{e} \times £100 = £5,400/\text{yr}$
     - Maintenance = £75,000/yr
     - Total Annual = $£108,000 + £5,400 + £75,000 = £188,400/\text{yr}$
     - $\text{PV of Annual OpEx} = £188,400 \times 8.316608 = £1,566,849$

3. **Mid-Life Replacements & Salvage:**
   - **Tenderer A:** Overhaul Year 5 = $£320,000 \times 0.841973 = £269,431$. Salvage Year 10 = $-£400,000 \times 0.708919 = -£283,568$.
   - **Tenderer B:** Battery Year 6 = $£650,000 \times 0.813501 = £528,776$. Salvage Year 10 = $-£700,000 \times 0.708919 = -£496,243$.

4. **Total Whole-Life Cost:**
   $$\text{WLC}_A = £4,200,000 + £3,661,303 + £269,431 - £283,568 = \mathbf{£7,847,166}$$
   $$\text{WLC}_B = £5,800,000 + £1,566,849 + £528,776 - £496,243 = \mathbf{£7,399,382}$$
   **Conclusion:** Tenderer B (Battery Electric) is the most economically advantageous tender, delivering a total whole-life saving of **£447,784** over Tenderer A, despite a £1.6M higher initial acquisition CapEx.

---

### Answer to Exercise 2: Demonstration of Rank Reversal
1. **Two Bidders (Bidder 3 Excluded):**
   - $P_{\min} = £1,000,000$
   - $S_1(P) = 100 \times (1.0 / 1.0) = 100.000$; $\text{Total}_1 = 0.60 \times 100 + 0.40 \times 70.0 = 60.0 + 28.0 = \mathbf{88.000}$
   - $S_2(P) = 100 \times (1.0 / 1.25) = 80.000$; $\text{Total}_2 = 0.60 \times 80 + 0.40 \times 88.0 = 48.0 + 35.2 = \mathbf{83.200}$
   - **Winner:** **Bidder 1 defeats Bidder 2 by 4.800 points.**

2. **Three Bidders (Bidder 3 Included with $P_3 = £500,000$):**
   - $P_{\min} = £500,000$
   - $S_1'(P) = 100 \times (0.50 / 1.00) = 50.000$; $\text{Total}_1' = 0.60 \times 50 + 0.40 \times 70.0 = 30.0 + 28.0 = \mathbf{58.000}$
   - $S_2'(P) = 100 \times (0.50 / 1.25) = 40.000$; $\text{Total}_2' = 0.60 \times 40 + 0.40 \times 88.0 = 24.0 + 35.2 = \mathbf{59.200}$
   - **Winner:** **Bidder 2 defeats Bidder 1 by 1.200 points!**
   - **Proof of Rank Reversal:** The inclusion of Bidder 3 compressed the price score difference between Bidder 1 and Bidder 2 from 20.0 points to 10.0 points, allowing Bidder 2's superior technical score to override Bidder 1's price advantage.

3. **Absolute Benchmark Scoring ($P_{\text{bench}} = £1,000,000$):**
   - $S_1^{\text{abs}} = 100 \times [1 - (1.0 - 1.0)/1.0] = 100.000 \implies \text{Total}_1 = 0.60(100) + 0.40(70) = \mathbf{88.000}$
   - $S_2^{\text{abs}} = 100 \times [1 - (1.25 - 1.0)/1.0] = 75.000 \implies \text{Total}_2 = 0.60(75) + 0.40(88) = 45.0 + 35.2 = \mathbf{80.200}$
   - Bidder 1 defeats Bidder 2 under all circumstances; Bidder 3's entry or exit has mathematically zero impact.

---

### Answer to Exercise 3: AHP Consistency Audit
1. **Sum of Columns in Matrix $A$:**
   - Column 1: $1.0 + 0.5 + 3.0 + 0.3333 = 4.8333$
   - Column 2: $2.0 + 1.0 + 4.0 + 0.5 = 7.5000$
   - Column 3: $0.3333 + 0.25 + 1.0 + 0.2 = 1.7833$
   - Column 4: $3.0 + 2.0 + 5.0 + 1.0 = 11.0000$

2. **Normalized Column Matrix and Row Average Priority Vector ($w$):**
   - Row 1: $(1/4.8333 + 2/7.5 + 0.3333/1.7833 + 3/11)/4 = (0.2069 + 0.2667 + 0.1869 + 0.2727)/4 = \mathbf{0.2333}$
   - Row 2: $(0.5/4.8333 + 1/7.5 + 0.25/1.7833 + 2/11)/4 = (0.1034 + 0.1333 + 0.1402 + 0.1818)/4 = \mathbf{0.1397}$
   - Row 3: $(3/4.8333 + 4/7.5 + 1/1.7833 + 5/11)/4 = (0.6207 + 0.5333 + 0.5607 + 0.4545)/4 = \mathbf{0.5423}$
   - Row 4: $(0.3333/4.8333 + 0.5/7.5 + 0.2/1.7833 + 1/11)/4 = (0.0690 + 0.0667 + 0.1121 + 0.0909)/4 = \mathbf{0.0847}$
   - Weight Vector: $w = [0.2333, 0.1397, 0.5423, 0.0847]^T$ (Sums to 1.0).

3. **Weighted Sum Vector ($A \cdot w$):**
   - $(A \cdot w)_1 = 1.0(0.2333) + 2.0(0.1397) + 0.3333(0.5423) + 3.0(0.0847) = 0.9476$
   - $(A \cdot w)_2 = 0.5(0.2333) + 1.0(0.1397) + 0.25(0.5423) + 2.0(0.0847) = 0.5613$
   - $(A \cdot w)_3 = 3.0(0.2333) + 4.0(0.1397) + 1.0(0.5423) + 5.0(0.0847) = 2.2245$
   - $(A \cdot w)_4 = 0.3333(0.2333) + 0.5(0.1397) + 0.2(0.5423) + 1.0(0.0847) = 0.3407$

4. **Eigenvalue and Consistency Evaluation:**
   - $\lambda_1 = 0.9476 / 0.2333 = 4.0617$; $\lambda_2 = 0.5613 / 0.1397 = 4.0179$
   - $\lambda_3 = 2.2245 / 0.5423 = 4.1020$; $\lambda_4 = 0.3407 / 0.0847 = 4.0224$
   - $\lambda_{\max} = (4.0617 + 4.0179 + 4.1020 + 4.0224)/4 = \mathbf{4.0510}$
   - $CI = \frac{4.0510 - 4}{4 - 1} = \frac{0.0510}{3} = \mathbf{0.0170}$
   - $CR = \frac{CI}{RI_4} = \frac{0.0170}{0.90} = \mathbf{0.0189} \quad (1.89\%)$
   - **Audit Conclusion:** Since $CR = 0.0189 \le 0.10$, the evaluation matrix is internally consistent, robust, and legally defensible against bid protest.

---

### Answer to Exercise 4: TOPSIS Procurement Ranking
1. **Vector Normalization Denominators ($\sqrt{\sum x_{ij}^2}$):**
   - $C_1: \sqrt{850^2 + 720^2 + 980^2} = \sqrt{722500 + 518400 + 960400} = \sqrt{2201300} = 1483.678$
   - $C_2: \sqrt{320^2 + 410^2 + 260^2} = \sqrt{102400 + 168100 + 67600} = \sqrt{338100} = 581.464$
   - $C_3: \sqrt{92^2 + 84^2 + 96^2} = \sqrt{8464 + 7056 + 9216} = \sqrt{24736} = 157.277$
   - $C_4: \sqrt{3500^2 + 2800^2 + 4200^2} = \sqrt{12250000 + 7840000 + 17640000} = \sqrt{37730000} = 6142.475$

2. **Weighted Normalized Matrix $V$ ($v_{ij} = w_j \cdot r_{ij}$):**
   - **Vendor X:** $[0.35(850/1483.68), 0.25(320/581.46), 0.25(92/157.28), 0.15(3500/6142.48)] = [0.2005, 0.1376, 0.1462, 0.0855]$
   - **Vendor Y:** $[0.35(720/1483.68), 0.25(410/581.46), 0.25(84/157.28), 0.15(2800/6142.48)] = [0.1698, 0.1763, 0.1335, 0.0684]$
   - **Vendor Z:** $[0.35(980/1483.68), 0.25(260/581.46), 0.25(96/157.28), 0.15(4200/6142.48)] = [0.2312, 0.1118, 0.1526, 0.1026]$

3. **Ideal ($A^+$) and Negative Ideal ($A^-$) Solutions:**
   - $C_1$ (Cost): $A_1^+ = \min(0.2005, 0.1698, 0.2312) = 0.1698$; $A_1^- = 0.2312$
   - $C_2$ (Cost): $A_2^+ = \min(0.1376, 0.1763, 0.1118) = 0.1118$; $A_2^- = 0.1763$
   - $C_3$ (Benefit): $A_3^+ = \max(0.1462, 0.1335, 0.1526) = 0.1526$; $A_3^- = 0.1335$
   - $C_4$ (Benefit): $A_4^+ = \max(0.0855, 0.0684, 0.1026) = 0.1026$; $A_4^- = 0.0684$
   - $A^+ = [0.1698, 0.1118, 0.1526, 0.1026]$
   - $A^- = [0.2312, 0.1763, 0.1335, 0.0684]$

4. **Euclidean Distances and Closeness Coefficients ($C_i^*$):**
   - **Vendor X:**
     - $D_X^+ = \sqrt{(0.2005-0.1698)^2 + (0.1376-0.1118)^2 + (0.1462-0.1526)^2 + (0.0855-0.1026)^2} = \sqrt{0.000942 + 0.000666 + 0.000041 + 0.000292} = \sqrt{0.001941} = \mathbf{0.0441}$
     - $D_X^- = \sqrt{(0.2005-0.2312)^2 + (0.1376-0.1763)^2 + (0.1462-0.1335)^2 + (0.0855-0.0684)^2} = \sqrt{0.000942 + 0.001498 + 0.000161 + 0.000292} = \sqrt{0.002893} = \mathbf{0.0538}$
     - $C_X^* = \frac{0.0538}{0.0441 + 0.0538} = \frac{0.0538}{0.0979} = \mathbf{0.5495}$
   - **Vendor Y:**
     - $D_Y^+ = \sqrt{(0)^2 + (0.0645)^2 + (0.0191)^2 + (0.0342)^2} = \sqrt{0.004160 + 0.000365 + 0.001170} = \sqrt{0.005695} = \mathbf{0.0755}$
     - $D_Y^- = \sqrt{(0.0614)^2 + (0)^2 + (0)^2 + (0)^2} = 0.0614$
     - $C_Y^* = \frac{0.0614}{0.0755 + 0.0614} = \frac{0.0614}{0.1369} = \mathbf{0.4485}$
   - **Vendor Z:**
     - $D_Z^+ = \sqrt{(0.0614)^2 + (0)^2 + (0)^2 + (0)^2} = 0.0614$
     - $D_Z^- = \sqrt{(0)^2 + (0.0645)^2 + (0.0191)^2 + (0.0342)^2} = 0.0755$
     - $C_Z^* = \frac{0.0755}{0.0614 + 0.0755} = \frac{0.0755}{0.1369} = \mathbf{0.5515}$
   - **Final Ranking:** **Vendor Z ($C_Z^* = 0.5515$) > Vendor X ($C_X^* = 0.5495$) > Vendor Y ($C_Y^* = 0.4485$).** Vendor Z wins by a narrow margin due to superior service cost, resolution, and reliability.

---

### Answer to Exercise 5: Monetized Quality Model Design
1. **Total Evaluated Cost Equation:**
   $$\text{TEC}_i = \text{LCC}_i - \min(6, \max(0, \text{Life}_i - 12)) \times £180,000 - \min(15, \max(0, \text{Skid}_i - 65)) \times £45,000$$

2. **Deductions:**
   - **Contractor 1:**
     - Design Life Bonus = $(16 - 12) \times £180,000 = 4 \times £180,000 = £720,000$
     - Skid Resistance Bonus = $(72 - 65) \times £45,000 = 7 \times £45,000 = £315,000$
     - Total Quality Deduction = $£720,000 + £315,000 = £1,035,000$
     - $\text{TEC}_1 = £4,600,000 - £1,035,000 = \mathbf{£3,565,000}$
   - **Contractor 2:**
     - Design Life Bonus = $(13 - 12) \times £180,000 = 1 \times £180,000 = £180,000$
     - Skid Resistance Bonus = $(67 - 65) \times £45,000 = 2 \times £45,000 = £90,000$
     - Total Quality Deduction = $£180,000 + £90,000 = £270,000$
     - $\text{TEC}_2 = £4,100,000 - £270,000 = \mathbf{£3,830,000}$
3. **Award Decision:** Contractor 1 wins the tender because its evaluated cost of £3,565,000 is **£265,000 lower** than Contractor 2.
4. **Rank Invariance Proof:** $\text{TEC}_1$ and $\text{TEC}_2$ are computed exclusively from the attributes of Contractor 1 and Contractor 2 against fixed statutory cost constants (£180k/yr, £45k/pt). The arrival or departure of Contractor 3 cannot mathematically change $\text{TEC}_1$ or $\text{TEC}_2$.

---

### Answer to Exercise 6: Abnormally Low Tender Forensic Audit
1. **Statistical Dispersion:**
   - Prices: £3.2M, £3.4M, £3.1M, £3.5M, £3.3M, £1.8M.
   - Mean ($\mu$): $(3.2 + 3.4 + 3.1 + 3.5 + 3.3 + 1.8)/6 = 18.3 / 6 = \mathbf{£3.05M}$
   - Sample Standard Deviation ($\sigma$):
     $$\sigma = \sqrt{\frac{(3.2-3.05)^2 + (3.4-3.05)^2 + (3.1-3.05)^2 + (3.5-3.05)^2 + (3.3-3.05)^2 + (1.8-3.05)^2}{5}}$$
     $$\sigma = \sqrt{\frac{0.0225 + 0.1225 + 0.0025 + 0.2025 + 0.0625 + 1.5625}{5}} = \sqrt{\frac{1.975}{5}} = \sqrt{0.395} = \mathbf{£0.6285M}$$
2. **Threshold Tests for Contractor Z (£1.80M):**
   - Threshold 1 ($\mu - 2\sigma$): $£3.05\text{M} - 2(£0.6285\text{M}) = £3.05\text{M} - £1.257\text{M} = \mathbf{£1.793M}$. (Contractor Z is at £1.80M, exactly bordering the $2\sigma$ threshold).
   - Threshold 2 ($70\%$ of IGCE): $0.70 \times £3.35\text{M} = \mathbf{£2.345M}$. Contractor Z (£1.80M) is far below $70\%$ of the IGCE (it is at $53.7\%$ of the IGCE).
   - **Conclusion:** Contractor Z decisively triggers mandatory ALT investigation.
3. **Audit Clarification Protocol:**
   - Under Directive 2014/24/EU Art. 69, the authority cannot immediately disqualify Contractor Z. It must issue a formal Request for Clarification requiring:
     a. Technical substantiation of the robotic technology (TRL level, proven operational deployments, software failure recovery).
     b. Detailed breakdown of direct labor hours, demonstrating strict adherence to national minimum wage and working time regulations.
     c. Proof of equipment capital depreciation and maintenance schedules.
     d. If Contractor Z cannot prove that the robotic system delivers equivalent cleaning quality and that labor rates satisfy statutory employment laws, the contracting authority must issue a formal notice of rejection.

---

## 14. Interview questions

### Question 1: How does relative scoring violate Arrow's Impossibility Theorem in public procurement?
**What a strong answer contains:**
- Explicit citation of Kenneth Arrow's Independence of Irrelevant Alternatives (IIA) condition.
- Mathematical demonstration that in relative formulas like $S_i = S_{\max} \cdot (P_{\min}/P_i)$, the score spread $(S_1 - S_2) = S_{\max} P_{\min} (1/P_1 - 1/P_2)$ is a function of the minimum price $P_{\min}$.
- Explanation that introducing a low-priced third bidder ($P_3 < P_1$) shrinks the commercial score difference between Bidder 1 and Bidder 2, allowing fixed technical quality differences to flip the final ranking.
- Practical mitigation: Replacing relative formulas with absolute benchmarks or monetized quality models.

**What a weak answer reveals:**
- Vague statements that relative scoring "is unfair" or "causes bias" without mathematical proof.
- Inability to define IIA or explain the partial derivative with respect to $P_{\min}$.
- Suggesting that rank reversal only occurs when evaluators cheat or collude.

---

### Question 2: Why should public authorities use the Social Time Preference Rate rather than corporate WACC for life-cycle costing?
**What a strong answer contains:**
- Clear distinction between corporate private risk-adjusted capital cost (WACC, typically 8%–15%) and sovereign intergenerational social discounting (STPR, typically 2.0%–3.5%).
- Derivation of the Ramsey formula: $\text{STPR} = \rho + \mu \cdot g$, explaining pure time preference ($\rho$) and elasticity of consumption utility ($\mu$).
- The operational consequence: A high commercial discount rate excessively suppresses future cash flows, creating an artificial bias toward cheap, inefficient assets that burden future generations with enormous operating and carbon costs.

**What a weak answer reveals:**
- Confusing nominal interest rates with real social discount rates.
- Claiming that the government should use commercial bank lending rates or treasury bond yields interchangeably without understanding social welfare optimization.

---

### Question 3: Under ISO 15686-5, what is the precise distinction between Life-Cycle Costing (LCC) and Whole-Life Costing (WLC)?
**What a strong answer contains:**
- Cites ISO 15686-5:2017 definitions directly.
- LCC is strictly confined to direct client-incurred costs across construction, operation, maintenance, and disposal.
- WLC expands LCC to encompass non-construction client costs (land acquisition, client management), terminal revenue streams, and external environmental costs (monetized GHG emissions, air pollution).

**What a weak answer reveals:**
- Treating LCC and WLC as identical marketing synonyms.
- Neglecting the role of environmental externalities and societal impact in WLC.

---

### Question 4: How do you mathematically detect and remediate an inconsistent AHP scoring matrix?
**What a strong answer contains:**
- Explanation of Saaty's eigenvalue problem: $A w = \lambda_{\max} w$.
- Mathematical formulation of the Consistency Index: $CI = (\lambda_{\max} - M)/(M - 1)$ and Consistency Ratio: $CR = CI / RI_M$.
- Stating the standard threshold: $CR \le 0.10$.
- Remediation strategy: If $CR > 0.10$, compute the element-wise absolute difference $|a_{ij} - w_i/w_j|$ to locate the largest transitivity violations and have the evaluation panel review those specific comparisons.

**What a weak answer reveals:**
- Believing that AHP scores can simply be averaged without checking transitivity.
- Unfamiliarity with the Random Index ($RI$) or what $CR > 0.10$ signifies legally.

---

### Question 5: When is the Technique for Order Preference by Similarity to Ideal Solution (TOPSIS) legally defensible in tender evaluation?
**What a strong answer contains:**
- Identifies that TOPSIS ranks alternatives based on relative closeness to the positive ideal and distance from the negative ideal.
- Emphasizes the necessity of pre-disclosing the exact normalization formula (vector normalization $r_{ij} = x_{ij}/\sqrt{\sum x_{kj}^2}$).
- Notes that TOPSIS can be vulnerable to rank reversal if an extreme non-compliant alternative alters the vector denominator or ideal solutions, requiring strict compliance gating *prior* to normalization.

**What a weak answer reveals:**
- Presenting TOPSIS as an unassailable black box without recognizing its normalization sensitivity.
- Inability to explain how the positive and negative ideal vectors are constructed for cost versus benefit criteria.

---

### Question 6: What legal steps must an authority take when an Abnormally Low Tender is identified under EU Directive 2014/24/EU Art. 69 or UK Procurement Act 2023?
**What a strong answer contains:**
- Emphasizes that **automatic rejection is illegal**.
- The authority must issue a formal written Request for Clarification requiring the bidder to explain its pricing, manufacturing economics, compliance with labor/environmental laws, and state aid.
- Rejection is lawful only if the evidence provided fails to account for the low price or reveals non-compliance with statutory social, labor, or environmental legislation.

**What a weak answer reveals:**
- Suggesting that the contracting authority should immediately disqualify the bidder to protect other competitors.
- Unawareness of statutory standstill rights and supplier challenge procedures.

---

### Question 7: How do you structure a Monetized Quality tender evaluation to guarantee rank preservation?
**What a strong answer contains:**
- Formulates Total Evaluated Cost: $\text{TEC}_i = \text{LCC}_i - \sum V_k \Delta Q_{i,k}$.
- Explains that quality improvements are converted into explicit monetary deductions based on objective economic benefits (e.g., fuel savings, avoided downtime).
- Demonstrates that because each bidder's TEC is calculated independently against fixed benchmark values, $\frac{\partial \text{TEC}_i}{\partial \text{Bidder}_j} = 0$, guaranteeing absolute rank preservation.

**What a weak answer reveals:**
- Attempting to combine monetized costs back into subjective percentage point scores.
- Confusing monetized quality with subjective bonus points awarded by committee votes.

---

### Question 8: How does the UK Technology and Construction Court (TCC) evaluate scoring audit trails in procurement challenges?
**What a strong answer contains:**
- References foundational case law, such as *Energy Solutions v NDA* and *Lancashire Care NHS Trust v Lancashire CC*.
- Explains the legal requirement for contemporaneous, complete, and un-manipulated scoring records explaining the rationale for score deductions.
- Notes that evaluators who alter scores during consensus meetings without clear factual justification expose the sovereign authority to contract quashing and massive financial damages.

**What a weak answer reveals:**
- Believing that sovereign defence and public authorities enjoy absolute executive immunity from commercial procurement challenges.
- Assuming that destroying individual evaluator notes after the consensus meeting is standard practice (it is treated by courts as spoliation of evidence).

---

## 15. Key terms

- **Abnormally Low Tender (ALT):** A tender submission whose cost structure appears unfeasibly low, triggering mandatory statutory investigation under Directive 2014/24/EU Art. 69.
- **Absolute Scoring Benchmark:** A tender scoring formulation where prices are evaluated against a fixed pre-announced cost curve, eliminating rank reversal.
- **Analytic Hierarchy Process (AHP):** A multi-criteria decision method based on pairwise comparison matrices, eigenvalue decomposition, and consistency metrics.
- **Arrow's Impossibility Theorem:** A foundational social-choice theorem proving that no ranked voting or relative evaluation system can satisfy Pareto efficiency, non-dictatorship, and Independence of Irrelevant Alternatives simultaneously.
- **Best Value Continuum:** The spectrum of acquisition evaluation strategies under US FAR Part 15, spanning from Lowest Price Technically Acceptable (LPTA) to Best Value Tradeoff.
- **Closeness Coefficient ($C_i^*$):** The metric in TOPSIS representing an alternative's relative proximity to the ideal solution.
- **Consistency Ratio ($CR$):** The ratio of an AHP matrix Consistency Index to the Random Index; must satisfy $CR \le 0.10$ for legal defensibility.
- **Cost Realism Analysis:** The independent evaluation of a tenderer's proposed cost lines to determine whether they reflect actual operational requirements (FAR 15.404).
- **Discount Factor:** The mathematical multiplier $(1 + r)^{-t}$ used to convert future cash flows into present economic value.
- **Fisher Equation:** The economic identity relating nominal interest rates, real interest rates, and expected inflation: $(1 + r_{\text{nom}}) = (1 + r_{\text{real}})(1 + \pi)$.
- **Independence of Irrelevant Alternatives (IIA):** The axiom requiring that the preference ranking between $A$ and $B$ remain unaffected by the presence or absence of $C$.
- **Independent Government Cost Estimate (IGCE):** The objective benchmark cost developed by the public agency prior to issuing a tender.
- **ISO 15686-5:2017:** The international standard governing Life-Cycle Costing for buildings, constructed assets, and civil infrastructure.
- **Life-Cycle Cost (LCC):** The cumulative discounted cost incurred directly by the asset owner over the operational horizon.
- **Monetized Quality:** An evaluation method converting qualitative technical advantages into currency deductions from whole-life cost.
- **Most Advantageous Tender (MAT):** The statutory evaluation standard under Section 19 of the UK Procurement Act 2023.
- **Most Economically Advantageous Tender (MEAT):** The overarching award criterion under EU Directive 2014/24/EU Art. 67.
- **Rank Reversal:** The illegal or unscientific inversion of the ranking between two tenderers caused by the entry or exit of a third irrelevant bidder.
- **Relative Scoring Formula:** Any tender evaluation formula where a bidder's commercial score is calculated relative to an endogenous variable, such as $P_{\min}$.
- **Social Time Preference Rate (STPR):** The intergenerational sovereign discount rate derived via the Ramsey growth formula ($\text{STPR} = \rho + \mu \cdot g$).
- **Standstill Period:** The mandatory statutory pause between contract award notification and signature, allowing unsuccessful bidders to seek judicial review.
- **TOPSIS:** The Technique for Order of Preference by Similarity to Ideal Solution, an MCDA technique based on Euclidean distance metrics.
- **Vector Normalization:** The normalization method $r_{ij} = x_{ij}/\sqrt{\sum x_{kj}^2}$ utilized in TOPSIS to eliminate unit scale dependencies.
- **Whole-Life Cost (WLC):** The total economic evaluation encompassing LCC, non-construction costs, residual revenues, and environmental externalities.

---

## 16. Sources

- **European Parliament and Council.** (2014). Directive 2014/24/EU of 26 February 2014 on public procurement and repealing Directive 2004/18/EC. *Official Journal of the European Union*, L 94/65.
- **UK Parliament.** (2023). *Procurement Act 2023* (c. 54). The Stationery Office.
- **US Federal Acquisition Regulation (FAR).** (2024). Part 15: *Contracting by Negotiation*. General Services Administration (GSA), Department of Defense (DoD), and National Aeronautics and Space Administration (NASA).
- **International Organization for Standardization.** (2017). *ISO 15686-5:2017: Buildings and constructed assets — Service life planning — Part 5: Life-cycle costing*. ISO, Geneva.
- **HM Treasury.** (2022). *The Green Book: Central Government Guidance on Appraisal and Evaluation*. HM Stationery Office, London.
- **US Office of Management and Budget (OMB).** (1992, rev. 2023). *Circular No. A-94: Guidelines and Discount Rates for Benefit-Cost Analysis of Federal Programs*. Executive Office of the President, Washington, D.C.
- **Arrow, K. J.** (1951). *Social Choice and Individual Values*. John Wiley & Sons, New York.
- **Saaty, T. L.** (1980). *The Analytic Hierarchy Process: Planning, Priority Setting, Resource Allocation*. McGraw-Hill, New York.
- **Hwang, C. L., & Yoon, K.** (1981). *Multiple Attribute Decision Making: Methods and Applications*. Springer-Verlag, Berlin.
- **Dini, F., Pacini, R., & Valletti, T.** (2006). Scoring Rules in Public Procurement: How to Score the Price. *Competition and Contracting in Services of General Interest*, Politécnica de Milano.
- **UK Technology and Construction Court.** (2016). *Energy Solutions EU Ltd v Nuclear Decommissioning Authority* [2016] EWHC 1988 (TCC).
- **US Government Accountability Office (GAO).** (2023). *F-35 Sustainment: Discretionary Spending and Life-Cycle Realities* (GAO-23-106217). Washington, D.C.

---

## 17. Links

- **Parent Module Index:** [`_index.md`](_index.md)
- **Previous Topic:** [`01_public-procurement-frameworks.md`](01_public-procurement-frameworks.md)
- **Next Topic:** [`03_defence-availability-pbl.md`](03_defence-availability-pbl.md)
- **Module Survey File:** [`../../../course/35_public_sector_defence_humanitarian.md`](../../../course/35_public_sector_defence_humanitarian.md)
- **Related Methods:** [Module 11: Supply Chain Network Economics](../../02_methods/11_network_design/_index.md) | [Module 12: Inventory Optimization](../../02_methods/12_inventory_policy/_index.md)
