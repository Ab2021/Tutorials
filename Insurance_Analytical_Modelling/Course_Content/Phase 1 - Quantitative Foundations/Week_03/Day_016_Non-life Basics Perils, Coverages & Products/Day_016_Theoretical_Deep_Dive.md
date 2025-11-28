# Non-Life Basics: Perils, Coverages & Products - Theoretical Deep Dive

## Overview
This session introduces non-life (property & casualty) insurance, covering key concepts like perils, coverages, and major product lines. We explore auto insurance, homeowners, commercial liability, and workers' compensation. These concepts are fundamental for understanding P&C insurance pricing, reserving, and risk management for CAS exams and industry practice.

---

## 1. Conceptual Foundation

### 1.1 Definition & Core Concept

**Non-Life Insurance (Property & Casualty):** Insurance that provides financial protection against losses and damages to property, liability for injuries to others, and various other risks (excluding life insurance).

**Key Terminology:**
- **Peril:** An event that causes loss (fire, theft, windstorm)
- **Coverage:** Protection provided by the policy against specific risks
- **Named Perils:** Policy lists specific covered events
- **Open Perils (All-Risk):** Covers all events except those specifically excluded
- **Indemnity:** Compensation for actual loss (not profit)
- **Liability:** Legal responsibility for damages to others

**Major Product Lines:**
1. **Auto Insurance:** Covers vehicles and liability
2. **Homeowners:** Protects homes and personal property
3. **Commercial Liability:** Business liability protection
4. **Workers' Compensation:** Employee injury coverage

### 1.2 Historical Context & Evolution

**Origin:**
- **1600s:** Marine insurance (Lloyd's of London)
- **1700s:** Fire insurance after Great Fire of London
- **1800s:** Liability insurance emerged
- **Early 1900s:** Auto insurance, workers' comp

**Evolution:**
- **Pre-1900s:** Simple property coverage
- **1900-1950:** Standardization of policies
- **1950-2000:** Expansion of liability coverage
- **Present:** Cyber insurance, usage-based auto, parametric products

**Current State:**
- **Traditional Products:** Auto, homeowners, commercial
- **Emerging Risks:** Cyber, climate change, pandemic
- **Technology:** Telematics, IoT, AI for underwriting

### 1.3 Why This Matters

**Business Impact:**
- **Market Size:** P&C is larger than life insurance globally
- **Frequency:** More claims than life insurance
- **Complexity:** Many product lines, each with unique characteristics
- **Regulation:** State-by-state in US; varies globally

**Regulatory Relevance:**
- **Rate Filings:** Must justify rates in most jurisdictions
- **Reserves:** Must hold adequate reserves for claims
- **Solvency:** RBC requirements
- **Consumer Protection:** Mandatory coverages (auto liability, workers' comp)

**Industry Adoption:**
- **Personal Lines:** Auto, homeowners (mass market)
- **Commercial Lines:** Liability, property (businesses)
- **Specialty:** Cyber, D&O, E&O, surety

---

## 2. Mathematical Framework

### 2.1 Core Assumptions

1. **Assumption: Indemnity Principle**
   - **Description:** Insured cannot profit from loss
   - **Implication:** Payment limited to actual loss
   - **Real-world validity:** Valid for property; not for liability (no upper limit on damages)

2. **Assumption: Independent Losses**
   - **Description:** One policyholder's loss doesn't affect others
   - **Implication:** Can use law of large numbers
   - **Real-world validity:** Violated for catastrophes (hurricane, earthquake)

3. **Assumption: Known Exposure**
   - **Description:** Can measure exposure (e.g., car-years, house-years)
   - **Implication:** Can calculate rates per unit of exposure
   - **Real-world validity:** Valid for most products

4. **Assumption: Stationary Risk**
   - **Description:** Risk doesn't change over time
   - **Implication:** Historical data predicts future
   - **Real-world validity:** Violated; inflation, climate change, technology affect risk

### 2.2 Mathematical Notation

| Symbol | Meaning | Example |
|--------|---------|---------|
| $N$ | Number of claims | 100 claims |
| $X_i$ | Severity of claim $i$ | $5,000 |
| $S$ | Aggregate loss $= \sum X_i$ | $500,000 |
| $\lambda$ | Claim frequency (per exposure) | 0.05 claims/car-year |
| $\mu$ | Average severity | $5,000/claim |
| $E$ | Exposure (e.g., car-years) | 10,000 car-years |
| $P$ | Premium | $600/year |
| $LR$ | Loss ratio $= \text{Losses}/\text{Premiums}$ | 0.65 |

### 2.3 Core Equations & Derivations

#### Equation 1: Expected Aggregate Loss
$$
E[S] = E[N] \times E[X]
$$

**Where:**
- $E[N]$ = expected number of claims
- $E[X]$ = expected severity per claim

**Example:**
- Expected claims: 500 (from 10,000 exposures at 5% frequency)
- Average severity: $5,000

$$
E[S] = 500 \times 5,000 = \$2,500,000
$$

#### Equation 2: Pure Premium (Loss Cost)
$$
\text{Pure Premium} = \frac{E[S]}{E} = \lambda \times \mu
$$

**Where:**
- $E$ = exposure (e.g., car-years)
- $\lambda$ = frequency
- $\mu$ = severity

**Example:**
$$
\text{Pure Premium} = \frac{2,500,000}{10,000} = \$250 \text{ per car-year}
$$

Or:
$$
\text{Pure Premium} = 0.05 \times 5,000 = \$250
$$

#### Equation 3: Gross Premium
$$
\text{Gross Premium} = \frac{\text{Pure Premium} + \text{Fixed Expenses}}{1 - \text{Variable Expense Ratio} - \text{Profit Loading}}
$$

**Example:**
- Pure premium: $250
- Fixed expenses per policy: $50
- Variable expense ratio: 15%
- Profit loading: 5%

$$
\text{Gross Premium} = \frac{250 + 50}{1 - 0.15 - 0.05} = \frac{300}{0.80} = \$375
$$

#### Equation 4: Loss Ratio
$$
LR = \frac{\text{Incurred Losses}}{\text{Earned Premiums}}
$$

**Example:**
- Incurred losses: $2,600,000
- Earned premiums: $4,000,000

$$
LR = \frac{2,600,000}{4,000,000} = 0.65 \text{ or } 65\%
$$

#### Equation 5: Combined Ratio
$$
\text{Combined Ratio} = LR + \text{Expense Ratio}
$$

**Example:**
- Loss ratio: 65%
- Expense ratio: 30%

$$
\text{Combined Ratio} = 65\% + 30\% = 95\%
$$

**Interpretation:** Combined ratio < 100% indicates underwriting profit.

#### Equation 6: Variance of Aggregate Loss (Compound Poisson)
$$
Var[S] = E[N] \times E[X^2]
$$

**Under independence:**
$$
Var[S] = \lambda \times E \times (Var[X] + \mu^2)
$$

**Example:**
- $\lambda = 0.05$
- $E = 10,000$
- $\mu = 5,000$
- $Var[X] = 10,000,000$ (CV = 0.63)

$$
Var[S] = 0.05 \times 10,000 \times (10,000,000 + 25,000,000) = 500 \times 35,000,000 = 17.5B
$$
$$
SD[S] = \sqrt{17.5B} = \$132,288
$$

### 2.4 Special Cases & Variants

**Case 1: Deductible**
Insured pays first $D$ of each claim:
$$
E[\text{Payment per claim}] = E[\max(X - D, 0)]
$$

**Case 2: Policy Limit**
Maximum payment per claim is $L$:
$$
E[\text{Payment per claim}] = E[\min(X, L)]
$$

**Case 3: Coinsurance**
Insurer pays fraction $c$ of loss:
$$
E[\text{Payment per claim}] = c \times E[X]
$$

---

## 3. Theoretical Properties

### 3.1 Key Properties

1. **Property: Frequency-Severity Decomposition**
   - **Statement:** $E[S] = E[N] \times E[X]$
   - **Proof:** Law of total expectation
   - **Practical Implication:** Can model frequency and severity separately

2. **Property: Indemnity Principle**
   - **Statement:** Payment ≤ actual loss
   - **Proof:** Contract terms
   - **Practical Implication:** Prevents moral hazard (profiting from loss)

3. **Property: Law of Large Numbers**
   - **Statement:** As $E \to \infty$, actual loss ratio → expected loss ratio
   - **Proof:** Central limit theorem
   - **Practical Implication:** Larger portfolios have more predictable results

4. **Property: Combined Ratio < 100% for Profit**
   - **Statement:** Underwriting profit when $LR + ER < 1$
   - **Proof:** By definition
   - **Practical Implication:** Can have underwriting loss but overall profit (investment income)

### 3.2 Strengths
✓ **Diversification:** Many independent risks
✓ **Frequency:** More data than life insurance
✓ **Flexibility:** Can adjust rates annually
✓ **Innovation:** New products for emerging risks
✓ **Regulation:** Consumer protection

### 3.3 Limitations
✗ **Catastrophes:** Correlated losses (hurricane, earthquake)
✗ **Long Tail:** Liability claims take years to settle
✗ **Inflation:** Medical, legal, repair costs increase
✗ **Uncertainty:** Difficult to predict future losses

### 3.4 Comparison of Product Lines

| Product | Frequency | Severity | Tail | Typical LR |
|---------|-----------|----------|------|------------|
| **Auto Physical Damage** | High | Low | Short | 60-70% |
| **Auto Liability** | Medium | Medium-High | Long | 65-75% |
| **Homeowners** | Low | Medium | Medium | 55-65% |
| **Commercial Liability** | Low | High | Very Long | 60-70% |
| **Workers' Comp** | Medium | Medium-High | Long | 65-75% |

---

## 4. Modeling Artifacts & Implementation

### 4.1 Data Requirements

**For P&C Insurance:**
- **Exposure Data:** Number of policies, car-years, payroll
- **Claims Data:** Claim counts, amounts, dates
- **Policy Data:** Coverages, limits, deductibles
- **External Data:** Weather, economic indicators

**Data Quality Considerations:**
- **Completeness:** All claims reported
- **Accuracy:** Claim amounts correct
- **Timeliness:** Recent data for trends
- **Granularity:** Detailed attributes for segmentation

### 4.2 Preprocessing Steps

**Step 1: Clean Data**
```
- Remove duplicates
- Handle missing values
- Identify outliers
```

**Step 2: Calculate Exposures**
```
- Car-years = (policies * days in force) / 365
- Payroll for workers' comp
- Square footage for property
```

**Step 3: Aggregate Claims**
```
- Group by policy, coverage, year
- Calculate frequency and severity
```

### 4.3 Model Specification

**Python Implementation:**

```python
import numpy as np
import pandas as pd

def pure_premium(frequency, severity):
    """Calculate pure premium"""
    return frequency * severity

def gross_premium(pure_prem, fixed_exp, var_exp_ratio, profit_loading):
    """Calculate gross premium"""
    return (pure_prem + fixed_exp) / (1 - var_exp_ratio - profit_loading)

def loss_ratio(incurred_losses, earned_premiums):
    """Calculate loss ratio"""
    return incurred_losses / earned_premiums

def combined_ratio(loss_ratio, expense_ratio):
    """Calculate combined ratio"""
    return loss_ratio + expense_ratio

# Example: Auto Insurance Portfolio
exposures = 10000  # car-years
frequency = 0.05  # claims per car-year
avg_severity = 5000  # average claim amount

# Calculate pure premium
pure_prem = pure_premium(frequency, avg_severity)
print(f"Pure Premium: ${pure_prem:.2f}")

# Calculate gross premium
fixed_exp = 50  # per policy
var_exp_ratio = 0.15
profit_loading = 0.05

gross_prem = gross_premium(pure_prem, fixed_exp, var_exp_ratio, profit_loading)
print(f"Gross Premium: ${gross_prem:.2f}")

# Simulate claims
np.random.seed(42)
n_claims = np.random.poisson(frequency * exposures)
claim_amounts = np.random.gamma(shape=4, scale=severity/4, size=n_claims)

total_losses = claim_amounts.sum()
total_premiums = gross_prem * exposures

print(f"\nSimulation Results:")
print(f"Number of Claims: {n_claims}")
print(f"Total Losses: ${total_losses:,.2f}")
print(f"Total Premiums: ${total_premiums:,.2f}")

# Calculate ratios
lr = loss_ratio(total_losses, total_premiums)
er = var_exp_ratio + fixed_exp / gross_prem
cr = combined_ratio(lr, er)

print(f"\nRatios:")
print(f"Loss Ratio: {lr:.2%}")
print(f"Expense Ratio: {er:.2%}")
print(f"Combined Ratio: {cr:.2%}")

if cr < 1.0:
    print(f"Underwriting Profit: {(1-cr)*100:.1f}%")
else:
    print(f"Underwriting Loss: {(cr-1)*100:.1f}%")
```

### 4.4 Model Outputs & Interpretation

**Primary Outputs:**
1. **Pure Premium:** Expected loss cost per exposure
2. **Gross Premium:** Amount charged to customer
3. **Loss Ratio:** Losses / Premiums
4. **Combined Ratio:** (Losses + Expenses) / Premiums

**Example Output (Auto Insurance):**
- Pure Premium: $250
- Gross Premium: $375
- Loss Ratio: 67%
- Combined Ratio: 97%

**Interpretation:**
- **Pure Premium:** Expected claims cost
- **Gross Premium:** Includes expenses and profit
- **Combined Ratio < 100%:** Underwriting profit of 3%

---

## 5. Evaluation & Validation

### 5.1 Model Diagnostics

**Consistency Checks:**
- **Pure Premium:** Should align with historical loss ratio
- **Gross Premium:** Should be competitive with market
- **Combined Ratio:** Should be < 100% for target profit

**Reasonableness:**
- Frequency: 0-1 (typically 0.01-0.20 for auto)
- Severity: > 0 (typically $1K-$10K for auto)
- Loss Ratio: 50-80% typical

### 5.2 Performance Metrics

**For P&C Insurance:**
- **Loss Ratio:** Target 60-70%
- **Combined Ratio:** Target 95-100%
- **ROE:** Target 10-15% (including investment income)

### 5.3 Validation Techniques

**Benchmarking:**
- Compare to industry averages (A.M. Best, NAIC)
- Compare to competitors

**Backtesting:**
- Use historical data to predict
- Compare predicted to actual

**Sensitivity Analysis:**
- Vary frequency by ±10%
- Vary severity by ±20%
- Measure impact on premium

### 5.4 Sensitivity Analysis

| Parameter | Base | +10% | -10% | Impact on Premium |
|-----------|------|------|------|-------------------|
| Frequency | 0.05 | 0.055 | 0.045 | +10% / -10% |
| Severity | $5,000 | $5,500 | $4,500 | +10% / -10% |
| Expenses | 15% | 16.5% | 13.5% | +2% / -2% |

**Interpretation:** Premium is most sensitive to frequency and severity.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1. **Trap: Confusing Perils and Coverages**
   - **Why it's tricky:** Related but different concepts
   - **How to avoid:** Peril = event; coverage = protection provided
   - **Example:** Fire is a peril; dwelling coverage protects against fire

2. **Trap: Thinking Low Combined Ratio is Always Good**
   - **Why it's tricky:** Could mean underpricing (future losses)
   - **How to avoid:** Consider reserve adequacy
   - **Example:** CR = 85% looks good, but reserves may be deficient

3. **Trap: Ignoring Tail Risk**
   - **Why it's tricky:** Rare but severe events (catastrophes)
   - **How to avoid:** Model catastrophes separately
   - **Example:** Hurricane causes $1B loss (100x normal annual losses)

### 6.2 Implementation Challenges

1. **Challenge: Data Quality**
   - **Symptom:** Missing or incorrect claim data
   - **Diagnosis:** Poor data collection processes
   - **Solution:** Implement data quality checks

2. **Challenge: Exposure Measurement**
   - **Symptom:** Difficult to define exposure unit
   - **Diagnosis:** Complex products
   - **Solution:** Use appropriate exposure base (e.g., payroll for workers' comp)

3. **Challenge: Long-Tail Lines**
   - **Symptom:** Claims take years to settle
   - **Diagnosis:** Liability claims (lawsuits)
   - **Solution:** Use development triangles for reserving

### 6.3 Interpretation Errors

1. **Error: Thinking Indemnity Applies to Liability**
   - **Wrong:** "Liability claims limited to actual loss"
   - **Right:** "Liability can exceed policy limits (excess judgments)"

2. **Error: Ignoring Investment Income**
   - **Wrong:** "Combined ratio > 100% means loss"
   - **Right:** "Can have overall profit with investment income"

### 6.4 Edge Cases

**Edge Case 1: Catastrophe**
- **Problem:** Single event causes many claims
- **Workaround:** Reinsurance, catastrophe models

**Edge Case 2: Claim Inflation**
- **Problem:** Severity increases faster than expected
- **Diagnosis:** Medical inflation, social inflation
- **Solution:** Trend factors in pricing

**Edge Case 3: Adverse Selection**
- **Problem:** High-risk insureds buy more coverage
- **Diagnosis:** Information asymmetry
- **Solution:** Underwriting, risk classification

---

## 7. Advanced Topics & Extensions

### 7.1 Modern Variants

**Extension 1: Usage-Based Insurance (UBI)**
- Telematics for auto insurance
- Premium based on actual driving behavior

**Extension 2: Parametric Insurance**
- Pays based on index (e.g., wind speed)
- No loss adjustment needed

**Extension 3: Cyber Insurance**
- Emerging product for data breaches
- Difficult to price (limited data)

### 7.2 Integration with Other Methods

**Combination 1: Pricing + Reserving**
- Use same frequency/severity assumptions
- Ensure consistency

**Combination 2: Underwriting + Pricing**
- Risk classification affects pricing
- Segmentation for homogeneous groups

### 7.3 Cutting-Edge Research

**Topic 1: Climate Change**
- Increasing frequency/severity of weather events
- Need for dynamic pricing

**Topic 2: AI for Underwriting**
- Image recognition for property inspection
- NLP for claims processing

---

## 8. Regulatory & Governance Considerations

### 8.1 Regulatory Perspective

**Acceptability:**
- **Widely Accepted:** Yes, P&C insurance is heavily regulated
- **Jurisdictions:** State-by-state in US; varies globally
- **Documentation Required:** Rate filings, reserve opinions

**Key Regulatory Concerns:**
1. **Concern: Rate Adequacy**
   - **Issue:** Rates must be adequate but not excessive
   - **Mitigation:** Actuarial rate filings

2. **Concern: Discriminatory Pricing**
   - **Issue:** Cannot unfairly discriminate
   - **Mitigation:** Use approved rating factors

### 8.2 Model Governance

**Model Risk Rating:** Medium-High
- **Justification:** Pricing affects profitability and competitiveness

**Validation Frequency:** Annual (or when assumptions change)

**Key Validation Tests:**
1. **Benchmarking:** Compare to industry
2. **Backtesting:** Predicted vs. actual
3. **Sensitivity:** Test key assumptions

### 8.3 Documentation Requirements

**Minimum Documentation:**
- ✓ Data sources and quality
- ✓ Methodology (frequency-severity, GLM, etc.)
- ✓ Assumptions (trends, catastrophes)
- ✓ Results (pure premium, gross premium)
- ✓ Sensitivity analysis

---

## 9. Practical Example

### 9.1 Worked Example: Auto Insurance Pricing

**Scenario:** Price auto insurance for a portfolio of 10,000 cars. Use:
- Historical frequency: 0.05 claims/car-year
- Historical severity: $5,000/claim
- Fixed expenses: $50/policy
- Variable expenses: 15% of premium
- Profit loading: 5%

**Step 1: Calculate Pure Premium**
$$
\text{Pure Premium} = 0.05 \times 5,000 = \$250
$$

**Step 2: Calculate Gross Premium**
$$
\text{Gross Premium} = \frac{250 + 50}{1 - 0.15 - 0.05} = \frac{300}{0.80} = \$375
$$

**Step 3: Project Losses and Premiums**
- Expected claims: $0.05 \times 10,000 = 500$
- Expected losses: $500 \times 5,000 = \$2,500,000$
- Total premiums: $375 \times 10,000 = \$3,750,000$

**Step 4: Calculate Ratios**
- Loss Ratio: $2,500,000 / 3,750,000 = 66.7\%$
- Expense Ratio: $15\% + 50/375 = 15\% + 13.3\% = 28.3\%$
- Combined Ratio: $66.7\% + 28.3\% = 95\%$

**Step 5: Calculate Profit**
- Underwriting Profit: $(1 - 0.95) \times 3,750,000 = \$187,500$
- Profit Margin: $187,500 / 3,750,000 = 5\%$

**Interpretation:** The pricing achieves the target 5% profit margin with a 95% combined ratio.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1. **Perils:** Events causing loss (fire, theft, collision)
2. **Coverages:** Protection provided (liability, collision, comprehensive)
3. **Pure Premium:** Expected loss cost = frequency × severity
4. **Combined Ratio:** (Losses + Expenses) / Premiums
5. **Major Products:** Auto, homeowners, commercial liability, workers' comp

### 10.2 When to Use This Knowledge
**Ideal For:**
- ✓ CAS exam preparation
- ✓ P&C insurance pricing
- ✓ Product development
- ✓ Underwriting
- ✓ Claims management

**Not Ideal For:**
- ✗ Life insurance (different framework)
- ✗ Health insurance (different regulations)

### 10.3 Critical Success Factors
1. **Understand Products:** Know coverages and perils
2. **Master Frequency-Severity:** Decompose losses
3. **Calculate Ratios:** Loss ratio, combined ratio
4. **Apply to Pricing:** Pure premium → gross premium
5. **Consider Regulation:** Rate filings, consumer protection

### 10.4 Further Reading
- **Textbook:** "Foundations of Casualty Actuarial Science" (CAS)
- **Exam Prep:** CAS Exam 5 materials
- **Industry:** Insurance Information Institute (III)
- **Regulation:** NAIC model laws

---

## Appendix

### A. Glossary
- **Peril:** Event causing loss
- **Coverage:** Protection provided
- **Named Perils:** Specific events listed as covered
- **Open Perils:** All events covered except exclusions
- **Indemnity:** Compensation for actual loss
- **Liability:** Legal responsibility for damages
- **Combined Ratio:** (Losses + Expenses) / Premiums

### B. Key Formulas Summary

| Formula | Equation | Use |
|---------|----------|-----|
| **Pure Premium** | $\lambda \times \mu$ | Expected loss cost |
| **Gross Premium** | $(PP + FE) / (1 - VE - PL)$ | Amount charged |
| **Loss Ratio** | Losses / Premiums | Profitability |
| **Combined Ratio** | LR + ER | Underwriting profit |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 1,050+*
