# Data Architecture for Pricing & Reserving - Theoretical Deep Dive

## Overview
Actuarial models are only as good as the data feeding them. In the old days, actuaries manually copied data from Excel to Excel. Today, we build **Data Pipelines (ETL)** that transform raw **Transaction Logs** into **Actuarial Triangles**. This session covers the architecture of a modern Actuarial Data Warehouse (ADW).

---

## 1. Conceptual Foundation

### 1.1 The Source of Truth: Transaction Logs

*   **Policy Admin System (PAS):** Records every policy change (New Business, Endorsement, Cancellation).
*   **Claims System:** Records every financial movement (Reserve Change, Payment, Recovery).
*   **The "Log":** A table with billions of rows.
    *   `ClaimID | Date | TransactionType | Amount | User`
    *   `123 | 2023-01-01 | ReserveSet | 10000 | Bob`
    *   `123 | 2023-02-01 | PartialPay | 2000 | Alice`

### 1.2 The Destination: Triangles

*   **Triangle:** An aggregation of the log.
*   **Cell (AY 2020, Age 12):** Sum of all transactions for accidents in 2020, occurring within 12 months of the accident date.
*   **The Gap:** Converting the Log to the Triangle is the hardest part of actuarial data engineering.

### 1.3 Data Quality Dimensions

1.  **Completeness:** Are all claims in the warehouse? (Reconciliation).
2.  **Accuracy:** Is the Paid Amount correct?
3.  **Consistency:** Does "Cause of Loss = Fire" mean the same thing in 2010 and 2020?
4.  **Timeliness:** Is the data available by Workday 1?

---

## 2. Mathematical Framework

### 2.1 Triangle Generation Algorithm

Let $T$ be the set of transactions. Each transaction $k$ has:
*   $A_k$: Accident Date.
*   $R_k$: Report Date.
*   $D_k$: Transaction Date.
*   $V_k$: Value (Amount).

**Incremental Paid Triangle:**
$$ C_{i, j} = \sum_{k \in T} V_k \cdot \mathbb{I}(\text{Year}(A_k)=i) \cdot \mathbb{I}(\text{Lag}(A_k, D_k)=j) \cdot \mathbb{I}(\text{Type}_k=\text{Payment}) $$

**Incurred Triangle:**
*   Requires "Snapshotting".
*   Incurred = Paid + Case Reserve.
*   Case Reserve at time $t$ = Sum of all "ReserveSet" transactions up to time $t$.

### 2.2 Reconciliation Logic

$$ \text{General Ledger (GL) Paid} = \sum \text{Claims System Paid} = \sum \text{Actuarial Triangle Paid} $$
*   **The "Penny Check":** If the Actuarial Paid doesn't match the Finance Paid, the reserves are wrong.
*   **Timing Differences:** GL books on "Process Date". Actuaries book on "Transaction Date".

---

## 3. Theoretical Properties

### 3.1 The "As-Of" Date Problem

*   **Issue:** A claim is paid on Jan 31. The system processes it on Feb 1.
*   **Financial Reporting:** It's a Feb expense.
*   **Actuarial Analysis:** It happened in Jan.
*   **Solution:** Store both dates. `TransDate` (Actuarial) and `BookDate` (Finance).

### 3.2 Slowly Changing Dimensions (SCD)

*   **Scenario:** A claim is "Open" in Jan. "Closed" in Feb. "Reopened" in Mar.
*   **Type 2 Dimension:** We need a history of the status.
    *   `ClaimID | Status | ValidFrom | ValidTo`
    *   `123 | Open | Jan 1 | Jan 31`
    *   `123 | Closed | Feb 1 | Feb 28`

---

## 4. Modeling Artifacts & Implementation

### 4.1 SQL for Triangle Generation

```sql
-- 1. Transaction Table (Raw)
CREATE TABLE Claims_Transactions (
    ClaimID INT,
    AccidentDate DATE,
    TransDate DATE,
    TransType VARCHAR(20), -- 'Payment', 'ReserveChange'
    Amount DECIMAL(18, 2)
);

-- 2. Aggregate to Incremental Triangle Format
SELECT
    YEAR(AccidentDate) AS AccidentYear,
    DATEDIFF(month, AccidentDate, TransDate) / 12 AS DevYear,
    SUM(Amount) AS IncrementalPaid
FROM
    Claims_Transactions
WHERE
    TransType = 'Payment'
GROUP BY
    YEAR(AccidentDate),
    DATEDIFF(month, AccidentDate, TransDate) / 12
ORDER BY
    1, 2;

-- 3. Cumulative Triangle (Window Function)
-- (Requires wrapping the above in a CTE)
SELECT
    AccidentYear,
    DevYear,
    SUM(IncrementalPaid) OVER (PARTITION BY AccidentYear ORDER BY DevYear) AS CumulativePaid
FROM
    Incremental_CTE;
```

### 4.2 Python ETL Pipeline (Pandas)

```python
import pandas as pd
import numpy as np

# Mock Transaction Log
data = pd.DataFrame({
    'ClaimID': [1, 1, 2],
    'AccidentDate': pd.to_datetime(['2020-01-01', '2020-01-01', '2021-06-01']),
    'TransDate': pd.to_datetime(['2020-02-01', '2021-02-01', '2021-07-01']),
    'Amount': [1000, 500, 2000]
})

# 1. Calculate Lag (Development Period)
data['AccidentYear'] = data['AccidentDate'].dt.year
data['DevLag'] = ((data['TransDate'] - data['AccidentDate']) / np.timedelta64(1, 'Y')).astype(int)

# 2. Pivot to Triangle
triangle = data.pivot_table(
    index='AccidentYear',
    columns='DevLag',
    values='Amount',
    aggfunc='sum'
).fillna(0)

# 3. Cumulate
cumulative_triangle = triangle.cumsum(axis=1)

print("Incremental:")
print(triangle)
print("\nCumulative:")
print(cumulative_triangle)
```

---

## 5. Evaluation & Validation

### 5.1 Automated Data Quality Checks (DQ)

*   **Check 1:** `Paid > Incurred`? (Impossible, unless Case Reserve < 0).
*   **Check 2:** `Paid < 0`? (Possible for Salvage/Subrogation, but check magnitude).
*   **Check 3:** `AccidentDate > ReportDate`? (Impossible).

### 5.2 The "Control Total" Dashboard

*   A daily report showing:
    *   Yesterday's Total Paid.
    *   Today's New Payments.
    *   Today's Total Paid.
    *   **Variance:** Must be 0.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Replacing History**
    *   **Issue:** The Claims System fixes a typo in 2015.
    *   **Result:** The 2015 diagonal changes in the 2023 triangle.
    *   **Rule:** Actuarial data should be **Immutable**. If you fix history, you must issue a "reversing transaction" in the current period, or strictly version control the snapshots.

2.  **Trap: Currency Conversion**
    *   **Issue:** Claims in EUR, GBP, JPY.
    *   **Rule:** Store Original Currency. Convert to Reporting Currency using the exchange rate **at the Transaction Date** (for Paid) or **at the Valuation Date** (for Reserves).

### 6.2 Implementation Challenges

1.  **Big Data Volume:**
    *   Telematics data (driving behavior) is TBs per day.
    *   You can't put it in the SQL transaction log.
    *   **Solution:** Data Lake (Parquet/Avro) for the raw sensor data. Aggregate features (e.g., "Miles Driven") to the Warehouse.

---

## 7. Advanced Topics & Extensions

### 7.1 The "One-Click" Reserving System

*   **Goal:** Pipeline runs at night. Reserving software (Arius/ResQ) picks up the data automatically. Models run. Dashboard updates.
*   **Reality:** Actuaries spend 2 weeks cleaning data because the pipeline broke.

### 7.2 IFRS 17 Granularity

*   IFRS 17 requires data at the "Group of Contracts" level (Cohort).
*   **Challenge:** Old systems don't track "Cohorts".
*   **Fix:** You need a "Grouping Engine" in the ETL layer to tag every claim with its IFRS 17 Cohort ID.

---

## 8. Regulatory & Governance Considerations

### 8.1 GDPR / PII

*   Claims data contains names, medical info, addresses.
*   **Rule:** Anonymize or Pseudonymize data in the Actuarial Warehouse. Actuaries don't need to know the claimant's name, just their age and injury.

---

## 9. Practical Example

### 9.1 Worked Example: The "Ghost" Claim

**Scenario:**
*   Actuary sees a \$1M spike in 2020 AY.
*   Drills down to Claim #999.
*   Claims System says Claim #999 is closed with \$0 pay.
*   **Cause:** The ETL logic treated a "Reserve = Null" as "Reserve = \$1M" due to a default value error.
*   **Lesson:** `Null` handling is the root of all evil.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **ETL** transforms Logs to Triangles.
2.  **Reconciliation** is mandatory.
3.  **Data Quality** is an engineering problem, not an actuarial one.

### 10.2 When to Use This Knowledge
*   **System Migration:** Moving from Legacy Mainframe to Guidewire.
*   **Automation:** Removing manual Excel work.

### 10.3 Critical Success Factors
1.  **Partner with IT:** Don't build "Shadow IT" databases in Access.
2.  **Version Control:** Code your SQL/Python.
3.  **Document the Mapping:** What does "Status Code 4" mean?

### 10.4 Further Reading
*   **Kimball:** "The Data Warehouse Toolkit" (The Bible of Dimensional Modeling).
*   **CAS:** "Actuarial Data Science" Working Party papers.

---

## Appendix

### A. Glossary
*   **ETL:** Extract, Transform, Load.
*   **SCD:** Slowly Changing Dimension.
*   **Data Lake:** Storage for raw, unstructured data.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Incremental** | $\sum V_k$ | Triangle Cell |
| **Recon** | $GL - Actuarial = 0$ | Quality Check |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
