# Advanced Data Engineering & Pipelines (Part 1) - Modern Data Stack - Theoretical Deep Dive

## Overview
"Data Scientists spend 80% of their time cleaning data."
The **Modern Data Stack (MDS)** aims to fix this. By moving from ETL to **ELT** (Extract, Load, Transform) and using tools like **Snowflake** and **dbt**, we bring Software Engineering best practices to Data.

---

## 1. Conceptual Foundation

### 1.1 ETL vs. ELT

*   **ETL (Old Way):** Extract -> Transform (in Python/Informatica) -> Load to Warehouse.
    *   *Problem:* Compute is limited. If you need a new column, you must re-run the whole pipeline.
*   **ELT (New Way):** Extract -> Load (Raw) -> Transform (in Warehouse).
    *   *Benefit:* Storage is cheap (S3/Blob). Compute is scalable (Snowflake). We keep the Raw data forever.

### 1.2 The Medallion Architecture

1.  **Bronze (Raw):** Exact copy of source systems (JSON dumps, CSVs). "Warts and all".
2.  **Silver (Clean):** Deduplicated, typed, filtered. (e.g., `claim_date` is now a Date, not a String).
3.  **Gold (Business):** Aggregated metrics. (e.g., `monthly_loss_ratio`).

---

## 2. Mathematical Framework

### 2.1 Idempotency

*   **Definition:** $f(x) = f(f(x))$. Running the pipeline twice should not duplicate data.
*   **Implementation:**
    *   `DELETE FROM target WHERE date = '2023-01-01'`
    *   `INSERT INTO target SELECT ... WHERE date = '2023-01-01'`
*   *Crucial for Insurance:* You cannot double-count a \$1M claim.

### 2.2 Slowly Changing Dimensions (SCD)

*   **Type 1:** Overwrite. (We lose history).
*   **Type 2:** Add a new row with `valid_from` and `valid_to`.
    *   *Scenario:* Policyholder moves from NY to FL.
    *   *Row 1:* NY, valid 2020-2022.
    *   *Row 2:* FL, valid 2022-Present.

---

## 3. Theoretical Properties

### 3.1 ACID in the Lakehouse

*   **Data Lake:** Cheap storage, but no transactions. (If a write fails, you get corrupt files).
*   **Data Warehouse:** ACID transactions, but expensive.
*   **Lakehouse (Delta Lake / Iceberg):** Brings ACID to the Lake.
    *   *Log-based:* A `_delta_log` folder tracks every transaction.

---

## 4. Modeling Artifacts & Implementation

### 4.1 dbt (Data Build Tool)

*   **Philosophy:** "Everything is a SELECT statement."
*   **DAG (Directed Acyclic Graph):** dbt automatically figures out the order of execution.

```sql
-- models/silver/claims_clean.sql

{{ config(materialized='incremental', unique_key='claim_id') }}

SELECT
    claim_id,
    CAST(incurred_date AS DATE) as incurred_date,
    CAST(paid_amount AS FLOAT) as paid_amount
FROM {{ source('raw', 'claims_bronze') }}
WHERE incurred_date > (SELECT max(incurred_date) FROM {{ this }})
```

### 4.2 Airflow (Orchestration)

*   **DAGs in Python:** Define dependencies as code.

```python
from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG('insurance_pipeline', schedule_interval='@daily') as dag:
    extract = BashOperator(task_id='extract', bash_command='python extract_claims.py')
    dbt_run = BashOperator(task_id='dbt_run', bash_command='dbt run')
    
    extract >> dbt_run
```

---

## 5. Evaluation & Validation

### 5.1 Data Quality Tests (Great Expectations)

*   **Assertion:** `paid_amount` must be >= 0.
*   **Assertion:** `policy_id` must not be NULL.
*   **dbt Tests:**
    ```yaml
    columns:
      - name: paid_amount
        tests:
          - not_null
          - dbt_utils.expression_is_true:
              expression: ">= 0"
    ```

### 5.2 Data Lineage

*   **Graph:** Source -> Bronze -> Silver -> Gold -> Tableau Dashboard.
*   *Utility:* If "Source System A" breaks, we know exactly which Dashboards are affected.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Business Logic in Bronze**
    *   "Let's filter out cancelled policies during ingestion."
    *   *Risk:* If the business definition of "Cancelled" changes, you have to re-ingest everything.
    *   *Fix:* Bronze must be Raw. Filter in Silver.

2.  **Trap: The "Spaghetti" DAG**
    *   Everything depends on everything.
    *   *Fix:* Strict layering. Gold should not depend on Bronze.

### 6.2 Implementation Challenges

1.  **Schema Evolution:**
    *   Source system adds a column `driver_rating`.
    *   *Delta Lake:* `mergeSchema=true` handles this automatically.

---

## 7. Advanced Topics & Extensions

### 7.1 Streaming Ingestion

*   **Kappa Architecture:** Treat everything as a stream.
*   **Snowpipe:** Auto-ingest files into Snowflake as soon as they land in S3.

### 7.2 Data Mesh

*   **Decentralization:** Instead of one central Data Team, each domain (Claims, Underwriting, Finance) owns their own data products.
*   **Federated Governance:** Global standards, local ownership.

---

## 8. Regulatory & Governance Considerations

### 8.1 GDPR & The Right to be Forgotten

*   **Challenge:** If a customer asks to be deleted, you must find them in Bronze, Silver, Gold, and Backups.
*   **Solution:** "Tokenization". Store PII in a secure vault. Delete the token, and the data becomes anonymous everywhere.

---

## 9. Practical Example

### 9.1 Worked Example: The "Loss Ratio" Pipeline

**Scenario:**
*   **Source:** Claims DB (Postgres) and Policy DB (Oracle).
*   **Bronze:** Replicate tables to Snowflake `RAW_SCHEMA`.
*   **Silver:**
    *   Join Claims and Policy on `policy_id`.
    *   Calculate `earned_premium` (Time-based).
*   **Gold:**
    *   Group by `product_line` and `month`.
    *   Calculate `Loss Ratio = Incurred Claims / Earned Premium`.
*   **Result:** A dashboard that updates every morning.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **ELT** leverages the power of the Cloud Warehouse.
2.  **dbt** brings version control to SQL.
3.  **Medallion Architecture** ensures data quality.

### 10.2 When to Use This Knowledge
*   **Migration:** Moving from On-Prem (SAS/Mainframe) to Cloud.
*   **Scalability:** When Excel crashes.

### 10.3 Critical Success Factors
1.  **Testing:** If you don't test your data, it is wrong.
2.  **Documentation:** dbt generates docs automatically. Use them.

### 10.4 Further Reading
*   **Reis & Housley:** "Fundamentals of Data Engineering".

---

## Appendix

### A. Glossary
*   **DAG:** Directed Acyclic Graph.
*   **Idempotent:** Safe to run multiple times.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **SCD Type 2** | `valid_from`, `valid_to` | History Tracking |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
