# Day 148: Garbage In, Garbage Out: Data Quality Frameworks
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 22: Data & Model Quality

---

> **🎯 Focus Area:** Your model failed in production. Why? Because the upstream Data Engineering team changed the column name from `user_id` to `userId`, or because `age` suddenly contains `-1`. **Great Expectations (GX)** prevents this data sewage from entering your clean pipeline.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Define** an Expectation Suite using Great Expectations (GX).
2.  **Validate** a Pandas DataFrame against the suite.
3.  **Generate** Data Docs (HTML reports) to visualize the schema and failures.
4.  **Integrate** validation into an Airflow/Prefect/Kubeflow pipeline task.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `pip install great_expectations pandas`.

---

## 📖 Theoretical Foundation

### 1. The Quality Gate
Data enters the ML system at multiple points:
1.  **Training:** Historical data dump.
2.  **Inference:** Live request JSON.
3.  **Batch:** Nightly CSV from Data Warehouse.

At *each* point, we must assert:
*   **Schema:** Do columns exist? Are types correct?
*   **Completeness:** Are there Nulls?
*   **Distributions:** Is absolute value of Z-Score < 3? (Outlier detection).

### 2. Frameworks
*   **Great Expectations (GX):** Heavyweight. Good for Data Engineering. Generates HTML docs.
*   **Pandera:** Lightweight. Decorator-based validation for Pandas. Good for runtime checks.
*   **Pydantic:** Good for single-row JSON validation (FastAPI).

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Great Expectations

We will define a suite for a "Census Dataset".

#### 📁 `src/01_gx_setup.py`
```python
import great_expectations as gx
import pandas as pd

# 1. Setup Context
context = gx.get_context()

# 2. Create Validator from Data
df = pd.DataFrame({
    "age": [30, 42, 12, 99],
    "income": [50000, 80000, 0, 120000],
    "country": ["US", "US", "CA", "US"]
})

datasource_name = "census_data"
data_asset_name = "raw_df"
validator = context.sources.pandas_default.read_dataframe(df)

# 3. Define Expectations (Interactive Mode)
print("Defining Expectations...")

# A. Age must be reasonable (0-120)
validator.expect_column_values_to_be_between(
    column="age", min_value=0, max_value=120
)

# B. Income cannot be negative
validator.expect_column_values_to_be_between(
    column="income", min_value=0
)

# C. Country must be in set
validator.expect_column_values_to_be_in_set(
    column="country", value_set=["US", "CA", "MX"]
)

# D. No Nulls in Age
validator.expect_column_values_to_not_be_null("age")

# 4. Save the Suite
validator.save_expectation_suite(discard_failed_expectations=False)
checkpoint = context.add_or_update_checkpoint(
    name="census_checkpoint",
    validator=validator,
)
```

### 👨‍💻 Core Implementation: Validation in Pipeline

#### 📁 `src/02_validate.py`
```python
import great_expectations as gx
import sys

def validate_data(df):
    context = gx.get_context()
    
    # Load existing suite
    # In V3 API, we usually use Checkpoints
    results = context.run_checkpoint(
        checkpoint_name="census_checkpoint",
        batch_request={
            "datasource_name": "census_data",
            "data_asset_name": "raw_df",
            "dataframe": df
        }
    )
    
    if not results["success"]:
        print("Data Quality Check Failed!")
        # print details
        sys.exit(1)
        
    print("Data Passed Gates.")

# Test with Bad Data
bad_df = pd.DataFrame({"age": [-5], "income": [100], "country": ["XX"]})
# validate_data(bad_df) # Should exit 1
```

### 👨‍💻 Alternative: Pandera (Lightweight)

Better for Inference services where GX is too slow.

```python
import pandera as pa
from pandera.typing import DataFrame, Series

class CensusSchema(pa.SchemaModel):
    age: Series[int] = pa.Field(ge=0, le=120)
    income: Series[int] = pa.Field(ge=0)
    country: Series[str] = pa.Field(isin=["US", "CA"])

@pa.check_types
def process_data(df: DataFrame[CensusSchema]):
    return df.groupby("country").mean()

# Usage
# process_data(bad_df) # Raises pa.errors.SchemaError
```

---

## 🔬 Lab Exercise: "Data Docs"

### Task
Visualize the failure.
1.  Run the GX setup script.
2.  Run the Checkpoint with invalid data.
3.  GX automatically builds HTML reports in `gx/uncommitted/data_docs`.
4.  Open `index.html`.
5.  **Observation:** You see a red bar. "Column `country`: Unexpected value 'XX'. Found 100% unexpected."
6.  **Value:** Send this link to the Data Engineering team instead of a log snippet.

---

## 📖 Advanced Theory: Semantic Checks
Simple checks (Type, Range) catch typos.
**Semantic Checks** catch logic errors.
*   `expect_column_kl_divergence_to_be_less_than(threshold)`
*   `expect_column_pair_values_a_to_be_greater_than_b("end_date", "start_date")`

---

## 📝 Daily Summary

### Key Takeaways
1.  **Shift Left:** Validate data as early as possible (Ingestion). Don't wait until the Model crashes 4 hours later.
2.  **Schema Evolution:** If valid data changes (e.g., new Country added), the Expectation Suite must be updated. This should be a PR process "Update Data Contracts".
3.  **Overhead:** Validation scans the dataset. For 1TB data, run validation on a random sample (1%), or use Spark/Ray backend for GX.

### API Summary
```python
validator.expect_column_values_to_be_between()
```

---

**Day 148 Complete** ✅

*Next: Day 149 - Model Quality Metrics - Beyond Accuracy.*
