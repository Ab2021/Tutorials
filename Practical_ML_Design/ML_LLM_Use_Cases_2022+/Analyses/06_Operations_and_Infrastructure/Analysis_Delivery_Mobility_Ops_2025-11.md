# Delivery & Mobility Industry Analysis: MLOps & Infrastructure (2023-2025)

**Analysis Date**: November 2025  
**Category**: 05_MLOps_and_Infrastructure  
**Industry**: Delivery & Mobility  
**Articles Analyzed**: 13+ (Uber, DoorDash, Lyft, Swiggy, Careem)  
**Period Covered**: 2023-2025  
**Research Method**: Web search synthesis + Folder Content

---

## PART 1: USE CASE OVERVIEW

### 1.1 Basic Information

**Category**: MLOps & Infrastructure  
**Industry**: Delivery & Mobility  
**Companies**: Uber, DoorDash, Lyft, Swiggy, Careem  
**Years**: 2024-2025 (Primary focus)  
**Tags**: Feature Stores, Real-time Serving, Orchestration, Anomaly Detection

**Use Cases Analyzed**:
1.  **Uber**: Michelangelo 2.0 & Palette Feature Store (2024)
2.  **DoorDash**: Real-time Feature Store (Flink + Redis) (2024)
3.  **Lyft**: Flyte (Kubernetes-native Orchestration) (2024)
4.  **Swiggy**: GenAI Codegen & Fraud Detection (2024)
5.  **Careem**: Fraud Anomaly Detection (2024)

### 1.2 Problem Statement

**What business problem are they solving?**

1.  **Latency Sensitivity**: "ETA prediction" must happen in milliseconds. If the feature store is slow, the app lags.
2.  **Data Freshness**: "Is the driver *currently* stuck in traffic?" Batch features from yesterday are useless. DoorDash needs real-time stream processing.
3.  **Scale**: Uber serves 10 million predictions *per second*. A standard Flask app would melt.
4.  **Reproducibility**: Lyft has hundreds of data scientists. If "Model A" works on a laptop but fails in production, it's a waste. Flyte ensures consistent environments.

**What makes this problem ML-worthy?**

-   **Throughput**: Handling exabytes of data (Uber DataK9) requires specialized distributed systems.
-   **Complexity**: Managing thousands of features across hundreds of models requires a **Feature Store** to prevent "feature drift" and "training-serving skew".

---

## PART 2: SYSTEM DESIGN DEEP DIVE

### 2.1 High-Level Architecture (The "Real-Time" Stack)

Delivery MLOps is defined by **Streaming-First** architectures.

```mermaid
graph TD
    A[Event Stream (Kafka)] --> B[Stream Processor (Flink)]
    
    subgraph "Feature Engineering (DoorDash)"
    B --> C[Real-time Features]
    C --> D[Online Store (Redis)]
    C --> E[Offline Store (Iceberg/S3)]
    end
    
    subgraph "Orchestration (Lyft)"
    F[Data Scientist Code] --> G[Flyte Spec]
    G --> H[Kubernetes Pods]
    end
    
    subgraph "Serving (Uber)"
    D --> I[Prediction Service]
    I --> J[Michelangelo Model]
    J --> K[User App]
    end
```

### 2.2 Detailed Architecture: Uber Michelangelo & Palette (2024)

Uber set the standard for the **Centralized ML Platform**.

**The Components**:
-   **Palette (Feature Store)**:
    -   **Scale**: Hosts 20,000+ features.
    -   **Dual-Write**: Writes features to **Cassandra** (for low-latency online serving) and **Hive** (for offline training) simultaneously to ensure consistency.
    -   **Metadata**: A centralized repository tells data scientists *who* owns a feature and *how* it's calculated.
-   **Serving**:
    -   **Throughput**: 10M QPS (Queries Per Second).
    -   **Shadow Modeling**: New models run in "shadow mode" (predicting but not acting) to verify performance before promotion.

### 2.3 Detailed Architecture: DoorDash Feature Store (2024)

DoorDash optimized for **Speed and Freshness**.

**The Pipeline**:
-   **Frameworks**: **Riviera** and **Fabricator** (Declarative Flink wrappers).
-   **Mechanism**: Data scientists write a high-level config (YAML/Python). The framework compiles this into a Flink job.
-   **Storage**:
    -   **Hot Storage**: **Redis** (sharded and compressed) for <10ms retrieval.
    -   **Cold Storage**: **Snowflake/S3** for historical training data.
-   **Benefit**: Decouples "Feature Logic" from "Infrastructure Plumbing".

### 2.4 Detailed Architecture: Lyft Flyte (2024)

Lyft solved the **"It works on my machine"** problem.

**The Platform**:
-   **Concept**: **Workflow-as-Code**.
-   **Architecture**: Built on **Kubernetes**. Every task (data prep, training, evaluation) runs in its own isolated container.
-   **Caching**: If Task A (Data Prep) takes 5 hours and succeeds, but Task B (Training) fails, re-running the workflow *skips* Task A and uses the cached output.
-   **Type Safety**: Flyte enforces strong typing between tasks (e.g., Task A *must* output a Pandas DataFrame), catching errors early.

---

## PART 3: MLOPS & INFRASTRUCTURE

### 3.1 Training & Serving

**Swiggy**:
-   **GenAI Ops**: Uses **DevNet** with Variational Loss for fraud detection. The infrastructure supports rapid iteration of Generative AI models for code generation, helping developers ship features faster.

**Uber (DataK9)**:
-   **Data Categorization**: An ML system that scans **Exabytes** of data to tag PII (Personally Identifiable Information).
-   **Scale**: Runs on massive Spark clusters to ensure GDPR/CCPA compliance across the entire data lake.

### 3.2 Evaluation Metrics

| Metric | Purpose | Company |
| :--- | :--- | :--- |
| **P99 Latency** | Serving speed (target <10ms) | DoorDash |
| **Feature Freshness** | Time from event to availability | Uber |
| **Workflow Success Rate** | Reliability of pipelines | Lyft |
| **Fraud Catch Rate** | Accuracy of risk models | Careem |

---

## PART 4: KEY ARCHITECTURAL PATTERNS

### 4.1 The "Lambda Architecture" for Features
**Used by**: Uber, DoorDash.
-   **Concept**: Process data twice. Once via **Stream** (Speed) and once via **Batch** (Accuracy/Correction).
-   **Why**: Streaming is fast but can be lossy. Batch is slow but accurate. You need both for a robust Feature Store.

### 4.2 Declarative ML Pipelines
**Used by**: Lyft (Flyte), DoorDash (Fabricator).
-   **Concept**: Define *what* you want (inputs, outputs, resources), not *how* to run it.
-   **Why**: Allows the platform team to swap out the backend (e.g., move from AWS to GCP) without breaking user code.

### 4.3 Shadow Serving
**Used by**: Uber.
-   **Concept**: Run the new model alongside the old one on live traffic. Log the new model's predictions but return the old model's result to the user.
-   **Why**: Safe testing. If the new model crashes or predicts "Free Rides for Everyone", no users are affected.

---

## PART 5: LESSONS LEARNED

### 5.1 "Features are the API" (Uber)
-   The contract between Data Engineering and Data Science is the **Feature**.
-   **Lesson**: Invest heavily in the **Feature Store** metadata. If you don't know what "driver_score_v3" means, you can't use it.

### 5.2 "Abstraction Saves Sanity" (DoorDash)
-   Asking Data Scientists to write raw Flink Java code is a recipe for failure.
-   **Lesson**: Build **DS-friendly wrappers** (Python/YAML) around complex infra tools.

### 5.3 "Isolation is Key" (Lyft)
-   One bad memory leak shouldn't crash the whole cluster.
-   **Lesson**: **Containerize everything**. Flyte's pod-per-task model ensures robust isolation.

---

## PART 6: QUANTITATIVE METRICS

| Metric | Result | Company | Context |
| :--- | :--- | :--- | :--- |
| **Throughput** | 10M QPS | Uber | Michelangelo Serving |
| **Feature Count** | 20,000+ | Uber | Palette Store |
| **Latency** | <10ms | DoorDash | Redis Feature Fetch |
| **Data Volume** | Exabytes | Uber | DataK9 Scanning |

---

## PART 7: REFERENCES

**Uber (3)**:
1.  Michelangelo 2.0 & Palette (2024)
2.  DataK9 PII Scanning (2024)
3.  uVitals Anomaly Detection (2023)

**DoorDash (1)**:
1.  Real-time Feature Store with Flink/Redis (2024)

**Lyft (1)**:
1.  Flyte Orchestration Platform (2024)

**Swiggy (1)**:
1.  GenAI & Fraud Detection (2024)

---

**Analysis Completed**: November 2025  
**Total Companies**: 5 (Uber, DoorDash, Lyft, Swiggy, Careem)  
**Use Cases Covered**: Feature Stores, Real-time Serving, Orchestration, Anomaly Detection  
**Status**: Comprehensive Analysis Complete
