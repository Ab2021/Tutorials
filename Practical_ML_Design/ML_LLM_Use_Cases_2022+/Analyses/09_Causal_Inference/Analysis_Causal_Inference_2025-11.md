# ML Use Case Analysis: Causal Inference in Tech (Comprehensive)

**Analysis Date**: November 2025  
**Category**: Causal Inference  
**Industry**: Multi-Industry (Media, Delivery, Ride-Sharing, Social Platforms)  
**Articles Analyzed**: 6 (Full Directory Coverage)

---

## PART 1: USE CASE OVERVIEW

### 1.1 Basic Information

**Category**: Causal Inference  
**Industries**: 
1.  **Media & Streaming** (Netflix)
2.  **Delivery & Mobility** (Gojek, Lyft, Uber)
3.  **Social Platforms** (LinkedIn)

**Use Cases Analyzed**:
-   **Media**: Content Valuation (Netflix).
-   **Delivery**: Voucher Allocation (Gojek), Marketing Measurement (Lyft), Pricing (Uber).
-   **Social**: Observational Causal Inference "Ocelot" (LinkedIn).

### 1.2 Problem Statement

**What business problem are they solving?**

Standard Machine Learning predicts **correlation** ("People who watch X also watch Y"). Causal Inference predicts **causality** ("If we *make* them watch X, will they stay subscribed?").

-   **Social (LinkedIn)**: "The Un-Testable".
    -   You can't run an A/B test for everything. E.g., "Does having a complete profile cause you to get a job?" You can't *force* people to have incomplete profiles (unethical/bad UX).
    -   *Solution*: Observational Causal Inference. Use existing data but "adjust" it to look like an experiment.
-   **Media (Netflix)**: "Content Investment".
    -   Should we renew 'Stranger Things'? Not just "did people watch it?", but "did it *cause* them to retain their subscription?"
-   **Delivery (Gojek/Uber)**: "Voucher Efficiency".
    -   Should we give User A a 50% discount? Only if it *changes* their behavior (Uplift).

**What makes this problem ML-worthy?**

1.  **Counterfactuals**: We can never observe the counterfactual (what would have happened to User A *if* we didn't give the voucher?). We must estimate it.
2.  **Confounding**: In observational data (LinkedIn), "Job Seekers" might just be more active than "Passive Candidates". Correlation != Causation. We need ML to control for these confounders.
3.  **Network Effects**: In ride-sharing (Uber), treating one user affects others. Standard A/B tests fail here.

---

## PART 2: SYSTEM DESIGN DEEP DIVE

### 2.1 High-Level Architecture

**LinkedIn Ocelot (Observational Causal Inference)**:
```mermaid
graph TD
    Data[Observational Data (Tracking)] --> Ocelot[Ocelot Platform]
    
    subgraph "Ocelot Compute Engine (Spark)"
        Ocelot --> Covariates[Covariate Selection]
        Ocelot --> Matching[Propensity Matching]
        Ocelot --> Weighting[IPW Weighting]
        
        Matching & Weighting --> Effect[ATE Estimation]
    end
    
    Effect --> Dashboard[Product Insights]
    Dashboard --> PM[Product Manager Decision]
```

**Gojek Uplift Modeling Architecture**:
```mermaid
graph TD
    User[User Features] --> Model[Uplift Model (T-Learner)]
    
    subgraph "T-Learner Architecture"
        User --> Model_0[Base Model (Control)]
        User --> Model_1[Base Model (Treatment)]
        
        Model_0 --> Pred_0[P(Buy | No Voucher)]
        Model_1 --> Pred_1[P(Buy | Voucher)]
    end
    
    Pred_1 -- Minus --> Pred_0
    Pred_0 --> Lift[Estimated Lift (CATE)]
    
    Lift --> Optimizer[Budget Optimizer]
    Optimizer --> Decision{Allocate Voucher?}
```

### Tech Stack Identified

| Industry | Component | Technology/Tool | Purpose | Company |
|:---|:---|:---|:---|:---|
| **Social** | **Platform** | Ocelot (Custom on Spark) | Scaling observational inference | LinkedIn |
| **Delivery** | **Uplift Model** | CausalML / EconML | Estimating CATE | Gojek, Uber |
| **Media** | **Double ML** | DoubleML | Removing bias from high-dimensional controls | Netflix |
| **Mobility** | **Synthetic Control** | CausalImpact | Measuring impact of geo-targeted events | Lyft |
| **All** | **Orchestrator** | Metaflow / Kubeflow | Pipeline Management | Netflix, LinkedIn |

### 2.2 Data Pipeline

**LinkedIn (Ocelot)**:
-   **Input**: "Tracking" data (User clicks, views, profile updates).
-   **Treatment Definition**: "Did the user use Feature X?" (Binary).
-   **Outcome Definition**: "Did the user apply for a job within 7 days?" (Binary).
-   **Confounder Adjustment**: Select hundreds of features (Industry, Seniority, Activity Level) that might influence both Treatment and Outcome.

**Netflix (Content Valuation)**:
-   **Observational Data**: Users who watched Show X vs. Users who didn't.
-   **Adjustment**: Use Double Machine Learning to control for confounders (viewing history, device, region) to isolate the *causal* impact of Show X on retention.

### 2.3 Feature Engineering

**Key Features**:

**Propensity Score (LinkedIn)**:
-   The probability `P(T=1 | X)` that a user *would have* taken the action naturally.
-   Used to "match" users. A treated user with Propensity 0.6 is compared to an untreated user with Propensity 0.6.

**Interference Features (Lyft)**:
-   "Fraction of neighbors treated". If my neighbors get discounts, they book cars, reducing supply for me. Crucial for network effect modeling.

### 2.4 Model Architecture

**Meta-Learners (T-Learner, S-Learner, X-Learner)**:
-   **S-Learner**: Single model `Y = f(X, T)`. Simple but often misses subtle effects.
-   **T-Learner**: Two models. `Y0 = f0(X)` (Control) and `Y1 = f1(X)` (Treatment). `Lift = f1(X) - f0(X)`.
-   **X-Learner**: Complex multi-stage learner. Best for imbalanced datasets (e.g., only 1% get treatment).

**Double Machine Learning (DML)**:
-   Used when you have high-dimensional confounders.
-   Stage 1: Predict Outcome `Y` from Confounders `X`. Predict Treatment `T` from Confounders `X`.
-   Stage 2: Regress the *residuals* of Y on the *residuals* of T.

---

## PART 3: MLOPS & INFRASTRUCTURE

### 3.1 Model Deployment & Serving

**Batch vs. Real-Time**:
-   **Strategic (LinkedIn/Netflix)**: **Batch**. "Does this feature work?" is a quarterly question. Ocelot runs big Spark jobs.
-   **Operational (Gojek/Uber)**: **Real-Time**. "Should I give a discount *now*?" requires sub-second inference.

**Constraint Optimization**:
-   **Gojek**: It's not just `Lift > 0`. It's `Maximize Total Lift` subject to `Budget < $10k`.
-   **Solution**: Linear Programming (LP) solvers running on top of the Uplift scores.

### 3.2 Monitoring & Observability

**Metrics**:
-   **Qini Curve / AUUC**: "Area Under Uplift Curve". Standard metric for uplift models.
-   **Covariate Balance (LinkedIn)**: After matching, are the Treated and Control groups statistically identical? (Standardized Mean Difference < 0.1).

### 3.3 Operational Challenges

**The "Universal Control Group"**:
-   To monitor long-term lift, you need a "Holdout Group" that *never* receives any campaigns.
-   **Cost**: This group generates less revenue.
-   **Benefit**: It's the only way to measure the true incremental value of your entire marketing program.

**Interference (SUTVA Violation)**:
-   In 2-sided marketplaces (Uber/Lyft), treating one user affects others.
-   **Solution**: **Switchback Testing** (Time-based randomization) or **Cluster Randomization** (Geo-based).

---

## PART 4: EVALUATION & VALIDATION

### 4.1 Offline Evaluation

**Placebo Tests (LinkedIn)**:
-   Run the causal model on a "fake" treatment (e.g., a random date). The estimated effect should be zero. If it's not, the model is biased.

### 4.2 Online Evaluation

**Stratified A/B Tests**:
-   Test the "Uplift Model Policy" vs. "Random Policy".
-   Metric: **Incremental ROI**. (Revenue from Treatment - Revenue from Control) / Cost of Treatment.

### 4.3 Failure Cases

-   **Sleeping Dogs**: Users who *churn* if you disturb them (e.g., spammy notifications).
    -   *Fix*: Uplift models explicitly identify this group (Negative Lift) and suppress messages.

---

## PART 5: LESSONS LEARNED & KEY TAKEAWAYS

### 5.1 Technical Insights

1.  **Correlation != Causation**: Standard Churn Prediction models target users *most likely to churn*. Uplift models target users *most likely to be saved*. These are different groups!
2.  **Observational is Scalable (LinkedIn)**: You can run 1000 observational studies in the time it takes to run 1 A/B test. It's a "pre-flight check" for product ideas.

### 5.2 Operational Insights

1.  **Budget Efficiency**: Gojek saved millions by stopping vouchers to "Sure Things" (users who buy anyway).
2.  **Democratization**: Netflix and LinkedIn built "Causal Platforms" so non-ML people (PMs) can run causal queries without writing code.

---

## PART 6: REFERENCE ARCHITECTURE (CAUSAL ML)

```mermaid
graph TD
    subgraph "Data Source"
        RCT[Randomized Experiments] --> Traj[Training Data]
        Obs[Observational Data] --> Traj
    end

    subgraph "Causal Engine"
        Traj --> Learner[Meta-Learner (T/X/S)]
        Traj --> Matching[Propensity Matching]
        
        Learner --> CATE[CATE Estimates]
        Matching --> ATE[ATE Estimates]
    end

    subgraph "Decision Layer"
        CATE --> Calibration[Isotonic Calibration]
        Calibration --> Scores[Uplift Scores]
        Scores --> Solver[LP Solver (Budget)]
        Solver --> Allocation[Final Allocation]
    end
    
    Allocation --> Campaign[Marketing Campaign]
```

### Estimated Costs
-   **Compute**: Moderate. Spark clusters for Ocelot.
-   **Data Cost**: High. Running RCTs (giving random discounts) costs real money (Opportunity Cost).
-   **Team**: Specialized. PhD-level Statisticians/Economists + ML Engineers.

---

*Analysis completed: November 2025*
