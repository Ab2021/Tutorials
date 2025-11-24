# ML Use Case Analysis: Delivery & Mobility Causal Inference

**Analysis Date**: November 2025  
**Category**: Causal Inference  
**Industry**: Delivery & Mobility  
**Articles Analyzed**: 3 (Gojek, Lyft, Uber)

---

## PART 1: USE CASE OVERVIEW

### 1.1 Basic Information

**Category**: Causal Inference  
**Industry**: Delivery & Mobility  
**Companies**: Gojek, Lyft, Uber  
**Years**: 2023-2025  
**Tags**: Uplift Modeling, Synthetic Control, Network Effects, Switchback Testing, Econometrics

**Use Cases Analyzed**:
1.  [Gojek - Personalised Vouchers at Scale](https://www.gojek.io/blog/how-gojek-allocates-personalised-vouchers-at-scale)
2.  [Lyft - Causal Inference for Marketing](https://eng.lyft.com/causal-forecasting-at-lyft-part-1-1466cab0655d)
3.  [Uber - CausalML Library](https://github.com/uber/causalml)

### 1.2 Problem Statement

**What business problem are they solving?**

This category addresses the **"Counterfactual Question"**: *What would have happened if we didn't do X?*

-   **Gojek (Voucher Allocation)**: "The Free Lunch Problem".
    -   *The Challenge*: Gojek has a budget of $1M for vouchers. If they give a voucher to everyone, they go bankrupt. If they give it to no one, they lose growth.
    -   *The Friction*: Traditional ML predicts "Who will buy?". It targets loyal users who would have bought anyway. This wastes money.
    -   *The Goal*: Predict "Who will buy *only if* they get a voucher?" (The "Persuadables").

-   **Lyft (Marketing Measurement)**: "The Billboard Problem".
    -   *The Challenge*: Lyft spends millions on TV ads. You can't click a TV ad. How do you measure ROI?
    -   *The Friction*: You can't A/B test a TV ad (everyone in New York sees it).
    -   *The Goal*: Use **Synthetic Control** to construct a "fake New York" from a weighted combination of Chicago, LA, and Boston, and compare the real New York (with ads) to the fake one (without ads).

**What makes this problem ML-worthy?**

1.  **Unobservable Ground Truth**: You can never observe the counterfactual. A user either got the voucher or didn't. You can't see both worlds. ML must *infer* the missing potential outcome.
2.  **Network Effects (Interference)**: In ride-sharing, treating User A affects User B. If A gets a discount and books a car, B waits longer. Standard A/B testing assumptions (SUTVA) fail.
3.  **Heterogeneity**: The treatment effect varies wildly. A 10% discount works for students but insults business travelers.

---

## PART 2: SYSTEM DESIGN DEEP DIVE

### 2.1 High-Level Architecture

**Gojek Uplift Modeling Pipeline**:
```mermaid
graph TD
    User[User Features] --> TLearner[Meta-Learner (T-Learner)]
    
    subgraph "Causal Inference Model"
        TLearner --> Model0[Base Model (Control)]
        TLearner --> Model1[Base Model (Treatment)]
        
        Model0 --> P0[Prob(Buy | No Voucher)]
        Model1 --> P1[Prob(Buy | Voucher)]
        
        P1 -- Minus --> P0
        P0 --> CATE[Conditional Average Treatment Effect]
    end
    
    CATE --> Optimizer[Linear Programming Solver]
    Optimizer --> Budget[Budget Constraint]
    Optimizer --> Allocation[Final Allocation List]
```

**Lyft Synthetic Control**:
```mermaid
graph TD
    Cities[Time Series of All Cities] --> Preproc[Normalize & Detrend]
    
    subgraph "Synthetic Control Construction"
        Preproc --> Weights[Learn City Weights]
        Weights --> Synthetic[Construct Synthetic 'Control' City]
    end
    
    Real[Real 'Treated' City Data] --> Compare
    Synthetic --> Compare
    
    Compare --> Lift[Calculate Lift (ATT)]
    Lift --> ROI[Marketing ROI Dashboard]
```

### Tech Stack Identified

| Component | Technology/Tool | Purpose | Company |
|-----------|----------------|---------|---------|
| **Causal Library** | CausalML (Uber) / EconML | Uplift modeling algorithms | Uber, Gojek |
| **Optimization** | Pyomo / OR-Tools | Solving budget allocation (Knapsack) | Gojek |
| **Experimentation** | Wasabi (Intuit) / PlanOut | Managing A/B tests | Lyft |
| **Data Warehouse** | BigQuery / Hive | Storing experiment logs | All |
| **Model** | XGBoost / LightGBM | Base learners for T-Learners | Gojek |

### 2.2 Data Pipeline

**Gojek (Uplift)**:
-   **Training Data Collection**:
    -   Run a **Randomized Control Trial (RCT)**.
    -   Randomly assign vouchers to 5% of users.
    -   Log: `{User_Features, Treatment_Flag, Outcome_Conversion}`.
-   **Feature Engineering**:
    -   *Recency*: Days since last order.
    -   *Sensitivity*: Did they use a voucher last time?
-   **Inference**:
    -   Score *all* eligible users.
    -   Output: `Predicted_Lift` for each user.

### 2.3 Feature Engineering

**Key Features**:

-   **Pre-Treatment Covariates**: Features that exist *before* the intervention. (Age, Location, History).
-   **Contextual Features**: Time of day, Raining/Sunny. (Vouchers work better when it's raining).
-   **Interaction Terms**: `Is_Student * Discount_Amount`. (Students are more price-sensitive).

### 2.4 Model Architecture

**The T-Learner (Two-Model Learner)**:
-   **Concept**: Train two separate models.
    -   `M0(X)`: Predicts conversion given Control (No Voucher).
    -   `M1(X)`: Predicts conversion given Treatment (Voucher).
-   **Prediction**: `Lift = M1(X) - M0(X)`.
-   **Pros**: Simple, uses standard libraries (XGBoost).
-   **Cons**: If the treatment effect is small, the errors in M0 and M1 can drown out the signal.

**The X-Learner**:
-   **Concept**: A more advanced meta-learner that handles "Imbalanced" data (e.g., Control group is much larger than Treatment group).
-   **Used by**: Uber (implemented in CausalML) for cases where RCTs are expensive/small.

---

## PART 3: MLOPS & INFRASTRUCTURE

### 3.1 Model Deployment & Serving

**Batch Inference**:
-   **Gojek**: Vouchers are usually sent as "Push Notifications" at a specific time (e.g., 11 AM before lunch).
-   **Workflow**:
    1.  Nightly Batch Job: Score 10M users.
    2.  Optimization Job: Select top N users s.t. `Sum(Cost) < Budget`.
    3.  Campaign Manager: Send pushes.

### 3.2 Experimentation Platform

**Switchback Testing (Network Effects)**:
-   **Problem**: You can't randomize by User in a marketplace.
-   **Solution**: Randomize by **Time-Region Units**.
    -   *Example*: "San Francisco" is Treatment from 10:00-10:20, then Control from 10:20-10:40.
-   **Infrastructure**: The experiment service must handle "Carryover Effects" (a ride booked at 10:19 affects supply at 10:21). They often discard data from the "Buffer Minutes" between switches.

### 3.3 Monitoring & Observability

**Metrics**:
-   **Incremental ROAS**: Return on Ad Spend *due to the model*.
-   **Uplift Deciles**:
    -   Sort users by predicted lift.
    -   Check if the top decile actually has the highest lift in a validation RCT.
    -   *Visual*: **Qini Curve** (Cumulative Uplift).

### 3.4 Operational Challenges

**Calibration**:
-   **Issue**: Causal models are notoriously uncalibrated. `Lift = 0.05` might actually mean `0.02`.
-   **Solution**: **Continuous RCTs**. Always hold out a small "Global Control Group" (never treated) and a "Global Treatment Group" (randomly treated) to ground-truth the models.

---

## PART 4: EVALUATION & VALIDATION

### 4.1 Offline Evaluation

**Qini Coefficient (AUUC)**:
-   Measures the "Area Under the Uplift Curve".
-   A random model has a straight diagonal line.
-   A perfect model bows upward (captures all persuadables first).
-   **Gojek** uses this to compare T-Learner vs S-Learner.

### 4.2 Online Evaluation

**Budget-Constrained Tests**:
-   Run the model with $10k budget vs Random Selection with $10k budget.
-   Compare total *Incremental* Orders.

### 4.3 Failure Cases

-   **The "Sleeping Dogs"**:
    -   *Failure*: Sending a notification to a dormant user reminds them to *unsubscribe*.
    -   *Fix*: The model should predict *negative* lift for these users. T-Learners can capture this (Lift < 0).
-   **Interference Bias**:
    -   *Failure*: A pricing experiment in "Manhattan" bleeds into "Brooklyn" because drivers cross bridges.
    -   *Fix*: **Cluster Randomization**. Treat the whole city as one unit, or use natural geographic barriers.

---

## PART 5: KEY ARCHITECTURAL PATTERNS

### 5.1 Common Patterns

-   [x] **Meta-Learners**: Using standard ML (XGBoost) as building blocks for Causal Inference (T/S/X-Learners).
-   [x] **Optimization Layer**: ML output (Lift) is not the final decision. It feeds into a Linear Programming solver (Knapsack Problem) to respect budgets.
-   [x] **Switchback Testing**: Temporal randomization to handle network effects.

### 5.2 Industry-Specific Insights

-   **Delivery**: **Margins are Thin**. You cannot afford to subsidize users who don't need it. Uplift modeling is a survival tool, not just optimization.
-   **Mobility**: **Supply is Finite**. Increasing demand (Vouchers) without increasing supply (Driver Pay) just causes surge pricing. Causal models must account for *Market Balance*.

---

## PART 6: LESSONS LEARNED & TAKEAWAYS

### 6.1 Technical Insights

1.  **Correlation != Causation**: Standard Churn Models predict "High spenders are less likely to churn". Causal Models might find "High spenders are *immune* to discounts".
2.  **Data Quality**: Causal inference requires **Propensity Scores** to correct for selection bias in observational data. If you don't log the *probability of treatment*, you can't debias later.

### 6.2 Operational Insights

1.  **Global Holdouts**: You must sacrifice some short-term revenue (by not treating a control group) to gain long-term knowledge (calibrated models).
2.  **Ethics**: Is it fair to charge User A more than User B based on an algorithm? Companies must set "Guardrails" (e.g., Max price difference < 10%).

---

## PART 7: REFERENCE ARCHITECTURE

### 7.1 System Diagram (Uplift Platform)

```mermaid
graph TD
    subgraph "Experimentation"
        RCT[Randomized Control Trial] --> Logs[Interaction Logs]
    end

    subgraph "Training Pipeline"
        Logs --> FeatureEng[Feature Eng]
        FeatureEng --> Split[Split Treatment/Control]
        
        Split --> Train0[Train Model(Control)]
        Split --> Train1[Train Model(Treatment)]
        
        Train0 & Train1 --> Eval[Evaluate Qini]
        Eval --> Registry[Model Registry]
    end

    subgraph "Inference & Allocation"
        Users[Target User Base] --> Score[Score Users (Lift)]
        Score --> Opt[Optimization Service]
        
        Config[Budget & Constraints] --> Opt
        Opt --> List[Final List]
    end

    subgraph "Execution"
        List --> Push[Push Notification Service]
        Push --> App[User App]
    end
```

### 7.2 Estimated Costs
-   **Compute**: Low. Batch processing.
-   **Experimentation**: High. The "Cost of Experimentation" (lost revenue from control groups) is significant.
-   **Team**: Specialized. Requires Economists + ML Engineers.

### 7.3 Team Composition
-   **Causal Inference Scientists**: 2-3 (PhD in Econ/Stats).
-   **ML Engineers**: 3-4 (Building the platform).
-   **Product Managers**: 1-2 (Designing the incentives).

---

*Analysis completed: November 2025*
