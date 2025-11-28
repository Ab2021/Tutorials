

# **Strategic Curriculum for Advanced Actuarial Intelligence and Insurance Analytics**

## **Executive Summary and Pedagogical Framework**

The insurance industry stands at a pivotal juncture where the deterministic rigor of traditional actuarial science converges with the predictive power of modern data science. This report outlines a comprehensive, expert-level course structure designed to master the full spectrum of insurance analytics. It bridges the gap between classical methods—such as Generalized Linear Models (GLMs) and Chain Ladder reserving—and the frontier of machine learning, including Gradient Boosting Machines (GBMs), Neural Networks, and Natural Language Processing (NLP).

The curriculum is rigorously mapped to the professional examination syllabi of the world's leading actuarial bodies: the **Casualty Actuarial Society (CAS)**, the **Society of Actuaries (SOA)**, and the **Institute and Faculty of Actuaries (IFoA)**. It integrates theoretical mathematics with practical implementation, ensuring that learners not only understand the derivation of a Tweedie loss function but can also implement it in an XGBoost pipeline for production-grade pricing.

This course is structured into eight intensive modules. Each module functions as a distinct unit of competency, moving from foundational statistical theory to advanced predictive modeling, and finally to the strategic application of these models in pricing, reserving, fraud detection, and customer value management. The design philosophy emphasizes the "Why" (mathematical derivation) and the "How" (computational implementation), preparing the actuary of the future to be a bilingual expert fluent in both risk theory and algorithmic logic.

---

## **Module 1: Mathematical Foundations of Actuarial Learning**

### **1.1 The Actuarial Statistical Paradigm**

Before approaching any specific insurance application, a robust statistical foundation is required. This module harmonizes the discrete probability concepts found in early actuarial exams with the continuous, high-dimensional frameworks used in machine learning. The objective is to transition the learner from "statistics for inference" (explaining the past) to "statistics for prediction" (forecasting the future).

Traditional statistics often focuses on asymptotic properties and hypothesis testing. However, in modern insurance analytics, the focus shifts towards generalization performance on unseen data. This module grounds the student in the mathematical structures that allow for this shift, specifically the Exponential Dispersion Family (EDF), which underpins both GLMs and many modern machine learning objective functions.

### **1.2 The Exponential Dispersion Family (EDF) and Loss Functions**

A deep understanding of the EDF is non-negotiable for modern actuaries. It unifies the disparate distributions used in insurance—Poisson for frequency, Gamma for severity, Tweedie for pure premium—under a single mathematical umbrella. This unification helps explain why modern machine learning libraries like XGBoost and LightGBM can be adapted for insurance simply by changing the objective function.

The probability density function for a random variable $Y$ in the EDF is defined as:

$$f(y; \\theta, \\phi) \= \\exp\\left( \\frac{y\\theta \- b(\\theta)}{a(\\phi)} \+ c(y, \\phi) \\right)$$  
Here, $\\theta$ is the canonical parameter related to the location (mean), and $\\phi$ is the dispersion parameter related to the scale (variance). The function $b(\\theta)$, known as the cumulant function, is the engine of the distribution. Its derivatives define the moments of the distribution:

* **Mean:** $E \= \\mu \= b'(\\theta)$  
* **Variance:** $\\text{Var}(Y) \= b''(\\theta) \\cdot a(\\phi)$

**The Variance Function:** The relationship between the mean and variance, $V(\\mu) \= b''(\\theta)$, uniquely identifies the distribution. This concept is central to "Tweedie" regression, where $V(\\mu) \= \\mu^p$.

* $p=0$: Normal Distribution (constant variance).  
* $p=1$: Poisson Distribution (variance equals mean).  
* $p=2$: Gamma Distribution (variance proportional to mean squared).  
* $p=3$: Inverse Gaussian Distribution (variance proportional to mean cubed).  
* $1 \< p \< 2$: Tweedie Compound Poisson-Gamma (undefined for simple distributions, but crucial for modeling pure premium with a point mass at zero).1

**Link to Machine Learning:** In a neural network or GBM, the "loss function" is simply the negative log-likelihood of these distributions. For example, minimizing the "Poisson Deviance" in a neural network is mathematically identical to maximizing the likelihood of a Poisson GLM. This mathematical equivalence is what allows actuaries to apply deep learning to claim counts without violating fundamental statistical principles.3

### **1.3 Bayesian Inference and Credibility Theory**

While frequentist Maximum Likelihood Estimation (MLE) dominates large-data pricing, Bayesian methods are the theoretical backbone of "Credibility"—the weighting of a specific risk's experience against the class average. This is critical when data is sparse, a frequent occurrence in commercial lines or reinsurance.

Conjugate Priors: The course explores why certain prior-likelihood pairs (e.g., Poisson-Gamma, Normal-Normal) are mathematically convenient, allowing for closed-form posterior distributions.  
Bühlmann-Straub Credibility: The linear approximation of the Bayesian posterior mean:

$$Z \= \\frac{n}{n \+ k}$$

where $n$ is the volume of data (exposure) and $k$ is the ratio of process variance to parameter variance.  
MCMC and Stan: Moving beyond analytical solutions, we introduce Markov Chain Monte Carlo (MCMC) methods using software like Stan or JAGS. This allows for the estimation of complex hierarchical models where no conjugate prior exists, often required in reserving for latent claims (e.g., asbestos) or stochastic mortality modeling.

### **1.4 Mapping to Professional Examinations**

This theoretical grounding is directly tested in the initial fellowship exams.

| Exam Body | Exam | Relevant Syllabus Content | Cognitive Level |
| :---- | :---- | :---- | :---- |
| **CAS** | **MAS-I** | Probability Models, Stochastic Processes, Bayesian Inference, MCMC | Apply & Analyze |
| **CAS** | **MAS-II** | Credibility Theory, Extended Linear Models, MCMC algorithms | Analyze & Evaluate |
| **SOA** | **FAM** | Bayesian Estimation, Credibility Theory, Loss Models | Understand & Apply |
| **IFoA** | **CS1** | Bayesian Statistics, GLM Theory, Empirical Bayes Credibility | Apply |

The "Cognitive Level" indicates the depth of testing. CAS MAS-II, for instance, moves beyond rote calculation (Remember) to "Analyze and Evaluate," requiring candidates to justify model choices and interpret MCMC convergence diagnostics.5

---

## **Module 2: The Generalized Linear Model (GLM) Framework – The Industry Standard**

### **2.1 The Actuary's Workhorse**

Despite the hype around AI, the Generalized Linear Model remains the primary tool for regulatory filing and rating plan development. Its transparency, interpretability, and multiplicative structure align perfectly with insurance requirements. This module covers the end-to-end construction of a GLM-based tariff.

### **2.2 Mathematical Structure of GLMs**

A GLM decomposes the modeling problem into three components:

1. **Random Component:** The distribution of the response variable $Y$ (from the EDF discussed in Module 1).  
2. Systematic Component: The linear combination of predictors $X$:

   $$\\eta\_i \= \\beta\_0 \+ \\beta\_1 x\_{i1} \+ \\dots \+ \\beta\_p x\_{ip} \= \\mathbf{x}\_i^T \\mathbf{\\beta}$$  
3. Link Function: The function $g(\\cdot)$ that connects the expected value $\\mu \= E$ to the linear predictor $\\eta$:

   $$g(\\mu\_i) \= \\eta\_i \\implies \\mu\_i \= g^{-1}(\\mathbf{x}\_i^T \\mathbf{\\beta})$$

The Log-Link Function: In insurance, the log-link $g(\\mu) \= \\ln(\\mu)$ is ubiquitous. It ensures predictions are always positive (crucial for premiums) and creates a multiplicative rating structure:  
$$ \\text{Rate} \= \\text{Base} \\times \\text{Relativity}{\\text{Age}} \\times \\text{Relativity}{\\text{Geo}} $$  
This structure is preferred because risk factors in insurance often interact multiplicatively; a risky driver in a risky territory is usually worse than the sum of the individual risks.

### **2.3 Frequency and Severity Modeling**

The standard approach involves modeling claim frequency ($N$) and claim severity ($X$) separately.

Frequency (Poisson/Negative Binomial):  
The target variable is claim counts, but the exposure varies per policy. We handle this via an "offset" term.

$$\\ln(\\mu) \= \\ln(\\text{Exposure}) \+ \\mathbf{x}^T \\mathbf{\\beta}$$

Here, the coefficient for $\\ln(\\text{Exposure})$ is constrained to be 1\. We explore the Negative Binomial distribution to handle overdispersion—the phenomenon where the variance of claim counts exceeds the mean, violating the strict Poisson assumption.  
Severity (Gamma):  
The target variable is the average cost per claim. The Gamma distribution is ideal because it is right-skewed and defined for positive reals.

$$E\[X\] \= \\exp(\\mathbf{x}^T \\mathbf{\\beta}\_{\\text{sev}})$$

Crucially, standard severity models are conditional on a claim occurring ($N \> 0$).

### **2.4 Model Diagnostics and Validation**

Building the model is only half the battle; validating it is where actuarial expertise shines.

* **Deviance Analysis:** Using Analysis of Deviance (ANODEV) tables to determine the statistical significance of adding a variable (Type I/III tests).  
* **One-Way Analysis:** A visual comparison of actual vs. predicted loss ratios across different cuts of the data. A robust model should track the actual experience closely across all univariate dimensions.  
* **Consistency Checks:** Are the coefficients monotonic? Does the "Young Driver" surcharge decrease smoothly as age increases? If not, the model may be overfitting noise.  
* **Gini Index and Lorenz Curves:** Measures the "lift" of the model—its ability to distinguish between best and worst risks. A higher Gini coefficient implies better segmentation.

### **2.5 Practical Implementation**

Students will implement GLMs in R using the glm() and statmod packages.

* *Key Exercise:* Replicating the "French Motor Third-Party Liability" case study.1  
* *Data:* Loading 600k+ rows of policy data.  
* *Preprocessing:* Binning continuous variables (Age, Vehicle Power) into categorical factors.  
* *Modeling:* Fitting separate Frequency and Severity GLMs.  
* *Combining:* Calculating Pure Premium \= Predicted Frequency $\\times$ Predicted Severity.

### **2.6 Mapping to Professional Examinations**

| Exam Body | Exam | Relevant Syllabus Content |
| :---- | :---- | :---- |
| **CAS** | **Exam 5** | Basic Ratemaking, GLM construction, One-way analysis, Validation |
| **CAS** | **Exam 8** | Advanced Ratemaking, Penalized Regression, Offset terms |
| **IFoA** | **SP8** | General Insurance Pricing, Rating factor selection, GLM diagnostics |
| **SOA** | **Exam PA** | GLM implementation in R, Variable selection, Interpretation |

**Note on CAS Exam 8:** The syllabus now includes "Penalized Regression and Lasso Credibility," signaling a shift toward regularized GLMs, which bridge the gap to machine learning.9

---

## **Module 3: Advanced Pricing – Machine Learning and Non-Linear Models**

### **3.1 Beyond Linearity**

While GLMs are interpretable, they struggle with complex interactions and non-linearities (e.g., the interaction between "Vehicle Power" and "Driver Age" might be highly non-linear). This module introduces the modern data science toolkit, moving from Generalized Additive Models (GAMs) to "Black Box" algorithms.

### **3.2 Generalized Additive Models (GAMs)**

GAMs relax the linearity assumption by replacing $\\beta x$ with a smooth function $f(x)$:

$$g(\\mu) \= \\beta\_0 \+ f\_1(x\_1) \+ f\_2(x\_2) \+ \\dots$$

* **Splines:** We use cubic or thin-plate regression splines to learn the shape of the relationship from the data.  
* **Interpretability:** GAMs occupy a "sweet spot." They are more accurate than GLMs but retain the additivity that allows for easy visualization (e.g., plotting the risk curve for Age).  
* **Software:** The mgcv package in R is the industry standard for fitting GAMs.11

### **3.3 Tree-Based Ensembles: The State of the Art**

For tabular insurance data, tree-based ensembles are currently superior to almost all other methods, including deep learning.

**Random Forests:**

* **Bagging:** Training multiple trees on bootstrapped subsets of data to reduce variance.  
* **Feature Randomness:** Selecting a random subset of features at each split to de-correlate the trees.  
* **Use Case:** Excellent for benchmarking GLMs (finding the "theoretical maximum" signal in the data) and determining variable importance.

Gradient Boosting Machines (GBMs):  
The core of modern pricing competitions and advanced R\&D.

* **Mechanism:** Training trees sequentially. Each new tree attempts to predict the *residuals* (errors) of the previous ensemble.  
* **Algorithms:**  
  * **XGBoost:** The pioneer of efficient, scalable boosting. Supports "monotonicity constraints," allowing actuaries to force the model to respect logic (e.g., more accidents \= higher price).11  
  * **LightGBM:** Developed by Microsoft. Uses "Leaf-wise" growth which is faster and often more accurate for large datasets. Handles categorical features natively without one-hot encoding.1  
  * **CatBoost:** Developed by Yandex. Optimized for categorical data (e.g., Vehicle Make/Model) using "ordered boosting" to prevent target leakage.  
* **Objective Functions:** This is where the EDF from Module 1 returns. We do not minimize Mean Squared Error. We set the objective to **Tweedie Loss** or **Poisson Deviance**.  
  * *Code Example (Python):* xgb.train(params={'objective': 'reg:tweedie', 'tweedie\_variance\_power': 1.5}).2

### **3.4 Neural Networks for Pricing**

While less common for tabular data, Neural Networks are powerful for "representation learning"—creating features from unstructured data.

* **Embeddings:** Using an "Embedding Layer" to transform high-cardinality categorical variables (like ZIP codes or Diagnosis Codes) into dense vectors.  
* **Combined Models:** A "GLM-NN" hybrid where the neural network learns the complex features, which are then fed into a final GLM layer for interpretability.3  
* **Loss Functions:** Implementing the Negative Multinomial Deviance or Poisson-Gamma loss directly in TensorFlow/PyTorch.3

### **3.5 Explainable AI (XAI)**

Regulators will not approve a "Black Box." We must open it.

* **SHAP (Shapley Additive Explanations):** A game-theoretic approach to attribute the prediction to each feature.  
  * *Global Interpretability:* Which variables matter most?  
  * *Local Interpretability:* Why did *this specific* policy get a 15% rate increase?  
* **Partial Dependence Plots (PDP):** Visualizing the marginal effect of a variable.  
* **Accumulated Local Effects (ALE):** A more robust alternative to PDP that handles correlated features better.  
* **Exam Mapping:** SOA Exam PA and CAS MAS-II explicitly test the interpretation of these plots.6

---

## **Module 4: Reserving and Claims Analytics – The Stochastic Revolution**

### **4.1 From Triangles to Granularity**

Traditional reserving relies on "Triangles"—aggregated data (Accident Year vs. Development Year). This destroys information. Modern analytics moves to **Individual Claims Reserving (ICR)**.

### **4.2 Deterministic & Stochastic Triangle Methods**

We begin with the classics, as they are the benchmark.

* **Chain Ladder Method (CLM):** The assumption that future development is proportional to past development.  
* **Bornhuetter-Ferguson (BF):** A credibility-weighted average of the CLM and an *a priori* expectation. Essential for unstable, immature years.  
* Mack's Model: A stochastic formulation of Chain Ladder. It provides a formula for the Mean Squared Error of Prediction (MSEP), estimating the volatility of the reserve.

  $$\\text{MSEP} \= \\text{Process Variance} \+ \\text{Parameter Estimation Error}$$  
* **Bootstrapping:** Resampling the residuals of the triangle to generate a full distribution of possible outcomes. This is critical for calculating the "Risk Margin" in Solvency II and IFRS 17\.

### **4.3 Machine Learning for Individual Claims**

Instead of aggregating, we model the lifecycle of *each claim*.

* **Granular Features:** We can now use claimant age, injury type, attorney involvement, and adjuster notes—data lost in a triangle.  
* **DeepTriangle:** A deep learning architecture that treats the development of a triangle as a sequence-to-sequence learning problem.15  
* **Dual-Modeling Approach:**  
  1. **Time-to-Event (Survival) Model:** Predicting *when* the next payment will occur and *when* the claim will close.  
  2. **Payment Severity Model:** Predicting *how much* the payment will be.  
* **Hierarchical Bayesian Models:** Modeling the dependency structure between different lines of business (e.g., Auto Liability and Auto Physical Damage correlations).

### **4.4 IFRS 17 and Solvency II Implications**

Modern regulation requires a "Best Estimate" plus a "Risk Adjustment."

* **Discounting:** Future cash flows must be discounted. The "yield curve" becomes a critical input.  
* **Risk Adjustment:** Calculated using the VaR (Value at Risk) or TVaR (Tail VaR) of the stochastic reserve distribution generated by our Bootstrap or ML models.  
* **Mapping to Examinations:**  
  * **IFoA SA3:** Covers the practical application of these standards in general insurance.  
  * **CAS Exam 7:** Deep focus on Mack, Bootstrap, and the theoretical underpinnings of reserve risk.17

---

## **Module 5: Severity Modeling, Large Losses, and Reinsurance**

### **5.1 The Tail Wagging the Dog**

In insurance, the "average" claim is irrelevant for solvency; the "extreme" claim matters. This module focuses on Heavy-Tailed distributions and Extreme Value Theory (EVT).

### **5.2 Extreme Value Theory (EVT)**

Standard distributions (Gamma, Lognormal) often underestimate the probability of extreme events (the "Dragon King" or "Black Swan" events).

* **Block Maxima (GEV):** Modeling the maximum loss in a given period (e.g., worst hurricane of the year).  
* **Peaks Over Threshold (POT):** Modeling all losses that exceed a high threshold $u$.  
  * **Generalized Pareto Distribution (GPD):** The limit distribution for excesses.  
  * *Key Parameter:* The shape parameter $\\xi$. If $\\xi \> 0$, the distribution has a "heavy tail" (polynomial decay), implying infinite variance is possible. This is common in Liability and Cyber insurance.

### **5.3 Splicing and Mixture Models**

A robust severity model often "splices" two distributions:

1. **Body:** A Lognormal or Gamma distribution for attritional claims (\< $1M).  
2. **Tail:** A GPD for large claims (\> $1M).  
* **Mixture Neural Networks:** Using a neural network to estimate the parameters of a mixture density dynamically based on claim features. For example, a claim with "Attorney Rep \= Yes" might have a much heavier tail than one without.19

### **5.4 Reinsurance Pricing**

* **Excess of Loss (XoL):** Pricing a layer (e.g., $5M excess of $5M).  
  * *Experience Rating (Burning Cost):* Trending historical large losses to present value.  
  * *Exposure Rating:* Using "Increased Limit Factors" (ILFs) derived from industry curves (e.g., ISO or Swiss Re curves) when own data is insufficient.  
* **Catastrophe Modeling:** Integrating scientific models (Seismology, Meteorology) with financial modules to price earthquake and hurricane risk.  
* **Exam Mapping:** IFoA SP8 and CAS Exam 9 focus heavily on XoL pricing and the mathematics of limit/deductible adjustments.21

---

## **Module 6: Fraud Detection, Litigation, and Unstructured Data**

### **6.1 The Hidden Cost**

Fraud and litigation are massive leakage points. This module explores how to use "Alternative Data"—text and graphs—to detect them.

### **6.2 Fraud Detection: The Class Imbalance Problem**

Fraud is rare (e.g., 1% of claims). A model that predicts "No Fraud" every time is 99% accurate but useless.

* **Sampling Techniques:** SMOTE (Synthetic Minority Over-sampling Technique) to create synthetic fraud examples for training.23  
* **Cost-Sensitive Learning:** Modifying the objective function to penalize False Negatives (missing a fraud) 100x more than False Positives.  
* **Unsupervised Anomaly Detection:**  
  * **Isolation Forests:** Identifying claims that are "few and different" in the feature space.  
  * **Autoencoders:** A neural network trained to compress and reconstruct "normal" claims. Claims with high reconstruction error are likely anomalies.

### **6.3 Social Network Analysis (SNA)**

Organized fraud rings involve collisions of connected entities.

* **Graph Construction:** Nodes \= {Person, Car, Doctor, Lawyer, Address}. Edges \= {Shared relationship}.  
* **BiRank / PageRank:** Identifying central nodes. A doctor connected to 500 claimants across different accidents is suspicious.  
* **Community Detection:** Algorithms like Louvain or Leiden to find clusters of colluding entities.24

### **6.4 Litigation Prediction with NLP**

Using Natural Language Processing to read adjuster notes and predict legal escalation.

* **TF-IDF & Embeddings:** Converting text to numbers.  
* **BERT / RoBERTa:** Fine-tuning Large Language Models (LLMs) to classify claim complexity.  
* **Predictive Task:** Binary classification (Litigation Y/N). Early identification allows the insurer to settle quickly, avoiding years of legal fees.26

---

## **Module 7: Customer Analytics – Strategy and Lifetime Value**

### **7.1 Beyond Risk: The Asset Side**

Actuaries are increasingly involved in Customer Relationship Management (CRM). We must understand the value of the customer, not just the risk.

### **7.2 Customer Lifetime Value (LTV)**

$$\\text{LTV} \= \\sum\_{t=0}^T \\frac{P\_t \- C\_t \- E\_t}{(1+i)^t} \\cdot S(t)$$

* **$P\_t$ (Premium):** Forecasted using our GLM/GBM pricing models (Module 3).  
* **$C\_t$ (Cost):** Forecasted using our Frequency/Severity models (Module 2).  
* **$S(t)$ (Retention):** The probability the customer stays.

### **7.3 Churn Prediction and Survival Analysis**

* Survival Analysis: Instead of a simple binary "Churn/No Churn" prediction, we use Cox Proportional Hazards models to predict when a customer will lapse.

  $$h(t|x) \= h\_0(t) \\exp(\\mathbf{x}^T \\mathbf{\\beta})$$

  This allows us to simulate the impact of a price increase on retention duration.  
* **Uplift Modeling:** Identifying customers who are "Persuadable"—they will only stay *if* we offer a discount. This prevents wasting budget on "Sure Things" (who stay anyway) or "Lost Causes".28

### **7.4 Marketing Attribution**

Which ad channel gets credit for the sale?

* **Multi-Touch Attribution (MTA):** Moving beyond "Last Click" attribution.  
* **Markov Chains:** Modeling the customer journey as a sequence of states (Facebook \-\> Google Search \-\> Website \-\> Quote). We calculate the "Removal Effect" of each channel to determine its true contribution to conversion.29

---

## **Module 8: Professionalism, Standards, and Ethics**

### **8.1 The Guardrails**

Technical brilliance without professional governance is dangerous.

### **8.2 Actuarial Standards of Practice (ASOPs)**

* **ASOP 12 (Risk Classification):** How to group risks fairly.  
* **ASOP 23 (Data Quality):** Responsibilities when using "dirty" or third-party data.  
* **ASOP 56 (Modeling):** The gold standard for model governance. It requires the actuary to understand the model's "intended purpose," "limitations," and "dependencies." You cannot just run xgboost and walk away.

### **8.3 Fairness and Bias in AI**

* **Proxy Discrimination:** A "No Race" policy is insufficient if ZIP code or Credit Score proxies for race.  
* **Fairness Metrics:**  
  * *Demographic Parity:* Acceptance rates must be equal across groups.  
  * *Equalized Odds:* True Positive and False Positive rates must be equal across groups.  
* **Mitigation:** Pre-processing (re-weighting data), In-processing (adding fairness regularization terms to the loss function), or Post-processing (adjusting thresholds).

---

## **Recommended Textbooks and Resources**

### **Core Texts (Must-Haves)**

1. **"Statistical Foundations of Actuarial Learning and its Applications"** by **Mario V. Wüthrich and Michael Merz** (2023).  
   * *Relevance:* The definitive text for modern actuarial data science. It provides the rigorous mathematical bridge between GLMs and Neural Networks. Essential for Module 1, 3, and 4\.31  
2. **"Predictive Modeling Applications in Actuarial Science"** (Volumes 1 & 2\) by **Edward W. Frees** et al.  
   * *Relevance:* Published by Cambridge University Press and sponsored by the CAS/SOA. It covers regression, longitudinal data, and practical case studies in R. A primary reference for CAS Exam 8 and SOA Exam PA.33  
3. **"Pricing in General Insurance"** by **Pietro Parodi**.  
   * *Relevance:* A comprehensive guide to the practicalities of the pricing cycle, heavily referenced in IFoA exams.35  
4. **"Loss Models: From Data to Decisions"** by **Klugman, Panjer, and Willmot**.  
   * *Relevance:* The classic text for distributional theory, constructing the EDF, and aggregate loss modeling. Essential for CAS Exam 5 and SOA FAM.36

### **Software Stack**

* **R:** The language of traditional statistics. Packages: tweedie (distributions), ChainLadder (reserving), actuar (loss distributions), mgcv (GAMs).  
* **Python:** The language of machine learning. Libraries: pandas (data), scikit-learn (ML pipelines), xgboost/lightgbm (boosting), shap (explanation), tensorflow/pytorch (deep learning).  
* **Stan:** For Bayesian probabilistic programming.

---

## **Conclusion**

This curriculum represents the convergence of two disciplines. It respects the historical wisdom of the actuarial profession—financial prudence, rigorous probability, and professional ethics—while aggressively adopting the superior predictive capabilities of modern computer science.

By mastering the derivation of the Tweedie loss function alongside the hyperparameters of a Gradient Boosting Machine, the actuary transitions from a "calculator of averages" to an "architect of risk." This course provides the blueprint for that transformation, ensuring readiness not just for the exams of the CAS, SOA, and IFoA, but for the demanding, data-rich reality of the modern insurance marketplace.

#### **Works cited**

1. Full article: Advancing Zero-Inflated Tweedie Models and Evaluating Gradient Boosting Libraries for Auto Claims, accessed November 28, 2025, [https://www.tandfonline.com/doi/full/10.1080/10920277.2025.2454460?src=](https://www.tandfonline.com/doi/full/10.1080/10920277.2025.2454460?src)  
2. Tweedie Loss Function. An example: Insurance pricing | by Sathesan Thavabalasingam, accessed November 28, 2025, [https://sathesant.medium.com/tweedie-loss-function-395d96883f0b](https://sathesant.medium.com/tweedie-loss-function-395d96883f0b)  
3. A Neural Network Approach for Pricing Correlated Health Risks \- MDPI, accessed November 28, 2025, [https://www.mdpi.com/2227-9091/13/5/82](https://www.mdpi.com/2227-9091/13/5/82)  
4. A neural network-based frequency and severity model for insurance claims \- arXiv, accessed November 28, 2025, [https://arxiv.org/html/2106.10770v2](https://arxiv.org/html/2106.10770v2)  
5. Modern Actuarial Statistics I (MAS-I), accessed November 28, 2025, [https://www.casact.org/sites/default/files/2025-05/MASI\_ContentOutline\_2025.pdf](https://www.casact.org/sites/default/files/2025-05/MASI_ContentOutline_2025.pdf)  
6. Modern Actuarial Statistics II (MAS-II), accessed November 28, 2025, [https://www.casact.org/sites/default/files/2025-05/MASII\_ContentOutline\_F\_2025.pdf](https://www.casact.org/sites/default/files/2025-05/MASII_ContentOutline_F_2025.pdf)  
7. Actuarial Statistics (CS1) Core Principles \- Associateship Qualification, accessed November 28, 2025, [https://actuaries.org.uk/document-library/qualify/curriculum/2026-associate-qualification-syllabi/cs1/](https://actuaries.org.uk/document-library/qualify/curriculum/2026-associate-qualification-syllabi/cs1/)  
8. Tweedie regression on insurance claims — scikit-learn 1.7.2 documentation, accessed November 28, 2025, [https://scikit-learn.org/stable/auto\_examples/linear\_model/plot\_tweedie\_regression\_insurance\_claims.html](https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html)  
9. Syllabus/Content Outline Updates \- Casualty Actuarial Society, accessed November 28, 2025, [https://www.casact.org/syllabuscontent-outline-updates](https://www.casact.org/syllabuscontent-outline-updates)  
10. CAS Exam 8 Content Outline 2024 \- FULL | PDF | Actuary | Risk \- Scribd, accessed November 28, 2025, [https://www.scribd.com/document/742865599/CAS-Exam-8-Content-Outline-2024-FULL](https://www.scribd.com/document/742865599/CAS-Exam-8-Content-Outline-2024-FULL)  
11. Claim Frequency Modeling in Insurance Pricing using GLM, Deep Learning, and Gradient Boosting \- Deutsche Aktuarvereinigung e.V., accessed November 28, 2025, [https://aktuar.de/en/knowledge/specialist-information/detail/claim-frequency-modeling-in-insurance-pricing-using-glm-deep-learning-and-gradient-boosting/](https://aktuar.de/en/knowledge/specialist-information/detail/claim-frequency-modeling-in-insurance-pricing-using-glm-deep-learning-and-gradient-boosting/)  
12. Auto XGBoost: Advanced Modeling Simplified for Insurers and Banks \- Earnix, accessed November 28, 2025, [https://earnix.com/blog/auto-xgboost-effortless-integration-of-advanced-modeling-in-production/](https://earnix.com/blog/auto-xgboost-effortless-integration-of-advanced-modeling-in-production/)  
13. GLM, Neural Nets and XGBoost for Insurance Pricing \- Kaggle, accessed November 28, 2025, [https://www.kaggle.com/code/floser/glm-neural-nets-and-xgboost-for-insurance-pricing](https://www.kaggle.com/code/floser/glm-neural-nets-and-xgboost-for-insurance-pricing)  
14. April 2024 Predictive Analytics (PA) Exam – Syllabus \- SOA, accessed November 28, 2025, [https://www.soa.org/4adf8d/globalassets/assets/files/edu/2024/spring/syllabi/2024-04-exam-pa-syllabus.pdf](https://www.soa.org/4adf8d/globalassets/assets/files/edu/2024/spring/syllabi/2024-04-exam-pa-syllabus.pdf)  
15. Advancing the Use of Deep Learning in Loss Reserving: A Generalized DeepTriangle Approach \- MDPI, accessed November 28, 2025, [https://www.mdpi.com/2227-9091/12/1/4](https://www.mdpi.com/2227-9091/12/1/4)  
16. Recurrent Neural Networks for Multivariate Loss Reserving and Risk Capital Analysis \- arXiv, accessed November 28, 2025, [https://arxiv.org/html/2402.10421v1](https://arxiv.org/html/2402.10421v1)  
17. Exam 8 \- Advanced Ratemaking \- Casualty Actuarial Society, accessed November 28, 2025, [https://www.casact.org/sites/default/files/2025-05/Exam\_8\_ContentOutline\_2025\_F.pdf](https://www.casact.org/sites/default/files/2025-05/Exam_8_ContentOutline_2025_F.pdf)  
18. General Insurance (SA3) Specialist Advanced \- Fellowship Qualification, accessed November 28, 2025, [https://actuaries.org.uk/document-library/qualify/curriculum/2026-fellow-qualification-syllabi/sa3/](https://actuaries.org.uk/document-library/qualify/curriculum/2026-fellow-qualification-syllabi/sa3/)  
19. PHASE-TYPE DISTRIBUTIONS FOR CLAIM SEVERITY REGRESSION MODELING | ASTIN Bulletin: The Journal of the IAA \- Cambridge University Press, accessed November 28, 2025, [https://www.cambridge.org/core/journals/astin-bulletin-journal-of-the-iaa/article/phasetype-distributions-for-claim-severity-regression-modeling/C611378A6E2543ED40099160C947F21C](https://www.cambridge.org/core/journals/astin-bulletin-journal-of-the-iaa/article/phasetype-distributions-for-claim-severity-regression-modeling/C611378A6E2543ED40099160C947F21C)  
20. Composite and Mixture Distributions for Heavy-Tailed Data—An Application to Insurance Claims \- MDPI, accessed November 28, 2025, [https://www.mdpi.com/2227-7390/12/2/335](https://www.mdpi.com/2227-7390/12/2/335)  
21. General Insurance \- IFoA, accessed November 28, 2025, [https://actuaries.org.uk/qualify/curriculum/general-insurance/](https://actuaries.org.uk/qualify/curriculum/general-insurance/)  
22. General Insurance Pricing (SP8) Specialist Principles \- Fellowship Qualification, accessed November 28, 2025, [https://actuaries.org.uk/document-library/qualify/curriculum/2026-fellow-qualification-syllabi/sp8/](https://actuaries.org.uk/document-library/qualify/curriculum/2026-fellow-qualification-syllabi/sp8/)  
23. Predictive Analytics For Insurance Fraud Detection \- Wipro, accessed November 28, 2025, [https://www.wipro.com/analytics/comparative-analysis-of-machine-learning-techniques-for-detectin/](https://www.wipro.com/analytics/comparative-analysis-of-machine-learning-techniques-for-detectin/)  
24. Social network analytics for supervised fraud detection in insurance \- ePrints Soton, accessed November 28, 2025, [https://eprints.soton.ac.uk/448443/1/SNAfraud.pdf](https://eprints.soton.ac.uk/448443/1/SNAfraud.pdf)  
25. \[2009.08313\] Social network analytics for supervised fraud detection in insurance \- arXiv, accessed November 28, 2025, [https://arxiv.org/abs/2009.08313](https://arxiv.org/abs/2009.08313)  
26. Framework of BERT-Based NLP Models for Frequency and Severity in Insurance Claims, accessed November 28, 2025, [https://variancejournal.org/article/89002-framework-of-bert-based-nlp-models-for-frequency-and-severity-in-insurance-claims](https://variancejournal.org/article/89002-framework-of-bert-based-nlp-models-for-frequency-and-severity-in-insurance-claims)  
27. AI-powered decision-making in facilitating insurance claim dispute resolution \- Pure, accessed November 28, 2025, [https://pure-oai.bham.ac.uk/ws/portalfiles/portal/211337219/ZhangW2023AI-powered.pdf](https://pure-oai.bham.ac.uk/ws/portalfiles/portal/211337219/ZhangW2023AI-powered.pdf)  
28. Multi-state Modelling of Customer Transitions \- Actuaries Digital, accessed November 28, 2025, [https://www.actuaries.asn.au/research-analysis/multi-state-modelling-of-customer-transitions](https://www.actuaries.asn.au/research-analysis/multi-state-modelling-of-customer-transitions)  
29. Marketing Attribution Models: The Ultimate Guide for 2025 \- Improvado, accessed November 28, 2025, [https://improvado.io/blog/marketing-attribution-models](https://improvado.io/blog/marketing-attribution-models)  
30. What is Attribution Modeling and How Does it Work? \- Hightouch, accessed November 28, 2025, [https://hightouch.com/blog/attribution-modeling](https://hightouch.com/blog/attribution-modeling)  
31. Statistical Foundations of Actuarial Learning and its Applications \- OAPEN Home, accessed November 28, 2025, [https://library.oapen.org/handle/20.500.12657/60157](https://library.oapen.org/handle/20.500.12657/60157)  
32. New Release "Statistical Foundations of Actuarial Learning and its Applications", accessed November 28, 2025, [https://risklab.ethz.ch/news-and-events/risklab-news/2022/11/new-release-statistical-foundations-of-actuarial-learning-and-its-applications.html](https://risklab.ethz.ch/news-and-events/risklab-news/2022/11/new-release-statistical-foundations-of-actuarial-learning-and-its-applications.html)  
33. Predictive Modeling Applications in Actuarial Science: Volume One Casualty Actuarial Society and the Canadian Institute of Actuaries, accessed November 28, 2025, [https://www.casact.org/publications-research/publications/predictive-modeling-applications-actuarial-science-volume-one](https://www.casact.org/publications-research/publications/predictive-modeling-applications-actuarial-science-volume-one)  
34. Predictive Modeling Applications in Actuarial Science: Volume II, Case Studies in Insurance, accessed November 28, 2025, [https://books.apple.com/us/book/predictive-modeling-applications-in-actuarial-science/id1138685262](https://books.apple.com/us/book/predictive-modeling-applications-in-actuarial-science/id1138685262)  
35. Pricing in General Insurance \- 2nd Edition \- Pietro Parodi \- Routledge, accessed November 28, 2025, [https://www.routledge.com/Pricing-in-General-Insurance/Parodi/p/book/9780367769031](https://www.routledge.com/Pricing-in-General-Insurance/Parodi/p/book/9780367769031)  
36. Actuarial Books \- SOA, accessed November 28, 2025, [https://www.soa.org/publications/books/](https://www.soa.org/publications/books/)