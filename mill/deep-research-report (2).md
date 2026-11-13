# General Mills — Senior Data Scientist / AI Interview Preparation Guide

> *Sources validated from Glassdoor, InterviewQuery, LockedIn AI, DataLemur, and candidate-shared interview reviews (2024–2026). Questions are grounded in actual GM interview reports and official job descriptions.*

---

## 1. About the Role & What GM Looks For

General Mills’ Senior Data Scientists (D&T group & GIC) focus on **CPG/FMCG analytics**: demand forecasting, supply chain optimization, price elasticity, consumer insights, and e-commerce personalization. Key skills (from GM JDs and peer reviews) include:

- **Machine Learning:** supervised/unsupervised models, regression/classification, ensemble methods, neural networks.
- **Time-Series & Forecasting:** ARIMA/SARIMA, Prophet, Exponential Smoothing, hierarchical (top-down/bottom-up) forecasting.
- **Operations Research:** Linear/Integer Programming (OR-Tools, PuLP, Gurobi), constraint optimization in supply chain.
- **Generative AI / LLM:** RAG pipelines, embedding search, agentic AI (tool use, ReAct), fine-tuning, prompt engineering, Hallucination mitigation.
- **Cloud & MLOps:** GCP (Vertex AI, BigQuery, Dataflow), Docker/Kubernetes, CI/CD for ML, model monitoring.
- **Domain Knowledge:** CPG supply chain (distribution, warehousing, shelf replenishment), market mix modeling, pricing strategy, marketing mix.
- **Communication:** Storytelling to non-technical stakeholders, translating data findings to business action.

*Experience required:* 10+ years in analytics, 5+ in supply chain or CPG analytics, advanced degree in CS/Stats/Engineering. Fluency in SQL, Python, GCP is a must.

---

## 2. Interview Process Overview

Multiple candidate reports (Glassdoor, LockedIn AI) outline this process:

1. **Screening (30-45 min):** Recruiter or phone screen – resume background, high-level motivation, salary check, basic SQL/Python questions.
2. **Technical Screen – Live Coding (60 min):** Shared coding environment (e.g. Codility or VS Code Live Share). Questions on Python or SQL. Common topics: window functions, time-series feature generation, algorithmic logic (e.g., calculate mutual information). Also discuss ML model evaluation metrics (MAPE, WAPE, RMSE).
3. **Take-Home Case Study (1–3 days):** Exploratory Data Analysis → modeling → business recommendation. Could be predicting sales or optimizing inventory with provided data. Alternatively, critique a colleague’s DS notebook (to test code review skills, as reported on Glassdoor).
4. **On-site/Virtual Panel (Half-day):**
   - **Presentation to Stakeholders:** Present take-home results or proposed solutions to a mixed audience (marketing, supply chain, finance). Emphasize business impact.
   - **Technical Deep Dive:** One-on-one with a senior DS or principal DS. Heavy ML/statistics questions (supervised vs unsupervised, evaluation, advanced modeling, generative AI). Possibly whiteboard coding or pseudocode.
   - **Supply Chain Interview:** With a supply chain planner/manager. Questions on demand planning, forecasting accuracy, OR optimization constraints.
   - **Behavioral Interview (Hiring Manager):** STAR-style questions on leadership, ambiguity, project impact.
   - **Culture Fit/Peer Interview:** With other DS or team leads, focusing on teamwork, GM values (“bold thinking, collaboration”) and adaptability.
5. **(Optional) Personality Assessment:** For leadership-track, may have Hogan or similar.

**Tip:** In early rounds, emphasize core technical strength and curiosity. In later rounds, stress cross-functional communication and business insights (GM highly values **stakeholder trust and Agile delivery**).

---

## 3. Technical Basics You Must Know

Be thoroughly prepared on these fundamentals:

- **ML Concepts:** Bias-variance tradeoff, overfitting vs. underfitting, regularization (L1/L2), and cross-validation strategies (k-fold, stratified, time-series splits). Know ensemble methods (Random Forest vs. Boosting: how they combine trees to reduce variance/bias). Familiarity with scikit-learn, XGBoost/LightGBM, TensorFlow/PyTorch (for advanced roles).
- **Evaluation Metrics:** For classification – accuracy, precision, recall, F1-score, AUC-ROC, AUC-PR. For regression/forecasting – MAPE, sMAPE, WAPE, RMSE, MAE, bias. Understand threshold trade-offs (precision vs. recall) in context (e.g., avoiding stockouts vs. overstocks).
- **Statistics:** Hypothesis tests (t-test, chi-square, ANOVA), p-values, confidence intervals, Type I/II errors. Central Limit Theorem, law of large numbers. A/B testing design: power analysis, significance levels, multiple test correction (Bonferroni, Benjamini-Hochberg). Fixed vs. random effects in mixed models (used for hierarchical or panel data).
- **SQL:** Joins (INNER, LEFT, RIGHT, FULL, CROSS). Window functions: `ROW_NUMBER()`, `RANK()`, `LAG()/LEAD()`, `SUM()/AVG() OVER()`. CTEs, subqueries. GROUP BY, HAVING, and CASE statements. Understand query execution plans; using indexes, filtering early, and query refactoring for performance.
- **Time-Series Forecasting:** Stationarity tests (ADF/KPSS), differencing series for stationarity. ARIMA/SARIMA components (p,d,q)(P,D,Q)[s]. Seasonal decomposition (STL). Prophet (trend/seasonality/holidays). Exponential smoothing models (Holt-Winters). Hierarchical forecasting (top-down, bottom-up, middle-out) and reconciliation (MinT). Always split data chronologically (no random shuffle). Key metrics: MAPE/WAPE (avoid division by zero issues with WAPE).
- **Generative AI / LLM:** Transformer basics (self-attention, positional encoding). RAG (Retrieval-Augmented Generation) pipeline: **chunk documents → embed → vector store → retrieve relevant chunks → feed to LLM**. Fine-tuning vs. prompt/RAG (see differences below). Prompt engineering (zero/few-shot, chain-of-thought, instruction tuning). Evaluate LLM outputs for factuality (RAGAS framework: **Relevance, Answerability, Groundedness**). Multi-agent orchestration (planning, action, observation loops; LangChain, LangGraph, CreAI patterns).
- **Operations Research / Optimization:** Linear Programming, Mixed Integer Programming, constraints. Common problems (transportation, knapsack, scheduling). Python OR tools (OR-Tools, PuLP, Pyomo). Heuristics (greedy, genetic, simulated annealing) for large combinatorial cases.
- **Cloud & MLOps:** Google Cloud Platform – BigQuery (SQL analytics), Vertex AI (training pipelines, hyperparameter tuning, feature store), Dataflow, Pub/Sub. Containerization (Docker), basic Kubernetes. CI/CD pipelines (GitHub Actions, Jenkins, Cloud Build). Model versioning/experiment tracking (MLflow, Vertex AI Experiments). Monitoring (data drift detection, model accuracy over time). Data lineage and governance (Privacy, PII safeguards are especially noted at GM for compliance).

---

## 4. 30+ Interview Questions with Answers

### A. Machine Learning & Modeling

**Q1. Explain the ROC curve and AUC, and why it matters.** *(Common DS interview topic)*

**Answer:** The ROC curve plots the *True Positive Rate (Recall)* against the *False Positive Rate (1 – Specificity)* as the classification threshold varies. It shows the trade-off between correctly identifying positives and incorrectly flagging negatives. The AUC (area under ROC) is a scalar summarizing this performance: 1.0 is perfect separation; 0.5 is random guessing. Crucially, AUC is *threshold-agnostic*, letting you compare models independent of any specific cutoff. This is useful at GM when, for example, deciding customer churn predictors or promotion responders where the business threshold can be set later. A higher AUC means that, across all thresholds, the model ranks true positives higher than false ones more often.

**Answering Strategy:** 
- Define TPR and FPR clearly (e.g., "TPR: fraction of actual positives correctly identified; FPR: fraction of negatives mislabeled as positives").
- Explain how changing the decision threshold moves along the ROC curve.
- Emphasize AUC’s advantage: no need to pre-commit to a threshold, so it gives an overall model quality metric.
- Relate to GM context: e.g., in marketing, choosing threshold later based on ROI, but still wanting an AUC-optimized model.
- **What interviewers check:** clarity on metrics, understanding of trade-offs (Precision/Recall vs ROC), and application relevance. 
- **Pitfalls:** Don’t say “AUC = accuracy.” Also, clarify it’s not sensitive to class imbalance by itself (which can be a trap).
- **Senior vs Junior:** A senior candidate might discuss when ROC is less informative (e.g., extreme class imbalance, prefer Precision-Recall curves) and mention related metrics like AUC-PR.

**Q2. Precision vs. recall – when would you prioritize each?**

**Answer:** Precision measures *“of all positives predicted, how many were correct”*; recall measures *“of all actual positives, how many we caught”*. You prioritize precision when false positives are costly. E.g., at GM, if you’re targeting high-value coupons, precision is key so you don’t waste spend on uninterested customers. Prioritize recall when false negatives are more harmful, e.g. an inventory anomaly detection: missing a true stockout (false negative) is worse than a false alarm, so we want high recall. In practice, the business context drives this choice.

**Answering Strategy:**
- Give concrete definitions with an example (e.g., email spam filter analogy: precision=avoid flagging good mails as spam; recall=catch all spam).
- Provide GM-specific scenarios: high recall for demand spikes (to avoid stockouts) vs. high precision for promo targeting (to save marketing budget).
- Note that often we tune threshold to balance them (show understanding of the precision-recall trade-off curve).
- **What they test:** understanding business costs of errors and metric selection beyond raw accuracy. Also, ability to tie metric to stakeholder priorities.
- **Common mistake:** Ignoring class imbalance or using accuracy as proxy. A senior answer might mention optimization techniques (e.g., using F1 or custom cost functions when balancing both matters).

**Q3. How does logistic regression work? Its assumptions and differences from linear regression?**

**Answer:** Logistic regression models a binary outcome by fitting the log-odds of “success” as a linear combination of features, then applying the logistic (sigmoid) function to output a probability (0–1). Its key assumptions are: (1) linear relationship between features and log-odds, (2) independent observations, (3) no multicollinearity among predictors, (4) sufficient sample size. Unlike linear regression, it doesn’t assume a normal distribution of residuals or equal error variance. It does assume the data are correctly labeled (no label noise) and the classes are well-separated by the predictors in that log-odds space. 

**Answering Strategy:**
- Briefly derive/logistic formula or intuition (logit link).
- Contrast to linear regression assumptions (you must highlight the difference: e.g., “no requirement of normal residuals”).
- Mention how to interpret coefficients (change in log-odds).
- Mention output as probabilities thresholded for class assignment, and how one can set threshold beyond default 0.5 if needed.
- **Interviewer is evaluating** if you truly understand not just the math, but when logistic vs. linear is used (binary vs. continuous target).
- **Pitfalls:** Don’t confuse logistic regression with decision trees or neural nets; and emphasize it’s for classification (not predicting continuous Y).
- **Senior tip:** Could mention regularization (L1/L2) and handling of imbalanced classes (class weights).

**Q4. What is PCA? How is it related to K-Means clustering?** *(This was explicitly mentioned in a GM interview.)*

**Answer:** PCA (Principal Component Analysis) reduces data to orthogonal components that capture maximum variance. It transforms possibly correlated features into uncorrelated principal components (linear combinations of original features). After PCA, K-Means clustering often works better because the data are de-noised and uncorrelated, so Euclidean distance (which K-Means uses) is more meaningful. In fact, K-Means clustering in original space is equivalent to K-Means on the top principal components if data are centered. The advantage: reduced computation and potentially clearer clusters. The disadvantage: PCA components are harder to interpret in business terms, so clusters might become "black boxes" to stakeholders if they only see PC1, PC2 instead of original features.

**Answering Strategy:**
- First define PCA (maximizes variance along orthogonal axes) and K-Means (partitions data by minimizing within-cluster variance).
- Explain how PCA can serve as a preprocessing step for clustering: it removes noise and redundancy (high-dimensional to low-dimensional).
- Use a small example: “imagine 2 highly correlated features – PCA compresses them, then K-Means runs on that 1D projection”.
- **Interviewer cares about:** depth of understanding (not just surface definitions). They explicitly asked this, so they value nuance. Mention the limitation of interpretability.
- **Common oversight:** forgetting to center/scale data before PCA; not mentioning that PCA is unsupervised as is K-Means (no labels).
- **Follow-up to anticipate:** They might ask about how many components to choose or about variance explained vs cluster validity.

**Q5. Describe building a demand forecasting model (for GM products).**

**Answer:** 

- **Problem Framing:** Define scope (e.g., forecast daily/weekly SKU sales per region). Determine horizon and frequency.
- **Data Gathering:** Historical sales from ERP/WMS, inventory levels, promotions, prices, distribution channels, seasonality patterns, holidays, marketing spend, competitor actions, weather (if relevant).
- **EDA:** Plot time series for trends, seasonality (e.g., monthly, weekly, events). Check for promotions spikes, outliers, missing weeks. Stationarity checks (ADF test) to decide differencing.
- **Feature Engineering:** Create lag features (t-1, t-7, etc.), rolling means (to smooth noise), encode promotions (binary flag, discount depth), holidays (special events).
- **Modeling:** Start simple (Naïve, seasonal Naïve). Try classical stats: SARIMA (if adequate data) or Prophet (good for known holidays). Then ML: XGBoost/LightGBM on tabular data with the engineered features often outperforms (GBMs handle many predictors). For ample data, deep learning (N-BEATS, TFT) might capture complex patterns.
- **Validation:** Time-based split (last N periods as test). Use appropriate metrics: MAPE/WAPE (avoiding division by zero) or Mean Absolute Scaled Error (MASE). Check bias (Forecast – Actual). Perform backtesting rolling validation for robustness.
- **Deployment:** Build pipeline (e.g., Vertex AI Pipelines) to retrain periodically. Automate feature extraction from live systems. Monitor drift (when errors consistently rise above threshold). 
- **Business Delivery:** Present results aggregated and SKU-level. Provide confidence intervals or prediction intervals. Highlight actionable insights (e.g., “Forecast is 10% above last year for SKUX during holiday week due to a new TV campaign”).

**Answering Strategy:**
- Structure the answer in clear phases (data, features, model, validation, deployment).
- Emphasize business alignment (what “forecast accuracy” means for inventory, lost sales, etc.).
- Mention how you’d handle special cases: new SKUs (use category analogs), intermittent demand (Croston’s method).
- **Interviewer evaluation:** They want to see end-to-end thinking (not just “I’d train X model”) and awareness of deployment issues (not forgetting data drift).
- **Common pitfalls:** Mixing up metrics (e.g., using classification metrics), ignoring baseline (always mention “compare to a simple baseline like last year’s same period”).
- **Senior edge:** Talk about hierarchical forecasting (if needed, e.g., store-level up to region) or incorporation of causal variables (promotion lift, cannibalization between products).

**Q6. If training and test data distributions differ (covariate shift), what do you do?** *(Glassdoor report mentioned this scenario.)*

**Answer:** This is known as covariate shift or concept drift. Steps to handle it:

- **Detect:** First, quantify the shift by comparing feature distributions (e.g., KS-test, Population Stability Index (PSI), or visualization). Also check target stability.
- **Reweighting:** Use importance weighting: train model but weight training samples by the ratio of test/training distribution (via density estimation or propensity scores).
- **Model Updating:** Retrain more frequently with recent data; use online learning or time-decay weighting. Use a sliding-window validation to ensure model sees the newest patterns.
- **Domain Adaptation:** If extreme, use techniques like CORAL (align covariances) or adversarial domain adaptation to make model invariant to domain.
- **Feature Engineering:** Identify invariant features; drop those that leaked shift. Add explicit “time” or “period” feature to let model learn drift.
- **Validation:** Always use *time-aware* splits to mimic real scenario (never randomize). For deployment, implement drift detection alerts (e.g., model error surge triggers investigation).

**Answering Strategy:**
- Clarify *covariate shift vs. label shift*: here features distribution changed, not the target mapping.
- Mention detection tools (KS-test, PSI) to show data-savviness.
- Highlight that at GM (time-series context), “old vs new” always differ; using most recent data is crucial.
- **Interviewer checking:** knowledge of distribution concepts and practical mitigation. Also, awareness that ignoring this leads to over-optimistic performance.
- **Pitfalls:** Random train/test split in time series (explicitly avoid this mistake). Also, not considering business context that might cause shift (e.g., pandemic effects on sales).
- **Senior nuance:** Could mention robust models (e.g., gradient boosting is somewhat robust, or mention re-training triggers in MLOps pipeline).

**Q7. How do you determine feature importance?**

**Answer:** Feature importance quantifies a variable’s impact on the model. Common methods:

- **Tree-based importance:** Many tree models provide built-in importance (e.g., mean decrease in Gini). Quick but biased to high-cardinality or continuous features.
- **Permutation importance:** Shuffle one feature’s values and measure the drop in model performance. Model-agnostic and more reliable than raw tree importance.
- **SHAP values:** Calculate Shapley values for each feature per prediction to see exact contribution (positive or negative). Best practice for stakeholder explanations because it’s consistent and local (explains individual predictions).
- **Coefficients (linear models):** For linear/logistic regression, standardized coefficients can indicate importance. Assumes features are scaled.
- **PDP/ICE plots:** Partial dependence or Individual Conditional Expectation graphs to see how prediction changes with a feature.

At GM, SHAP or permutation are valuable: e.g. explaining why a certain SKU’s forecast increased (“Holiday season + promotion flag” with SHAP) helps a planner trust the model.

**Answering Strategy:**
- Enumerate methods and their trade-offs (speed vs. bias, global vs. local, etc.).
- Mention why explainability matters: regulatory, business acceptance.
- **Interview highlight:** They want to see you pick the right tool for the job (e.g., “I’d use SHAP if I need precise local explanation for specific inventory decisions”).
- **Common mistake:** Relying solely on coefficient magnitudes without accounting for scaling or collinearity.
- **Edge:** Could note LIME as another local explainer, or write “for neural nets, use integrated gradients or attention scores”.

**Q8. How would you build a recommendation model (or predict churn)?**

**Answer:** *For product recommendations (as GM has e-commerce/retail):* 

- **Content-based:** Compute embeddings for products (ingredients, categories, brand) and users (past purchases, preferences). Recommend items with high cosine similarity to user profile.
- **Collaborative filtering:** Build user-item interaction matrix (purchases, ratings, clicks). Use matrix factorization (ALS, SVD) to learn latent user and item factors. Recommend items with high predicted user-item score.
- **Hybrid:** Combine both to handle cold-start: e.g., use content features for new products.
- **Evaluation:** Offline use metrics like Precision@K, Recall@K, NDCG. Online, A/B test lift in conversion or basket size.
  
*For churn prediction:* 

- Use classification (logistic regression, XGBoost) on features like tenure, engagement (loyalty program use, site visits), past purchase frequency drop. 
- Balance classes via weighting or resampling if churners are rare.
- Tune threshold based on cost: a false negative (missed churner) means lost revenue, false positive costs retention offer.
- **Metrics:** Focus on recall or F1 rather than accuracy if imbalance. Use confusion matrix cost analysis (same concept as precision/recall above).
- **Deployment:** Score users monthly; integrate with marketing to trigger retention campaigns for high-risk.

**Answering Strategy:**
- Clarify which problem (recommendation vs churn) they meant. The question wording suggests either, but it's safe to briefly outline both.
- Cover methods quickly and state why each is used (e.g., collaborative for pattern discovery).
- **Interview key:** Demonstrate product sense (why recommenders matter to GM’s D2C channel), and class imbalance handling for churn.
- **Pitfalls:** Ignoring cold start or using accuracy on imbalanced churn data.
- **Senior angle:** Mention matrix factorization vs. more advanced deep models (e.g., neural collaborative filtering) if relevant.

**Q9. Difference between bagging and boosting?**

**Answer:** Both are ensemble methods to improve base learners, but:

- **Bagging (Bootstrap Aggregating):** Trains multiple models independently on random bootstrap samples and averages (for regression) or votes (for classification). It reduces variance (stabilizes noisy models). Example: Random Forest (bagged decision trees with random feature selection). Strength: robust, parallelizable. Limitation: may still have high bias if base model is weak.
- **Boosting:** Trains models sequentially; each new model focuses on the errors of the previous ones by re-weighting instances. It reduces both bias and variance. Examples: AdaBoost, Gradient Boosting, XGBoost, LightGBM. Tends to have better accuracy on tabular data but is more prone to overfitting if not regularized.

At GM, boosting (XGBoost/LightGBM) is often preferred for tabular analytics (after careful tuning), because it handles complex feature interactions well; bagging (Random Forest) is simpler to tune and less prone to overfit without tuning.

**Answering Strategy:**
- Define each process (independent vs sequential).
- Mention the effect (bagging: reduces variance; boosting: reduces bias and variance).
- Give examples of algorithms.
- **Check they want:** If you know when to use which. Perhaps mention computation (bagging is parallelizable, boosting is inherently sequential but some implementations allow parallel tree building).
- **Common pitfall:** Saying “bagging is random forest vs boosting is XGBoost” without explaining mechanism.
- **Senior detail:** Could mention that boosting can accommodate sample weights and that hyperparameters like learning rate control how much new models learn from previous errors.

**Q10. How do you handle class imbalance in classification?**

**Answer:** Common approaches:

- **Resampling:** Undersample majority or oversample minority (e.g., SMOTE to create synthetic minority examples).
- **Class weights:** In algorithms (e.g., `class_weight='balanced'` in scikit-learn or `scale_pos_weight` in XGBoost) to penalize mistakes on minority more heavily.
- **Threshold adjustment:** Train on probabilistic output, then choose a non-default threshold that balances precision/recall as needed.
- **Ensemble methods:** Use balanced random forest or boosting variants that incorporate imbalance handling.
- **Evaluation:** Use metrics insensitive to imbalance, like F1-score, Matthews correlation coefficient, or ROC-AUC (though ROC-AUC can be misleading if classes are extremely skewed – then use precision-recall AUC).
- **Use case:** If predicting equipment failure (rare event), tune recall. If predicting user churn (rare), consider cost of false alarms vs misses.

**Answering Strategy:**
- Quickly list techniques (sampling, weights, thresholds).
- Explain conceptually why accuracy is not enough here.
- Tie to metric choice (e.g., “GM would never hire a model that’s 98% accurate but always predicts the majority class”).
- **What they evaluate:** Understanding that technical fix is only part; business context will determine how aggressive we need to be with capturing rare class.
- **Mistake to avoid:** Simply oversampling blindly without cross-validation (can overfit).
- **Senior nuance:** Possibly mention anomaly detection techniques if class is extremely rare, or that sometimes a two-stage model (screening + classify) can help.

---

### B. Statistics & Probability

**Q11. What are the assumptions of linear regression? (The “LINE” acronym is a common mnemonic.)** 

**Answer (LINE):**  
- **L**inearity: The relationship between predictors and outcome is linear (in parameters).  
- **I**ndependence: Observations (and errors) are independent; no autocorrelation (check Durbin-Watson in time-series).  
- **N**ormality: The residuals are normally distributed (especially important for confidence intervals on coefficients).  
- **E**qual variance (Homoscedasticity): Residuals have constant variance across all levels of predictors (no pattern in residuals vs fitted plot).  
- Also, no perfect multicollinearity (predictors not exact linear combos of each other). 

These ensure valid inference and interpretation. However, linear regression is fairly robust to mild violations (especially with large sample sizes).

**Answering Strategy:**
- Spell out LINE clearly. If time, give a quick check method (e.g., Q-Q plot for normality, plot residuals for variance).
- Mention if something is violated: e.g., “If non-linearity, consider polynomial terms or transform variables; if heteroscedasticity, use robust standard errors or weighted regression.”
- **Check:** They might want exactly LINE acronym. 
- **Pitfall:** Forgetting independence (time-series often violate it).
- **Senior note:** Could mention exceptions (e.g., GLMs if normality fails, or Ridge/Lasso if multicollinearity is severe).

**Q12. Explain hypothesis testing and A/B test design.** 

**Answer:**  

- **Hypothesis Testing:** Formulate null (H₀: no effect/difference) vs alternative (H₁). Choose significance level (α, e.g., 0.05). Collect data and compute a test statistic (e.g., t or z) to see how extreme the observed effect is under H₀. Calculate p-value (probability of seeing data as extreme under H₀). If p<α, reject H₀. Report confidence intervals (e.g., 95% CI means if we repeated the experiment 100 times, 95 of those CIs would contain the true difference).
- **A/B Test Design:** Define metric (conversion rate, revenue per user, etc.) and baseline. Ensure random assignment to control (A) and test (B) groups to avoid selection bias. Calculate required sample size: depends on baseline rate, minimum detectable effect (lift), α, and power (1–β, e.g., 0.8). Run test until enough samples. Then apply appropriate test (z-test for proportions, t-test for continuous means, χ² for categories), checking assumptions (e.g., normality or sample size for CLT). If multiple tests/variants, apply corrections to maintain overall α (Bonferroni or BH correction).  
- For example, at GM, an A/B test might check if a new recipe label (B) increases sales vs old label (A). The lift needed may be small, so power and sample size are critical.  

**Answering Strategy:**  
- Define H₀/H₁, significance, p-value, type I/II errors, and power. 
- Walk through an A/B setup step by step: hypothesis, sample size, execution, and conclusion. 
- Emphasize pre-planning (no peeking) and post-hoc correction if needed. 
- **What they want:** Candidate should mention power analysis and sample size (often missing). Also, alignment of test metric with business KPI. 
- **Common trap:** Data scientists sometimes skip assumption checks; mention verifying equal variances or using non-parametric tests if needed.
- **Senior add-on:** Could mention sequential testing or bandit designs if advanced, but that’s usually beyond GM’s typical interviews.

**Q13. Random effects vs. fixed effects models – what’s the difference?** 

**Answer:**  

- **Fixed Effects:** Treat group-level characteristics as non-random constants. Useful when you want to control for specific, known groups (e.g., each store or region has its own intercept). Interpretation is limited to those groups in the data (no inference to others). Captures all between-group variation in the intercepts.  
- **Random Effects (Mixed Models):** Assume group-level effects are drawn from a distribution (usually normal). Best when groups (e.g., stores, batches) are a sample from a larger population. Allows inference to new groups. Mixed models include both fixed effects (global predictors) and random effects (group-specific deviations).  

At GM, a mixed model might be used in price optimization: treat each store as a random effect (since there are thousands, we want a generalizable effect), but treat price sensitivity globally. A fixed-effect approach would require separate coefficients per store (if limited stores) and won’t generalize.

**Answering Strategy:**  
- Clarify scenario of panel data or hierarchical data. 
- Relate to variance: fixed-effects soak up all group differences, random-effects share strength across groups. 
- **Interviewer looks for:** Understanding which to use. If asked in context, tie to GM domain (e.g., brand campaigns measured across store chains).
- **Pitfall:** Confusing terminology (some use “fixed effect” generically for any group dummy).
- **Senior insight:** May mention Hausman test (to choose RE vs FE) or that “mixed model” encompasses both.

**Q14. Bayesian vs. frequentist probability – explain the difference and use-case.** 

**Answer:**  

- **Frequentist:** Probability is long-run frequency of events. Model parameters are fixed but unknown. You compute p-values and confidence intervals from hypothetical repeated samples.  
- **Bayesian:** Probability is a degree of belief. Parameters have probability distributions. We start with a prior (e.g., belief about a promotion’s uplift based on past campaigns), update with data (likelihood) to get a posterior distribution. Outputs are credible intervals (region of parameter space with given probability).  

Use **Bayesian** when: incorporating expert knowledge or previous studies (priors), dealing with small data (priors stabilize estimates), or needing full uncertainty quantification (posterior distribution for each outcome). Use **Frequentist** for large-scale A/B tests where p-values and CIs are the norm in the organization. GM might prefer frequentist for standard marketing A/B tests (industry standard), but use Bayesian if modeling under uncertainty (e.g., forecasting for new products with scarce data).

**Answering Strategy:**  
- Provide a simple analogy (e.g., coin flip: frequentist says “50% since 100 flips 50 heads”; Bayesian could incorporate a belief that coin is fair or loaded).
- Give an example of Bayesian advantage (e.g., dynamic Bayesian updating in inventory forecasting).
- **Why they ask:** Demonstrates depth in statistical thinking, not just formula recall. 
- **Pitfalls:** Say you use the approach that fits the team culture. (They often still use frequentist results).
- **Advanced point:** Bayesian credible intervals vs confidence intervals interpretation (probability of parameter in interval vs. probability of data).

**Q15. How to handle outliers in production data?** 

**Answer:**  

1. **Detect:** Methods include statistical (IQR rule: beyond 1.5×IQR from quartiles, z-score > 3) or model-based (isolation forest, DBSCAN for anomalies). Also domain checks (e.g., negative sales, dates in future).
2. **Investigate:** Determine cause. Is it data error (typo, data pipeline issue)? Or a real extreme (e.g., a sudden promo spike or seasonal event)? 
3. **Treat:** 
   - If data error, correct it if possible (e.g., fixing a misplaced decimal) or remove the record. 
   - If genuine, model it explicitly (add a “promo” indicator) rather than removing. For forecasting, you might cap extremely large values or use robust regression (Huber, RANSAC). For classification, mark as separate class or use tree-based models (which are less sensitive).
4. **In Production:** Implement automated checks (e.g., Great Expectations or Deequ rules: “sales >= 0, <= some max based on capacity”). If flagged, route to data stewards or pipeline fallback (e.g., use last valid data). 

At GM, an unusual sales surge might be flagged as a promotional anomaly. We wouldn’t blindly drop it – that’s business insight (maybe put a dummy variable “holiday promo”).

**Answering Strategy:**  
- Outline a systematic approach: detect → categorize → handle. 
- Emphasize communication with domain experts (the anomaly might be an important event).
- **Interview interest:** Approach to data quality shows maturity. 
- **Common mistake:** Ignoring outliers or treating them all the same. 
- **Senior nuance:** Could mention robust metrics (median, IQR) or that some ML algorithms (tree ensembles) inherently downweight outliers.

---

### C. SQL & Data Engineering

**Q16. *SQL:* Get the current salary for each employee, given an ETL appends new salary rows (no updates).**

**Answer:** Use `ROW_NUMBER()` to pick the latest record per employee:

```sql
SELECT first_name, last_name, salary
FROM (
    SELECT 
        first_name,
        last_name,
        salary,
        ROW_NUMBER() OVER (
            PARTITION BY first_name, last_name 
            ORDER BY id DESC
        ) AS rn
    FROM employees
) t
WHERE rn = 1;
```

This subquery partitions by employee (here by name; in practice use employee_id) and orders by the insert `id` (or timestamp) descending. The newest row gets `rn=1`. We then filter to get each employee’s latest salary. 

**Answering Strategy:**  
- Clarify assumptions: “We assume higher `id` = more recent salary entry (or there’s a timestamp column).”
- Mention that partitioning keys should uniquely identify employees (first+last used here for simplicity, but better to use an `emp_id` column).
- **Interviewer check:** Knowledge of window functions vs. GROUP BY. `ROW_NUMBER` is a common solution.
- **Mistake to avoid:** Using `MAX(id)` with join (works but more complex). This approach is clean.
- **Senior note:** Could also mention `RANK()` or `DENSE_RANK()` variations if there are ties, though unlikely if `id` is unique.

**Q17. *SQL:* Calculate a moving average (7-day) of sales per product.**

**Answer:** Use a window frame with `ROWS BETWEEN`. Example:

```sql
SELECT
  product_id,
  sale_date,
  quantity,
  AVG(quantity) OVER (
    PARTITION BY product_id
    ORDER BY sale_date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ) AS mov_avg_7d
FROM daily_sales
ORDER BY product_id, sale_date;
```

This computes a 7-day rolling average (including current day) by partitioning by product and ordering by date. Explicit framing is used to ensure we take exactly the 6 prior days plus current (7 total). 

**Answering Strategy:**  
- Emphasize the use of window functions with `ORDER BY` and `ROWS BETWEEN`.
- A common pitfall is forgetting to specify `ROWS BETWEEN`; without it, `AVG OVER` defaults to the entire partition.
- **What they watch:** Knowledge of window functions and correct frame specification.
- **Senior mention:** If gaps in dates, note this calculates over actual rows, not calendar days – mention how to handle missing dates (generate date series join).

**Q18. *SQL:* Average monthly sales per product.** 

**Answer:** Group by product and month. In PostgreSQL:

```sql
SELECT
  product_id,
  DATE_TRUNC('month', transaction_date) AS sale_month,
  AVG(quantity) AS avg_monthly_quantity
FROM sales
GROUP BY product_id, DATE_TRUNC('month', transaction_date)
ORDER BY product_id, sale_month;
```

This query truncates each `transaction_date` to the first of its month, then groups by product and month to compute the average. It matches the DataLemur solution that “extracts the month… then calculates the average quantity of each product for each month”. 

**Answering Strategy:**  
- Mention the need to use a date function (here `DATE_TRUNC`) to group by month.
- Note ordering by both product and month ensures sorted output.
- **Interviewer interest:** Proper date handling in SQL. Could also mention how to format month (`MM-YYYY` or keep as date).
- **Edge:** For other DBs, the function differs (e.g., `strftime` in SQLite or `TO_CHAR` in Oracle).

**Q19. *SQL:* How do you optimize a slow query?** 

**Answer:** Systematically:
1. **Analyze:** Use `EXPLAIN`/`EXPLAIN ANALYZE` to see where time is spent (full scans, large joins).
2. **Indexing:** Ensure indexes on filter/join columns (WHERE and ON clauses). Covering indexes can speed select queries.
3. **Reduce data early:** Apply `WHERE` filters before joins (in subqueries or CTEs) to limit rows. 
4. **Avoid `SELECT *`:** Only fetch needed columns to reduce I/O.
5. **Rewrite queries:** Replace correlated subqueries with joins/CTEs if possible. Use window functions over self-joins for things like “top N per group”.
6. **Partitioning:** If data is huge (e.g., sales by year), partition tables by date or key.
7. **Materialize intermediate results:** If a CTE is used multiple times, consider temp tables or indexes on it.
8. **Database-specific:** In BigQuery or Redshift, use nested queries vs CTEs differently (CTEs might be optimization barriers). Use clustering/sorting keys. 
9. **Approximate functions:** If 100% accuracy isn’t needed, use `APPROX_COUNT_DISTINCT` or similar.
10. **Server tuning:** As last resort, adjust DB config (memory, parallelism).

**Answering Strategy:**  
- Show a holistic approach (not just one trick). It’s about critical thinking: measure, then improve.
- Mention specific SQL performance tips (e.g., analyzing plans, push down predicates).
- **What they care about:** Ability to debug and improve large data queries. They might drill on one step (e.g., “how to index?”).
- **Pitfalls:** Forgetting to consider data volumes (e.g., indexing a small table won’t help).
- **Senior insight:** A good answer could include an example: “I once sped up a join from 30s to 0.5s by adding a composite index on (date, status) and filtering pre-join.”

**Q20. *SQL:* What is a CROSS JOIN and when would you use it?** 

**Answer:** A **CROSS JOIN** (Cartesian join) pairs every row of Table A with every row of Table B. The result has |A|×|B| rows. It’s rarely needed, but useful for generating combinations. For example, to create a baseline prediction dataset, you might generate all (customer, product) pairs to score purchase probability. In supply chain, you might cross all dates with all SKUs to ensure no date-sku is missing (and fill zeros for no sales). Beware: results grow multiplicatively, e.g. 10,000 customers × 500 products = 5M rows.

**Answering Strategy:**  
- Define it succinctly (pairs everything with everything) and warn of combinatorial explosion.
- Give a concrete use-case: typically for scaffolding or all-pairs analysis.
- **Check:** They want to see understanding of danger (e.g., causing memory issues).
- **Senior note:** Might mention “CROSS JOIN without a WHERE is a full cartesian product; a similar effect can happen accidentally with a missing join condition.”

---

### D. Time-Series Forecasting & Supply Chain

**Q21. What are ARIMA/SARIMA models? How do you choose p,d,q?** 

**Answer:**  

- **ARIMA(p,d,q):** p = autoregressive lags, d = degree of differencing, q = moving-average lags. It models a time series as AR terms plus MA terms on residuals, after differencing d times to achieve stationarity.  
- **SARIMA(P,D,Q)[s]:** Adds seasonal components with seasonal p, D, Q and seasonality period s (e.g., s=12 for monthly or 7 for daily weekly patterns).  

To select parameters: first ensure stationarity (test with ADF/KPSS). The **d** (and D) are set by how many differences needed (often 1). Then inspect ACF/PACF plots: ACF tailing off suggests AR part for q, PACF cutoff suggests p (or vice versa depending on which chart). Or use automated tools (`pmdarima.auto_arima`) to search for lowest AIC/BIC model. 

**Answering Strategy:**  
- Explain what each parameter does and why differencing is needed (for trend/seasonality).
- Mention diagnostic approach (ACF/PACF, information criteria).
- **What they expect:** Knowledge of identifying stationarity and order, not just running auto_arima blindly.
- **Pitfall:** Not addressing seasonality explicitly (must mention SARIMA).
- **Senior detail:** Could compare to Prophet (handles seasonality/holidays differently) or ML alternatives (LightGBM on lag features often beats ARIMA in practice).

**Q22. What is WAPE and why prefer it to MAPE in supply chain?** *(LockedIn AI supply chain interview mentioned WAPE.)* 

**Answer:**  

- **MAPE** (Mean Absolute Percentage Error) = mean(|(actual – forecast)/actual|). Problem: if actual is 0 (common for new or out-of-stock items), MAPE is undefined or huge.  
- **WAPE** (Weighted Absolute Percentage Error) = sum(|actual – forecast|) / sum(actual). It weights errors by actual volumes, effectively giving a volume-weighted error. This avoids divide-by-zero issues and aligns errors to business impact (errors on high-volume SKUs count more). For GM’s supply chain, WAPE is preferred because we care about aggregate forecast accuracy (e.g., total volume error) and must handle zeros gracefully (e.g., new product launches have no history).

**Answering Strategy:**  
- Define both metrics. Stress that WAPE uses totals rather than per-period percentages.
- Note business interpretation: e.g., a 10% error on a 1M-case product hurts more than on a 100-case product.
- **Interview angle:** Shows supply chain domain awareness (they explicitly asked this).
- **Pitfall:** Saying “we never use MAPE” – better to say WAPE is better when zeros appear.
- **Senior note:** Could mention sMAPE (symmetric APE) as another variant, but GM specifically uses WAPE historically.

**Q23. How to model seasonality and promotional effects in demand forecasting?**

**Answer:**  

- **Seasonality:** Use indicators or features: sine/cosine Fourier terms or seasonal dummies (month, week-of-year). Decompose data (STL) to isolate seasonal component or let models capture it (Prophet has built-in seasonality). For multiple seasonalities (daily+weekly+yearly), consider TBATS or NeuralForecast.
- **Promotions:** Include dummy flags (`is_promo`), promotion depth (% discount), and promotion type (BOGO, display ads) as features. Create lagged promo features (effects can carry over). Use a separate model or a hierarchical approach where baseline forecast is adjusted by a learned "promo lift factor".
- **Special events:** Tag known holidays (Thanksgiving, Ramadan for certain markets) or major launches with separate features. Models like Prophet allow adding holiday regressors.
- **Evaluation:** Always compare promo periods to the baseline model’s counterfactual (causal inference if possible). For a predictive model, ensure you have non-promo training data; consider excluding heavy promo periods when fitting a baseline demand model, then add promo effect explicitly.
- For new SKUs with no history, use category/brand analogs or Bayesian shrinkage to incorporate group-level priors.

**Answering Strategy:**  
- Distinguish how to encode season (regular, seasonal decomposition) vs. promotional (external events). 
- Emphasize encoding rather than trying to “remove” these patterns: the model should see them as signals.
- **What they check:** Domain-specific approach (promotions are huge in CPG); a generic answer (“seasonality is periodic”) is not enough.
- **Common oversight:** Forgetting that promotions change demand in a non-linear way (e.g., saturating effect).
- **Edge:** Could bring up uplift modeling or difference-in-differences if asked how to measure true promotional lift from observational data.

**Q24. Explain hierarchical forecasting (top-down, bottom-up, middle-out).**

**Answer:**  

- **Bottom-up:** Forecast at the most granular level (SKU × store), then sum up to higher levels. Captures local patterns well but is noisy (each low-level forecast has error).
- **Top-down:** Forecast at total (company or category) level, then allocate down by historical proportions (e.g., this brand historically accounts for 30% of portfolio). Stable aggregate forecast but may miss local trends.
- **Middle-out:** Forecast at an intermediate level (e.g., category-level by region), then disaggregate up and down. Balances granularity and stability. 
- **Reconciliation:** Regardless, ensure forecasts sum consistently (e.g., using MinT or bottom-up adjustments). 

**Trade-offs:** Bottom-up uses all data but is sensitive to noise and can overfit. Top-down is smoother but assumes stable proportions. Middle-out is a compromise (often used by brand managers). GM likely uses middle-out (brand × division) and then “top-down from brand aggregates” to reconcile with total sales.

**Answering Strategy:**  
- Clearly define each approach and its pros/cons.
- Tie to business: e.g., “If divisional GM gives brand targets, then middle-out fits that frame.”
- Show awareness of coherence (forecasts should add up correctly).
- **Watch for:** Mentioning practical issues: bottom-up can’t forecast new SKUs well; top-down ignores store-level differences (one region may season differently).
- **Senior note:** If asked further, discuss “optimal reconciliation” (MinT) or cross-series models (like Prophet with grouping).

**Q25. Why is a random train-test split wrong for time-series?**

**Answer:** Random splitting in time series causes data leakage: future values end up in training, so model learns “future information” and performance is unrealistically high. This violates how the model will be used (always forecast ahead in time). Instead, always split by time (e.g., train on Jan–Nov and test on Dec). For cross-validation, use time-series CV (rolling windows), where each fold’s training period is earlier in time than its test period. This better simulates actual forecasting conditions.

**Answering Strategy:**  
- Highlight “time order is sacred” concept.  
- Mention specific methods: “TimeSeriesSplit” or expanding windows.  
- **They expect:** Immediate recognition that chronological order matters.  
- **Pitfall:** Thinking you can shuffle time-series like cross-sectional data.  
- **Senior touch:** Could mention backtesting residuals or using walk-forward validation.

---

### E. Generative AI / LLM / RAG / Agentic AI

*(Note: this is explicitly required in the 2026 GM job description for senior roles.)*

**Q26. What is RAG (Retrieval-Augmented Generation)? Describe the pipeline.** *(Listed in GM Senior DS JD.)*

**Answer:**  
RAG augments an LLM with external knowledge retrieval. Pipeline steps:

1. **Document Ingestion:** Gather relevant corpora (e.g., company docs, manuals, contracts). Pre-process text (cleaning, OCR if needed).
2. **Chunking:** Split long texts into manageable chunks (e.g., ~500 tokens) to fit model context. Can use fixed token counts or semantic splits (e.g., paragraphs or overlapping chunks).
3. **Embedding:** Convert chunks into vector embeddings using a sentence transformer or embedding model (e.g., OpenAI’s text-embedding-3 or similar).
4. **Vector Store/Index:** Store these embeddings in a database (FAISS, Pinecone, ChromaDB, or BigQuery/AlloyDB for GCP). Include metadata (document ID, source).
5. **Retrieval:** When a query arrives, embed the query and retrieve top-k similar chunks via cosine similarity or ANN search. Possibly combine with a traditional keyword search (hybrid).
6. **Generation:** Provide retrieved chunks to the LLM (like GPT-4o or Gemini) in the prompt as context, then ask it to answer the query. This anchors the response to actual content, reducing hallucination.
7. **Post-processing:** Optionally, have the model cite sources or formulate final answer with references. Evaluate output with RAGAS metrics (Relevance, Answer, Groundedness).
8. **Monitoring:** Track which documents are used and user feedback (thumbs up/down) to refine the index or prompt strategy.

*Example:* GM product team queries “Show me the ingredients in the new cereal recipe.” The RAG system finds the specification sheet chunk and has the LLM answer using that ground truth.

**Answering Strategy:**  
- Step through each component clearly (ingest → embed → retrieve → generate).  
- Emphasize why retrieval helps (covers up-to-date enterprise knowledge vs static LLM).
- **Interviewer focus:** Understanding of end-to-end architecture and limitations (need to keep index updated for freshness).
- **Pitfalls:** Not mentioning chunking or that vector store is needed. 
- **Senior detail:** Could note index update pipeline (how to re-embed on new doc arrivals), and evaluation (RAGAS ensures answers are faithful).

**Q27. Fine-tuning vs. RAG – when to use each?**

**Answer:**  

| Aspect            | Fine-tuning                    | RAG                            |
| ------------------| ------------------------------|--------------------------------|
| **Approach**      | Update model weights on domain data. | Keep base model fixed, augment with retrieved context. |
| **Use-case**      | Specialized tasks/format (e.g., model outputs JSON in required schema, or captures company voice/brand rules) or when you have many labeled examples. | Info retrieval tasks (e.g., Q&A on private docs, latest data) where updating weights for new facts is impractical. |
| **Data needs**    | Labeled examples (question-answer pairs). | Unlabeled corpora to retrieve from. |
| **Pros**          | Directly embeds knowledge/ style. | Always uses up-to-date info; lower risk of stale or incorrect generation. |
| **Cons**          | Expensive (GPU time), knowledge gets out-of-date after cut-off. | Requires maintenance of vector DB; can still hallucinate if context is insufficient. |

At GM, use **RAG** for retrieving from dynamic sources (e.g., fresh supply-chain reports, HR policies). Use **fine-tuning** when you need the model to reliably follow a template or reasoning chain (e.g., a chatbot that formats answers as structured JSON for an app), and you have enough quality examples to train on.

**Answering Strategy:**  
- Compare key differences in bullet or table form (as above).
- Give scenarios from GM domain to illustrate (fresh product info = RAG; specialized regulatory compliance checker = fine-tune).
- **They test:** Knowledge of LLM operations, cost trade-offs.
- **Pitfalls:** Over-simplifying (both have hallucination risks).
- **Senior tip:** Might mention parameter-efficient fine-tuning (LoRA) or prompt-engineering as lighter alternatives.

**Q28. What is an Agentic AI system? How does it differ from a standard LLM call?** *(GM JD requires Agentic AI expertise.)*

**Answer:**  

An **Agentic AI** (AI Agent) augments an LLM with tools, memory, and planning. Instead of a single prompt-response, the agent interacts in a loop:
1. **Observe:** Agent receives input (e.g., a user query or environment data).
2. **Reason/Plan:** The LLM (or a separate planning LLM) decides on next actions (e.g., “call database”, “compute sum”, “search web”).
3. **Act:** Agent executes a tool/API call (SQL query, API fetch, calculator, web search) using the plan.
4. **Receive:** Gets results from tools, appends to context.
5. Repeat reasoning and acting until final answer or goal achieved.

Example patterns: **ReAct** (chain-of-thought with API calls), **Plan-and-Execute** (two LLMs: one plans a sequence of actions, one executes them), **Multi-agent orchestration** (Supervisory agent delegates tasks to specialized sub-agents).

This differs from a standard LLM: the base LLM just predicts text in one shot. An agent can perform computation or I/O during generation. For GM: an agent might parse a user’s request (“Generate a purchase order for SKU X”) → retrieve current inventory via an API, calculate optimal order via linear programming tool, and then draft a PO document, all orchestrated step-by-step.

**Answering Strategy:**  
- Clearly delineate “Agent vs plain LLM”. Emphasize the tool-calling loop.
- Mention examples of tools (database, HTTP API, code execution).
- **What interviewers look for:** Awareness of modern AI techniques and architecture (especially if listed in JD).
- **Pitfall:** Confusing “multi-step prompt” (still not an agent unless it calls external tools).
- **Senior nuance:** Could mention frameworks like LangChain or AutoGen that support agents.

**Q29. What are hallucinations in LLMs, and how to mitigate them in production?**

**Answer:**  

**Hallucinations** are plausible-sounding but false or unsupported outputs by an LLM. Mitigation strategies:

- **Retrieval Grounding (RAG):** Provide context docs so answer is anchored in facts (and force citation of sources).
- **Constrained Decoding:** Use lower temperature (0–0.2) to make model more deterministic on factual tasks. Use stop sequences or JSON schema to constrain output format.
- **Self-Consistency:** Generate multiple answers and choose the majority answer for factual queries.
- **LLM-as-Judge:** Use a second LLM to compare answer vs. sources to flag hallucination (e.g., RAGAS scoring).
- **Prompt Engineering:** Clearly instruct the model to admit uncertainty (“If not sure, say “I don’t know” and provide no answer”).
- **Tool/Augmentation:** Use calculators for math, databases for facts (agentic approach).
- **Human-in-the-Loop:** For high-stakes answers (like regulatory or contract info), require human verification before final use.
- **Monitoring:** In deployment, track when the model’s confidence (if available) is low or when user feedback flags false info.

At GM, for something like a supplier contract Q&A, we'd use RAG to ensure each statement is backed by a contract clause and likely have a compliance reviewer double-check the final answer.

**Answering Strategy:**  
- Provide definition and why it’s a concern (ER risk or brand trust at GM).
- List concrete mitigations (especially RAG with citations, as relevant to GM’s pipeline).
- **Focus:** The interviewer likely expects RAG + evaluation metrics (e.g., mention RAGAS) because it’s recent research and in GM JD.
- **Avoid:** Over-reliance on “more training data solves it”. The point is production readiness.

**Q30. (System Design) Sketch an end-to-end RAG pipeline for GM’s supply chain knowledge base.**

**Answer:**  

```
1. DATA INGESTION
   - Sources: ERP manuals, supplier contracts, SOP documents, product specs.
   - Process: OCR/Doc AI for PDFs → text extraction (via GCP Document AI for scanned files).

2. CHUNKING
   - Method: Recursive splitter ~500 tokens, with ~10% overlap to preserve context.
   - Special handling: Keep each table or bullet list in one chunk if possible.

3. EMBEDDING & INDEXING
   - Model: GCP Vertex Embeddings (e.g., text-embedding-gecko-001) for high-quality vectors.
   - Database: AlloyDB (as vector DB) or Pinecone/Chroma for similarity search.
   - Metadata: Add tags (doc type, date, keyword tags like “inventory”, “lot size”).

4. RETRIEVAL
   - Query embed + ANN search (cosine similarity, top k=5).
   - Hybrid: also perform keyword search (BigQuery full-text) to catch exact part numbers or legal terms.

5. GENERATION
   - LLM: Use Gemini 1.5 or GPT-4o on Vertex AI.
   - Prompt: System + “Given the following relevant excerpts, answer the question. Provide citations.”
   - Ensure response includes source references (RAGAS).
   - Possibly format output as JSON for downstream tools.

6. EVALUATION (RAGAS)
   - Groundedness: Check if claims match cited chunks.
   - Relevance: Did it answer the user’s question?
   - Context Precision/Recall: How much of provided context was actually used.

7. MLOps & Monitoring
   - Vertex AI Pipelines: Automate reindex when new docs arrive (monthly or event-driven).
   - Logging: Track query latency, retrieval click-through.
   - Feedback: User thumbs up/down to improve retriever tuning.

8. SECURITY/Scalability
   - Run retrieval and generation in VPC with IAM.
   - Horizontal scaling on heavy query load.
   - Caching common queries.
```

**Answering Strategy:**  
- Outline each component clearly and logically (ETL → index → query → answer).
- Mention specific GM context: what docs (SOPs, product sheets), and relevant tech (GCP services as GM uses).
- **Check:** They care about architecture sense: modularity, monitoring, evaluation.
- **Pitfall:** Too vague. Include specifics (model names, vector DB, etc.) to show applied knowledge.
- **Senior insight:** Could note trade-offs (embedding dimensionality vs speed, or on-prem vs cloud storage for PII docs).

**Q31. What are embeddings? How used in NLP and recommender systems?**

**Answer:**  

Embeddings map discrete entities (words, sentences, products, users) into dense vectors in ℝⁿ such that semantic similarity corresponds to vector similarity. In NLP, word embeddings (Word2Vec, GloVe) or contextual embeddings (BERT, OpenAI embeddings) capture meaning: similar words/sentences have similar vectors. They enable similarity search or as features for downstream tasks. 

In recommendation, embeddings are critical. Content-based recommenders embed items (e.g., product features) so that similar products are near. Collaborative filtering (matrix factorization) learns item and user embeddings so that dot(user, item) predicts preference (e.g., implicit factor models like ALS). Hybrid systems may fuse these (e.g., two-tower models with user & item text). 

For GM: we might embed product descriptions and user purchase histories. New product launches (cold start) can get an embedding from their feature vector, enabling recommendations even without historical sales.

**Answering Strategy:**  
- Define "embedding" clearly (vector representation of concepts).
- Give NLP example (word vectors) and RS example (user/item factors).
- Highlight why embeddings are useful (semantic similarity, dimensionality reduction).
- **Interviewer interest:** They want to see you know how modern similarity search works behind the scenes (relevant to RAG as well).
- **Pitfall:** Only describing word2vec, not linking to recommender context.
- **Edge:** Could mention vector search (FAISS) and nearest-neighbors used in both NLP and RS.

---

### F. Python Coding

**Q32. *Python:* Compute mutual information (MI) between two discrete variables.** *(Glassdoor – GM codility test)*

```python
import numpy as np
from collections import Counter

def mutual_information(x, y):
    """Compute mutual information between two discrete variables."""
    assert len(x) == len(y), "Variables must have same length"
    n = len(x)
    joint_counts = Counter(zip(x, y))
    x_counts = Counter(x)
    y_counts = Counter(y)
    mi = 0.0
    for (xi, yi), count in joint_counts.items():
        p_xy = count / n
        p_x = x_counts[xi] / n
        p_y = y_counts[yi] / n
        mi += p_xy * np.log(p_xy / (p_x * p_y))
    return mi

# Example
x = [1, 0, 1, 1, 0, 0, 1, 0]
y = [1, 0, 1, 0, 0, 1, 1, 0]
print(f"MI: {mutual_information(x, y):.4f}")
```

This code computes $\sum_{x,y} p(x,y) \log\frac{p(x,y)}{p(x)p(y)}$. It counts joint and marginal frequencies, then uses the MI formula. 

**Answering Strategy:**  
- Use a clear, commented function. Check dimensions (len match).  
- Explain key steps: building joint/marginal distributions.  
- **Interview check:** Efficiency (Counters for O(n) passes) and numerical stability (log of division, fine for probabilities >0).  
- **Pitfalls:** Forgetting to normalize counts; not handling zero probabilities (here avoided since log is only taken on observed joint pairs).
- **Senior tip:** Might mention how to handle continuous data (binning or using sklearn’s `mutual_info_score` for discrete).

**Q33. *Python:* How to handle missing values in a pipeline?** *(GM manager round)*

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def handle_missing(df: pd.DataFrame, threshold: float = 0.4) -> pd.DataFrame:
    """
    Handles missing values:
    - Drops columns with >threshold fraction missing.
    - Numeric: median imputation.
    - Categorical: mode imputation.
    """
    df = df.copy()
    # Drop high-missingness columns
    miss_frac = df.isna().mean()
    drop_cols = miss_frac[miss_frac > threshold].index.tolist()
    df.drop(columns=drop_cols, inplace=True)
    # Separate numeric and categorical
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    # Impute numeric with median
    if len(num_cols) > 0:
        imp_num = SimpleImputer(strategy='median')
        df[num_cols] = imp_num.fit_transform(df[num_cols])
    # Impute categorical with most frequent
    if len(cat_cols) > 0:
        imp_cat = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = imp_cat.fit_transform(df[cat_cols])
    return df
```

This function drops any column where >40% values are missing (threshold configurable). Then, numeric columns get median imputation (robust to outliers) and categoricals get mode imputation. More advanced pipelines might chain `sklearn.pipeline` steps or use `KNNImputer` for numeric correlation.

**Answering Strategy:**  
- Show use of `sklearn.impute` for clarity. 
- Mention *why* median for numeric (outliers) and mode for categorical. 
- Address the high-missing cutoff (discuss with stakeholder: too much missing = unreliable feature).
- **Interviewer aspect:** Whether you mention maintaining consistency (fitting on train only, then applying to test). Here, a simple function gets the idea across.
- **Pitfalls:** Ignoring new categories in test; forgetting to fit on train subset. 
- **Senior nuance:** Could discuss more complex methods (multiple imputation, using algorithms that handle NaNs, e.g., XGBoost’s internal handling).

**Q34. *Python:* Given a time-series DataFrame, create lag and rolling-mean features.** *(Supply chain screen, LockedIn AI)*

```python
import pandas as pd

def create_ts_features(df: pd.DataFrame, date_col: str, group_col: str, target_col: str,
                       lags=[1,2,4,8], windows=[4,8]) -> pd.DataFrame:
    """
    Adds lag and rolling mean features for each group.
    """
    df = df.copy()
    # Ensure sorted for accurate shifts
    df.sort_values([group_col, date_col], inplace=True)
    
    # Lag features
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df.groupby(group_col)[target_col].shift(lag)
    # Rolling mean features
    for window in windows:
        df[f'{target_col}_roll_{window}'] = (
            df.groupby(group_col)[target_col]
              .transform(lambda x: x.shift(1).rolling(window).mean())
        )
    # Drop rows with NaN due to shifting
    df.dropna(inplace=True)
    return df

# Example usage:
# df = create_ts_features(sales_df, date_col='week', group_col='SKU', target_col='sales')
```

We use `groupby` and `shift` to create lag features per group (e.g., SKU). The rolling mean uses `shift(1)` to avoid using the current row in its own rolling window. Finally we drop NaNs (common for initial periods).

**Answering Strategy:**  
- Ensure to sort by date within each group.  
- Show `groupby.shift()` for lags and `.rolling()` for moving statistics.  
- **They check:** Grouping logic is correct (so features don’t mix between SKUs).
- **Pitfall:** Forgetting `shift(1)` inside rolling (double-counting current row).
- **Senior detail:** Could extend to multiple group keys (SKU, store) or other aggregates if needed.

---

### G. Behavioral & Stakeholder Management

**Q35. Describe a time you made a decision with incomplete information.** *(Glassdoor #1 reported GM DS interview Q)*

**Answer (STAR):**  
- **Situation:** In the annual demand plan submission, I discovered 3 months of sales data missing for 200 SKUs (ERP migration glitch) with deadline looming.  
- **Task:** Provide forecasts anyway, with an acceptable confidence level for finance.  
- **Action:** I used a multi-pronged approach:
   1. **Borrowed Similar-SKU Data:** Clustered SKUs by brand/category, pulled data from nearest analogs, and scaled by historic month-over-month rates.
   2. **Bayesian Priors:** Incorporated category-level growth priors from past years as Bayesian priors for missing series, updating them with whatever partial data we had.
   3. **Uncertainty Flagging:** Outputted wide confidence intervals for those SKUs, and marked them for manual review.
- **Result:** Finance accepted the forecasts, acknowledging higher uncertainty. After launch, we tracked that those SKUs had a 20% forecast error (vs. 10% for full-history SKUs), which was understandable. The “proxy forecasting” method was documented and later automated for any new-SKU cold starts.

**Answering Strategy:**  
- Use STAR: clearly state Situation/Task/Action/Result with metrics.  
- Highlight risk mitigation and communication (“flagged uncertainty”).
- **What they look for:** Comfort with ambiguity, structured thinking, balancing speed vs. accuracy. 
- **Tip:** Emphasize collaboration (e.g., did I loop in supply chain planners?), though space is short.
- **Senior element:** Show learning – “we later built a formal cold-start model from this experience.”

**Q36. How to handle dirty/bad data in a production model?**

**Answer:**  

- **Prevention & Monitoring:** Implement validation rules (schema checks, value ranges) as early as possible (e.g., using Great Expectations).
- **Detection:** Monitor incoming data stats (e.g., PSI, statistical process control) to catch sudden shifts or anomalies.
- **Fallback Logic:** In scoring pipeline, if a check fails (e.g., X is negative where it should be positive), either use last known good value or a default (e.g., historical average) rather than breaking the model.
- **Alerts:** Generate alerts (Slack/email) for significant data quality drops.
- **Retraining/Correction:** If bad data is fed inadvertently, block retraining on it until fix. Document issues and fix upstream (bug in ETL, vendor data glitch, etc.).
- **Communication:** Have a quick manual override (e.g., switch to a simpler baseline model) if data corruption can't be resolved immediately.

**Answering Strategy:**  
- Emphasize *defense-in-depth*: upfront checks + ongoing monitoring.
- **Interview focus:** Practical DevOps/MLOps mindset (not just “filter out outliers”).
- **Common response to improve:** Mention specific tools or frameworks (Datafold, Deequ, Airflow DAGs).
- **Senior perspective:** Might discuss “DataOps” team collaboration and maintaining trust in predictions through transparency when data issues occur.

**Q37. Describe a project where you used optimization (OR) to solve a business problem.**

**Answer (STAR):**  
- **Situation:** GM had 15 warehouses and 500 retail customers with manual replenishment, causing high freight costs (22% overweight capacity).  
- **Task:** Automate network allocation to minimize transportation cost while meeting service levels.  
- **Action:** I formulated a Mixed Integer Linear Program: Minimize cost subject to supply = demand, warehouse capacity, and service-day constraints. Implemented in Python with Google OR-Tools CP-SAT solver. For small instances, CP-SAT found exact optimum. For daily large runs, we used a heuristic (greedy initial solution + local search) to get near-optimal quickly. The model ran nightly to set next-day allocations.  
- **Result:** 14% reduction in freight costs and improved fill rates by 5%. Stakeholders (logistics managers) were thrilled; the solution scaled to 1000+ SKUs automatically. This project was deployed on Vertex AI pipelines to re-optimize each week as demands changed.

**Answering Strategy:**  
- Clearly articulate objective, variables, and constraints in writing (but briefly).
- Highlight tools (OR-Tools, Gurobi) and any performance tuning needed.
- **What they seek:** Ability to translate business constraints to math formulation, and pragmatic solving (using heuristics when exact is too slow).
- **Pitfall:** Avoid just “we used linear programming” – explain the business terms (capex, service levels, etc.).
- **Senior extra:** Could mention decomposition if network was huge, or risk modeling in objective.

**Q38. How do you prioritize projects with multiple stakeholders?**

**Answer:**  

1. **Impact vs Effort:** Score projects on expected business impact (revenue/ROI, risk mitigation, strategic importance) and complexity (data availability, stakeholder buy-in). Use a simple matrix to align with leadership on priorities.
2. **Stakeholder Alignment:** Early discussions with each sponsor to understand their goals (RACI chart). Often different functions (Sales, Finance, Ops) have different KPIs – seek overlap.
3. **Agile Iteration:** Break projects into 2-week sprints delivering incremental value (e.g., initial forecast dashboard or a prototype model). This builds trust.
4. **Roadmap & Transparency:** Maintain a visible roadmap (e.g., in JIRA/Confluence) with status, so everyone knows what’s next. Update stakeholders weekly.
5. **Say No Tactfully:** If resources are limited, be prepared to explain the trade-off: “To build Model B, we need to defer analysis A, which is a $X opportunity vs. $Y”.

**Answering Strategy:**  
- Use a framework answer (Impact/Effort, Agile, communication).
- Emphasize stakeholder management as much as technical skill – GM values “develop trust with business leaders”.
- **They check:** Whether you understand business value, not just coding everything.
- **Mistake:** Saying “I do whatever they tell me” – you must have a systematic approach.
- **Senior view:** Might mention KPIs for your own team or organization (OKRs) to align projects.

**Q39. How do you explain a complex model to non-technical stakeholders?**

**Answer:**  

- **Business-first narrative:** Start with what problem the model solves (e.g., “This model helps us decide which promotion will yield 15% more sales”) rather than “I used XGBoost with 100 trees.” 
- **Simple visuals:** Use charts/plots. For instance, SHAP plots highlight which features (e.g., price, season) drove a forecast up. Visual examples resonate more than tables of numbers.
- **Analogies:** Compare the model to something familiar (e.g., “Think of it like an experienced planner who’s reviewed 10,000 product histories.”).
- **Highlight uncertainties:** Always present confidence intervals or expected error range (e.g., “We’re 95% sure the sales will be between 500 and 550 units”), so decisions account for risk.
- **Check understanding:** Encourage questions. Sometimes sketching out “This feature was most important, meaning if it changes, our forecast shifts by X%”.

**Answering Strategy:**  
- They want to see communication skills. State how you *translate* tech to business.  
- Mention specific tools: “I use SHAP or partial dependence plots because they’re intuitive”.
- **Common trap:** Talking in jargon (VIP vs MVP).
- **Senior addition:** Possibly note localization (explain at different detail levels to exec vs analyst audience).

**Q40. (On-site) Walk through a given DS notebook and critique it.** *(Glassdoor confirmed question.)* 

**Approach:**  
Interviewers often give a partially completed Jupyter notebook. They expect you to review it for best practices and correctness. Key points:

- **Data handling:** Check if missing values or duplicates are handled. Are datasets merged correctly? Look for data leakage (e.g., if they normalized before a train/test split, that’s a red flag).
- **Exploratory Analysis:** See if they did sufficient EDA. Are distributions and outliers examined? Correlations? If not, mention it.
- **Feature engineering:** Are new features created sensibly? Or are raw IDs/features used without justification?
- **Modeling:** Check that train/test split is appropriate (no shuffle for time-series). Look at hyperparameter tuning (grid search CV?). Is there a baseline comparison (like a simple model or naive forecast)? 
- **Evaluation:** Are the metrics suitable for the problem? (e.g., using accuracy on imbalanced data is wrong).
- **Code quality:** Is code modular, documented, without magic numbers? Are random seeds set for reproducibility?
- **Interpretation:** Does the notebook just spit results, or does it conclude with actionable insights? Did they answer the business question or just minimize error?
- **Recommendations:** At the end of review, suggest improvements: e.g., “I’d add cross-validation here, and consider [X] metric for business alignment.”

**Answering Strategy:**  
- Methodically go section by section in your description. 
- Point out at least one positive and one improvement in each area (data, features, model, evaluation).
- **They check:** Critical thinking and coding standards, not just raw answers.
- **Red flags:** Directly point out if you see leakage or misused metrics. They want to hear your thought process: e.g., “I see no validation set—this means their accuracy likely doesn’t reflect true performance.”
- **Senior stance:** Demonstrate understanding of software practices: “We should break this into functions or use a pipeline, rather than one long notebook.”

---

## 5. Tips to Ace the GM Interview

- **Know the GM Business:** Tie examples to CPG context (supply chain, forecasting, promotions). Read GM news (recipes, new products) to sound credible.  
- **STAR Format:** Always structure behavioral answers (Situation, Task, Action, Result) with concrete outcomes (numbers if possible).  
- **Stakeholder Communication:** Practice explaining technical work in plain language. GM explicitly values “educating stakeholders on analytic practices”.  
- **Prepare Deep-Dive Projects:** They will probe your past projects. Be ready to explain why you chose a method, how you validated it, what you’d improve.  
- **Cloud Fluency:** GM uses GCP heavily. If your experience is AWS/Azure, know the equivalents (BigQuery vs Redshift, Vertex AI vs SageMaker).  
- **Generative AI Savvy:** The JD cites RAG/LLM/Agentic AI as expert-level. Be prepared to discuss any experience with embeddings, prompt engineering, or agents.  
- **Practice Coding:** GM technical screens use platforms like Codility. Brush up on Python/SQL fundamentals: sliding windows, data structures, and common interview algorithms (like mutual information calculation or custom probability functions as seen in their tests).  
- **Balanced Metrics:** Never just default to accuracy. Demonstrate using appropriate metrics for the problem (precision/recall, WAPE, ROC-AUC, etc.).  
- **Baselines and Causality:** Always mention a simple baseline (like “last year’s sales”) and be clear about correlation vs causation (important in A/B tests and marketing analytics).  
- **MLOps Awareness:** As a senior hire, highlight how you’d put models into production: version control, CI/CD pipelines, monitoring model drift, etc.

**❌ Common Pitfalls to Avoid:**  
- No baseline model comparison (always compare against naive or simple solution).  
- Overfitting your answer to tech details; remember the business why.  
- Using class imbalance without justification (never ignore minority class impact).  
- Conflating correlation with causation when talking A/B or uplift.  
- Ignoring model deployment/monitoring - at senior level, they expect awareness of maintaining models, not just building them.

---

## 📚 Quick Reference: Key Frameworks & Tools

| **Category**           | **Examples/Tools** |
|------------------------|--------------------|
| **ML/Python**          | scikit-learn, XGBoost, LightGBM, TensorFlow, PyTorch |
| **Time-Series**        | statsmodels (ARIMA), Prophet, pmdarima, NeuralForecast, Darts |
| **GenAI/LLM**          | LangChain, LlamaIndex, Vertex AI (Gemini/GPT-4o), OpenAI API |
| **Vector DB**          | FAISS, Pinecone, ChromaDB, AlloyDB (GCP) |
| **Optimization (OR)**  | Google OR-Tools, Gurobi, PuLP, Pyomo |
| **Cloud/Big Data**     | GCP: Vertex AI, BigQuery, Dataflow, Cloud Composer (Airflow) |
| **MLOps/DevOps**       | MLflow, Vertex AI Experiments, Docker, Kubernetes, Airflow, Great Expectations |
| **SQL/Databases**      | PostgreSQL/BigQuery (window functions, CTEs), Redshift |
| **Evaluation/Explain** | SHAP, LIME, RAGAS (for RAG) |
| **Experiment Tracking**| MLflow, W&B, Vertex AI Experiments |

*Last updated: Nov 2026 – Compiled from Glassdoor interviews, DataLemur’s GM interview guides, GM job descriptions, and candidate experiences.*