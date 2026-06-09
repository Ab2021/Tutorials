# Ensemble Health -- Comprehensive Interview Questionnaire

> **150 Questions Covering Every Aspect of the Denial Prediction System**
>
> Focus: Reasoning, trade-offs, alternatives considered, and the logical selection process

---

## Section 1: Business Context & Problem Understanding (15 Questions)

### 1.1 Problem Framing

**Q1.** Why is pre-bill denial prediction valuable for a healthcare revenue cycle company like Ensemble Health? What is the cost of NOT predicting denials?

**Q2.** What is the difference between pre-bill denial prediction and post-bill denial management? Why is the former harder to implement?

**Q3.** The assessment specifies a 25% review capacity. Why is this constraint important for model evaluation? How would your approach change if the capacity were 10% or 50%?

**Q4.** If Ensemble Health processes 100,000 claims per month, what would be the estimated annual savings from a model that captures 45.7% of denials in the top 25%? Walk through your calculation.

**Q5.** Why did you choose to predict denial probability rather than classify claims as "deny" or "not deny"? What operational advantage does a probability provide over a binary prediction?

### 1.2 Stakeholder Alignment

**Q6.** How would you explain the value of this system to a hospital CFO who has never heard of machine learning? What metrics would you lead with?

**Q7.** A biller asks: "Your model says this claim has a 60% chance of denial. Should I fix it or submit it?" How do you interpret model probabilities for non-technical users?

**Q8.** What is the difference between a business metric (Capture@25%) and an academic metric (ROC-AUC)? Why did you prioritize the business metric?

**Q9.** If a stakeholder demanded 90% accuracy, how would you explain why that's not the right target for this problem?

**Q10.** The model predicts 125 claims as High-risk but only ~67 are actually denied. A VP asks: "Why is your model wrong 58 times out of 125?" How do you respond?

### 1.3 Constraints & Requirements

**Q11.** What are the three most important constraints for a production healthcare ML system? How did you address each one?

**Q12.** Why is data leakage the cardinal sin of ML in healthcare? Give three examples of what WOULD constitute leakage in this project.

**Q13.** The assessment requires a single-command pipeline. Why is this important? What problems could arise from a multi-step manual process?

**Q14.** Why did you choose to make the system HIPAA-compliant even though this is a hiring assessment with synthetic data? What specific decisions reflect this concern?

**Q15.** If Ensemble Health asked you to deploy this model to production tomorrow, what three things would you add before going live?

---

## Section 2: Exploratory Data Analysis (15 Questions)

### 2.1 Data Understanding

**Q16.** Walk me through your EDA process. What was the first thing you looked at and why?

**Q17.** You found a 21.6% overall denial rate. Why is this number important? What does it tell you about the modeling approach you should take?

**Q18.** The test set has a 26.0% denial rate vs 21.1% in training. What does this gap suggest? How did it influence your production deployment strategy?

**Q19.** You noted that `total_billed` is heavily right-skewed. What specific problems does this create for linear models? How did you address it?

**Q20.** Explain the difference between mean and median for `total_billed` ($12,164 vs $7,972). What does this gap tell you about the data distribution? Why does it matter?

**Q21.** You found that `prior_auth_required=1216` but `has_prior_auth=1262`. What does the extra 46 claims with auth but no requirement tell you? Is this a data quality issue or a business pattern?

**Q22.** Why did you analyze denial rates by payer type? What patterns did you find? How did these patterns influence feature engineering?

### 2.2 Gap Analysis Deep Dive

**Q23.** You identified `auth_gap` as having the highest lift (+27.6 pp). Walk me through exactly how you computed this metric. Why is it more meaningful than raw correlation?

**Q24.** The cumulative gap analysis shows 0 gaps = 13.4% denial, 3 gaps = 69.2% denial. Is this relationship perfectly linear? How would you test for linearity? What are the implications of any deviation?

**Q25.** Why did you compute both individual gap lifts AND cumulative gap denial rates? What different insights do they provide?

**Q26.** `network_issue` only has a +4.0 pp lift. Why is it so much lower than `auth_gap` (+27.6 pp)? What does this tell you about out-of-network claims?

**Q27.** Only 4 claims have 4 gaps and all were denied. Is this statistically significant? How would you handle such small sample sizes in your analysis?

**Q28.** If you could only use ONE feature to predict denials, which would it be and why? Back up your answer with the data.

### 2.3 Data Quality & Validation

**Q29.** You found 2,509 missing values in `denial_reason`. Is this a problem? Why did you choose to exclude rather than impute?

**Q30.** How would you detect if the current claims (2025-01) have a different distribution than the historical data (2024)? What statistical tests would you run?

---

## Section 3: Feature Engineering (20 Questions)

### 3.1 Design Philosophy

**Q31.** What is the single most important principle guiding your feature engineering? Why?

**Q32.** You engineered 53 features from 17 raw columns. Walk me through your thought process for deciding which features to create. How did the EDA inform your choices?

**Q33.** What is the difference between a feature that is "predictive" and one that causes "data leakage"? Give an example of each from this project.

**Q34.** Why did you create `total_admin_gaps` when you already have individual gap flags? Doesn't this create redundancy? Explain the modeling rationale.

**Q35.** You log-transformed `total_billed` and `expected_payment` but not `num_procedures` or `days_to_submit`. Why? What is the decision rule for when to log-transform?

### 3.2 Specific Feature Decisions

**Q36.** Explain the difference between `payment_ratio` and `payment_gap`. Why create both? What different aspects of a claim do they capture?

**Q37.** You binned `days_to_submit` into 5 ordinal categories. Why not use the raw continuous value? What are the pros and cons of binning?

**Q38.** Why did you create `double_deficit` (auth_gap AND doc_issue) as a separate feature? Why not let the model learn this interaction automatically?

**Q39.** What is the purpose of `is_high_risk_combo`? How did EDA lead you to create this specific feature? Could the model have discovered this pattern on its own?

**Q40.** You used `OneHotEncoder(handle_unknown='ignore')`. Why is `handle_unknown='ignore'` critical for production? What would happen without it?

**Q41.** Why did you choose `StandardScaler` over `MinMaxScaler` or `RobustScaler`? What are the trade-offs?

**Q42.** What is the dimension of your feature matrix after one-hot encoding? Walk through the calculation: 3 categorical features with cardinalities 4, 4, 12.

### 3.3 Leakage Prevention

**Q43.** List every column you excluded from the feature matrix and explain why each one would constitute data leakage or bias.

**Q44.** If a junior developer added `denial_reason` as a feature and the model's ROC-AUC jumped from 0.69 to 0.95, how would you detect and explain this problem?

**Q45.** The `split` column marks train/validation/test. Why can't this be used as a feature? What specific bias would it introduce?

**Q46.** You excluded `service_month` but created `service_month_num` and `service_quarter`. Why is one leakage and the other acceptable? Where do you draw the line?

### 3.4 Feature Selection

**Q47.** You kept all 53 features rather than doing feature selection. Why? Under what circumstances would you reduce the feature set?

**Q48.** How would you determine which features are most important in the final model? What metric would you use?

**Q49.** If you had to reduce to only 10 features, which would you keep? Justify each choice with EDA evidence.

**Q50.** What is the curse of dimensionality? Does 53 features on 2,240 training samples risk this? When would it become a concern?

---

## Section 4: Model Selection & Experimentation (20 Questions)

### 4.1 Experimental Design

**Q51.** Why did you run 10 experiments instead of just picking a model and tuning it? What is the value of breadth over depth in model exploration?

**Q52.** Walk me through your train/validation/test split strategy. Why 70/15/15? What would change if you had 100,000 samples instead of 3,200?

**Q53.** You used `class_weight='balanced'` for LR. What does this do mathematically? How does the class weight of 2.32 for the minority class affect the loss function?

**Q54.** Why did you search C values [0.001, 0.01, 0.1, 1.0, 10.0, 100.0] rather than a finer grid? What is the relationship between C and model complexity?

**Q55.** Explain what happens to LR's coefficients as C increases from 0.001 to 100.0. Why did C=0.1 work best on this data?

### 4.2 Model Architecture Decisions

**Q56.** Why did LR achieve 49.51% capture while XGBoost only reached 35.92%? This is a 13.6 pp gap -- explain WHY this happened using the data properties you discovered.

**Q57.** GBM (48.54%) significantly outperformed XGBoost (35.92%). Both are gradient-boosted trees. What architectural differences explain this gap?

**Q58.** You weighted LR 2x in the Voting ensemble. Why not equal weights? What is the mathematical consequence of overweighting the best model?

**Q59.** Stacking had the BEST ROC-AUC (0.718) and Brier (0.136) but only ranked 6th on capture@25 (46.60%). Explain this paradox. Why would you NOT deploy the model with the best ROC-AUC?

**Q60.** MLP with (64,32) architecture achieved only 43.69%. Is this a fair comparison given only 2,240 training samples? How many parameters does this network have? What is the samples-per-parameter ratio?

**Q61.** Why did SVC with RBF kernel underperform LR? What property of the data makes RBF kernels unnecessary?

**Q62.** RF got 39.81% with 200 trees and depth=12. If you increased trees to 1,000 and depth to 20, would it catch up to LR? Why or why not?

### 4.3 Model Comparison & Selection

**Q63.** Rank the 10 experiments by: (a) capture@25, (b) ROC-AUC, (c) Brier, (d) training time. Which model wins each category?

**Q64.** If you had to choose between a model with capture=49% and Brier=0.14 vs capture=48% and Brier=0.10, which would you pick? Why?

**Q65.** The three LR variants all got identical capture@25. Does this mean feature engineering is irrelevant and model architecture is all that matters? Why or why not?

**Q66.** How do you know you haven't overfit to the validation set with 10 experiments? What safeguards prevent this?

**Q67.** If you could only run 3 experiments instead of 10, which would you pick and why?

**Q68.** What experiment would you add as #11 if you had another week? Justify your choice with a specific hypothesis.

### 4.4 Hyperparameter Sensitivity

**Q69.** How sensitive is LR to the choice of C? If C changed from 0.1 to 0.05, what would happen to coefficients and capture@25?

**Q70.** XGBoost has `scale_pos_weight=3.6`. How did you arrive at this value? What is the theoretical basis for this number?

---

## Section 5: Evaluation Metrics (15 Questions)

### 5.1 Metric Selection

**Q71.** Define Capture@25% in plain English. How is it calculated? Why is it more relevant than accuracy for this problem?

**Q72.** A model with 78.4% accuracy sounds great until you learn the denial rate is 21.6%. Explain why accuracy is misleading for imbalanced classification.

**Q73.** What is the difference between ROC-AUC and PR-AUC? When would you prefer one over the other? Which is more informative for this problem?

**Q74.** Your model has ROC-AUC=0.691 and PR-AUC=0.522. Why is PR-AUC so much lower? Is this a problem?

**Q75.** Explain Brier score in simple terms. What does a Brier score of 0.137 mean? What would a perfect model score? A random model?

### 5.2 Metric Trade-offs

**Q76.** Your model captures 45.7% of denials at 25% review. A colleague suggests lowering the threshold to capture more denials. What would happen to precision? At what point does this become counterproductive?

**Q77.** If you could only optimize for ONE metric throughout the entire project, which would you choose? Defend your answer.

**Q78.** How would you detect if your model is performing worse in production than in testing? What metrics would you monitor?

**Q79.** The validation capture@25 is 49.51% but test capture@25 is 45.71%. Is this gap concerning? What could cause it?

**Q80.** How many denials per 500 claims does your model catch vs random? How many additional denials is that?

### 5.3 Statistical Rigor

**Q81.** How would you put confidence intervals on your test metrics? Why are point estimates insufficient?

**Q82.** If you ran the pipeline 100 times with different random seeds, how much would capture@25 vary? What factors influence this variance?

**Q83.** How do you know the 13.6 pp gap between LR and XGBoost is statistically significant and not just noise? What test would confirm this?

**Q84.** The 49.51% capture represents ~103 out of ~208 validation denials caught. If you caught 100 or 106 instead, would your conclusions change? At what threshold?

**Q85.** What is the minimum detectable effect size for your experiment setup? How many samples would you need to detect a 2 pp improvement with 80% power?

---

## Section 6: Calibration & Thresholds (10 Questions)

### 6.1 Calibration Theory

**Q86.** What does it mean for a model to be "well-calibrated"? Give an example of a well-calibrated vs poorly calibrated prediction.

**Q87.** Why does regularized LR (C=0.1) need calibration? Doesn't LR theoretically output probabilities? What breaks this property?

**Q88.** Explain Platt scaling mathematically. What are the two parameters (A and B) and what does each do to the probability distribution?

**Q89.** You chose sigmoid calibration over isotonic regression. What are the trade-offs? When would isotonic be preferable?

**Q90.** Why `CalibratedClassifierCV(cv=5)` instead of a single split? What problem does cross-validation solve in calibration?

### 6.2 Threshold Selection

**Q91.** Your validation threshold is 0.252. Why does a "denial prediction" threshold of 25.2% make sense when the denial rate is 21.6%? Shouldn't it be 50%?

**Q92.** Early versions of this project computed the threshold from test set probabilities. Why is this wrong? What specific harm does it cause?

**Q93.** If you changed the review capacity to 10%, your threshold would rise. How would this affect capture@10%, precision@10%, and the number of denials caught?

**Q94.** The threshold is frozen after validation. Why not recompute it on current claims? What assumption would that violate?

**Q95.** How would you explain to a stakeholder that 125 claims get flagged as High-risk when only ~67 are actually denied? Is the system "wrong" 46% of the time?

---

## Section 7: GenAI & LLM Integration (15 Questions)

### 7.1 Architecture & Design

**Q96.** Walk me through the full flow from a claim row to a plain-English explanation. What happens at each step?

**Q97.** Why use Pydantic models for both input and output validation? What specific failure modes does this two-layer approach prevent?

**Q98.** You chose gemma4:31b-cloud via ollama.chat(). What alternatives did you consider? Why this specific model?

**Q99.** The prompt instructs the LLM to "Return ONLY a valid JSON object." Why is this constraint necessary? What happens without it?

**Q100.** You inline risk factors with their permitted actions rather than just listing names. Why does this reduce hallucination?

**Q101.** What specific rules did you include in the prompt to prevent the LLM from mentioning ICD codes, dollar amounts, or patient identifiers? Why are these rules important for healthcare?

### 7.2 Validation & Fallback

**Q102.** Explain the three-tier fallback strategy: JSON parse -> regex extraction -> deterministic template. When does each tier activate?

**Q103.** The `@field_validator` for disclaimer checks for "estimate", "statistical", "not guaranteed". Why is this specific validation critical? What would happen if a disclaimer was missing?

**Q104.** Your deterministic template produces explanations like "This claim has an estimated denial risk of 26% -- this is a statistical estimate, not a guaranteed outcome." How was this template designed? What factors went into making it sound natural?

**Q105.** If the API returns a perfectly valid JSON with a disclaimer that says "This claim will be denied," would your Pydantic validator catch it? Why or why not?

### 7.3 Production Considerations

**Q106.** Why do you only send 125 claims to the API instead of all 500? What is the cost-benefit calculation?

**Q107.** Estimate the token cost for 500 API calls vs 125. If each 1M tokens costs $0.50, what's the difference?

**Q108.** How do you handle API rate limits? What happens if Ollama Cloud is down during pipeline execution?

**Q109.** The audit log records 500 records but only 125 are from the API. What do the other 375 records contain? Why log bypassed calls?

**Q110.** How would you detect if the LLM's explanations are degrading in quality over time? What metrics would you monitor?

---

## Section 8: Production & Code Architecture (10 Questions)

### 8.1 System Design

**Q111.** Walk me through the modular package structure. Why did you separate code into `config/`, `utils/`, `prompts/`, and top-level modules? What principle does this follow?

**Q112.** You have `build_features()` in `feature_engineering.py` and `engineer = build_features` in other files. Why the alias? What problem does this solve?

**Q113.** Why did you replace `print()` statements with `logging.getLogger()`? What are three advantages of structured logging?

**Q114.** You use `if/raise ValueError` instead of `assert` for validation. Why? What happens to `assert` statements in optimized Python?

**Q115.** The tier assignment uses `n // 4` instead of hardcoded `.loc[:124]`. Why is this important? What if current claims grow to 1,000?

**Q116.** You added pre-flight validation for file existence. What specific checks would you add for a production deployment?

### 8.2 Testing & Quality

**Q117.** You have 82 tests across 6 modules. How did you decide what to test? What is NOT tested that should be?

**Q118.** Your tests use `conftest.py` for shared fixtures. What is the advantage of fixtures over creating data in each test?

**Q119.** How would you test the LLM integration without making actual API calls? What mocking strategy would you use?

**Q120.** What code quality issues did you identify and fix in your codebase? What tools would you add to a CI pipeline?

---

## Section 9: Risk Factors & Explainability (10 Questions)

### 9.1 Extraction Logic

**Q121.** Walk through the algorithm that converts LR coefficients into human-readable risk factors. What are the three steps?

**Q122.** Why use coefficient * feature_value for attribution? Why not SHAP or LIME? What are the trade-offs?

**Q123.** How do you handle the case where both `auth_gap` and `double_deficit` map to "Missing Required Prior Authorization"? What is your deduplication strategy?

**Q124.** You map raw feature names like `payer_id_P008` to "High-risk segment: Payer ID: P008". Why is this a reasonable label? What information does it convey to a biller?

**Q125.** For claims with prob < 0.25, you skip attribution and output "No actionable pre-submission risk flags detected." Why is this the right approach? What would be wrong with extracting factors anyway?

### 9.2 Business Value

**Q126.** A biller sees "Missing Required Prior Authorization, Compound deficit: missing auth + documentation, High-risk segment: Payer ID: P008." What should they DO with this information? Walk through the workflow.

**Q127.** How would you measure whether explanations are actually useful to billers? What feedback mechanism would you implement?

**Q128.** If a claim has 5 positive contributions but you only show 3, how do you choose which to display? What is the risk of showing too many factors?

**Q129.** Your risk factor labels were hand-crafted. How would you maintain these mappings if you added 50 new features? What process would you follow?

**Q130.** Explain the connection between model interpretability and regulatory compliance. Why do healthcare models need to be explainable?

---

## Section 10: Error Analysis (10 Questions)

### 10.1 Understanding Mistakes

**Q131.** Your model missed 73 denials (false negatives). What characterizes these claims? What do they have in common?

**Q132.** Your model flagged 79 non-denials (false positives). Are these "mistakes" or "reasonable precautions"? How do you distinguish?

**Q133.** If you could add ONE feature to reduce false negatives, what would it be? Base your answer on the error analysis.

**Q134.** The model is most confident (best calibration) in the 0.50-0.60 range. Why might this be? What does this tell you about the model's strengths?

**Q135.** Claims with 0 admin gaps are 38% of false negatives. Why can't the model catch these? Is this a model problem or a data problem?

### 10.2 Improvement Strategies

**Q136.** If you had access to ICD-10 diagnosis codes and CPT procedure codes, how would you expect model performance to change? What specific patterns would they capture?

**Q137.** Medicaid MCO claims are overrepresented in false positives (35% vs 24% population). What does this suggest about a single global model? How would you address this?

**Q138.** How would you implement a feedback loop where biller corrections improve future predictions? What data would you need to collect?

**Q139.** If you noticed model performance degrading month-over-month, what investigation would you conduct? What root causes would you check?

**Q140.** What is the theoretical maximum capture@25 for this dataset? How close are we to the ceiling? How would you estimate this?

---

## Section 11: Scenario-Based & Synthesis (10 Questions)

### 11.1 Hypotheticals

**Q141.** Your model is in production. A new payer joins the network with radically different denial patterns. Your model's capture@25 drops from 45% to 30%. What happened? What do you do?

**Q142.** A data engineer accidentally swaps `has_prior_auth` and `prior_auth_required` columns in the pipeline. How would your validation catch this? What would happen to model performance?

**Q143.** The business wants explanations for ALL 500 claims, not just 125. What changes would you make? What are the cost implications?

**Q144.** You're asked to reduce training time by 50%. What would you change? What is the performance impact?

**Q145.** A regulator asks you to prove your model doesn't discriminate against Medicaid patients. What analysis would you conduct? What metrics would you report?

### 11.2 Synthesis

**Q146.** If you were to rebuild this project from scratch with unlimited resources, what would you do differently? What mistakes would you avoid?

**Q147.** What is the single most important decision you made in this project? Why was it pivotal?

**Q148.** Rank these in order of impact on final model performance: feature engineering, model selection, hyperparameter tuning, calibration. Defend your ranking.

**Q149.** A peer claims "logistic regression is obsolete; everyone uses neural networks now." How do you respond using evidence from this project?

**Q150.** Looking back, what surprised you most about this project? What finding contradicted your initial assumptions?
