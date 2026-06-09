# Ensemble Health -- Interview Question Answers

> **Detailed Answers to All 150 Questions**
>
> Companion to `INTERVIEW_QUESTIONS.md` -- every answer rooted in project data, decisions, and trade-offs

---

## Section 1: Business Context & Problem Understanding

### 1.1 Problem Framing

**Q1. Why is pre-bill denial prediction valuable for a healthcare revenue cycle company like Ensemble Health? What is the cost of NOT predicting denials?**

Pre-bill denial prediction transforms healthcare billing from reactive to proactive. Currently, most revenue cycle teams discover denials 14-45 days after claim submission when the payer returns a remittance advice with reason codes. At that point, the biller must research the denial, gather missing documentation, potentially appeal, and resubmit -- a process costing $25-$118 per claim in administrative labor alone. The financial cost extends beyond labor: denied claims delay revenue by 30-90 days (impacting working capital), some are never recovered (permanent revenue loss), and high denial rates damage provider-payer relationships and CMS Star ratings.

The cost of NOT predicting denials is multiplicative. For 100,000 claims per month at a 21.6% denial rate, there are 21,600 denials monthly. Even with no model -- a random review strategy catching ~25% of those -- billers still face 16,200 denials to rework. At $25-$118 per rework, that's $405K-$1.9M monthly in avoidable rework costs, plus uncountable revenue delays and write-offs. A model that captures even 45% of denials at 25% review prevents ~9,720 denials from becoming rework items. The ROI is clear: every $1 spent on prediction infrastructure returns multiple dollars in avoided rework and accelerated revenue.

**Q2. What is the difference between pre-bill denial prediction and post-bill denial management? Why is the former harder to implement?**

Pre-bill denial prediction occurs BEFORE the claim leaves the provider's system. The model has access only to claim attributes available at the point of coding and billing: patient demographics, visit type, procedure codes, prior authorization status, and payer information. The target variable (will the claim be denied?) is unknown -- the model must predict it.

Post-bill denial management occurs AFTER the payer responds. The system knows EXACTLY which claims were denied and the specific denial reason codes. This is fundamentally easier: it's a deterministic matching problem ("code this denial reason to the appropriate workflow") rather than a predictive one. The challenge is workflow optimization, not prediction.

Pre-bill is harder for three reasons: (1) The target is unknown at prediction time, requiring statistical modeling rather than rules; (2) Available features are limited to pre-submission data -- no payer response codes, no adjudication history for this specific claim; (3) The model must generalize across claim types, payers, and visit scenarios that all have different denial patterns. However, the business value of pre-bill is 10x higher because prevention is always cheaper than cure.

**Q3. The assessment specifies a 25% review capacity. Why is this constraint important for model evaluation? How would your approach change if the capacity were 10% or 50%?**

The 25% review capacity is the binding operational constraint. It defines the denominator of our primary metric (capture@25%) and fundamentally shapes model evaluation. Without this constraint, a model could achieve 100% capture by flagging every claim -- but that defeats the purpose since billers can't review everything. The constraint creates the classic precision-recall trade-off: at a fixed review budget, maximizing recall (denial capture) is the optimization problem.

At 10% capacity (50 claims), the task becomes extreme. The threshold rises to the 90th percentile of validation probabilities -- probably around 0.50-0.60. Only the highest-risk claims are reviewed. Capture might drop to 25-30%, but precision would rise above 60%. I would prioritize calibration heavily (to ensure tier assignments are rock-solid) and consider an ensemble approach that trades some ranking ability for better tail probability estimation.

At 50% capacity (250 claims), the task becomes much easier. The threshold drops to the 50th percentile, and capture might hit 65-75%. At this point, the model's calibration matters less because many claims are reviewed regardless. I might switch to optimizing for precision (fewer false positives in the review queue) and consider a simpler model that prioritizes speed and interpretability over marginal capture improvements.

**Q4. If Ensemble Health processes 100,000 claims per month, what would be the estimated annual savings from a model that captures 45.7% of denials in the top 25%? Walk through your calculation.**

Let me break this down step by step:

**Monthly baseline (no model, random review):**
- Total claims: 100,000
- Denial rate: 21.6% → 21,600 denials
- Random review catches: 25% of claims × 21,600 = 5,400 denials caught proactively
- Denials missed: 21,600 - 5,400 = 16,200 (enter rework pipeline)

**Monthly with our model (45.7% capture):**
- 25% review = 25,000 claims reviewed
- Denials caught: 21,600 × 0.457 = 9,871 denials caught proactively
- Denials missed: 21,600 - 9,871 = 11,729 (enter rework)
- Improvement: 9,871 - 5,400 = 4,471 additional denials caught

**Cost savings:**
| Rework Cost/Denial | Monthly Savings | Annual Savings |
|---|---|---|
| Conservative ($25) | $111,775 | $1,341,300 |
| Average ($70) | $312,970 | $3,755,640 |
| Aggressive ($118) | $527,578 | $6,330,936 |

**Additional non-labor savings (harder to quantify):**
- Reduced write-offs for unrecovered denials
- Faster cash flow (claims submitted correctly first time get paid ~14-21 days faster)
- Reduced biller burnout and turnover from repetitive rework
- Improved payer-provider relationship scores

The annual savings range of $1.3M-$6.3M per 100K monthly claims makes this a high-ROI investment even at the conservative end. Note that these are just the DIRECT rework labor savings; the indirect benefits (accelerated revenue, fewer write-offs, improved staff retention) multiply the impact.

**Q5. Why did you choose to predict denial probability rather than classify claims as "deny" or "not deny"? What operational advantage does a probability provide over a binary prediction?**

Probability predictions offer five critical operational advantages over binary classification:

**1. Ranking and prioritization:** Binary predictions sort claims into exactly two groups. Probabilities allow continuous ranking -- the biller can start with the 75% probability claim, then the 72%, then the 68%, working from highest to lowest risk. This is fundamentally more useful than a flat list of "denies."

**2. Capacity-aware thresholds:** With binary classification, you're locked into the model's hard decision boundary. With probabilities, you can set the threshold at whatever review capacity exists: 10%, 25%, 50%. The model adapts to operational reality rather than forcing operations to adapt to the model.

**3. Calibration and confidence:** A probability of 0.60 vs 0.90 conveys very different levels of confidence. Binary predictions collapse this rich information into a single bit. Billers can triage based on confidence: double-check the 0.60 claim, fast-track fix the 0.90 claim.

**4. Audit and monitoring:** Probability distributions can be monitored for drift over time. If the mean predicted probability shifts from 0.25 to 0.35 month-over-month, something changed (new denial patterns, feature drift). Binary prediction rates changing from 25% to 35% tells you less about the underlying signal.

**5. Integration with cost-sensitive decisions:** Not all false positives cost the same. A probability score allows a downstream system to incorporate the cost of review vs. the cost of a missed denial into the decision function. Binary predictions bake in a single cost assumption.

### 1.2 Stakeholder Alignment

**Q6. How would you explain the value of this system to a hospital CFO who has never heard of machine learning? What metrics would you lead with?**

I would lead with dollars, not AUC. Here's the conversation:

"Your billers review claims before submitting them to insurance companies. Right now, they probably review them in submission order, or by dollar amount, or by some rule of thumb. Our system ranks all claims by how likely they are to be denied. Your billers review the riskiest 25% -- about 25,000 claims per month if you process 100,000.

Before our system, reviewing a random 25% would catch about 5,400 denials before they become problems. After our system, that same 25% review catches about 9,900 denials. That's 4,500 fewer denials per month that become rework.

Each denied claim costs roughly $25-$118 in biller time to fix. That means we're saving $110,000-$530,000 per month in rework costs. Over a year, that's $1.3-$6.3 million.

And that's just the labor cost. Clean claims get paid faster -- typically 14-21 days faster. Clean submissions mean fewer provider complaints. And your billers spend less time on rework and more time on new claims.

The system runs automatically. Feed it a CSV of claims, it produces a ranked list with explanations for why the high-risk ones are flagged. Your billers open the list, start at the top, and work down. That's it."

I lead with the dollar savings because CFOs think in ROI. Capture@25% is the metric that translates directly to dollars saved. Technical metrics (AUC, Brier) are for the data science team, not the finance team.

**Q7. A biller asks: "Your model says this claim has a 60% chance of denial. Should I fix it or submit it?" How do you interpret model probabilities for non-technical users?**

"A 60% probability means: out of 100 claims that look exactly like yours -- same payer, same visit type, same authorization status, same documentation completeness -- about 60 of them get denied. Your claim is in that higher-risk group.

Think of it like a weather forecast. When the weather app says 60% chance of rain, it means on days with similar conditions, it rained 60% of the time. You probably bring an umbrella. Similarly, for this claim, I'd recommend you review it -- check the authorization, verify the documentation, confirm the referral -- before submitting. The 5-10 minutes of review now is almost always worth avoiding 2-3 hours of rework later.

The system also tells you WHY it's flagged: 'Missing Required Prior Authorization' and 'Missing Supporting Documentation.' Start with those two items. If you can resolve them, the claim's real risk drops significantly.

One more thing: 60% doesn't mean the claim WILL be denied. 40% of similar claims go through fine. But that's not a bet you want to make 100 times a day. Fixing 60 at-risk claims prevents roughly 36 denials. The math works in your favor."

The key is: (1) ground the probability in frequencies (60 out of 100), (2) use an everyday analogy (weather), (3) connect the number to an ACTION the biller can take, and (4) acknowledge the uncertainty honestly.

**Q8. What is the difference between a business metric (Capture@25%) and an academic metric (ROC-AUC)? Why did you prioritize the business metric?**

Capture@25% answers: "Of all the denials that will happen, what fraction did the biller catch in their first 125 reviews?" This is a direct operational question. If capture@25 = 45.7%, the biller prevents 45.7% of denials within their review capacity.

ROC-AUC answers: "Across all possible thresholds, how well does the model separate denied claims from non-denied claims?" This is a theoretical question. It averages over thresholds the business will never use -- including reviewing 100% of claims, which defeats the purpose.

The critical difference is that ROC-AUC is threshold-agnostic while the business has a hard threshold constraint (25% review). Our Stacking model (exp8) achieved the BEST ROC-AUC (0.718) but only ranked 6th on capture@25 (46.60%). Concretely, Stacking ranked claims slightly better overall but compressed the extreme tail -- the top 25% were less distinguishable. This directly harmed the business metric despite improving the academic metric.

I prioritized the business metric because the optimization problem is constrained: maximize denials caught at exactly 25% review. ROC-AUC is a diagnostic (is the model learning anything?), not an objective (is the model useful?). In ML engineering, the metric that maps to business value IS the metric you optimize.

**Q9. If a stakeholder demanded 90% accuracy, how would you explain why that's not the right target for this problem?**

"90% accuracy sounds reasonable, but let me show you why it's not the right target for this problem. Only 21.6% of claims are denied. This means if our model predicted 'no denial' for EVERY SINGLE CLAIM, it would be right 78.4% of the time -- nearly 80% accuracy with zero effort.

That model catches zero denials and saves zero dollars. But it's '80% accurate.'

Now let's say we build a model that correctly flags most high-risk claims but also has some false positives. It catches 45% of denials -- saving millions. But because it flags more claims as risky, its accuracy might drop to 75% -- 'worse' than the do-nothing model by 3 points.

The metric we should care about is denial capture at our review capacity. Our model catches ~46% of denials in the top 25% of claims, vs 25% from random review. That's an extra 4,500 denials prevented per 100,000 claims per month. That is the right target -- not a misleading accuracy number that the do-nothing model already achieves."

The deep issue is that accuracy weights false positives and false negatives equally. For denial prediction, a false negative (missing a denial) costs $25-118 while a false positive (reviewing a clean claim) costs ~2 minutes of biller time. These are NOT symmetric costs, and accuracy treats them as identical.

**Q10. The model predicts 125 claims as High-risk but only ~67 are actually denied. A VP asks: "Why is your model wrong 58 times out of 125?" How do you respond?**

"That's a fair question, and I'd reframe it. The model isn't 'wrong' 58 times -- it's appropriately cautious 58 times. Here's why:

Of those 58 flagged-but-not-denied claims, roughly 62% have 2 or more administrative gaps -- missing authorization, documentation issues, or referral problems. These claims LOOK like denials to the payer's automated audit system. The fact that they weren't denied doesn't mean the flagging was wrong; it means the biller caught and fixed the issues, or the payer made an exception.

Think of it like a smoke detector. It goes off when it detects smoke, not just fire. You'd rather have it go off for burnt toast than fail to go off for a real fire. Those 58 false positives are the 'burnt toast' -- an acceptable cost for catching the 67 real fires.

The alternative is to make the model less cautious. But every 1% we relax the threshold to reduce false positives, we miss real denials. The optimal balance -- and this is what the model finds automatically -- is the point where the cost of reviewing a false positive (about 2 minutes) is worth the benefit of catching an additional true positive ($25-118 saved).

If you want fewer flags, we can raise the threshold. But for every 10 fewer flags, we'll miss roughly 4-5 real denials. I'd recommend staying at the current threshold based on the cost-benefit analysis."

### 1.3 Constraints & Requirements

**Q11. What are the three most important constraints for a production healthcare ML system? How did you address each one?**

**1. Data leakage prevention:** Healthcare ML systems will leak if you're not obsessive. A model that uses post-submission data (denial reason, adjudication date) to predict pre-submission denial risk is cheating -- it looks great in testing but is useless in production because that data doesn't exist yet. Our solution: `prepare_model_matrix()` programmatically drops `is_denied`, `denial_reason`, `split`, and any intermediate temporal columns. The exclusion is automatic -- no developer can accidentally include them. We also compute the denial threshold from validation data alone, not test data, to prevent evaluation leakage.

**2. Interpretability and audit trail:** Healthcare models must be explainable to meet regulatory requirements (payers and auditors will ask "why was this claim flagged?"). Our solution: LR's linear coefficients provide native interpretability (every risk factor traces to a specific feature contribution), Pydantic-validated GenAI explanations provide plain-English narrative, and `LLMAuditLogger` records every prediction decision with timestamps, tokens, and validation results.

**3. Production safety and reliability:** The system must never crash silently on edge cases. A missing column, a new payer category, or a malformed API response must be handled gracefully. Our solution: `handle_unknown='ignore'` in the OneHotEncoder, three-tier fallback for LLM responses (JSON parse -> regex -> template), pre-flight file existence validation, and `if/raise ValueError` (not `assert`) for business-critical validations that cannot be disabled in production.

**Q12. Why is data leakage the cardinal sin of ML in healthcare? Give three examples of what WOULD constitute leakage in this project.**

Data leakage is catastrophic because it creates a model that appears brilliant in testing but fails completely in production. The model learns patterns from data that won't be available at prediction time, so its real-world performance collapses. In healthcare, this has real consequences: a model that silently underperforms leads to missed denials, wasted biller time, and potential compliance violations.

Three specific leakage scenarios in this project:

**Example 1: Using `denial_reason` as a feature.** This column is populated only when a claim IS denied, and contains the payer's stated reason. Including it would give the model direct access to the target -- the model would effectively see the answer before making its prediction. The ROC-AUC might hit 0.98. But in production, `denial_reason` is NULL for all current claims (they haven't been adjudicated yet), so the model's predictions would be garbage.

**Example 2: Computing the threshold from test set probabilities.** Early versions did `np.percentile(test_prob, 75)` to set the denial threshold. The test set represents unseen future data. Computing the threshold on it is statistically cheating -- you're using knowledge of the test distribution to define what "high risk" means. In production, you'd apply this threshold to truly unseen current claims, and your tier assignments would be biased. The correct approach is `np.percentile(val_prob, 75)` -- threshold computed strictly from validation data.

**Example 3: Using `split` as a feature.** The split column (train/val/test) encodes the temporal order of claims. Including it would allow the model to learn that "test split claims have a higher denial rate" (26.0% vs 21.1% in training). This is temporal leakage -- the model learns time period effects rather than claim-specific risk patterns. In production, there is no "split" label for new claims.

**Q13. The assessment requires a single-command pipeline. Why is this important? What problems could arise from a multi-step manual process?**

A single-command pipeline (`python src/run_pipeline.py`) eliminates entire categories of production failure:

**Eliminated failure modes:**
- **Step ordering errors:** Someone runs step 3 before step 2, producing results from stale intermediate data
- **Environment inconsistency:** Step 1 uses Python 3.12, step 2 accidentally uses 3.9, producing subtle bugs
- **Manual parameter drift:** Someone changes a threshold in step 2 but forgets to update it in step 4
- **Reproducibility loss:** A critical result can't be reproduced because nobody documented the exact sequence of manual steps
- **Onboarding friction:** New team members need days to learn a multi-step workflow; a single command takes 30 seconds

**Why this matters specifically for healthcare:**
Healthcare ML results are subject to audit. If a payer challenges a denial prediction, the provider must be able to reproduce the exact prediction pipeline. A multi-step manual process introduces variability that makes exact reproduction impossible. A single command with pinned dependency versions (`requirements.txt`) and deterministic random seeds ensures bit-for-bit reproducibility.

**Q14. Why did you choose to make the system HIPAA-compliant even though this is a hiring assessment with synthetic data? What specific decisions reflect this concern?**

I built with HIPAA principles because the assessment simulates a real healthcare system. Decisions made on a prototype have a way of surviving into production, and it's far easier to build compliance in from the start than to retrofit it later. Specific HIPAA-aligned decisions:

**No external telemetry:** The experiment tracker (`ExperimentTracker`) is entirely self-contained with local JSON artifacts under `data/output/experiments/`. No data leaves the system. Heavy MLflow was rejected specifically because it requires external dependencies and could inadvertently phone home with telemetry data.

**No PII in logs:** Audit logs (`LLMAuditLogger`) record claim IDs, timestamps, token counts, latency, and validation flags -- but never patient names, dollar amounts, diagnosis codes, or any PHI that might be in a real claim. The log structure is designed to be safe even on a real system.

**Prompt safety rules:** The LLM prompt explicitly instructs: "Do NOT mention ICD/CPT codes, dollar amounts, or patient identifiers." This prevents an LLM hallucination from accidentally surfacing sensitive data in an explanation that a biller then sees.

**Data separation:** `data/input/` and `data/output/` are both gitignored. Raw claims data (even synthetic) stays out of version control. Generated predictions are local artifacts, not committed to a shared repository.

**Audit trail completeness:** Every prediction, including template-based Medium/Low claims, generates an audit record. If a regulator asks "why was claim X flagged?" there is a timestamped, versioned record with the exact feature values that drove the prediction.

These practices cost nothing extra but build a compliance mindset that transfers directly to production.

**Q15. If Ensemble Health asked you to deploy this model to production tomorrow, what three things would you add before going live?**

**1. Model performance monitoring with automatic alerting.** The current system generates metrics but doesn't actively monitor them. I'd implement a scheduled job that computes capture@25, Brier score, and PR-AUC on a fixed holdout set weekly, with automatic alerts if any metric drops below a threshold (e.g., capture@25 drops below 40% or Brier rises above 0.25). This catches model degradation from concept drift (changing payer denial patterns) before billers waste weeks on an underperforming model.

**2. Online feedback loop for continuous improvement.** Currently the system makes predictions but never learns from outcomes. I'd build a feedback pipeline that, when a claim closes (paid or denied), records the actual outcome and retrains the model monthly. This solves the temporal drift problem (test set denial rate 26% vs train 21%) by keeping the model current. The feedback loop would also collect biller corrections -- "I reviewed claim X, the model said auth_gap, but the real issue was documentation" -- as training data for the explanation engine.

**3. Multi-tenant/payer-specific sub-models.** Our error analysis showed that Medicaid MCO claims are overrepresented in false positives (35% of FPs vs 24% of population), suggesting a single global model doesn't capture payer-specific denial patterns perfectly. I'd train per-payer Logistic Regression models using the same feature set but payer-specific coefficients. The global model becomes the fallback for rare payers. This should improve capture@25 by 2-5 percentage points and reduce the false positive rate for Medicaid MCO claims.

---

## Section 2: Exploratory Data Analysis

### 2.1 Data Understanding

**Q16. Walk me through your EDA process. What was the first thing you looked at and why?**

My EDA followed a structured, hypothesis-driven sequence:

**Step 1: Dataset dimensions and schemas.** First command: `df.shape`, `df.info()`, `df.describe()`. This established that I had 3,200 historical claims with 17 columns and 500 current claims. Zero missing values in most columns except `denial_reason`. This baseline understanding prevents embarrassing mistakes later (e.g., building a model for 50,000 rows when you actually have 3,200).

**Step 2: Target distribution.** `df['is_denied'].value_counts(normalize=True)` showed 21.6% denial rate. This immediately flagged three things: (a) class imbalance at ~4:1, meaning accuracy is a useless metric, (b) need for class_weight='balanced' or similar handling, (c) PR-AUC is more informative than ROC-AUC. Knowing the target distribution shapes every subsequent modeling decision.

**Step 3: Leakage column identification.** I audited every column against the "available at pre-submission?" test. `denial_reason` → POST-submission (exclude). `is_denied` → TARGET (exclude). `split` → data partition label (exclude). `claim_id` → unique identifier (exclude). `service_month` → convert to numeric, drop raw. This audit must happen before any modeling because even thinking about features with leakage columns biases the approach.

**Step 4: Feature distributions by target class.** For each numeric feature, I computed `df.groupby('is_denied')[col].describe()` to understand separation. `total_billed` is right-skewed in both classes → log-transform needed. `days_to_submit` is slightly higher for denied claims → `late_filing` flag might help. This step determines which raw features carry signal.

**Step 5: Categorical analysis.** Denial rates by `payer_type`, `visit_type`, and `payer_id`. Medicaid MCO at 28.6% vs Medicare Advantage at 15.0% was the standout find -- 13.6 pp spread. This drove the `is_high_risk_combo` feature.

**Step 6: Administrative gap analysis.** This was the aha moment. Computing denial rates by each gap flag (`auth_gap`, `doc_issue`, etc.) revealed strong monotonic relationships. Computing cumulative gap denial rates showed near-linear increases. This single analysis justified Logistic Regression as the primary model and administrative gap features as the core signal.

This ordered sequence is intentional: start broad (shapes, types), narrow to specific signals (target correlations), and drill into the most promising direction (gaps). Jumping to modeling without this sequence would miss the linear structure entirely.

**Q17. You found a 21.6% overall denial rate. Why is this number important? What does it tell you about the modeling approach you should take?**

The 21.6% denial rate is the single most important summary statistic in the project because it shapes every downstream decision:

**It defines the class imbalance ratio.** At 21.6:78.4 (roughly 1:4), this is moderate imbalance -- not the 1:100 of fraud detection, but sufficient to break any model optimized for accuracy. A dummy classifier predicting "no denial" scores 78.4% accuracy while being operationally worthless. This tells me: (a) never report accuracy, (b) use class_weight='balanced', and (c) optimize for recall at a fixed precision budget.

**It sets expectations for probability calibration.** A well-calibrated model should output mean predicted probability ≈ 0.216. If the mean prediction is 0.35, the model is overconfident. If it's 0.12, it's underconfident. This is a quick calibration sanity check.

**It bounds the capture@25 metric.** The theoretical maximum capture@25 is min(1.0, 0.25/0.216) ≈ 115% (i.e., 100%). If 100% of denials fall in the top 25%, capture reaches 100%. If denials are perfectly mixed, capture = 25%. A random model achieves exactly 25%. Our models achieved 35-50% -- roughly 1.4x to 2x random improvement. This gap between random and maximum defines the learnable signal.

**It informs the threshold selection.** The 25.2% validation threshold makes sense precisely because the baseline rate is 21.6%. The threshold is slightly above the population mean, meaning "high risk" is defined as "higher than average by a meaningful margin." If the denial rate were 5%, the 75th percentile threshold would be much lower.

**Q18. The test set has a 26.0% denial rate vs 21.1% in training. What does this gap suggest? How did it influence your production deployment strategy?**

This 4.9 percentage point gap suggests a temporal shift in denial patterns. The test set, being the chronologically latest data, has a meaningfully higher denial rate than the training data. Possible causes: (a) a payer changed denial policies mid-year, (b) the mix of claim types shifted seasonally, or (c) external factors (regulatory changes, new payer contracts) increased scrutiny.

This finding directly influenced three deployment decisions:

**1. Full-history retraining for production scoring.** Rather than deploying a model trained only on the training set (which would underestimate risk), the production pipeline retrains on ALL 3,200 historical claims (train + validation + test). This gives the model exposure to the full temporal range, making its predictions more representative of the most recent patterns.

**2. Calibration skepticism.** The 4.9 pp gap means probabilities calibrated on the 21.1% training set will systematically underestimate risk on the 26.0% test set. Platt scaling (CalibratedClassifierCV) partially corrects this by fitting the sigmoid to validation data, but the underlying issue is data distribution drift, not calibration error. This informed the decision to not over-invest in calibration beyond Platt scaling.

**3. Monitoring recommendation.** This gap is exactly why I'd recommend a monthly retraining cycle with feedback from actual outcomes. A model trained on January-December 2024 data scoring January 2025 claims is already using data that's 1-12 months stale. The 4.9 pp gap is the canary in the coal mine for temporal drift.

**Q19. You noted that `total_billed` is heavily right-skewed. What specific problems does this create for linear models? How did you address it?**

Right-skewed financial amounts create three specific problems for linear models:

**Problem 1: Outlier domination of coefficients.** In ordinary least squares and logistic regression, the loss function penalizes squared errors. A single $95,000 claim with 19x the mean value contributes 361x (19^2) to the gradient. The model's coefficients become overwhelmingly determined by the top 1-2% of claims, effectively ignoring the patterns in the other 98%. For a denial prediction model, this means a single high-dollar claim can distort the entire risk surface.

**Problem 2: Non-normal residuals.** Linear models assume normally distributed errors with constant variance. Right-skewed predictors violate this: the variance of predictions at high dollar amounts is much larger than at low amounts, creating heteroscedasticity. While this is less critical for logistic regression (which models log-odds), it still affects coefficient stability.

**Problem 3: Scale mismatch with other features.** After `StandardScaler`, the z-score of a $95,000 claim might be (95,000 - 12,164) / 13,173 ≈ 6.3 standard deviations. All other features live in the [-2, +2] range. This creates an implicit feature importance bias toward high-variance financial columns.

**Solution: `log1p` transformation.** `log1p(x) = ln(x + 1)` compresses the range dramatically: `log1p(95000) = 11.46`, `log1p(5000) = 8.52`. The ratio drops from 19x to 1.34x. After `StandardScaler`, the log-transformed values live in a similar z-score range as other features. The transformation is monotonic (preserves ordering) and handles zero values gracefully (unlike raw `log(x)` which is undefined at x=0).

**Q20. Explain the difference between mean and median for `total_billed` ($12,164 vs $7,972). What does this gap tell you about the data distribution? Why does it matter?**

The mean being 52% larger than the median ($12,164 vs $7,972) is the textbook signature of a right-skewed distribution. When the mean substantially exceeds the median, the distribution has a long right tail pulling the average upward. Specifically:

**What this tells us:** The "typical" claim bills about $8,000 (median), but there are enough very large claims ($35K, $50K, $95K) that the arithmetic average is dragged up to $12,164. More than 50% of claims bill below $8,000, but a small number of high-dollar claims disproportionately influences the mean.

**Why it matters for modeling:**
- **Central tendency:** Use median for reporting "typical" values; don't report mean for skewed data. A stakeholder hearing "average claim is $12,164" will overestimate what most claims look like.
- **Outlier sensitivity:** Any model using raw `total_billed` will be hypersensitive to the tail. A new $200,000 claim (entirely possible in healthcare, think complex surgeries) would completely change predictions for nearby claims.
- **Transformation necessity:** The mean-median gap is a direct measure of skew. A gap ratio > 1.5x is a strong indicator that log transformation is necessary. Our ratio of 1.53x crosses this threshold.
- **Sampling implications:** Train/test splits can accidentally concentrate high-dollar claims in one split, creating artificial performance differences. Stratified or careful random splitting is important.

After log transformation, the ratio of mean to median for `log_total_billed` should approach 1.0 -- that's how you verify the transformation worked.

**Q21. You found that `prior_auth_required=1216` but `has_prior_auth=1262`. What does the extra 46 claims with auth but no requirement tell you? Is this a data quality issue or a business pattern?**

The 46 claims where `has_prior_auth=1` but `prior_auth_required=0` represents approximately 3.7% of the dataset (46/1262). This could be either data quality or business pattern:

**Business pattern explanation (more likely):** Many providers obtain prior authorization proactively even when not strictly required. Reasons include: (a) payer policies are ambiguous and providers err on the side of caution, (b) the provider's standard workflow obtains auth for all services above a dollar threshold regardless of strict requirements, or (c) the auth was obtained for a related service and extended to this claim. In this interpretation, these 46 claims represent conscientious billing practices, not errors.

**Data quality explanation (less likely):** There might be a systematic encoding issue where `prior_auth_required` is under-reported. Perhaps certain departments or providers don't consistently set this flag, or there's a lag between auth requirement being determined and the flag being updated in the billing system.

**Modeling implications:** This pattern doesn't affect our `auth_gap` feature (which requires BOTH `prior_auth_required=1` AND `has_prior_auth=0`). These 46 claims have `auth_gap=0` regardless of which explanation is correct. However, for a production system, I'd recommend investigating with the billing operations team to understand the root cause. If it's a quality issue, fixing it could add 46 more training examples with correctly labeled auth requirements.

**Q22. Why did you analyze denial rates by payer type? What patterns did you find? How did these patterns influence feature engineering?**

I analyzed denial rates by payer type because in healthcare billing, the payer is the single biggest determinant of denial behavior. Each payer (and payer type) has distinct prior authorization requirements, documentation standards, timely filing limits, and adjudication rules. A claim that sails through Commercial might be summarily denied by Medicaid MCO for the exact same documentation.

**Patterns found:**

| Payer Type | Denial Rate | vs. Average |
|---|---|---|
| Medicaid MCO | 28.6% | +7.0 pp |
| BCBS | 22.8% | +1.2 pp |
| Commercial | 21.2% | -0.4 pp |
| Medicare Advantage | 15.0% | -6.6 pp |

The 13.6 percentage point spread between highest (Medicaid MCO, 28.6%) and lowest (Medicare Advantage, 15.0%) is massive. Medicaid MCO plans deny at nearly twice the rate of Medicare Advantage. This isn't random -- it reflects the operational reality that Medicaid managed care organizations have stricter prior authorization requirements, more aggressive utilization review, and narrower coverage policies.

**How this influenced feature engineering:**

**1. `payer_type` included in one-hot encoding.** All payer types are passed through OneHotEncoder, allowing the model to learn per-payer intercepts. The LR coefficient for `payer_type_Medicaid MCO` should be positive and large, reflecting the elevated baseline risk.

**2. `is_high_risk_combo` feature created.** `(payer_type == 'Medicaid MCO') & (visit_type in ['Inpatient', 'Emergency'])`. This captures a specific high-denial intersection: Medicaid MCO patients with expensive visit types. Inpatient stays are inherently complex and expensive; Emergency visits have EMTALA protections but still face scrutiny. Combining the strictest payer with the highest-cost visit types targets a cluster where denials are concentrated.

**3. `payer_id` preserved as a separate feature.** Beyond the 4 broad payer types, individual payer IDs (12 unique values) have their own policies. `payer_id_P008` might deny differently than `payer_id_P001` even within the same type. One-hot encoding captures this granularity.

Without this payer analysis, I would have missed the strongest categorical signal in the data and not created the `is_high_risk_combo` interaction feature.

### 2.2 Gap Analysis Deep Dive

**Q23. You identified `auth_gap` as having the highest lift (+27.6 pp). Walk me through exactly how you computed this metric. Why is it more meaningful than raw correlation?**

**Computation:**

```python
# Step 1: Define the gap
auth_gap = (df['prior_auth_required'] == 1) & (df['has_prior_auth'] == 0)

# Step 2: Compute denial rates for each group
denial_rate_gap_1 = df.loc[auth_gap, 'is_denied'].mean()      # 46.5%
denial_rate_gap_0 = df.loc[~auth_gap, 'is_denied'].mean()     # 18.9%

# Step 3: Compute lift
lift = denial_rate_gap_1 - denial_rate_gap_0                    # 46.5% - 18.9% = 27.6 pp
```

**Why lift is more meaningful than correlation:**

Correlation measures linear association between two continuous variables. But `auth_gap` is binary (0/1) and `is_denied` is binary (0/1). The Pearson correlation between binary variables has a maximum value far below 1.0 (it's bounded by the product of the class probabilities), making the raw correlation coefficient hard to interpret. A correlation of 0.25 between two binary variables might be "strong" by binary correlation standards, but the number 0.25 doesn't tell a non-technical person anything useful.

Lift in percentage points tells an operational story: "Claims missing prior authorization are denied 46.5% of the time, compared to 18.9% when auth is in place. That's a 27.6 percentage point increase in denial risk." A biller immediately understands this: "If I see a missing auth, I know this claim has nearly a 50-50 chance of denial."

Lift also directly feeds into the additive model. If `auth_gap` adds 27.6 pp and `doc_issue` adds 21.0 pp, the combined lift is approximately 48.6 pp (minus some overlap). This mental model matches exactly how LR coefficients work: each gap adds its marginal contribution to the log-odds.

**Q24. The cumulative gap analysis shows 0 gaps = 13.4% denial, 3 gaps = 69.2% denial. Is this relationship perfectly linear? How would you test for linearity? What are the implications of any deviation?**

The relationship is nearly but not perfectly linear. Plotting denial rate vs. number of gaps shows:

| Gaps | Denial Rate | Delta from Previous |
|---|---|---|
| 0 | 13.4% | -- |
| 1 | 23.0% | +9.6 pp |
| 2 | 40.3% | +17.3 pp |
| 3 | 69.2% | +28.9 pp |
| 4 | 100.0% | +30.8 pp |

The deltas INCREASE with each additional gap: +9.6, +17.3, +28.9, +30.8. This shows a slightly super-linear relationship -- each additional gap adds more risk than the previous one. A "perfectly linear" relationship would have constant deltas (e.g., +15 pp per gap).

**How to test formally:**
1. Fit a logistic regression of `is_denied ~ total_admin_gaps` (without `total_admin_gaps^2`). Check the residual deviance.
2. Add `total_admin_gaps^2` as a feature and perform a likelihood ratio test. If the squared term is significant, the relationship is non-linear.
3. Fit a GAM (Generalized Additive Model) with a smooth term for `total_admin_gaps`. Check the effective degrees of freedom of the smooth: edf=1 means linear, edf>1 means non-linear.

**Implications of the super-linear pattern:**
- The `total_admin_gaps` feature alone captures the cumulative effect, but the squared term (`total_admin_gaps^2`) could improve the LR model. Our interaction features (`double_deficit` = auth * doc) partially capture this.
- Tree models (XGBoost, RF) should theoretically handle the non-linearity better than LR -- yet they underperformed. This suggests the non-linearity is mild enough that LR's approximation is sufficient, and trees lose more from their axis-aligned split inefficiency than they gain from capturing the slight curvature.
- For a production system with more data, I'd add `total_admin_gaps^2` as a feature to explicitly capture the accelerating risk.

**Q25. Why did you compute both individual gap lifts AND cumulative gap denial rates? What different insights do they provide?**

Individual gap lifts answer: "Isolating everything else, how much does THIS specific gap increase denial risk?" This is the marginal effect of a single administrative deficiency. It tells you which gaps are most impactful when they occur in isolation or are the only issue on the claim. `auth_gap` at +27.6 pp tells you that missing authorization is, on its own, the single biggest red flag.

Cumulative gap denial rates answer: "How does the risk compound when multiple gaps coexist?" This captures the interaction effect -- the acceleration of risk when a claim has multiple problems. The jump from 1 gap (23.0%) to 2 gaps (40.3%) is larger than the jump from 0 to 1 (13.4% to 23.0%), indicating super-additive risk.

**Why you need both:**

Individual lifts tell you WHERE to focus feature engineering and what to include in the model. Cumulative rates tell you HOW the model should combine them.

If you only had individual lifts, you might assume the relationship is purely additive: auth_gap + doc_issue = 27.6 + 21.0 = 48.6 pp expected lift for a claim with both gaps. But cumulative analysis shows the actual lift is 40.3 - 13.4 = 26.9 pp from baseline to having 2 gaps. Wait -- that seems LOWER than the sum (26.9 vs 48.6). This reveals the overlap: claims with `auth_gap` are more likely to ALSO have `doc_issue`, so the individual lifts partially capture the same population.

This overlap is exactly why we include both individual flags AND `total_admin_gaps`. The individual flags learn the marginal effects; `total_admin_gaps` captures the compounding. The model figures out the right balance from the data.

**Q26. `network_issue` only has a +4.0 pp lift. Why is it so much lower than `auth_gap` (+27.6 pp)? What does this tell you about out-of-network claims?**

The 27.6 pp vs 4.0 pp gap reveals a fundamental difference in denial mechanisms:

**Authorization is a hard gate.** Most payers have automated systems that check for prior authorization before processing. If auth is missing, the system rejects the claim algorithmically -- no human review. This makes `auth_gap` a near-deterministic denial trigger: 46.5% denial rate. The other 53.5% might get through because the service didn't actually require auth despite the flag, or the auth was obtained but not recorded correctly.

**Out-of-network status is a soft gate.** OON claims can still be paid, just at lower rates or with higher patient responsibility. Payers don't automatically deny OON claims -- they process them under out-of-network benefits. The 4.0 pp lift suggests that while OON status increases scrutiny (higher dollar amounts, documentation requirements), it's not a denial trigger in the way missing auth is. Additionally, emergency services have EMTALA protections that require payers to cover OON emergency care at in-network rates.

**Operational insight:** A biller should prioritize fixing `auth_gap` above all else. An OON claim, while worth reviewing, is far less likely to be denied solely because of network status. The 4.0 pp lift represents nuisance denials from OON status, not systematic rejection.

This is also why our `oon_auth_gap` interaction feature matters: OON + missing auth is genuinely dangerous because it combines the worst of both worlds. The model needs to learn that OON matters most when auth is also missing.

**Q27. Only 4 claims have 4 gaps and all were denied. Is this statistically significant? How would you handle such small sample sizes in your analysis?**

**Statistical significance:** With only 4 claims and all denied, this result is suggestive but not statistically significant by conventional standards. A 95% confidence interval for a proportion with 4/4 successes is approximately [39.8%, 100.0%] using the Wilson score interval. This means the true denial rate for 4-gap claims could be as low as 40% -- nowhere near the certainty the 100% observed rate suggests.

A Fisher exact test comparing 4-gap vs 0-gap claims (4/4 vs 213/1587) would show p < 0.05, but with a sample of 4, the test is underpowered and the result is driven entirely by the denominator. This is statistical significance without practical reliability.

**How to handle small sample sizes:**

1. **Report uncertainty explicitly.** In the documentation, I noted "100.0% denial rate, but only 4 claims -- more anecdotal than statistically rigorous."

2. **Pool with adjacent categories.** For modeling purposes, 3-gap and 4-gap claims could be merged into "3+ gaps." This creates a bin with 82 claims (78 with 3 gaps + 4 with 4 gaps) and a denial rate of ~70%, which is statistically informative.

3. **Use regularization.** The LR model's L2 penalty (C=0.1) automatically shrinks coefficients for rare categories, preventing the model from learning "4 gaps = 100% denial" as a hard rule. The model will learn "4 gaps = very high risk" but won't assign probability 1.0.

4. **Acknowledge the limitation to stakeholders.** "We see a clear pattern that 4 gaps means near-certain denial, but we've only seen 4 such claims. We need more data to confirm. In the meantime, flag these as highest priority for review."

**Q28. If you could only use ONE feature to predict denials, which would it be and why? Back up your answer with the data.**

**`total_admin_gaps`** -- the cumulative count of administrative deficiencies.

**Data backing:**
| Gaps | Sample Size | Denial Rate | Separation from Baseline |
|---|---|---|---|
| 0 | 1,587 (49.6%) | 13.4% | Baseline |
| 1 | 1,134 (35.4%) | 23.0% | +9.6 pp |
| 2 | 397 (12.4%) | 40.3% | +26.9 pp |
| 3+ | 82 (2.6%) | 69.2%+ | +55.8+ pp |

**Why this single feature dominates:**

1. **Coverage:** It captures information from all 5 individual gap flags, so one feature summarizes the full administrative picture.

2. **Monotonicity:** The relationship is strictly increasing -- more gaps = more denials. No U-shaped curves, no thresholds where the relationship reverses. This makes it robust to any model that can learn monotonic relationships.

3. **Separation power:** A simple rule "if total_admin_gaps >= 2, flag as high risk" isolates 479 claims (15% of data) with roughly double the baseline denial rate. Even this naive rule outperforms some of our ML models.

4. **Interpretability:** A biller doesn't need to understand coefficients. "3 issues = high risk" is immediately actionable.

5. **Stability:** The relationship holds across payer types, visit types, and time periods. It's not a spurious pattern that might disappear with new data.

If I built a single-feature logistic regression model on `total_admin_gaps`, I'd expect capture@25 around 35-40% -- better than XGBoost and close to the more sophisticated models. This demonstrates that feature engineering dominates model architecture for this problem.

### 2.3 Data Quality & Validation

**Q29. You found 2,509 missing values in `denial_reason`. Is this a problem? Why did you choose to exclude rather than impute?**

The 2,509 missing values (78.4%) in `denial_reason` are not a data quality problem -- they're structural. `denial_reason` is populated ONLY for denied claims (691 rows) and is NULL for all non-denied claims (2,509 rows). The "missingness" IS the information: missing means "not denied."

**Why exclude rather than impute:**

**Imputation would create leakage.** If you impute `denial_reason` with "No Denial" or a special category, you've encoded the TARGET into the feature. Any model will learn "denial_reason != missing" = "denied" -- perfect separation on the training data, useless in production where all current claims have NULL denial reasons.

**Imputation would be meaningless.** The denial_reason contains post-submission payer communication (e.g., "Missing prior authorization," "Service not covered"). This text describes WHY the claim was denied -- it's the explanation of the outcome, not a predictor of it. Including it, even properly imputed, tells the model the answer.

**Exclusion is the ONLY correct choice.** `denial_reason` is excluded programmatically in `prepare_model_matrix()` along with `is_denied` and other leakage columns. The exclusion is automatic, making it impossible for a developer to accidentally include it.

**Q30. How would you detect if the current claims (2025-01) have a different distribution than the historical data (2024)? What statistical tests would you run?**

I'd implement a multi-level distribution drift detection system:

**Level 1: Univariate statistical tests.**
- **Kolmogorov-Smirnov test** for each continuous feature (e.g., `total_billed`, `days_to_submit`). Tests whether the current claim distribution differs from the historical distribution. A significant result (p < 0.05) flags potential drift.
- **Chi-squared test** for each categorical feature (e.g., `payer_type`, `visit_type`). Tests whether the proportions differ across time periods.
- **Z-test for proportions** for binary features (e.g., `auth_gap`, `doc_issue`). Tests whether gap rates have shifted.

**Level 2: Multivariate drift.**
- **Population Stability Index (PSI)** on the model's predicted probabilities. Bins the probability space and compares the proportion of current vs. historical claims in each bin. PSI > 0.25 signals significant drift.
- **Domain classifier:** Train an LR to distinguish historical vs. current claims. If a simple model can achieve ROC-AUC > 0.70, the distributions are meaningfully different.

**Level 3: Business context.**
- Plot feature means by service month within the historical data. Is there already a trend? If `days_to_submit` was increasing from January to December 2024, the January 2025 values continuing that trend is expected, not alarming.
- Interview operations: Did any payer change policies effective January 2025? Did a new EMR go live? Business context explains statistical anomalies.

**Action thresholds:**
- **PSI < 0.1:** No action. Distributions are stable.
- **PSI 0.1-0.25:** Flag for monitoring. Investigate specific features driving the drift.
- **PSI > 0.25:** Alert. Model may need retraining. Consider a "current claims" recalibration before scoring.

---

