# RAPID-FIRE INTERVIEW Q&A: THE RESUME MASTERCLASS (v1)
## The ultimate technical defense bank tailored to YOUR Resume (No Code)

> **Critical Context:** These are the aggressive, high-speed questions Lead Technical Interviewers use to break candidates who rely on buzzwords. Because you are interviewing for a Lead / AI Engineer role, they will drill into the exact decisions you made at Chubb, Axtria, and EXL. Your answers must be short, punchy, and highlight the architectural tradeoffs and mathematical realities.

---

## SECTION 1: CHUBB FRAUD DETECTION (RAG & MLOPS)

**Q1: "You used LightGBM for real-time scoring and XGBoost for batch scoring at Chubb. That adds massive DevOps complexity. Why not just use one?"**
*   **The Deep-Dive Answer:** "It does add CI/CD overhead, but the business SLA demanded it. LightGBM's leaf-wise tree growth and histogram binning make it blazing fast—sub-15ms for inference. This was mathematically mandatory for our 200ms real-time API SLA. However, for overnight batch scoring, we didn't care about millisecond speed; we cared about probability calibration. We use the batch scores to strictly tier claims (High/Medium/Low risk). XGBoost's level-wise growth produces significantly better-calibrated probabilities. I proved this by plotting Reliability Diagrams and calculating the Brier score (XGBoost 0.042 vs LightGBM 0.058). The dual-model architecture optimized for both strict constraints."

**Q2: "In your Chubb RAG pipeline, what if the adjuster's claim note is updated *after* the SIU confirms it's fraud? Doesn't that leak labels into your features?"**
*   **The Deep-Dive Answer:** "Exactly, that is the biggest trap in fraud modeling. If an adjuster types 'SIU confirmed fraud' into the notes, and we train our embedding pipeline on that, the model learns the future. It will hit 100% accuracy in testing but fail catastrophically in production. We architected a strict Temporal Cutoff. The data engineering pipeline is hardcoded with a SQL `AS_OF` join. It only ingests and embeds notes written within the first 72 hours of claim creation, absolutely preventing downstream data leakage."

**Q3: "How did you measure the success of the Chubb system? Did you use ROC-AUC?"**
*   **The Deep-Dive Answer:** "I explicitly did NOT use ROC-AUC. Fraud is a highly imbalanced dataset (e.g., < 2% of claims). ROC-AUC is misleading on imbalanced data because it rewards correctly classifying the massive 'majority' class (True Negatives). I optimized for PR-AUC (Precision-Recall Area Under Curve). 
Our business goal was to improve the referral rate to the Special Investigative Unit from 12% to 23%. But referral rate alone is dangerous—you can hit 100% just by flagging everything, which overwhelms the investigators. I tracked 'Precision@Referral' to ensure that of the extra claims we sent them, at least 60% were actually suspicious. We hit the 23% referral target while maintaining 63% precision."

**Q4: "What happens if your RAG pipeline extracts a fraud flag that isn't actually in the claim notes?"**
*   **The Deep-Dive Answer:** "Hallucinations happen, especially with LLMs. We solved this with CI/CD evaluation and UI design. First, we ran the Ragas framework (LLM-as-a-Judge) weekly on a Golden Dataset of 200 claims to track 'Faithfulness.' If Faithfulness dropped below 0.90, the pipeline deployment halted. Second, in the investigator UI, every extracted fraud flag (e.g., 'Inconsistent Timeline: True') had a clickable citation. It linked directly to the exact source sentence in the unstructured note, allowing the human to instantly verify the AI's logic."

**Q5: "How did you handle the cold-start problem in Kubernetes for your real-time LightGBM pods?"**
*   **The Deep-Dive Answer:** "If a pod scales up and tries to load a 2GB model into memory on the first user request, that request times out. We configured Kubernetes Readiness Probes. The pod spins up, loads the LightGBM model into global memory during the startup lifecycle hook, and ONLY THEN does the Readiness Probe return `200 OK` to the load balancer, indicating it is ready to receive traffic. This ensures zero latency spikes during Horizontal Pod Autoscaling (HPA) events."

---

## SECTION 2: AXTRIA MARKETING MIX MODELING (MMM)

**Q6: "Why did you use XGBoost for Marketing Mix Modeling? Everyone uses Linear Regression for MMM because it's interpretable."**
*   **The Deep-Dive Answer:** "Linear regression is the baseline, and its coefficients are perfectly interpretable. But it fails on two mathematical realities in Pharma marketing: Diminishing returns (saturation) and Channel Synergy (e.g., TV spend making Google Search ads cheaper and more effective). Linear models assume infinite straight-line returns and independence. XGBoost's tree structure naturally captures non-linear saturation curves and complex interactions. To maintain interpretability for the business stakeholders, I extracted SHAP (SHapley Additive exPlanations) values to calculate the actual channel attribution."

**Q7: "You used Markov Chains for attribution. Why not just use Shapley values or a deep neural network?"**
*   **The Deep-Dive Answer:** "Deep neural nets require massive data volume that we didn't have for weekly Pharma campaigns (n=156 weeks), and they are black boxes. Shapley values are theoretically perfect but computationally heavy to calculate across thousands of multi-touch permutations. Markov Chains were the sweet spot. They specifically model sequential journeys as a transition matrix (Email -> Search -> Web), allowing us to calculate the 'Removal Effect'—what happens to the overall conversion probability if a specific touchpoint is mathematically removed from the graph. It was statistically sound but still transparent enough to explain to marketing directors."

**Q8: "How did you handle highly correlated marketing channels (Multicollinearity)?"**
*   **The Deep-Dive Answer:** "If TV spend and Search spend always launch in the exact same week, the model struggles to assign isolated credit. First, I checked the Variance Inflation Factor (VIF). If channels were highly collinear (VIF > 10), I either mathematically combined them into a single 'Upper Funnel Media' feature, or I relied on XGBoost's L2 regularization, which handles collinearity vastly better than OLS regression. Crucially, I sanity-checked the final attribution against the business's historical priors to ensure the model wasn't generating artifacts."

**Q9: "Explain the Adstock transformation and how you tuned its decay rate."**
*   **The Deep-Dive Answer:** "Marketing spend has a memory effect. A commercial watched today influences a purchase next week. The Adstock formula is `Adstock(t) = Spend(t) + decay_rate * Adstock(t-1)`. The `decay_rate` (between 0 and 1) dictates how long the memory lasts. We did not guess this parameter. We ran a Grid Search across a range of decay values (e.g., 0.1 to 0.9) for each channel, optimizing for the lowest out-of-sample MAPE (Mean Absolute Percentage Error) on our holdout validation set."

---

## SECTION 3: EXL HEALTH (PATIENT READMISSION & SURVIVAL ANALYSIS)

**Q10: "Why did you use Kaplan-Meier survival analysis for patient engagement at EXL instead of a simple churn classification model?"**
*   **The Deep-Dive Answer:** "Because of 'Right-Censored' data. If we study patients for 6 months, and a patient is perfectly engaged at month 4 when our study arbitrarily ends, a standard classification model doesn't know what to do with them—did they eventually drop off? Survival analysis mathematically accounts for patients who haven't experienced the event yet. The Kaplan-Meier estimator built a survival curve that gave the business a precise time-to-event probability (e.g., '75% engaged at month 1, 40% at month 6'), which let the care managers precisely target their intervention timing."

**Q11: "For the readmission model, how did you balance model performance against clinical trust?"**
*   **The Deep-Dive Answer:** "I could have built a complex deep learning model to squeeze out 2 more points of AUC, but if a nurse doesn't understand *why* the model flagged a patient, they will ignore it. I stuck to Logistic Regression (for the baseline) and Random Forests, prioritizing clinical explainability via SHAP. Furthermore, I collaborated directly with clinicians to ensure the engineered input features (like the Charlson Comorbidity Index) were standardized medical metrics they already trusted and used in their daily workflows, rather than obscure mathematical features."

**Q12: "If your EXL readmission model predicted a 0.8 risk score, what does that actually mean?"**
*   **The Deep-Dive Answer:** "It MUST mean there is an 80% real-world probability of readmission. If a model is miscalibrated (e.g., the score says 0.8 but only 40% of those patients are actually readmitted), clinicians will suffer alarm fatigue and abandon the system. Random Forests are notoriously uncalibrated; they push probabilities toward the mean (0.5). I used Platt Scaling on a strictly held-out calibration dataset and rigorously monitored the Brier Score to guarantee that our model's confidence perfectly matched the real-world outcome distribution."

---

## SECTION 4: LEADERSHIP & STRATEGY

**Q13: "Tell me about a time you disagreed with a stakeholder on model architecture."**
*   **The Deep-Dive Answer:** "At EXL, a client demanded a complex Neural Network for a very small, structured tabular dataset. I knew it would overfit massively and act as a black box. I didn't argue from opinion; I built a Logistic Regression baseline in a few hours. It achieved 85% of their target AUC with 100% explainability. I presented the tradeoff in business terms: 'We can use this safe, explainable model in production today, or spend a month building a black box for a theoretical 3% gain.' They chose the regression model. Data beats opinions."

**Q14: "How do you ensure your junior engineers don't deploy data leakage into production?"**
*   **The Deep-Dive Answer:** "I enforce architectural guardrails at the CI/CD level. I teach them that all temporal features must be created using an `AS_OF` timestamp join. I mandate a code review policy where any PR touching feature engineering requires explicit proof of timestamp boundaries. We also run a baseline test: If a feature achieves an impossibly high correlation with the target variable during training, the pipeline automatically flags it as suspected leakage and halts the build for manual review."
