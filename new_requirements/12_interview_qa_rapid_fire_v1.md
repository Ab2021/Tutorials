# RAPID-FIRE INTERVIEW Q&A: THE MASTERCLASS (v1)
## The ultimate defense bank tailored to YOUR Resume (No Code)

> **Critical Context:** These are the aggressive, high-speed questions interviewers use to break candidates. Because you are interviewing for a Lead role, they will drill into the exact decisions you made at Chubb, Axtria, and EXL. Your answers must be short, punchy, and highlight the architectural tradeoffs.

---

## SECTION 1: CHUBB FRAUD DETECTION (RAG & XGBoost)

**Q1: "Why did you use LightGBM for real-time scoring but XGBoost for batch scoring at Chubb? That adds DevOps complexity."**
*   **Answer:** "It does, but the business SLA demanded it. LightGBM's leaf-wise growth makes it blazing fast—sub-15ms for inference—which was mandatory for our 200ms real-time API SLA. However, for overnight batch scoring, we didn't care about millisecond speed; we cared about probability calibration. We use the batch scores to tier claims (High/Medium/Low risk). XGBoost's level-wise growth produced significantly better-calibrated probabilities (lower Brier score). The dual-model architecture optimized for both constraints."

**Q2: "In your Chubb RAG pipeline, what if the adjuster's claim note is updated *after* the SIU confirms it's fraud? Doesn't that leak labels into your features?"**
*   **Answer:** "Exactly, that is the biggest trap in fraud modeling. If an adjuster types 'SIU confirmed fraud' into the notes, and we train our embedding pipeline on that, the model learns the future and gets 100% accuracy in testing but fails in production. We architected a strict Temporal Cutoff. The pipeline is hardcoded to only ingest and embed notes written within the first 72 hours of claim creation, absolutely preventing downstream data leakage."

**Q3: "How did you measure the success of the Chubb system?"**
*   **Answer:** "We improved the referral rate to the Special Investigative Unit from 12% to 23%. But referral rate alone is dangerous—you can hit 100% just by flagging everything, which overwhelms the investigators. I tracked 'Precision@Referral' to ensure that of the extra claims we sent them, at least 60% were actually suspicious. We achieved 23% referral at 63% precision."

**Q4: "What happens if your RAG pipeline extracts a fraud flag that isn't actually in the claim notes?"**
*   **Answer:** "Hallucinations happen. We solved this with evaluation and UI design. First, we ran an LLM-as-a-Judge weekly on a Golden Dataset of 200 claims to track 'Faithfulness.' If it dropped, we halted the pipeline. Second, in the investigator UI, every extracted fraud flag (e.g., 'Inconsistent Timeline: True') had a clickable citation linking directly to the exact sentence in the unstructured note, allowing the human to instantly verify the AI's logic."

---

## SECTION 2: AXTRIA MARKETING MIX MODELING (MMM)

**Q5: "Why did you use XGBoost for Marketing Mix Modeling? Everyone uses Linear Regression for MMM because it's interpretable."**
*   **Answer:** "Linear regression is the baseline, and its coefficients are perfectly interpretable. But it fails on two business realities in Pharma marketing: Diminishing returns (saturation) and Channel Synergy (TV boosting Digital). Linear models assume infinite straight-line returns and independence. XGBoost's tree structure naturally captures non-linear saturation curves and complex interactions. To maintain interpretability for the business, I extracted SHAP values to calculate the actual channel attribution."

**Q6: "You used Markov Chains for attribution. Why not just use Shapley values or a deep neural network?"**
*   **Answer:** "Deep neural nets require massive data volume that we didn't have for weekly Pharma campaigns, and they are black boxes. Shapley values are great but computationally heavy. Markov Chains were the sweet spot. They specifically model sequential journeys (Email -> Search -> Web), allowing us to calculate the 'removal effect'—what happens to conversion probability if a specific touchpoint is removed. It was mathematically sound but still transparent enough to explain to marketing stakeholders."

**Q7: "How did you handle highly correlated marketing channels (Multicollinearity)?"**
*   **Answer:** "If TV spend and Search spend always go up in the exact same week, the model struggles to assign credit. First, I checked Variance Inflation Factor (VIF). If channels were highly collinear, I either mathematically combined them into a single 'Upper Funnel Media' feature, or I relied on XGBoost's regularization (which handles collinearity better than OLS regression). Crucially, I sanity-checked the final attribution against the business's historical priors."

---

## SECTION 3: EXL HEALTH (PATIENT READMISSION & SURVIVAL ANALYSIS)

**Q8: "Why did you use Kaplan-Meier survival analysis for patient engagement at EXL instead of a simple churn classification model?"**
*   **Answer:** "Because of 'Right-Censored' data. If we study patients for 6 months, and a patient is engaged at month 4 when the study ends, a standard classification model doesn't know what to do with them—did they eventually drop off? Survival analysis mathematically accounts for patients who haven't experienced the event yet. The Kaplan-Meier curve gave the business a time-to-event probability (e.g., '75% engaged at month 1, 40% at month 6'), which let them precisely target their intervention timing."

**Q9: "For the readmission model, how did you balance model performance against clinical trust?"**
*   **Answer:** "I could have built a complex deep learning model to squeeze out 2 more points of AUC, but if a nurse doesn't understand why the model flagged a patient, they won't intervene. I stuck to Logistic Regression and Random Forests, prioritizing clinical explainability (SHAP). I also collaborated directly with clinicians to ensure the input features (like the Charlson Comorbidity Index) were metrics they already trusted and used in their daily workflows."

**Q10: "If your EXL readmission model predicted a 0.8 risk score, what does that actually mean?"**
*   **Answer:** "It MUST mean there is an 80% real-world probability of readmission. If a model is miscalibrated (e.g., the score is 0.8 but only 40% of those patients are readmitted), clinicians will suffer alarm fatigue and ignore the system. I used Platt Scaling on a hold-out validation set and rigorously monitored the Brier Score to guarantee that our model's confidence perfectly matched the real-world outcome distribution."

---

## SECTION 4: LEADERSHIP & CONSULTING STRATEGY

**Q11: "Tell me about a time you disagreed with a stakeholder on model architecture."**
*   **Answer:** "At EXL, a client wanted a complex Neural Network for a small, structured tabular dataset. I knew it would overfit and be uninterpretable. I didn't argue from opinion; I built a Logistic Regression baseline in a few hours. It achieved 85% of their target AUC with 100% explainability. I presented the tradeoff: 'We can use this safe, explainable model today, or spend a month building a black box for a 3% gain.' They chose the regression model. Data beats opinions."

**Q12: "How do you scope a new AI project for an SME client?"**
*   **Answer:** "I focus heavily on the 'Tolerance for Error.' If they want an LLM to generate legal contracts and send them automatically, the tolerance for error is zero. I will immediately reject an autonomous architecture and pivot them to a 'Copilot' architecture, where the AI drafts the contract and a human hits approve. I also audit their baseline data—if their PDFs are unreadable by a human, I stop the AI conversation and scope a data digitization phase first."
