# Day 35: Safety, Bias, and Fairness in LLMs
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the difference between bias and fairness?

**Answer:**
- **Bias:** Systematic errors or prejudices in the model's behavior (e.g., associating "nurse" with "female").
- **Fairness:** A normative concept about how the model should treat different groups. Multiple definitions exist (Demographic Parity, Equalized Odds, Calibration).
- **Relationship:** Bias often leads to unfairness, but they are distinct concepts. A model can be biased but still satisfy certain fairness criteria.

#### Q2: Why is it impossible to satisfy all fairness criteria simultaneously?

**Answer:**
- **Mathematical Impossibility:** For any imperfect classifier, Demographic Parity, Equalized Odds, and Calibration cannot all be satisfied at once (except in trivial cases where base rates are equal across groups).
- **Implication:** You must choose which fairness criterion is most appropriate for your application.
- **Example:** In lending, Calibration is critical (predicted default rate should match actual rate). In hiring, Equalized Odds may be more important (equal TPR/FPR across demographics).

#### Q3: What is "bias amplification" and how does it occur?

**Answer:**
- **Definition:** When a model exaggerates biases present in the training data.
- **Example:** Training data has 60% male CEOs. Model generates 90% male CEOs.
- **Cause:** The model learns strong correlations and extrapolates them beyond the training distribution.
- **Mitigation:** Counterfactual data augmentation, adversarial debiasing, balanced training data.

#### Q4: How do you detect PII (Personally Identifiable Information) leakage?

**Answer:**
- **Canary Extraction:** Insert unique strings into training data and check if the model reproduces them.
- **Regex Patterns:** Scan outputs for patterns (SSN, credit card numbers, emails).
- **NER (Named Entity Recognition):** Use a model to detect PERSON, ORG, LOC entities.
- **Mitigation:** Data scrubbing before training, output filtering, differential privacy.

#### Q5: What is the role of red teaming in LLM safety?

**Answer:**
- **Purpose:** Proactively find safety vulnerabilities before deployment.
- **Process:** Hire diverse adversarial testers to try to make the model generate harmful content.
- **Outcome:** Catalog successful attacks, add them to training data (with safe responses), update safety filters.
- **Iteration:** Continuous process. GPT-4 had 50+ external red teamers over 6 months.

---

### Production Challenges

#### Challenge 1: Balancing Safety and Utility

**Scenario:** Your safety filter is too aggressive. It refuses benign requests like "How to kill a process in Linux?"
**Root Cause:** Keyword-based filter flags "kill" as violent.
**Solution:**
- **Context-Aware Filtering:** Use a classifier that understands context (e.g., "kill" in a technical context is benign).
- **Fine-Tuning:** Train the safety model on examples of benign uses of sensitive words.
- **User Feedback:** Track false refusals. Adjust thresholds based on user reports.

#### Challenge 2: Measuring Bias in Open-Ended Generation

**Scenario:** You want to measure gender bias in your chatbot, but it generates free-form text (not classification).
**Solution:**
- **BOLD Benchmark:** Prompt with demographic descriptors ("The Black woman..."). Measure sentiment and toxicity of completions.
- **Pronoun Analysis:** Count gender pronouns in generated text about professions. Compare to baseline.
- **Human Evaluation:** Have diverse annotators rate outputs for bias.

#### Challenge 3: Debiasing Without Hurting Performance

**Scenario:** You applied counterfactual data augmentation. Bias decreased, but so did overall quality.
**Root Cause:** Synthetic counterfactuals are lower quality than real data.
**Solution:**
- **Hybrid Approach:** 80% real data + 20% high-quality counterfactuals.
- **Adversarial Debiasing:** Use adversarial training instead of data augmentation.
- **Post-Processing:** Apply debiasing only at inference time (e.g., re-rank outputs to balance demographics).

#### Challenge 4: Compliance with Regulations

**Scenario:** You're deploying in the EU. The AI Act requires safety assessments for high-risk systems.
**Requirements:**
- **Documentation:** Model card, training data provenance, safety evaluations.
- **Testing:** Bias audits, adversarial testing, human evaluation.
- **Monitoring:** Continuous logging of safety incidents.
**Solution:**
- **Automated Compliance:** Use tools like HELM for standardized evaluations.
- **Third-Party Audits:** Hire external auditors to validate safety claims.
- **Transparency Reports:** Publish regular reports on safety incidents and mitigations.

#### Challenge 5: Handling Disagreement in Safety Labels

**Scenario:** You have 3 annotators label the same output as toxic/non-toxic. They disagree 40% of the time.
**Analysis:** Safety is subjective. Different people have different thresholds.
**Solution:**
- **Majority Vote:** Use the majority label.
- **Confidence Weighting:** Weight labels by annotator confidence.
- **Disagreement as Signal:** High disagreement = ambiguous case. Flag for expert review or exclude from training.
- **Diverse Annotators:** Ensure annotators represent diverse demographics and perspectives.

### Summary Checklist for Production
- [ ] **Toxicity:** Use **Perspective API** to filter outputs.
- [ ] **Bias:** Run **BOLD** and **BBQ** benchmarks.
- [ ] **PII:** Scrub training data and filter outputs with **NER**.
- [ ] **Red Team:** Conduct **adversarial testing** before release.
- [ ] **Monitor:** Track **safety incidents** in production.
- [ ] **Compliance:** Maintain **documentation** for regulatory audits.
