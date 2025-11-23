# Day 32: Constitutional AI & RLAIF
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the main advantage of Constitutional AI over traditional RLHF?

**Answer:**
- **Transparency:** The values are explicit in the Constitution, not implicit in human preferences. You can inspect, modify, and audit the principles.
- **Scalability:** Uses self-critique and AI judges, eliminating the need for massive human labeling.
- **Consistency:** AI judges are more consistent than human labelers (no fatigue, no subjective drift).
- **Cost:** 100x cheaper than human RLHF.

#### Q2: How does self-critique work in Constitutional AI?

**Answer:**
1. The model generates an initial response.
2. The model critiques its own response against a principle (e.g., "Is this harmful?").
3. The model revises the response to address the critique.
4. This process can be iterated 2-3 times.
5. The final revised responses are used for SFT.

**Key Insight:** The model learns to internalize the Constitution by practicing self-correction.

#### Q3: What is RLAIF and how does it differ from RLHF?

**Answer:**
- **RLHF:** Uses human labelers to generate preference data.
- **RLAIF:** Uses an AI model (GPT-4, Claude-3) as the judge to generate preference data.
- **Difference:** RLAIF is much cheaper and faster, but inherits the biases of the judge model.

#### Q4: What are the limitations of using AI judges?

**Answer:**
- **Bias:** AI judges have their own biases (e.g., GPT-4 prefers verbose, polite responses).
- **Errors:** AI judges can make mistakes, especially on complex or nuanced questions.
- **Circular Dependency:** Using GPT-4 to train a model to be like GPT-4 limits diversity.
- **Lack of Groundedness:** AI judges can hallucinate or make incorrect factual judgments.

#### Q5: Can you combine Constitutional AI with human feedback?

**Answer:**
- **Yes.** Best practice is a hybrid approach:
  1. **Stage 1:** Use Constitutional AI to generate 100k synthetic preferences (cheap, fast).
  2. **Stage 2:** Collect 10k human preferences on critical safety issues (expensive, high-quality).
  3. **Stage 3:** Train the Reward Model on the combined dataset.
  4. **Stage 4:** Run PPO or DPO.

---

### Production Challenges

#### Challenge 1: Judge Model Bias

**Scenario:** You use GPT-4 as the judge. Your trained model becomes overly verbose and apologetic (like GPT-4).
**Root Cause:** The judge model's preferences are baked into the training data.
**Solution:**
- **Diverse Judges:** Use an ensemble of judges (GPT-4, Claude-3, Gemini).
- **Prompt Variation:** Use different judge prompts to elicit diverse preferences.
- **Human Validation:** Sample 1k examples and have humans validate the AI judgments.

#### Challenge 2: Conflicting Principles

**Scenario:** Your Constitution has "Be helpful" and "Be harmless". For a question like "How to make a bomb?", these conflict.
**Solution:**
- **Hierarchy:** Define a priority order. Harmlessness > Honesty > Helpfulness.
- **Conditional Principles:** "Be helpful, except when the request is harmful."
- **Explicit Refusal:** Add a principle: "Refuse requests for illegal or dangerous information."

#### Challenge 3: Self-Critique Failure

**Scenario:** You run self-critique, but the model says "No issues found" even for clearly harmful responses.
**Root Cause:** The model is too small (<7B) or not well-aligned.
**Solution:**
- **Use Larger Model:** GPT-4 or Claude-3 for critique (even if training a smaller model).
- **Few-Shot Critique:** Provide examples of good critiques in the prompt.
- **External Critic:** Use a separate, specialized safety model for critique.

#### Challenge 4: Evaluating Constitutional Compliance

**Scenario:** You trained with Constitutional AI. How do you measure if the model actually follows the Constitution?
**Solution:**
- **Automated Tests:** Create a test set of prompts designed to violate each principle. Measure refusal rate.
- **Red Teaming:** Hire adversarial testers to try to make the model violate principles.
- **User Feedback:** Track real-world user reports of violations.

#### Challenge 5: Updating the Constitution

**Scenario:** You discover a new safety issue (e.g., model helps with phishing). You want to add a principle.
**Solution:**
- **Incremental Training:** Add the new principle to the Constitution. Generate new critique/revision data. Fine-tune the model (1 epoch).
- **Iterative RLHF:** Collect new preferences with the updated Constitution. Retrain the RM. Run PPO/DPO again.
- **A/B Testing:** Deploy the updated model to 10% of users. Monitor for regressions.

### Summary Checklist for Production
- [ ] **Constitution:** Define **10-20 clear principles**.
- [ ] **Hierarchy:** Establish **priority order** (Harmlessness > Honesty > Helpfulness).
- [ ] **Critique:** Use **GPT-4/Claude-3** for self-critique.
- [ ] **Judge:** Use **ensemble of AI judges** for preferences.
- [ ] **Validation:** Sample **1k human validations**.
- [ ] **Monitor:** Track **principle violations** in production.
