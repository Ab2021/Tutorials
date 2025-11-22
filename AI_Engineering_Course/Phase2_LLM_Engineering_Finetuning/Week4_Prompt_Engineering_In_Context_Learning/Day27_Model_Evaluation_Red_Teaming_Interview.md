# Day 27: Model Evaluation & Red Teaming
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the difference between Static and Dynamic Red Teaming?

**Answer:**
- **Static:** Running a fixed dataset of known attack prompts (e.g., "JailbreakBench"). Good for regression testing.
- **Dynamic:** Using humans or AI agents to actively probe the model in real-time, adapting to its responses. Good for finding *new*, unknown vulnerabilities.

#### Q2: Explain "False Refusal Rate" (FRR). Why is it a problem?

**Answer:**
- **Definition:** The percentage of harmless prompts that the model refuses to answer (thinking they are harmful).
- **Example:** User: "How to kill a process?" Model: "I cannot help with killing."
- **Problem:** High FRR frustrates users and makes the model unusable for professional tasks. It indicates the safety filter is "over-fitted".

#### Q3: How does "Garak" work?

**Answer:**
- Garak is an LLM vulnerability scanner.
- It uses "Probes" (generators of attack prompts) and "Detectors" (analyzers of model output).
- It sends thousands of attacks (Injection, Leakage, Hallucination triggers) to the model and reports the success rate. It's like Nmap for LLMs.

#### Q4: What is the role of a "Safety Reward Model" in RLHF?

**Answer:**
- In standard RLHF, the Reward Model encourages helpfulness.
- A Safety RM is trained specifically to predict "Is this response safe?".
- During PPO training, we subtract the Safety Score from the total reward if the response is unsafe. This penalizes the model for generating toxic content, even if it is "helpful" to a malicious user.

#### Q5: Can Red Teaming guarantee a model is safe?

**Answer:**
- **No.** It can only prove the presence of vulnerabilities, not their absence.
- The input space of an LLM is infinite. There will always be an "adversarial example" that hasn't been found yet. Red Teaming just raises the bar for the attacker.

---

### Production Challenges

#### Challenge 1: The "Whack-a-Mole" Problem

**Scenario:** You fix one jailbreak ("DAN"), and users find another ("Developer Mode").
**Solution:**
- **Generalize:** Don't just train on specific jailbreak strings. Train on the *intent* or *style* of jailbreaks (e.g., roleplaying, hypothetical framing).
- **Constitutional AI:** Use RLAIF to teach the model broad principles ("Do not help with illegal acts") rather than specific pattern matching.

#### Challenge 2: Evaluating Hallucinations at Scale

**Scenario:** Users complain the model lies, but you can't manually check 1M logs.
**Solution:**
- **SelfCheckGPT:** For every response, sample 5 more responses. If they contradict each other, flag as hallucination.
- **NLI-based Eval:** Use a small Entailment model to check if the generated answer is supported by the retrieved context (in RAG).

#### Challenge 3: Safety Regression

**Scenario:** You fine-tuned the model on new data, and it suddenly became racist.
**Root Cause:** The new data wasn't scrubbed, or the fine-tuning broke the safety alignment (Catastrophic Forgetting).
**Solution:**
- **Golden Set:** Always run your Safety Benchmark (RealToxicityPrompts) before deploying any new checkpoint.
- **Mix-in:** Include safety demonstrations in your fine-tuning dataset.

#### Challenge 4: Cost of Automated Red Teaming

**Scenario:** Running Garak with GPT-4 as the judge is too expensive.
**Solution:**
- **Cheaper Judge:** Use a fine-tuned LLaMA-7B as the safety judge.
- **Tiered Testing:** Run cheap static tests (regex, keyword) first. Only run expensive AI-based attacks on major releases.

### Summary Checklist for Production
- [ ] **Scan:** Run **Garak** on every release.
- [ ] **Metric:** Track **False Refusal Rate**.
- [ ] **Data:** Scrub training data for toxicity.
- [ ] **Monitor:** Log user reports of safety issues.
