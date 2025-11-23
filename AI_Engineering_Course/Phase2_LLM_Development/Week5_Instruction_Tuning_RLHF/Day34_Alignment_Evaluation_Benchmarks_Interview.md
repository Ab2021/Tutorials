# Day 34: Alignment Evaluation & Benchmarks
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Why can't we just use MMLU to evaluate chatbot quality?

**Answer:**
- **MMLU** tests knowledge and reasoning on multiple-choice questions.
- **Chatbot Quality** requires helpfulness, conversational ability, safety, and instruction following.
- **Correlation:** MMLU correlates moderately (0.75) with chatbot quality, but misses key dimensions like tone, safety, and multi-turn coherence.
- **Conclusion:** MMLU is necessary but not sufficient. Use MT-Bench or Chatbot Arena for chatbot evaluation.

#### Q2: What is the "Position Bias" problem in LLM-as-a-Judge?

**Answer:**
- **Problem:** When comparing two responses (A and B), the judge LLM statistically prefers the first response (A) more often than it should.
- **Cause:** Training data bias (in many datasets, the first option is correct more often).
- **Fix:** Run evaluation twice: (A vs B) and (B vs A). If the judge prefers A both times, A wins. If it flips, it's a tie.

#### Q3: Explain the Elo rating system used in Chatbot Arena.

**Answer:**
- **Concept:** Borrowed from chess. Each model has a rating (e.g., 1300).
- **Matches:** Users chat with two anonymous models and vote for the better one.
- **Update:** Winner's rating increases, loser's decreases. The amount depends on the rating difference (upset wins give more points).
- **Convergence:** After ~1000 matches, ratings stabilize and reflect true model quality.

#### Q4: What is calibration and why does it matter?

**Answer:**
- **Calibration:** Does the model's confidence match its accuracy?
- **Example:** If the model says "I'm 90% confident" on 100 questions, it should get ~90 correct.
- **Why It Matters:** Overconfident models mislead users. Well-calibrated models help users trust the right answers and doubt the wrong ones.

#### Q5: How do you measure toxicity in LLM outputs?

**Answer:**
- **Perspective API:** A pre-trained classifier that scores text on toxicity (0.0-1.0).
- **RealToxicityPrompts:** A benchmark with 100k prompts designed to elicit toxic responses. Measure the Expected Maximum Toxicity over 25 generations.
- **Human Evaluation:** Sample outputs and have humans label them as toxic/non-toxic.

---

### Production Challenges

#### Challenge 1: MT-Bench Scores Don't Match User Feedback

**Scenario:** Your model scores 8.5 on MT-Bench, but users complain it's unhelpful.
**Root Cause:**
- **Judge Bias:** GPT-4 prefers verbose, polite responses. Your model is verbose but not actually helpful.
- **Domain Mismatch:** MT-Bench covers general topics. Your users ask domain-specific questions (e.g., legal, medical).
**Solution:**
- **Custom Benchmark:** Create a domain-specific eval set with real user queries.
- **Human Validation:** Run a user study. Compare MT-Bench scores with human preferences.

#### Challenge 2: Evaluating Safety at Scale

**Scenario:** You have 1M user interactions. You can't manually review them all for safety violations.
**Solution:**
- **Automated Filters:** Use Perspective API to flag high-toxicity responses (>0.7).
- **Sampling:** Review a random sample of 1000 interactions per day.
- **User Reports:** Track user "Report" button clicks. Prioritize reviewing reported interactions.
- **Anomaly Detection:** Flag unusual patterns (e.g., sudden spike in refusals).

#### Challenge 3: Benchmark Saturation

**Scenario:** Your model scores 95% on MMLU. Further improvements don't increase the score.
**Root Cause:** The benchmark is saturated. The remaining 5% are ambiguous or incorrectly labeled questions.
**Solution:**
- **Harder Benchmarks:** Switch to BBH (Big Bench Hard) or GPQA (Graduate-Level Questions).
- **Human Eval:** For saturated benchmarks, focus on human evaluation instead.

#### Challenge 4: Correlation vs. Causation

**Scenario:** Model A scores higher than Model B on MT-Bench. You deploy Model A, but users prefer Model B.
**Analysis:** MT-Bench is a proxy, not ground truth. It might miss dimensions users care about (e.g., speed, personality).
**Solution:**
- **A/B Testing:** Always validate with real users before full deployment.
- **Multi-Metric:** Don't rely on a single benchmark. Use MT-Bench + AlpacaEval + Human Eval.

#### Challenge 5: Evaluating Refusal Behavior

**Scenario:** You want the model to refuse harmful requests but answer benign ones. How do you measure this?
**Metrics:**
- **True Positive Rate (Sensitivity):** % of harmful requests correctly refused.
- **False Positive Rate:** % of benign requests incorrectly refused.
**Benchmark:**
- **Do Not Answer:** 939 harmful instructions. Measure refusal rate (should be ~100%).
- **Benign Set:** 1000 benign instructions. Measure refusal rate (should be ~0%).
**Trade-off:** Lowering FPR (fewer false refusals) often increases risk of missing harmful requests.

### Summary Checklist for Production
- [ ] **Automated:** Run **MT-Bench** and **AlpacaEval** daily.
- [ ] **Safety:** Run **TruthfulQA** and **RealToxicity** weekly.
- [ ] **Knowledge:** Run **MMLU** and **GSM8K** on releases.
- [ ] **Human:** Conduct **user studies** monthly.
- [ ] **Arena:** Deploy to **Chatbot Arena** for Elo rating.
- [ ] **Red Team:** Run **adversarial testing** before major releases.
