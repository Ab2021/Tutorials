# Day 49: MLOps for LLMs
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the difference between traditional MLOps and LLM MLOps?

**Answer:**
**Traditional MLOps:**
- Model training, deployment, monitoring.
- Deterministic outputs (same input → same output).
- Metrics: Accuracy, precision, recall.

**LLM MLOps:**
- Prompt management, fine-tuning, evaluation.
- Non-deterministic outputs (same input → different outputs).
- Metrics: User satisfaction, hallucination rate, cost.
- **Additional:** Prompt versioning, A/B testing, cost tracking.

#### Q2: How do you version and manage prompts in production?

**Answer:**
**Versioning:**
- Store prompts in version control (Git).
- Semantic versioning (v1.0, v1.1, v2.0).
- Track metadata (author, date, purpose).

**Management:**
- **Prompt Registry:** Central store for all prompts.
- **A/B Testing:** Test multiple prompt versions.
- **Rollback:** Revert to previous version if needed.

**Example:** `prompts/system/v1.txt`, `prompts/system/v2.txt`.

#### Q3: Explain canary deployment for LLM models.

**Answer:**
- **Concept:** Deploy new model to small % of traffic, monitor, gradually increase.
- **Process:**
  1. Deploy canary (5% traffic).
  2. Monitor metrics (latency, quality, cost).
  3. If OK, increase to 50%.
  4. If OK, increase to 100%.
  5. If issues, rollback to 0%.
- **Benefits:** Reduces risk, allows gradual rollout.
- **Metrics to Monitor:** Latency, error rate, user satisfaction.

#### Q4: How do you evaluate LLM quality in production?

**Answer:**
**Automated:**
- **Regression Tests:** Run on test set, check metrics (accuracy, hallucination rate).
- **LLM-as-Judge:** Use another LLM to evaluate outputs.
- **Benchmark Scores:** MT-Bench, AlpacaEval.

**User Feedback:**
- **Thumbs Up/Down:** Direct user feedback.
- **Ratings:** 1-5 star ratings.
- **A/B Testing:** Compare user satisfaction between versions.

**Continuous:** Evaluate every deployment, alert if metrics degrade >5%.

#### Q5: What is shadow deployment and when should you use it?

**Answer:**
- **Concept:** New model receives same requests as current model but doesn't serve users.
- **Purpose:** Compare outputs without affecting users.
- **Process:**
  1. Deploy shadow model.
  2. Send all requests to both current and shadow.
  3. Compare outputs (quality, latency).
  4. If shadow performs well, promote to canary.
- **When:** Testing major model changes, validating before canary.

---

### Production Challenges

#### Challenge 1: Prompt Drift

**Scenario:** Prompt v1 worked well. After 3 months, quality degraded (same prompt, worse outputs).
**Root Cause:** Model behavior changed (fine-tuning, updates) or user expectations changed.
**Solution:**
- **Monitor Quality:** Track user satisfaction over time.
- **Regression Tests:** Run same test set monthly, check if outputs changed.
- **Prompt Refresh:** Update prompts every 3-6 months.
- **Version Lock:** Pin to specific model version if prompt is critical.

#### Challenge 2: A/B Test Inconclusive

**Scenario:** A/B tested prompt v1 vs v2. Both have same user satisfaction (80%).
**Root Cause:** Insufficient sample size or no real difference.
**Solution:**
- **Increase Sample Size:** Run for longer (1 week → 1 month).
- **Statistical Significance:** Use t-test to check if difference is significant.
- **Secondary Metrics:** Check latency, cost (if same quality, choose cheaper).
- **Qualitative Analysis:** Manually review outputs to find subtle differences.

#### Challenge 3: Canary Rollout Failed

**Scenario:** Canary at 50% traffic. Latency spiked to 5s (vs 1s for current).
**Root Cause:** Canary model is larger or slower.
**Solution:**
- **Immediate Rollback:** Reduce canary to 0%.
- **Optimize Canary:** Quantize, use smaller model, or optimize serving.
- **Gradual Rollout:** Start with 1% instead of 5%.
- **Shadow First:** Test in shadow mode before canary.

#### Challenge 4: Regression Test False Positives

**Scenario:** Regression test fails (accuracy dropped from 85% to 84%) but outputs are actually better.
**Root Cause:** Test set is outdated or metrics don't capture quality.
**Solution:**
- **Update Test Set:** Refresh test set every 6 months.
- **Multiple Metrics:** Don't rely on single metric (use accuracy + user satisfaction).
- **Manual Review:** Manually review failed examples.
- **Threshold:** Allow 1-2% drop (not 0% tolerance).

#### Challenge 5: Model Registry Bloat

**Scenario:** Model registry has 100 versions. Storage cost is high.
**Root Cause:** Keeping all versions indefinitely.
**Solution:**
- **Retention Policy:** Delete versions older than 6 months (except production).
- **Archive:** Move old versions to cheaper storage (S3 Glacier).
- **Tagging:** Tag important versions (production, baseline) to prevent deletion.
- **Cleanup Script:** Automated script to delete old versions.

### Summary Checklist for Production
- [ ] **Prompt Management:** Version control, A/B testing, registry.
- [ ] **Model Versioning:** MLflow or W&B, semantic versioning.
- [ ] **Continuous Evaluation:** Automated tests, regression detection.
- [ ] **Deployment:** Canary or blue-green, shadow testing.
- [ ] **Monitoring:** Latency, cost, quality, user satisfaction.
- [ ] **CI/CD:** Automated testing, deployment pipelines.
- [ ] **Incident Response:** Runbooks, rollback procedures.
