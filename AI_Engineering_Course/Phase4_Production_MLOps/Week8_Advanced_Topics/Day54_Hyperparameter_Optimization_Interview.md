# Day 54: Hyperparameter Optimization
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the difference between grid search and random search?

**Answer:**
**Grid Search:**
- Exhaustively search predefined grid.
- **Example:** LR=[1e-5, 1e-4, 1e-3], BS=[16, 32] → 6 combinations.
- **Pros:** Guaranteed to find best in grid.
- **Cons:** Exponentially expensive, doesn't scale.

**Random Search:**
- Sample random combinations.
- **Pros:** More efficient, better coverage.
- **Cons:** No guarantee of finding best.

**When:** Random search is better for high-dimensional spaces (>3 hyperparameters).

#### Q2: How does Bayesian optimization work?

**Answer:**
1. **Build surrogate model** (Gaussian Process) of objective function.
2. **Acquisition function** selects next point to evaluate (balance exploration/exploitation).
3. **Evaluate** objective at selected point.
4. **Update** surrogate model with new observation.
5. **Repeat** until budget exhausted.

**Benefits:** Efficient for expensive evaluations (e.g., training large models).

#### Q3: What is Population-Based Training (PBT)?

**Answer:**
- **Concept:** Evolve population of models during training.
- **Process:**
  1. Train population in parallel.
  2. Periodically evaluate performance.
  3. **Exploit:** Replace worst with copies of best.
  4. **Explore:** Perturb hyperparameters.
- **Benefits:** Adapts hyperparameters during training, finds better solutions.

#### Q4: Explain the linear scaling rule for batch size.

**Answer:**
- **Rule:** If batch size increases by k, multiply learning rate by k.
- **Example:** BS=32, LR=1e-4 → BS=256 (8x), LR=8e-4.
- **Reason:** Larger batches have less noisy gradients, can use higher LR.
- **Limitation:** Breaks down for very large batches (>8K).

#### Q5: What is the purpose of learning rate warmup?

**Answer:**
- **Problem:** Large models unstable at start with high LR.
- **Solution:** Start with low LR, gradually increase to target LR.
- **Example:** Linear warmup over 1000 steps: `lr = max_lr * (step / 1000)`.
- **Benefits:** Stabilizes training, prevents divergence.

---

### Production Challenges

#### Challenge 1: Bayesian Optimization Stuck in Local Optimum

**Scenario:** Bayesian optimization keeps suggesting similar hyperparameters.
**Root Cause:** Exploitation too strong, not enough exploration.
**Solution:**
- **Increase xi:** Higher exploration parameter in Expected Improvement.
- **More Random Starts:** Increase initial random samples (5 → 10).
- **Different Acquisition:** Try UCB instead of EI.
- **Restart:** Run multiple independent studies, compare results.

#### Challenge 2: PBT Population Converges Too Quickly

**Scenario:** All population members have same hyperparameters after few iterations.
**Root Cause:** Exploit too aggressive, explore too weak.
**Solution:**
- **Reduce Exploit Threshold:** 20% → 10% (replace fewer members).
- **Increase Perturbation:** Perturb more hyperparameters (25% → 50%).
- **Larger Population:** 10 → 20 members.
- **Diversity Penalty:** Penalize similar configurations.

#### Challenge 3: Learning Rate Warmup Too Short

**Scenario:** Training unstable despite warmup.
**Root Cause:** Warmup too short for model size.
**Solution:**
- **Longer Warmup:** 1000 steps → 5000 steps (or 1-5% of total steps).
- **Lower Max LR:** Reduce target learning rate.
- **Gradient Clipping:** Clip gradients to prevent explosion.
- **Rule of Thumb:** Warmup steps ≈ 0.01 × total_steps for large models.

#### Challenge 4: Hyperband Wastes Resources

**Scenario:** Hyperband trains many bad configurations.
**Root Cause:** Too many initial configurations.
**Solution:**
- **Smaller Initial n:** Reduce initial configurations.
- **Better Sampling:** Use informed sampling instead of random.
- **Early Stopping:** Add early stopping within Hyperband.
- **Hybrid:** Combine with Bayesian optimization for initial sampling.

#### Challenge 5: Optuna Study Not Improving

**Scenario:** Optuna study plateaus after 20 trials.
**Root Cause:** Search space too large or poorly defined.
**Solution:**
- **Narrow Search Space:** Reduce bounds based on initial results.
- **Log Scale:** Use log scale for learning rate, batch size.
- **More Trials:** Run 100+ trials instead of 20.
- **Pruning:** Enable pruning to stop bad trials early.

### Summary Checklist for Production
- [ ] **Start Simple:** Tune **learning rate first** (most important).
- [ ] **Random Search:** Use for **>3 hyperparameters**.
- [ ] **Bayesian Optimization:** Use for **expensive evaluations**.
- [ ] **PBT:** Use for **long training runs** (>1 day).
- [ ] **Warmup:** Use **1-5% of total steps** for large models.
- [ ] **Batch Size Scaling:** Use **linear scaling rule** up to 8K batch size.
- [ ] **Early Stopping:** Use **patience=5-10 epochs** to save time.
