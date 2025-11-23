# Day 77: Agent Evaluation & Simulation
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Why are traditional NLP metrics (BLEU, ROUGE) useless for Agents?

**Answer:**
- **Semantics:** Agents perform actions. "Deleting file A" and "Deleting file B" are textually similar (high BLEU) but functionally opposite.
- **Reasoning:** BLEU measures n-gram overlap. It cannot measure if the agent followed the correct logical steps to solve a math problem.
- **Solution:** Use Functional Correctness (Pass/Fail) or LLM-as-a-Judge.

#### Q2: What is "Data Contamination" in benchmarks?

**Answer:**
- When the test set (benchmark questions) accidentally leaks into the training set of the LLM.
- **Result:** The model "cheats" by memorizing the answer.
- **Detection:** Check if the model can solve the problem if you slightly modify the numbers/names. If it fails, it was memorizing.

#### Q3: How do you evaluate "Safety" in agents?

**Answer:**
- **Red Teaming:** Use an adversarial agent to try to trick the target agent into doing something bad (e.g., "Delete the database").
- **Sandboxing:** Run the agent in a secure environment and monitor for forbidden system calls.
- **Metric:** Attack Success Rate (ASR).

#### Q4: Explain the "Simulacra" concept.

**Answer:**
- Creating a population of simulated users (agents) to test your system.
- **Use Case:** Testing a social network bot. Create 1000 agents with different personalities (Troll, Fan, Spammer) and let them interact with your bot.

#### Q5: What is the difference between "Process Reward" and "Outcome Reward"?

**Answer:**
- **Outcome Reward:** Did you get the right answer? (Binary). Hard to learn from if the task is long.
- **Process Reward:** Did you take a good step? (Dense). Provides feedback at every step of the chain.
- **ToT:** Uses Process Rewards to prune bad branches early.

---

### Production Challenges

#### Challenge 1: The "Flaky" Benchmark

**Scenario:** Agent scores 80% today, 60% tomorrow on the same benchmark.
**Root Cause:** LLM non-determinism (Temperature > 0).
**Solution:**
- **Greedy Decoding:** Set Temperature = 0 for eval (though this reduces creativity).
- **Multiple Runs:** Run eval 5 times and average the score (Pass@k).

#### Challenge 2: Evaluator Bias

**Scenario:** GPT-4 judge prefers long, polite answers, even if they are slightly wrong.
**Root Cause:** LLM bias ("Length bias").
**Solution:**
- **Reference-Free Eval:** Don't just ask "Is this good?". Ask "Does this match the Gold Standard Answer?".
- **Pairwise Comparison:** "Is A better than B?" is more reliable than "Rate A 1-10".

#### Challenge 3: Cost of Simulation

**Scenario:** Running a 100-turn simulation with GPT-4 costs $10 per run. 1000 tests = $10,000.
**Root Cause:** Expensive models.
**Solution:**
- **Small Simulators:** Use Llama-3-70B as the simulator/judge.
- **Early Stopping:** Stop simulation if the agent gets stuck in a loop.

#### Challenge 4: Overfitting to Benchmark

**Scenario:** You optimize your agent to pass GAIA. It fails in real world.
**Root Cause:** Goodhart's Law. The benchmark became the target.
**Solution:**
- **Private Eval Set:** Maintain a set of real user logs that the dev team never sees/optimizes for directly.

#### Challenge 5: Tool Mocking Complexity

**Scenario:** Mocking the entire AWS API for testing is impossible.
**Root Cause:** Complex external dependencies.
**Solution:**
- **VCR (Record/Replay):** Record real API interactions once, then replay them for tests.
- **LocalStack:** Use local emulators for cloud services.

### System Design Scenario: Agent CI/CD Pipeline

**Requirement:** Automatically test agent before deployment.
**Design:**
1.  **Commit:** Dev pushes code.
2.  **Unit Test:** Run deterministic logic tests (Mock tools).
3.  **Integration:** Run 50 scenarios in a Simulator (WebArena).
4.  **Eval:** GPT-4 grades the trajectories.
5.  **Gate:** If Score > Baseline, deploy to Staging.
6.  **Monitor:** Watch "User Thumbs Up" in Staging.

### Summary Checklist for Production
- [ ] **Benchmark:** Use **GAIA** or **SWE-bench**.
- [ ] **Simulator:** Build a **User Simulator** for stress testing.
- [ ] **Judge:** Use **GPT-4** to grade trajectories.
- [ ] **Determinism:** Set **Temp=0** for regression testing.
- [ ] **Safety:** Run **Red Team** attacks.
