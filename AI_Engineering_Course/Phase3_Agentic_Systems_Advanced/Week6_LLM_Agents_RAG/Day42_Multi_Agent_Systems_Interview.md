# Day 42: Multi-Agent Systems
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What are the advantages of multi-agent systems over single agents?

**Answer:**
- **Specialization:** Each agent can be expert in a specific domain.
- **Robustness:** If one agent fails, others can continue.
- **Diversity:** Multiple perspectives lead to better decisions.
- **Scalability:** Can add more agents for complex tasks.
- **Example:** Software development with PM, Architect, Coder, Tester agents.

#### Q2: Explain the debate pattern in multi-agent systems.

**Answer:**
- **Structure:** Two agents argue opposite sides (Pro vs Con).
- **Process:** Alternate arguments for N rounds.
- **Judge:** Third agent or human selects winner or synthesizes.
- **Benefit:** Explores both sides thoroughly, reduces bias.
- **Use Case:** Decision making, evaluating proposals.

#### Q3: What is the difference between hierarchical and peer-to-peer multi-agent systems?

**Answer:**
- **Hierarchical:** Manager agent coordinates worker agents. Clear control, single point of failure.
- **Peer-to-Peer:** All agents are equal, communicate directly. Democratic, robust, but complex coordination.
- **When to Use:** Hierarchical for clear task decomposition, P2P for collaborative problem-solving.

#### Q4: How do you prevent infinite loops in multi-agent systems?

**Answer:**
- **Max Iterations:** Hard limit on agent interactions (e.g., 10 rounds).
- **Convergence Detection:** If agents repeat the same messages, stop.
- **Timeout:** If task takes too long, terminate.
- **Deadlock Detection:** If agents are waiting for each other, intervene.

#### Q5: What are the cost implications of multi-agent systems?

**Answer:**
- **Multiple LLM Calls:** 5 agents = 5x cost per task.
- **Coordination Overhead:** Additional calls for communication.
- **Mitigation:**
  - Use smaller models for simple agents (GPT-3.5 for workers, GPT-4 for manager).
  - Cache results.
  - Limit number of agents and iterations.

---

### Production Challenges

#### Challenge 1: Agent Disagreement

**Scenario:** In a debate, Pro and Con agents never converge. Judge can't decide.
**Solution:**
- **Voting:** If multiple judges, use majority vote.
- **Confidence Scores:** Each agent provides confidence. Higher confidence wins.
- **Human-in-the-Loop:** Escalate to human for final decision.
- **Default:** If no consensus after N rounds, use a default answer.

#### Challenge 2: Coordination Overhead

**Scenario:** 10 agents spend more time coordinating than working.
**Root Cause:** Too many agents, inefficient communication.
**Solution:**
- **Reduce Agents:** Use 3-5 agents, not 10.
- **Hierarchical Structure:** Manager coordinates, workers don't talk to each other.
- **Asynchronous:** Agents work independently, synchronize at end.

#### Challenge 3: Quality Degradation

**Scenario:** Ensemble of 5 agents produces worse output than single GPT-4.
**Root Cause:** Aggregation dilutes quality (averaging mediocre answers).
**Solution:**
- **Weighted Aggregation:** Weight higher-quality agents more.
- **Best-of-N:** Select the best answer, don't average.
- **Filtering:** Remove low-quality answers before aggregation.

#### Challenge 4: Role Confusion

**Scenario:** In hierarchical system, worker agents try to act as managers.
**Root Cause:** Unclear role definitions in prompts.
**Solution:**
- **Explicit Roles:** "You are a researcher. Your ONLY job is to find information."
- **Constraints:** "Do NOT delegate tasks. Do NOT make decisions."
- **Validation:** Check if agent output matches its role.

#### Challenge 5: Cost Explosion

**Scenario:** Reflection system runs 10 iterations. Cost is 20x single agent.
**Analysis:** Generator + Critic = 2 calls per iteration. 10 iterations = 20 calls.
**Solution:**
- **Early Stopping:** If critique says "Satisfactory", stop immediately.
- **Max Iterations:** Limit to 3 iterations.
- **Cheaper Critic:** Use GPT-3.5 for critique, GPT-4 for generation.

### Summary Checklist for Production
- [ ] **Agents:** Use **3-5 agents**, not 10+.
- [ ] **Iterations:** Limit to **3-5 rounds** max.
- [ ] **Roles:** Define **explicit roles** and constraints.
- [ ] **Coordination:** Use **hierarchical** structure for efficiency.
- [ ] **Cost:** Use **smaller models** for simple agents.
- [ ] **Monitoring:** Track **number of calls** and **cost per task**.
