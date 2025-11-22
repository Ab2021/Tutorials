# Lab 29.2: Blameless Post-Mortems

## 🎯 Objective

Learn from failure. A **Post-Mortem** (or Incident Retrospective) is a document written after an incident to understand *what* happened, *why* it happened, and *how* to prevent it. It must be **Blameless**.

## 📋 Prerequisites

-   A text editor.

## 📚 Background

### The Blameless Culture
-   **Bad**: "Bob pushed bad code." (Bob gets fired, everyone hides mistakes).
-   **Good**: "The CI pipeline allowed bad code to be pushed." (We fix the pipeline, everyone learns).

---

## 🔨 Hands-On Implementation

### Part 1: The Scenario 💥

**Incident**: The Checkout Service went down for 45 minutes on Black Friday.
**Cause**: A database migration locked the Users table.
**Impact**: $50,000 lost revenue.

### Part 2: Write the Post-Mortem 📝

Create `post-mortem-001.md` and fill in the sections:

#### 1. Summary
*Brief description of the incident.*
> At 14:00 UTC, the Checkout Service began returning 500 errors. Traffic dropped to zero. Service restored at 14:45 UTC.

#### 2. Impact
*Who was affected?*
> 100% of users could not complete purchases. Estimated revenue loss: $50k. Support tickets: 500+.

#### 3. Timeline
*What happened when? (UTC)*
> - **14:00**: Deployment of v2.1 starts.
> - **14:01**: Monitoring alerts "High Error Rate".
> - **14:05**: On-Call (Alice) acknowledges.
> - **14:10**: Alice identifies DB lock.
> - **14:15**: Alice attempts rollback. Rollback fails (migration not reversible).
> - **14:30**: Bob (DBA) joins call. Kills the migration query.
> - **14:45**: System recovers.

#### 4. Root Cause Analysis (The 5 Whys)
*Drill down to the systemic issue.*
> 1. **Why did the service fail?** Database requests timed out.
> 2. **Why did they time out?** The `users` table was locked.
> 3. **Why was it locked?** A migration added a column with a default value.
> 4. **Why did that lock the table?** In Postgres < 11, adding a column with default requires a full table rewrite.
> 5. **Why was this not caught in testing?** Staging DB is small (100 rows). Prod DB is huge (10M rows). Lock was instant in Staging.

#### 5. Action Items
*Preventative measures.*
> - [ ] **Immediate**: Upgrade Postgres to v13 (where this is safe).
> - [ ] **Process**: Require DBA review for all migrations.
> - [ ] **Testing**: Create a "Load Test" environment with realistic data volume.

### Part 3: Review 🧐

1.  **Check for Blame:**
    Did you write "Alice failed to rollback"?
    Change to: "Rollback procedure failed due to missing down-migration script."

---

## 🎯 Challenges

### Challenge 1: Chaos Experiment (Difficulty: ⭐⭐⭐)

**Task:**
Create a Chaos Engineering experiment (Lab 25) that reproduces this failure.
Simulate a table lock and verify that the application handles it gracefully (or fails fast).

### Challenge 2: Automate the Timeline (Difficulty: ⭐⭐)

**Task:**
If you use Slack for incidents, use a bot (like Jeli or FireHydrant) to export the chat history as the initial timeline.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
Run `LOCK TABLE users IN ACCESS EXCLUSIVE MODE;` in a transaction and sleep.
</details>

---

## 🔑 Key Takeaways

1.  **Systemic Fixes**: You can't fix people. You can fix systems.
2.  **Documentation**: The Post-Mortem is a learning asset. Store it in a searchable repository.
3.  **Closure**: An incident isn't over until the Action Items are done.

---

## ⏭️ Next Steps

We learned from the past. Now let's deploy for the future.

Proceed to **Module 30: Production Deployment Strategies**.
