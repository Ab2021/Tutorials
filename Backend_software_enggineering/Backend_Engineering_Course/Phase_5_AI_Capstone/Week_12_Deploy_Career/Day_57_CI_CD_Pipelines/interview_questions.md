# Day 57: Interview Questions & Answers

## Conceptual Questions

### Q1: What is the difference between Continuous Delivery and Continuous Deployment?
**Answer:**
*   **Continuous Delivery**: Code is *ready* to deploy at any time. Deployment is a manual button click. (Common in regulated industries).
*   **Continuous Deployment**: Code is deployed *automatically* to production if tests pass. (Common in startups).

### Q2: How do you handle "Flaky Tests" in CI?
**Answer:**
*   **Definition**: A test that passes sometimes and fails sometimes (without code changes).
*   **Impact**: Destroys trust in the pipeline.
*   **Fix**:
    1.  **Isolate**: Move to a separate suite.
    2.  **Fix**: Root cause (Race condition? Network dependency?).
    3.  **Delete**: If not fixable, delete it. Better no test than a flaky one.

### Q3: What is "Trunk Based Development"?
**Answer:**
*   **Strategy**: Developers merge small changes to `main` (trunk) frequently (daily).
*   **Contrast**: GitFlow (Long-lived feature branches).
*   **Benefit**: Avoids "Merge Hell". Enables faster CI/CD.

---

## Scenario-Based Questions

### Q4: The deployment failed halfway. The DB migrated, but the code didn't update. What do you do?
**Answer:**
*   **Immediate**: Rollback isn't always possible (DB is changed).
*   **Fix Forward**: Push a hotfix to update the code.
*   **Prevention**: Use **Transactional DDL** (if DB supports it) or ensure migrations are backward compatible so old code works with new DB.

### Q5: How do you secure your CI/CD pipeline?
**Answer:**
*   **Least Privilege**: The CI runner shouldn't have root access to Prod.
*   **Secrets**: Don't print secrets in logs.
*   **Pin Actions**: Use `actions/checkout@v3` (SHA hash) instead of `@latest` to prevent supply chain attacks.

---

## Behavioral / Role-Specific Questions

### Q6: A developer bypasses CI and SSHs into Prod to fix a bug. How do you handle this?
**Answer:**
*   **Short Term**: Fix the bug.
*   **Long Term**: Remove SSH access.
*   **Culture**: "If it's not in Git, it doesn't exist." Manual changes are lost on the next deploy.
