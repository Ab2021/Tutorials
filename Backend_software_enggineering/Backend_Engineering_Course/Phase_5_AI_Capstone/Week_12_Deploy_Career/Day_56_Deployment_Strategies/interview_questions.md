# Day 56: Interview Questions & Answers

## Conceptual Questions

### Q1: What is Blue/Green Deployment?
**Answer:**
*   **Setup**: Two identical environments. Blue (Live) and Green (Idle).
*   **Deploy**: Deploy new version to Green. Test it.
*   **Switch**: Update Load Balancer to point to Green.
*   **Rollback**: If bugs found, switch back to Blue instantly.

### Q2: What is a "Canary Release"?
**Answer:**
*   **Concept**: Roll out the update to a small subset of users (e.g., 5%).
*   **Monitor**: Check error rates/latency.
*   **Expand**: If stable, increase to 20%, 50%, 100%.
*   **Origin**: "Canary in the coal mine".

### Q3: Horizontal vs Vertical Scaling. Which is better?
**Answer:**
*   **Vertical (Scale Up)**: Bigger CPU/RAM.
    *   *Limit*: Hardware limits. Single point of failure.
*   **Horizontal (Scale Out)**: More machines.
    *   *Limit*: Complexity (Distributed state).
    *   *Verdict*: Horizontal is preferred for cloud-native apps.

---

## Scenario-Based Questions

### Q4: Your Serverless (Lambda) API is slow for the first request. Why?
**Answer:**
*   **Cause**: **Cold Start**. AWS has to provision a container and start your runtime (Python/Java) before handling the request.
*   **Fix**:
    *   **Keep Warm**: Ping it every 5 mins.
    *   **Provisioned Concurrency**: Pay to keep instances ready.
    *   **Lighter Runtime**: Use Go/Rust instead of Java/Spring.

### Q5: How do you handle Database Migrations during a deployment?
**Answer:**
*   **Backward Compatibility**:
    *   *Bad*: Rename column `name` to `full_name`. (Old code breaks).
    *   *Good*: Add `full_name`. Deploy Code that writes to both. Backfill data. Remove `name`.
*   **Locking**: Avoid long-running migrations that lock tables during peak hours.

---

## Behavioral / Role-Specific Questions

### Q6: A startup wants to use Kubernetes for their MVP. Good idea?
**Answer:**
*   **No**.
*   **Overhead**: K8s requires a dedicated DevOps person/team to manage properly.
*   **Advice**: Use PaaS (Heroku, Render, AWS App Runner) for MVP. Move to K8s only when you need complex orchestration.
