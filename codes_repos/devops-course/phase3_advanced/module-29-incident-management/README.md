# Incident Management & On-Call

## 🎯 Learning Objectives

By the end of this module, you will have a comprehensive understanding of Incident Management, including:
- **Incident Response**: Following a structured process to resolve outages quickly.
- **On-Call**: Setting up rotations and escalation policies with **PagerDuty**.
- **Runbooks**: Writing step-by-step guides for common incidents.
- **Post-Mortems**: Conducting blameless retrospectives to learn from failures.
- **Culture**: Building a culture where failure is seen as an opportunity to improve.

---

## 📖 Theoretical Concepts

### 1. Incident Severity Levels

- **SEV-1 (Critical)**: Complete outage. All hands on deck. Page everyone.
- **SEV-2 (High)**: Partial outage. Core functionality degraded. Page on-call engineer.
- **SEV-3 (Medium)**: Minor issue. Non-core feature broken. Create ticket.
- **SEV-4 (Low)**: Cosmetic issue. Fix in next sprint.

### 2. Incident Response Roles

- **Incident Commander (IC)**: Coordinates the response. Makes decisions. Doesn't fix the problem.
- **Communications Lead**: Updates status page, notifies stakeholders.
- **Subject Matter Expert (SME)**: The engineer who actually fixes the issue.

### 3. The Incident Lifecycle

1.  **Detection**: Alert fires (Prometheus/PagerDuty).
2.  **Triage**: IC assesses severity and assembles team.
3.  **Mitigation**: Stop the bleeding (rollback, failover).
4.  **Resolution**: Fix the root cause.
5.  **Post-Mortem**: Document what happened and how to prevent it.

### 4. Blameless Post-Mortems

The goal is to learn, not to punish.
- **What Happened**: Timeline of events.
- **Why It Happened**: Root cause analysis (5 Whys).
- **Action Items**: Concrete steps to prevent recurrence (assign owners and due dates).

---

## 🔧 Practical Examples

### Runbook Template

```markdown
# Runbook: Database Connection Pool Exhausted

## Symptoms
- HTTP 500 errors
- Logs show "Too many connections"

## Diagnosis
1. Check Grafana dashboard: `db_connections_active`
2. If > 95% of max, pool is exhausted

## Mitigation
1. Restart app pods: `kubectl rollout restart deployment/app`
2. Increase pool size (temporary): `kubectl set env deployment/app DB_POOL_SIZE=50`

## Root Cause
- Slow queries holding connections open
- Missing connection timeout

## Prevention
- Add connection timeout (5s)
- Optimize slow queries
```

### PagerDuty Escalation Policy

```yaml
escalation_policy:
  name: "Platform Team"
  escalation_rules:
    - escalation_delay_in_minutes: 0
      targets:
        - type: user
          id: "P123ABC"  # On-call engineer
    - escalation_delay_in_minutes: 15
      targets:
        - type: user
          id: "P456DEF"  # Team lead
```

---

## 🎯 Hands-on Labs

- [Lab 29.1: Incident Response Simulation](./labs/lab-29.1-incident-response.md)
- [Lab 29.2: Blameless Post-Mortems](./labs/lab-29.2-post-mortem.md)
- [Lab 29.3: Pagerduty Integration](./labs/lab-29.3-pagerduty-integration.md)
- [Lab 29.4: Runbooks](./labs/lab-29.4-runbooks.md)
- [Lab 29.5: Post Mortems](./labs/lab-29.5-post-mortems.md)
- [Lab 29.6: Incident Communication](./labs/lab-29.6-incident-communication.md)
- [Lab 29.7: Escalation Policies](./labs/lab-29.7-escalation-policies.md)
- [Lab 29.8: Incident Metrics](./labs/lab-29.8-incident-metrics.md)
- [Lab 29.9: Blameless Culture](./labs/lab-29.9-blameless-culture.md)
- [Lab 29.10: Continuous Improvement](./labs/lab-29.10-continuous-improvement.md)

---

## 📚 Additional Resources

### Official Documentation
- [PagerDuty Documentation](https://support.pagerduty.com/)
- [Google SRE Book - Incident Management](https://sre.google/sre-book/managing-incidents/)

### Templates
- [Atlassian Post-Mortem Template](https://www.atlassian.com/incident-management/postmortem/templates)

---

## 🔑 Key Takeaways

1.  **Practice**: Run Game Days (simulate outages) to practice incident response.
2.  **Automate Runbooks**: If a runbook is run more than 3 times, automate it.
3.  **No Blame**: "Who broke it?" is the wrong question. "How do we prevent this?" is the right one.
4.  **MTTR > MTBF**: Mean Time To Recovery is more important than Mean Time Between Failures. Failures will happen. Recover fast.

---

## ⏭️ Next Steps

1.  Complete the labs to prepare for real incidents.
2.  Proceed to **[Module 30: Production Deployment](../module-30-production-deployment/README.md)** for the final capstone project.
