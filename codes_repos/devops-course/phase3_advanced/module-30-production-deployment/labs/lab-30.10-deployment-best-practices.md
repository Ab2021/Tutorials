# Lab 30.10: Deployment Best Practices

## Objective
Learn and implement deployment best practices.

## Learning Objectives
- Follow deployment checklist
- Implement safety measures
- Document procedures
- Measure deployment metrics

---

## Deployment Checklist

```markdown
## Pre-Deployment
- [ ] Code reviewed and approved
- [ ] All tests passing (unit, integration, e2e)
- [ ] Security scans completed
- [ ] Database migrations tested
- [ ] Rollback plan documented
- [ ] Stakeholders notified

## During Deployment
- [ ] Deploy to staging first
- [ ] Run smoke tests
- [ ] Monitor metrics (error rate, latency)
- [ ] Verify health checks
- [ ] Check logs for errors

## Post-Deployment
- [ ] Verify all features working
- [ ] Monitor for 30 minutes
- [ ] Update documentation
- [ ] Notify team of completion
- [ ] Schedule post-deployment review
```

## Deployment Metrics

```python
# Track deployment metrics
metrics = {
    "deployment_frequency": "10 per day",
    "lead_time": "2 hours",
    "mttr": "15 minutes",
    "change_failure_rate": "5%"
}

# DORA metrics
def calculate_dora_metrics():
    return {
        "elite": {
            "deployment_frequency": "Multiple per day",
            "lead_time": "< 1 hour",
            "mttr": "< 1 hour",
            "change_failure_rate": "< 15%"
        }
    }
```

## Safety Measures

```yaml
# PodDisruptionBudget
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: myapp-pdb
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: myapp
```

## Documentation

```markdown
# Deployment Runbook

## Normal Deployment
1. Merge PR to main
2. CI/CD automatically deploys to staging
3. Run smoke tests
4. Approve production deployment
5. Monitor for 30 minutes

## Emergency Rollback
```bash
kubectl rollout undo deployment/myapp
```

## Contacts
- On-call: #oncall-platform
- Escalation: platform-team@example.com
```

## Success Criteria
✅ Deployment checklist followed  
✅ Metrics tracked  
✅ Safety measures in place  
✅ Documentation complete  

**Time:** 40 min

---

## 🎉 Congratulations!

You've completed all 30 modules of the DevOps course!

You now have the skills to:
- Build and deploy containerized applications
- Manage infrastructure as code
- Implement CI/CD pipelines
- Monitor and observe systems
- Ensure security and compliance
- Operate production systems at scale

**Next Steps:**
- Apply these skills in real projects
- Contribute to open source
- Pursue DevOps certifications
- Continue learning and growing

**Good luck on your DevOps journey!** 🚀
