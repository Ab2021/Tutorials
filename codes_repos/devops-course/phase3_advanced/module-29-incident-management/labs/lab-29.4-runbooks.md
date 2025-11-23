# Lab 29.4: Runbooks

## Objective
Create effective runbooks for incident response.

## Learning Objectives
- Write clear runbooks
- Automate common tasks
- Test runbooks
- Keep documentation updated

---

## Runbook Template

```markdown
# Runbook: [Service Name] - [Issue Type]

## Overview
Brief description of the issue and its impact.

## Severity: [P0/P1/P2/P3]

## Symptoms
- Error rate > 5%
- Latency > 1s
- 500 errors in logs

## Investigation Steps

### 1. Check Metrics
```bash
# Prometheus query
rate(http_requests_total{status="500"}[5m])
```

### 2. Check Logs
```bash
kubectl logs -l app=myapp --tail=100 | grep ERROR
```

### 3. Check Recent Changes
```bash
kubectl rollout history deployment/myapp
git log --since="1 hour ago" --oneline
```

## Resolution Steps

### Quick Fix
```bash
# Restart pods
kubectl rollout restart deployment/myapp
```

### Rollback
```bash
kubectl rollout undo deployment/myapp
```

## Escalation
- 15 min: Page team lead
- 30 min: Page manager
- 60 min: Executive escalation

## Post-Incident
- [ ] Create incident report
- [ ] Schedule post-mortem
- [ ] Update runbook with learnings
```

## Automated Runbook

```python
# auto-remediate.py
import subprocess

def check_health():
    result = subprocess.run(['curl', 'http://api/health'], capture_output=True)
    return result.returncode == 0

def restart_service():
    subprocess.run(['kubectl', 'rollout', 'restart', 'deployment/api'])

if not check_health():
    print("Service unhealthy, restarting...")
    restart_service()
```

## Success Criteria
✅ Runbooks created for common issues  
✅ Clear investigation steps  
✅ Automation implemented  
✅ Runbooks tested  

**Time:** 40 min
