# Lab 30.5: Rollback Strategies

## Objective
Implement effective rollback strategies for production.

## Learning Objectives
- Automate rollback detection
- Execute rollback procedures
- Preserve data integrity
- Test rollback processes

---

## Kubernetes Rollback

```bash
# View rollout history
kubectl rollout history deployment/myapp

# Rollback to previous version
kubectl rollout undo deployment/myapp

# Rollback to specific revision
kubectl rollout undo deployment/myapp --to-revision=3

# Check rollback status
kubectl rollout status deployment/myapp
```

## Automated Rollback

```yaml
# ArgoCD automated rollback
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: myapp
spec:
  strategy:
    canary:
      steps:
      - setWeight: 20
      - pause: {duration: 5m}
      - setWeight: 40
      - pause: {duration: 5m}
      analysis:
        templates:
        - templateName: success-rate
        startingStep: 2
      trafficRouting:
        istio:
          virtualService:
            name: myapp
```

## Database Rollback

```sql
-- Use migrations with down scripts
-- V1__create_users.sql (up)
CREATE TABLE users (id INT, name VARCHAR(100));

-- V1__create_users_down.sql (down)
DROP TABLE users;
```

## Success Criteria
✅ Rollback procedures documented  
✅ Automated rollback working  
✅ Data integrity maintained  

**Time:** 40 min
