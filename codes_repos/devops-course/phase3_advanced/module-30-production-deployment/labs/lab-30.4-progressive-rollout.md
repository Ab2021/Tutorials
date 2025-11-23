# Lab 30.4: Progressive Rollout

## Objective
Implement progressive rollout strategies for safe deployments.

## Learning Objectives
- Configure progressive delivery
- Monitor rollout health
- Automate rollback
- Use traffic shaping

---

## Flagger Progressive Delivery

```yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: myapp
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: myapp
  progressDeadlineSeconds: 60
  service:
    port: 80
  analysis:
    interval: 1m
    threshold: 5
    maxWeight: 50
    stepWeight: 10
    metrics:
    - name: request-success-rate
      thresholdRange:
        min: 99
    - name: request-duration
      thresholdRange:
        max: 500
  webhooks:
    - name: load-test
      url: http://flagger-loadtester/
      timeout: 5s
      metadata:
        cmd: "hey -z 1m -q 10 -c 2 http://myapp/"
```

## Success Criteria
✅ Progressive rollout configured  
✅ Automated health checks  
✅ Rollback on failure  

**Time:** 40 min
