# Lab 30.2: Canary Releases

## Objective
Implement canary deployments for gradual rollouts.

## Learning Objectives
- Deploy canary version
- Route percentage of traffic
- Monitor canary metrics
- Promote or rollback

---

## Kubernetes Canary

```yaml
# stable-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp-stable
spec:
  replicas: 9
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
        version: stable
    spec:
      containers:
      - name: myapp
        image: myapp:v1.0
---
# canary-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp-canary
spec:
  replicas: 1  # 10% traffic
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
        version: canary
    spec:
      containers:
      - name: myapp
        image: myapp:v2.0
```

## Service (Both Versions)

```yaml
apiVersion: v1
kind: Service
metadata:
  name: myapp
spec:
  selector:
    app: myapp  # Matches both stable and canary
  ports:
  - port: 80
```

## Monitor Canary

```promql
# Error rate for canary
rate(http_requests_total{version="canary",status=~"5.."}[5m])
/
rate(http_requests_total{version="canary"}[5m])

# Compare to stable
rate(http_requests_total{version="stable",status=~"5.."}[5m])
/
rate(http_requests_total{version="stable"}[5m])
```

## Promote Canary

```bash
# If canary is healthy, promote
kubectl scale deployment myapp-canary --replicas=10
kubectl scale deployment myapp-stable --replicas=0
kubectl delete deployment myapp-stable

# Rename canary to stable
kubectl patch deployment myapp-canary -p '{"metadata":{"name":"myapp-stable"}}'
```

## Flagger (Automated Canary)

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
```

## Success Criteria
✅ Canary deployed  
✅ Traffic split working  
✅ Metrics monitored  
✅ Automated promotion/rollback  

**Time:** 50 min
