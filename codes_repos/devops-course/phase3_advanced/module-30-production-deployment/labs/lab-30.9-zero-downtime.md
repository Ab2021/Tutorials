# Lab 30.9: Zero Downtime

## Objective
Achieve zero-downtime deployments in production.

## Learning Objectives
- Implement rolling updates
- Configure readiness probes
- Use connection draining
- Test zero-downtime

---

## Kubernetes Rolling Update

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  replicas: 10
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2        # Max 2 extra pods during update
      maxUnavailable: 0  # Never go below desired count
  template:
    spec:
      containers:
      - name: myapp
        image: myapp:v2.0
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 15
          periodSeconds: 10
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 15"]
```

## Load Balancer Configuration

```yaml
# Service with connection draining
apiVersion: v1
kind: Service
metadata:
  name: myapp
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-connection-draining-enabled: "true"
    service.beta.kubernetes.io/aws-load-balancer-connection-draining-timeout: "60"
spec:
  type: LoadBalancer
  selector:
    app: myapp
  ports:
  - port: 80
    targetPort: 8080
```

## Test Zero Downtime

```bash
#!/bin/bash
# test-zero-downtime.sh

# Start continuous requests
while true; do
  curl -s http://myapp.example.com/health
  sleep 0.1
done &

# Deploy new version
kubectl set image deployment/myapp myapp=myapp:v2.0

# Wait for rollout
kubectl rollout status deployment/myapp

# Check if any requests failed
# (Should be zero)
```

## Success Criteria
✅ Rolling update configured  
✅ Readiness probes working  
✅ Zero failed requests during deployment  
✅ Connection draining enabled  

**Time:** 45 min
