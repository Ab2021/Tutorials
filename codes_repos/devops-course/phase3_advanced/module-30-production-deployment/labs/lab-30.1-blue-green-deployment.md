# Lab 30.1: Blue-Green Deployment

## Objective
Implement blue-green deployment for zero-downtime releases.

## Learning Objectives
- Set up blue-green infrastructure
- Deploy to green environment
- Switch traffic
- Rollback if needed

---

## Kubernetes Blue-Green

```yaml
# blue-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp-blue
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
      version: blue
  template:
    metadata:
      labels:
        app: myapp
        version: blue
    spec:
      containers:
      - name: myapp
        image: myapp:v1.0
```

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: myapp
spec:
  selector:
    app: myapp
    version: blue  # Points to blue initially
  ports:
  - port: 80
```

## Deploy Green

```bash
# Deploy green
kubectl apply -f green-deployment.yaml

# Test green
kubectl port-forward deployment/myapp-green 8080:80
curl localhost:8080

# Switch traffic to green
kubectl patch service myapp -p '{"spec":{"selector":{"version":"green"}}}'

# Verify
kubectl get svc myapp -o yaml | grep version

# Delete blue (after verification)
kubectl delete deployment myapp-blue
```

## AWS Blue-Green

```bash
# Create green target group
aws elbv2 create-target-group \
  --name myapp-green \
  --protocol HTTP \
  --port 80 \
  --vpc-id vpc-12345

# Register green instances
aws elbv2 register-targets \
  --target-group-arn arn:aws:elasticloadbalancing:... \
  --targets Id=i-green1 Id=i-green2

# Switch listener to green
aws elbv2 modify-listener \
  --listener-arn arn:aws:elasticloadbalancing:... \
  --default-actions Type=forward,TargetGroupArn=arn:...:targetgroup/myapp-green
```

## Success Criteria
✅ Blue and green environments running  
✅ Traffic switched successfully  
✅ Zero downtime achieved  
✅ Rollback tested  

**Time:** 45 min
