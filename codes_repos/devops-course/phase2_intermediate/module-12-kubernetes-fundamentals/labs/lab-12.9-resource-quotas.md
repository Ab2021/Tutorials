# Lab 12.9: Resource Quotas and Limits

## Objective
Manage resource allocation with quotas and limit ranges.

## Learning Objectives
- Create ResourceQuotas
- Set LimitRanges
- Monitor resource usage
- Prevent resource exhaustion

---

## ResourceQuota

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: compute-quota
  namespace: default
spec:
  hard:
    requests.cpu: "10"
    requests.memory: 20Gi
    limits.cpu: "20"
    limits.memory: 40Gi
    pods: "10"
    services: "5"
```

## LimitRange

```yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: resource-limits
spec:
  limits:
  - max:
      cpu: "2"
      memory: 4Gi
    min:
      cpu: "100m"
      memory: 128Mi
    default:
      cpu: "500m"
      memory: 512Mi
    defaultRequest:
      cpu: "250m"
      memory: 256Mi
    type: Container
```

## Pod with Resources

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: myapp
spec:
  containers:
  - name: app
    image: nginx
    resources:
      requests:
        memory: "256Mi"
        cpu: "250m"
      limits:
        memory: "512Mi"
        cpu: "500m"
```

## Monitor Usage

```bash
# Check quota
kubectl describe quota compute-quota

# Check namespace usage
kubectl top nodes
kubectl top pods
```

## Success Criteria
✅ ResourceQuotas enforced  
✅ LimitRanges applied  
✅ Resource usage monitored  
✅ Pods within limits  

**Time:** 40 min
