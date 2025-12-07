# Days 134-140: Week 20 - K8s Advanced Labs
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## K8s Advanced Quick Reference

### Custom Resource Definitions
```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: trainingjobs.ml.example.com
spec:
  group: ml.example.com
  names:
    kind: TrainingJob
    plural: trainingjobs
  scope: Namespaced
  versions:
  - name: v1
    served: true
    storage: true
```

### Priority Classes
```yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: ml-critical
value: 1000000
preemptionPolicy: PreemptLowerPriority
globalDefault: false
```

### Pod Disruption Budget
```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: ml-api-pdb
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: ml-api
```

### Vertical Pod Autoscaler
```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: ml-api-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-api
  updatePolicy:
    updateMode: Auto
```

---

## 📝 Week 20 Summary
| Day | Topic |
|-----|-------|
| 134 | CRDs |
| 135 | Operators |
| 136 | Priority |
| 137 | Preemption |
| 138 | VPA |
| 139 | Quotas |
| 140 | Project |
