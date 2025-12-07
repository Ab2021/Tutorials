# Days 162-168: Week 24 - Cost & Spot Labs
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## Cost Optimization Quick Reference

### Karpenter Provisioner
```yaml
apiVersion: karpenter.sh/v1beta1
kind: Provisioner
metadata:
  name: gpu-spot
spec:
  requirements:
    - key: karpenter.sh/capacity-type
      operator: In
      values: ["spot"]
    - key: node.kubernetes.io/instance-type
      operator: In
      values: ["g4dn.xlarge", "g4dn.2xlarge", "g5.xlarge"]
  limits:
    resources:
      nvidia.com/gpu: 100
  ttlSecondsAfterEmpty: 30
```

### Spot Interruption Handler
```bash
helm install aws-node-termination-handler \
  eks/aws-node-termination-handler \
  --set enableSpotInterruptionDraining=true
```

### Kubecost Queries
```bash
# Get namespace costs
kubectl cost namespace --window 7d

# Get pod costs
kubectl cost pod -n ml-production --window 24h
```

### Right-sizing
```yaml
# Vertical Pod Autoscaler recommendations
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
    updateMode: "Off"  # Recommendations only
```

### Cost Alerts
```yaml
# AWS Budget alert (Terraform)
resource "aws_budgets_budget" "gpu_budget" {
  name         = "gpu-monthly"
  budget_type  = "COST"
  limit_amount = "10000"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"
}
```

---

## 📝 Week 24 Summary
| Day | Topic |
|-----|-------|
| 162 | Spot basics |
| 163 | Karpenter |
| 164 | Interruption |
| 165 | Kubecost |
| 166 | Right-sizing |
| 167 | Budgets |
| 168 | Project |

---

## 🎓 Phase 6D Complete!
MLOps & Production Engineering (Weeks 19-24)
