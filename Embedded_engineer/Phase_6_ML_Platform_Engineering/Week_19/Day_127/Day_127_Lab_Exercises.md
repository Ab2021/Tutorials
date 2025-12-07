# Days 127-168: Weeks 19-24 - MLOps & Production Labs
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## Week 19: MLflow (Days 127-133)

### Tracking
```python
import mlflow

mlflow.set_tracking_uri("http://mlflow:5000")
with mlflow.start_run():
    mlflow.log_param("lr", 0.001)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.pytorch.log_model(model, "model")
```

### Model Registry
```python
mlflow.register_model("runs:/abc123/model", "production-model")
```

---

## Week 20: K8s Advanced (Days 134-140)

### Custom Operators
```yaml
apiVersion: ml.example.com/v1
kind: TrainingJob
metadata:
  name: bert-finetune
spec:
  model: bert-base
  epochs: 10
  gpus: 4
```

### Priority Classes
```yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: high-priority-training
value: 1000000
preemptionPolicy: PreemptLowerPriority
```

---

## Week 21: ArgoCD Advanced (Days 141-147)

### App of Apps
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: root
spec:
  source:
    path: apps
```

### Image Updater
```yaml
annotations:
  argocd-image-updater.argoproj.io/image-list: myapp=myorg/myapp
  argocd-image-updater.argoproj.io/myapp.update-strategy: semver
```

---

## Week 22: Prometheus/Grafana (Days 148-154)

### Custom Metrics
```python
from prometheus_client import Counter, Histogram

PREDICTIONS = Counter('predictions_total', 'Total predictions')
LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')

@LATENCY.time()
def predict(data):
    PREDICTIONS.inc()
    return model(data)
```

### Grafana Dashboard
```json
{
  "panels": [{
    "title": "Predictions/sec",
    "targets": [{
      "expr": "rate(predictions_total[5m])"
    }]
  }]
}
```

---

## Week 23: GPU Ops (Days 155-161)

### DCGM Metrics
```promql
DCGM_FI_DEV_GPU_UTIL  # GPU utilization
DCGM_FI_DEV_FB_USED   # Memory used
DCGM_FI_DEV_POWER_USAGE # Power draw
```

### MIG Monitoring
```bash
nvidia-smi mig -lgi
dcgmi discovery -c
```

---

## Week 24: Cost & Spot (Days 162-168)

### Karpenter
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
      values: ["g4dn.xlarge", "g4dn.2xlarge"]
```

---

## 📝 Weeks 19-24 Summary
| Week | Topic | Focus |
|------|-------|-------|
| 19 | MLflow | Tracking, Registry |
| 20 | K8s Advanced | Operators, Priority |
| 21 | ArgoCD | App of Apps |
| 22 | Monitoring | Prometheus/Grafana |
| 23 | GPU Ops | DCGM, MIG |
| 24 | Cost | Spot, Karpenter |

---

## 🎓 Phase 6D Complete!
MLOps & Production Engineering (Weeks 19-24)
