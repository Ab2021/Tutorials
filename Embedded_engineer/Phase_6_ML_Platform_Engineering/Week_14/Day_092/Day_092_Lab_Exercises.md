# Days 92-98: Week 14 - KubeRay Production Labs
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## Day 92-98: KubeRay Quick Reference

```yaml
# RayJob for batch training
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: training-job
spec:
  entrypoint: python train.py
  runtimeEnvYAML: |
    pip: ["torch", "transformers"]
  rayClusterSpec:
    headGroupSpec:
      template:
        spec:
          containers:
          - name: ray-head
            image: rayproject/ray:2.9.0-py310-gpu
    workerGroupSpecs:
    - replicas: 4
      groupName: gpu-workers
      template:
        spec:
          containers:
          - name: ray-worker
            resources:
              limits:
                nvidia.com/gpu: 1
```

---

## Key Labs

### Day 92: Custom RayCluster
```bash
kubectl apply -f raycluster.yaml
kubectl get rayclusters
```

### Day 93: RayJob Workflow
```bash
kubectl apply -f rayjob.yaml
kubectl logs job/training-job
```

### Day 94: HA with Redis GCS
```yaml
spec:
  headGroupSpec:
    rayStartParams:
      redis-password: "secret"
```

### Day 95: Autoscaling
```yaml
spec:
  enableInTreeAutoscaling: true
  autoscalerOptions:
    upscalingMode: Conservative
```

### Day 96: Multi-Tenancy
```yaml
# Namespace isolation
metadata:
  namespace: team-a
```

### Day 97: Security
```yaml
# Pod security context
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
```

### Day 98: Week 14 Project
```bash
# Production Ray platform checklist
# [ ] HA enabled
# [ ] Autoscaling configured
# [ ] RBAC per team
# [ ] Monitoring integrated
```

---

## 📝 Week 14 Summary
| Day | Topic |
|-----|-------|
| 92 | KubeRay CRDs |
| 93 | RayJob |
| 94 | GCS HA |
| 95 | Autoscaling |
| 96 | Multi-tenancy |
| 97 | Security |
| 98 | Project |
