# Days 176-182: Week 26 - Multi-Cluster & Federation Labs
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## Day 176-182: Multi-Cluster Quick Reference

### Karmada Setup
```bash
kubectl karmada init
kubectl karmada join member1 --kubeconfig member1.kubeconfig
kubectl karmada join member2 --kubeconfig member2.kubeconfig
```

### PropagationPolicy
```yaml
apiVersion: policy.karmada.io/v1alpha1
kind: PropagationPolicy
metadata:
  name: ml-api-propagation
spec:
  resourceSelectors:
    - apiVersion: apps/v1
      kind: Deployment
      name: ml-api
  placement:
    clusterAffinity:
      clusterNames:
        - us-east
        - eu-west
```

### Istio Multi-Cluster
```bash
istioctl install --set profile=minimal
istioctl x create-remote-secret --name cluster2 | kubectl apply -f -
```

### Service Mesh
```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: ml-api
spec:
  hosts:
  - ml-api
  http:
  - route:
    - destination:
        host: ml-api
        subset: v1
      weight: 90
    - destination:
        host: ml-api
        subset: v2
      weight: 10
```

---

## 📝 Week 26 Summary
| Day | Topic |
|-----|-------|
| 176 | Karmada |
| 177 | Federation |
| 178 | Istio Mesh |
| 179 | Traffic Split |
| 180 | Observability |
| 181-182 | Project |
