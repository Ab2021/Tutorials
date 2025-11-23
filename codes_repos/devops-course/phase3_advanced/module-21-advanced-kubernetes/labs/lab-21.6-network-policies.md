# Lab 21.6: Network Policies

## Objective
Secure Kubernetes networking with Network Policies.

## Learning Objectives
- Create network policies
- Implement ingress/egress rules
- Test policy enforcement
- Secure pod communication

---

## Default Deny

```yaml
# deny-all.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
```

## Allow Specific Traffic

```yaml
# allow-frontend.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-frontend
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    ports:
    - protocol: TCP
      port: 8080
```

## Egress Policy

```yaml
# allow-dns.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-dns
spec:
  podSelector: {}
  policyTypes:
  - Egress
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: UDP
      port: 53
```

## Test Policies

```bash
# Deploy test pods
kubectl run frontend --image=busybox --labels=app=frontend -- sleep 3600
kubectl run backend --image=busybox --labels=app=backend -- sleep 3600

# Test connectivity
kubectl exec frontend -- wget -O- backend:8080  # Should work
kubectl exec other -- wget -O- backend:8080     # Should fail
```

## Success Criteria
✅ Network policies created  
✅ Traffic restricted  
✅ Policies tested  
✅ DNS still works  

**Time:** 40 min
