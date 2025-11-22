# Advanced Kubernetes

## 🎯 Learning Objectives

By the end of this module, you will have a comprehensive understanding of advanced Kubernetes concepts, including:
- **Extensibility**: Creating Custom Resource Definitions (CRDs) and Operators.
- **Packaging**: Managing complex applications with **Helm**.
- **Service Mesh**: Controlling traffic, security, and observability with **Istio**.
- **Security**: Hardening clusters with Network Policies and Pod Security Standards.
- **Stateful Apps**: Running databases on K8s with StatefulSets.

---

## 📖 Theoretical Concepts

### 1. CRDs and Operators

Kubernetes is extensible. You can teach it new tricks.
- **CRD (Custom Resource Definition)**: Defines a new object type (e.g., `PrometheusRule`, `PostgresDB`).
- **Operator**: A controller that watches for your CRD and takes action. It encodes human operational knowledge into software (e.g., "How to backup a Postgres DB").

### 2. Helm (The Package Manager)

Writing 50 YAML files for one app is painful.
- **Chart**: A collection of templates.
- **Values**: Configuration values injected into templates.
- **Release**: A deployed instance of a chart.
- **Rollback**: `helm rollback my-app 1`.

### 3. Service Mesh (Istio)

A dedicated infrastructure layer for service-to-service communication.
- **Traffic Management**: Canary deployments (90% v1, 10% v2), Retries, Circuit Breakers.
- **Security**: mTLS (Mutual TLS) encryption between all pods automatically.
- **Observability**: Golden metrics (Latency, Traffic, Errors) for free.

### 4. Network Policies

By default, all pods can talk to all pods. This is bad.
- **NetworkPolicy**: A firewall rule for K8s.
- **Best Practice**: "Deny All" ingress by default, then whitelist specific traffic.

---

## 🔧 Practical Examples

### Helm Chart Structure

```text
my-chart/
  Chart.yaml
  values.yaml
  templates/
    deployment.yaml
    service.yaml
```

### Installing a Chart

```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install my-redis bitnami/redis --set architecture=standalone
```

### Network Policy (Deny All)

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
spec:
  podSelector: {}
  policyTypes:
  - Ingress
```

### Istio VirtualService (Canary)

```yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: my-app
spec:
  hosts:
  - my-app
  http:
  - route:
    - destination:
        host: my-app
        subset: v1
      weight: 90
    - destination:
        host: my-app
        subset: v2
      weight: 10
```

---

## 🎯 Hands-on Labs

- [Lab 21.1: Custom Resources](./labs/lab-21.1-custom-resources.md)
- [Lab 21.2: Operators](./labs/lab-21.2-operators.md)
- [Lab 21.3: Helm Charts](./labs/lab-21.3-helm-charts.md)
- [Lab 21.4: Helm Repositories](./labs/lab-21.4-helm-repositories.md)
- [Lab 21.5: Service Mesh Istio](./labs/lab-21.5-service-mesh-istio.md)
- [Lab 21.6: Network Policies](./labs/lab-21.6-network-policies.md)
- [Lab 21.7: Pod Security](./labs/lab-21.7-pod-security.md)
- [Lab 21.8: Cluster Autoscaling](./labs/lab-21.8-cluster-autoscaling.md)
- [Lab 21.9: Stateful Applications](./labs/lab-21.9-stateful-applications.md)
- [Lab 21.10: K8S Production](./labs/lab-21.10-k8s-production.md)

---

## 📚 Additional Resources

### Official Documentation
- [Helm Docs](https://helm.sh/docs/)
- [Istio Docs](https://istio.io/latest/docs/)
- [Kubernetes Network Policies](https://kubernetes.io/docs/concepts/services-networking/network-policies/)

### Tools
- [K9s](https://k9scli.io/) - Terminal UI for K8s.
- [Kubens/Kubectx](https://github.com/ahmetb/kubectx) - Switch namespaces/contexts fast.

---

## 🔑 Key Takeaways

1.  **Don't Write Raw YAML**: Use Helm or Kustomize for anything complex.
2.  **Zero Trust**: Assume the network is compromised. Use Network Policies and mTLS.
3.  **Operators are Powerful**: Use them for stateful apps (Databases, Queues) instead of managing StatefulSets manually.
4.  **Sidecars**: Istio injects a proxy (Envoy) into every pod to intercept traffic.

---

## ⏭️ Next Steps

1.  Complete the labs to master the K8s ecosystem.
2.  Proceed to **[Module 22: GitOps with ArgoCD](../module-22-gitops-argocd/README.md)** to automate your K8s deployments.
