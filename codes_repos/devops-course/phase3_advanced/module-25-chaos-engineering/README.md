# Module 25: Chaos Engineering

## 🎯 Learning Objectives

By the end of this module, you will:
- Understand the core principles of Chaos Engineering.
- Gain hands-on experience with **Chaos Mesh** and **LitmusChaos**.
- Design and execute chaos experiments to test system resilience.
- Implement automated chaos in CI/CD pipelines.
- Analyze blast radius and minimize production impact.

---

## 📖 Module Overview

**Duration:** 6-8 hours  
**Difficulty:** Advanced
**Prerequisites:**
- [Module 12: Kubernetes Fundamentals](../module-12-kubernetes-fundamentals/README.md)
- [Module 16: Monitoring & Observability](../module-16-monitoring-observability/README.md)

### Topics Covered

- **Principles of Chaos**: Hypothesis, Variances, Experiments.
- **Tools**: Chaos Mesh, LitmusChaos, Gremlin.
- **Experiments**: Pod Kill, Network Latency, CPU Stress, IO Faults.
- **Safety**: Abort conditions and Blast Radius.

---

## 📚 Theoretical Concepts

### Introduction

**Chaos Engineering** is the discipline of experimenting on a system in order to build confidence in the system's capability to withstand turbulent conditions in production. It is not about "breaking things randomly"; it is about **controlled experiments**.

### Key Concepts

#### 1. Steady State Hypothesis
Define "normal" behavior. For example, "The 99th percentile latency is < 300ms" or "Error rate is < 1%". If the system deviates from this during an experiment, the system is not resilient.

#### 2. Blast Radius
The subset of the system affected by the experiment. Start small (one container, one user) and expand (one node, one availability zone).

#### 3. Fault Injection
Intentionally introducing errors:
- **Resource**: CPU burn, Memory leak, Disk full.
- **Network**: Latency, Packet loss, DNS failure.
- **State**: Kill pods, Reboot nodes, Time skew.

### Best Practices

- **Start in Staging**: Don't run in Prod until you are confident.
- **Automate**: Run chaos experiments as part of the CI/CD pipeline.
- **Minimize Blast Radius**: Use namespaces and labels to target specific components.
- **Have a Kill Switch**: Stop the experiment immediately if business metrics (e.g., Sales) drop.

---

## 🔧 Practical Examples

### Example 1: Network Delay with Chaos Mesh

```yaml
apiVersion: chaos-mesh.org/v1alpha1
kind: NetworkChaos
metadata:
  name: network-delay
spec:
  action: delay
  mode: one
  selector:
    namespaces:
      - default
    labelSelectors:
      app: nginx
  delay:
    latency: "200ms"
  duration: "30s"
```

### Example 2: Pod Kill

```bash
# Imperative Pod Kill
kubectl delete pod -l app=nginx
# (Watch it restart if you have a Deployment)
```

---

## 🎯 Hands-on Labs

- [Lab 25.1: Chaos Mesh (Pod Kill)](./labs/lab-25.1-chaos-mesh.md)
- [Lab 25.1: Chaos Principles](./labs/lab-25.1-chaos-principles.md)
- [Lab 25.10: Building Resilience](./labs/lab-25.10-building-resilience.md)
- [Lab 25.2: Chaos Monkey](./labs/lab-25.2-chaos-monkey.md)
- [Lab 25.2: LitmusChaos](./labs/lab-25.2-litmuschaos.md)
- [Lab 25.3: Failure Injection](./labs/lab-25.3-failure-injection.md)
- [Lab 25.4: Resilience Testing](./labs/lab-25.4-resilience-testing.md)
- [Lab 25.5: Chaos Experiments](./labs/lab-25.5-chaos-experiments.md)
- [Lab 25.6: Blast Radius](./labs/lab-25.6-blast-radius.md)
- [Lab 25.7: Steady State Hypothesis](./labs/lab-25.7-steady-state-hypothesis.md)
- [Lab 25.8: Automated Chaos](./labs/lab-25.8-automated-chaos.md)
- [Lab 25.9: Chaos Reporting](./labs/lab-25.9-chaos-reporting.md)

---

## 📚 Additional Resources

### Official Documentation
- [Chaos Mesh Documentation](https://chaos-mesh.org/docs/)
- [LitmusChaos Documentation](https://litmuschaos.io/docs/)
- [Principles of Chaos Engineering](https://principlesofchaos.org/)

### Tutorials and Guides
- [CNCF Chaos Engineering Whitepaper](https://github.com/cncf/tag-app-delivery/blob/master/chaos-engineering/whitepaper/chaos-engineering-whitepaper.md)

---

## 🔑 Key Takeaways

1.  **Resilience is a Requirement**: Systems fail. Design for failure.
2.  **Observability is Prerequisite**: You cannot do Chaos Engineering if you cannot measure the impact.
3.  **Culture**: Move from "Who broke it?" to "How can we make it unbreakable?".

---

## ⏭️ Next Steps

1.  Complete the labs to practice Fault Injection.
2.  Proceed to **[Module 26: Multi-Cloud & Hybrid](../module-26-multi-cloud-hybrid/README.md)** to learn about managing infrastructure across clouds.
