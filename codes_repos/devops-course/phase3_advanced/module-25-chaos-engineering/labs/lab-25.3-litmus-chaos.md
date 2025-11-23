# Lab 25.3: Chaos Engineering with Litmus

## Objective
Implement chaos engineering experiments with LitmusChaos.

## Learning Objectives
- Install LitmusChaos
- Create chaos experiments
- Analyze system resilience
- Improve fault tolerance

---

## Install Litmus

```bash
# Install Litmus Operator
kubectl apply -f https://litmuschaos.github.io/litmus/litmus-operator-v2.0.0.yaml

# Verify installation
kubectl get pods -n litmus

# Install chaos experiments
kubectl apply -f https://hub.litmuschaos.io/api/chaos/2.0.0?file=charts/generic/experiments.yaml
```

## Pod Delete Experiment

```yaml
apiVersion: litmuschaos.io/v1alpha1
kind: ChaosEngine
metadata:
  name: nginx-chaos
  namespace: default
spec:
  appinfo:
    appns: 'default'
    applabel: 'app=nginx'
    appkind: 'deployment'
  engineState: 'active'
  chaosServiceAccount: litmus-admin
  experiments:
  - name: pod-delete
    spec:
      components:
        env:
        - name: TOTAL_CHAOS_DURATION
          value: '60'
        - name: CHAOS_INTERVAL
          value: '10'
        - name: FORCE
          value: 'false'
```

## Network Chaos

```yaml
apiVersion: litmuschaos.io/v1alpha1
kind: ChaosEngine
metadata:
  name: network-chaos
spec:
  experiments:
  - name: pod-network-latency
    spec:
      components:
        env:
        - name: NETWORK_LATENCY
          value: '2000'  # 2 seconds
        - name: TOTAL_CHAOS_DURATION
          value: '60'
```

## Success Criteria
✅ Litmus installed  
✅ Experiments running  
✅ System resilience tested  
✅ Improvements identified  

**Time:** 45 min
