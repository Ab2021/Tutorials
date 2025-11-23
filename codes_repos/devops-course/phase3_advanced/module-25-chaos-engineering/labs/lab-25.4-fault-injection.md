# Lab 25.4: Fault Injection

## Objective
Implement fault injection to test system resilience.

## Learning Objectives
- Inject network faults
- Simulate resource exhaustion
- Test failure scenarios
- Validate recovery mechanisms

---

## Network Fault Injection

```yaml
# Istio Fault Injection
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: ratings
spec:
  hosts:
  - ratings
  http:
  - fault:
      delay:
        percentage:
          value: 50
        fixedDelay: 5s
      abort:
        percentage:
          value: 10
        httpStatus: 500
    route:
    - destination:
        host: ratings
```

## CPU Stress

```yaml
apiVersion: litmuschaos.io/v1alpha1
kind: ChaosEngine
metadata:
  name: cpu-stress
spec:
  experiments:
  - name: pod-cpu-hog
    spec:
      components:
        env:
        - name: CPU_CORES
          value: '2'
        - name: TOTAL_CHAOS_DURATION
          value: '60'
```

## Memory Stress

```yaml
apiVersion: litmuschaos.io/v1alpha1
kind: ChaosEngine
metadata:
  name: memory-stress
spec:
  experiments:
  - name: pod-memory-hog
    spec:
      components:
        env:
        - name: MEMORY_CONSUMPTION
          value: '500'  # MB
        - name: TOTAL_CHAOS_DURATION
          value: '60'
```

## Disk Fill

```yaml
apiVersion: litmuschaos.io/v1alpha1
kind: ChaosEngine
metadata:
  name: disk-fill
spec:
  experiments:
  - name: disk-fill
    spec:
      components:
        env:
        - name: FILL_PERCENTAGE
          value: '80'
        - name: TOTAL_CHAOS_DURATION
          value: '60'
```

## Success Criteria
✅ Faults injected  
✅ System behavior observed  
✅ Recovery validated  
✅ Resilience improved  

**Time:** 45 min
