# Lab 25.2: Chaos Monkey

## Objective
Implement Chaos Monkey for random instance termination.

## Learning Objectives
- Install Chaos Monkey
- Configure termination rules
- Monitor impact
- Verify auto-recovery

---

## Chaos Toolkit

```bash
pip install chaostoolkit chaostoolkit-kubernetes
```

## Experiment

```yaml
# chaos-experiment.yaml
version: 1.0.0
title: "Kill random pod"
description: "Terminate a random frontend pod"

steady-state-hypothesis:
  title: "Application is healthy"
  probes:
    - type: probe
      name: "app-responds"
      tolerance: 200
      provider:
        type: http
        url: "http://myapp.com/health"

method:
  - type: action
    name: "terminate-pod"
    provider:
      type: python
      module: chaosk8s.pod.actions
      func: terminate_pods
      arguments:
        label_selector: "app=frontend"
        qty: 1
        ns: "default"
    pauses:
      after: 10

rollbacks:
  - type: action
    name: "scale-up"
    provider:
      type: python
      module: chaosk8s.deployment.actions
      func: scale_deployment
      arguments:
        name: "frontend"
        replicas: 3
```

## Run Experiment

```bash
chaos run chaos-experiment.yaml
```

## Success Criteria
✅ Chaos Toolkit installed  
✅ Experiment executed  
✅ System recovered  
✅ Metrics collected  

**Time:** 40 min
