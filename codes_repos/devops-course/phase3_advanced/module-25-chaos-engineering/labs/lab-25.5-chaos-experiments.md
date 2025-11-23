# Lab 25.5: Chaos Experiments

## Objective
Design and execute comprehensive chaos experiments.

## Learning Objectives
- Design experiments
- Test network failures
- Simulate resource exhaustion
- Validate recovery

---

## Network Latency

```yaml
# network-latency.yaml
version: 1.0.0
title: "Network latency injection"

method:
  - type: action
    name: "add-latency"
    provider:
      type: python
      module: chaosistio.fault.actions
      func: add_delay_fault
      arguments:
        virtual_service_name: "myapp"
        fixed_delay: "5s"
        percentage: 50
```

## CPU Stress

```yaml
# cpu-stress.yaml
method:
  - type: action
    name: "stress-cpu"
    provider:
      type: process
      path: "stress-ng"
      arguments: "--cpu 4 --timeout 60s"
```

## Disk Fill

```yaml
# disk-fill.yaml
method:
  - type: action
    name: "fill-disk"
    provider:
      type: python
      module: chaosaws.ec2.actions
      func: fill_disk
      arguments:
        instance_id: "i-1234567890"
        path: "/data"
        size: "5G"
```

## Success Criteria
✅ Network chaos tested  
✅ Resource exhaustion simulated  
✅ System recovered  
✅ Weaknesses identified  

**Time:** 50 min
