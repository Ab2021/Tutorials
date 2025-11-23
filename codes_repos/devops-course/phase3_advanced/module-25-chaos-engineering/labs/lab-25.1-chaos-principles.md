# Lab 25.1: Chaos Principles

## Objective
Understand chaos engineering principles and practices.

## Learning Objectives
- Understand chaos engineering
- Define steady state
- Design chaos experiments
- Minimize blast radius

---

## Chaos Engineering Principles

1. **Define Steady State** - Normal system behavior
2. **Hypothesize** - What could go wrong?
3. **Inject Failure** - Simulate real-world issues
4. **Observe** - Monitor system response
5. **Learn** - Improve resilience

## Experiment Template

```yaml
experiment:
  name: "Pod Failure Test"
  hypothesis: "System remains available when 1 pod fails"
  steady_state:
    - metric: http_success_rate
      threshold: 99%
  method:
    - action: delete_pod
      target: frontend
      count: 1
  rollback:
    - restore_pod
```

## Success Criteria
✅ Chaos principles understood  
✅ Steady state defined  
✅ Experiment designed  
✅ Blast radius minimized  

**Time:** 30 min
