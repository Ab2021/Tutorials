# Lab 1.4: DevOps Metrics and KPIs

## Objective
Measure DevOps success with key metrics and KPIs.

## Learning Objectives
- Define DORA metrics
- Track deployment frequency
- Measure lead time
- Monitor MTTR and change failure rate

---

## DORA Metrics

```python
# Track deployment frequency
deployments = {
    'week_1': 45,
    'week_2': 52,
    'week_3': 48,
    'week_4': 50
}

avg_deployments = sum(deployments.values()) / len(deployments)
print(f"Average deployments per week: {avg_deployments}")

# Lead time for changes
lead_times = [2.5, 3.0, 1.8, 2.2, 2.8]  # hours
avg_lead_time = sum(lead_times) / len(lead_times)
print(f"Average lead time: {avg_lead_time} hours")

# Change failure rate
total_deployments = 200
failed_deployments = 10
failure_rate = (failed_deployments / total_deployments) * 100
print(f"Change failure rate: {failure_rate}%")

# MTTR (Mean Time To Recovery)
recovery_times = [15, 30, 20, 25]  # minutes
mttr = sum(recovery_times) / len(recovery_times)
print(f"MTTR: {mttr} minutes")
```

## Metrics Dashboard

```yaml
# Grafana dashboard for DevOps metrics
dashboard:
  title: "DevOps Metrics"
  panels:
    - title: "Deployment Frequency"
      query: "rate(deployments_total[1w])"
    
    - title: "Lead Time"
      query: "avg(deployment_lead_time_seconds)"
    
    - title: "Change Failure Rate"
      query: "failed_deployments / total_deployments * 100"
    
    - title: "MTTR"
      query: "avg(incident_recovery_time_minutes)"
```

## Success Criteria
✅ DORA metrics tracked  
✅ Dashboard created  
✅ Trends analyzed  
✅ Improvements identified  

**Time:** 40 min
