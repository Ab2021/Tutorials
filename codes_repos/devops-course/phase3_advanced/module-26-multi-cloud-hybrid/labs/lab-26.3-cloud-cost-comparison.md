# Lab 26.3: Cloud Cost Comparison

## Objective
Compare and optimize costs across cloud providers.

## Learning Objectives
- Analyze cloud pricing
- Compare service costs
- Implement cost optimization
- Use cost management tools

---

## Cost Analysis

```python
# Cloud cost comparison
cloud_costs = {
    'aws': {
        'compute': {
            't2.micro': 0.0116,  # per hour
            't2.small': 0.023,
            't2.medium': 0.0464
        },
        'storage': {
            's3_standard': 0.023,  # per GB/month
            's3_ia': 0.0125
        }
    },
    'azure': {
        'compute': {
            'B1S': 0.0104,
            'B1MS': 0.0207,
            'B2S': 0.0416
        },
        'storage': {
            'blob_hot': 0.0184,
            'blob_cool': 0.01
        }
    },
    'gcp': {
        'compute': {
            'e2-micro': 0.0084,
            'e2-small': 0.0168,
            'e2-medium': 0.0336
        },
        'storage': {
            'standard': 0.020,
            'nearline': 0.010
        }
    }
}

def calculate_monthly_cost(provider, instance_type, storage_gb):
    compute_cost = cloud_costs[provider]['compute'][instance_type] * 730  # hours/month
    storage_cost = cloud_costs[provider]['storage'][list(cloud_costs[provider]['storage'].keys())[0]] * storage_gb
    return compute_cost + storage_cost

# Compare costs
for provider in ['aws', 'azure', 'gcp']:
    cost = calculate_monthly_cost(provider, list(cloud_costs[provider]['compute'].keys())[0], 100)
    print(f"{provider}: ${cost:.2f}/month")
```

## Cost Optimization

```python
# Reserved instances savings
on_demand_cost = 0.0464 * 730  # t2.medium
reserved_1yr = 0.0284 * 730
reserved_3yr = 0.0186 * 730

savings_1yr = (on_demand_cost - reserved_1yr) / on_demand_cost * 100
savings_3yr = (on_demand_cost - reserved_3yr) / on_demand_cost * 100

print(f"1-year RI savings: {savings_1yr:.1f}%")
print(f"3-year RI savings: {savings_3yr:.1f}%")
```

## Success Criteria
✅ Costs compared across clouds  
✅ Optimization opportunities identified  
✅ Cost savings calculated  
✅ Recommendations documented  

**Time:** 40 min
