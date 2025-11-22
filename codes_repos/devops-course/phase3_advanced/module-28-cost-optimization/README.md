# Cloud Cost Optimization & FinOps

## 🎯 Learning Objectives

By the end of this module, you will have a comprehensive understanding of Cost Optimization, including:
- **FinOps**: Understanding the cultural shift to make everyone responsible for cost.
- **Visibility**: Using tools like **Kubecost** and AWS Cost Explorer to see where money goes.
- **Optimization**: Rightsizing instances, using Spot/Reserved Instances, and eliminating waste.
- **Governance**: Implementing budgets, alerts, and tagging policies.
- **Kubernetes**: Optimizing K8s costs with **Karpenter** and resource limits.

---

## 📖 Theoretical Concepts

### 1. FinOps Principles

FinOps (Financial Operations) is a cultural practice that brings financial accountability to the variable spend model of cloud.
- **Collaborate**: Finance, Engineering, and Business work together.
- **Decide**: Everyone can make cost-aware decisions in real-time.
- **Optimize**: Continuously improve efficiency.

### 2. The 80/20 Rule

80% of your cloud bill comes from 20% of resources.
- **Compute**: EC2/VMs (biggest cost).
- **Data Transfer**: Moving data between regions/clouds.
- **Storage**: S3/EBS (cheap per GB, but adds up).

### 3. Pricing Models

- **On-Demand**: Pay by the hour. Most expensive. Most flexible.
- **Reserved Instances (RIs)**: 1-3 year commitment. 30-70% discount.
- **Spot Instances**: Bid on unused capacity. Up to 90% discount. Can be terminated with 2-min notice.
- **Savings Plans**: Flexible RIs. Commit to $/hour instead of instance type.

### 4. Kubernetes Cost Optimization

- **Right-Sizing**: Set `requests` = actual usage. Don't over-provision.
- **Cluster Autoscaler**: Add/remove nodes based on pending pods.
- **Karpenter**: AWS-native autoscaler. Provisions the exact instance type needed (vs fixed node groups).
- **Kubecost**: Shows cost per namespace/pod/label.

---

## 🔧 Practical Examples

### Tagging Strategy (Terraform)

```hcl
resource "aws_instance" "web" {
  ami           = "ami-123456"
  instance_type = "t3.micro"

  tags = {
    Environment = "Production"
    Team        = "Platform"
    CostCenter  = "Engineering"
    Project     = "WebApp"
  }
}
```

### Budget Alert (AWS CLI)

```bash
aws budgets create-budget \
  --account-id 123456789012 \
  --budget file://budget.json \
  --notifications-with-subscribers file://notifications.json
```

### Karpenter Provisioner

```yaml
apiVersion: karpenter.sh/v1alpha5
kind: Provisioner
metadata:
  name: default
spec:
  requirements:
    - key: karpenter.sh/capacity-type
      operator: In
      values: ["spot", "on-demand"]
  limits:
    resources:
      cpu: 1000
```

---

## 🎯 Hands-on Labs

- [Lab 28.1: Kubecost (Cost Visibility)](./labs/lab-28.1-kubecost.md)
- [Lab 28.2: Karpenter (Just-in-Time Scaling)](./labs/lab-28.2-karpenter.md)
- [Lab 28.3: Resource Tagging](./labs/lab-28.3-resource-tagging.md)
- [Lab 28.4: Rightsizing](./labs/lab-28.4-rightsizing.md)
- [Lab 28.5: Reserved Instances](./labs/lab-28.5-reserved-instances.md)
- [Lab 28.6: Spot Instances](./labs/lab-28.6-spot-instances.md)
- [Lab 28.7: Cost Allocation](./labs/lab-28.7-cost-allocation.md)
- [Lab 28.8: Budget Alerts](./labs/lab-28.8-budget-alerts.md)
- [Lab 28.9: Waste Elimination](./labs/lab-28.9-waste-elimination.md)
- [Lab 28.10: Cost Governance](./labs/lab-28.10-cost-governance.md)

---

## 📚 Additional Resources

### Official Documentation
- [Kubecost Documentation](https://docs.kubecost.com/)
- [AWS Cost Optimization](https://aws.amazon.com/aws-cost-management/)

### Tools
- [Infracost](https://www.infracost.io/) - See Terraform costs before deploying.

---

## 🔑 Key Takeaways

1.  **Visibility First**: You can't optimize what you can't measure.
2.  **Tag Everything**: Tags are the foundation of cost allocation.
3.  **Delete Unused Resources**: Orphaned EBS volumes, old snapshots, idle load balancers.
4.  **Automate Shutdowns**: Turn off Dev/Test environments at night and weekends.

---

## ⏭️ Next Steps

1.  Complete the labs to reduce your cloud bill by 30%.
2.  Proceed to **[Module 29: Incident Management](../module-29-incident-management/README.md)** to handle production outages.
