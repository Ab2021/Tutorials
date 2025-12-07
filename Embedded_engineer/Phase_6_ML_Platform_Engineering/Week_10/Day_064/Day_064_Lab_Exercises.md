# Days 64-70: Week 10 - Cloud Platforms for ML Labs
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## Day 64: AWS for ML

```bash
# Create EKS cluster with GPU nodes
eksctl create cluster \
  --name ml-platform \
  --node-type p3.2xlarge \
  --nodes 4 \
  --nodes-min 1 \
  --nodes-max 10

# Install GPU device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/deployments/static/nvidia-device-plugin.yml
```

---

## Day 65: GCP for ML

```bash
# Create GKE cluster with GPU
gcloud container clusters create ml-platform \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --machine-type n1-standard-4 \
  --num-nodes 4

# Install GPU drivers
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

---

## Day 66: Azure for ML

```bash
# Create AKS cluster with GPU
az aks create \
  --resource-group ml-rg \
  --name ml-platform \
  --node-count 3 \
  --node-vm-size Standard_NC6s_v3
```

---

## Day 67: Cost Optimization

```yaml
# Spot instance node group
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig
metadata:
  name: ml-platform
managedNodeGroups:
- name: spot-gpu
  instanceTypes: ["g4dn.xlarge", "g4dn.2xlarge"]
  spot: true
  minSize: 0
  maxSize: 10
```

---

## Day 68: High Availability

```yaml
# Pod Disruption Budget
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: ml-api-pdb
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: ml-api
```

---

## Day 69: Hybrid Cloud

```bash
# Arc-enabled Kubernetes (Azure)
az connectedk8s connect \
  --name on-prem-cluster \
  --resource-group hybrid-rg
```

---

## Day 70: Week 10 Project

```bash
# Multi-cloud deployment checklist
# [ ] EKS cluster with GPU
# [ ] ArgoCD installed
# [ ] Model registry (S3/MLflow)
# [ ] Monitoring (Prometheus/Grafana)
# [ ] Cost alerts configured
```

---

## 📝 Week 10 Summary
| Day | Topic | Platform |
|-----|-------|----------|
| 64 | AWS | EKS, EC2 |
| 65 | GCP | GKE, Compute |
| 66 | Azure | AKS, VMs |
| 67 | Cost | Spot, RI |
| 68 | HA | PDB, Multi-AZ |
| 69 | Hybrid | On-prem |
| 70 | Project | Cloud ML |
