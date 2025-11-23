# Lab 26.4: Cloud-Agnostic Tools

## Objective
Use cloud-agnostic tools for portability.

## Learning Objectives
- Use Kubernetes across clouds
- Implement Terraform providers
- Use cloud-agnostic databases
- Manage secrets universally

---

## Kubernetes on Multiple Clouds

```bash
# AWS EKS
eksctl create cluster --name my-cluster --region us-east-1

# Azure AKS
az aks create --resource-group myRG --name my-cluster

# GCP GKE
gcloud container clusters create my-cluster --zone us-central1-a
```

## Cloud-Agnostic Storage

```python
# Using MinIO (S3-compatible)
from minio import Minio

client = Minio(
    "minio.example.com:9000",
    access_key="ACCESS_KEY",
    secret_key="SECRET_KEY"
)

# Works with AWS S3, MinIO, GCS, Azure Blob
client.fput_object("bucket", "object", "file.txt")
```

## Universal Secrets

```bash
# Vault works everywhere
vault kv put secret/db password=secret

# Application reads from Vault regardless of cloud
```

## Success Criteria
✅ K8s deployed on multiple clouds  
✅ Cloud-agnostic storage working  
✅ Secrets managed universally  
✅ Applications portable  

**Time:** 45 min
