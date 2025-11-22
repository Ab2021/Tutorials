# Multi-Cloud & Hybrid Strategies

## 🎯 Learning Objectives

By the end of this module, you will have a comprehensive understanding of Multi-Cloud and Hybrid architectures, including:
- **Strategy**: Understanding when (and when not) to go multi-cloud.
- **Portability**: Using cloud-agnostic tools like Terraform and Crossplane.
- **Networking**: Connecting AWS, Azure, and GCP with VPN/Direct Connect.
- **Hybrid**: Integrating on-premises data centers with the cloud.
- **Cost**: Comparing pricing models across providers.

---

## 📖 Theoretical Concepts

### 1. Why Multi-Cloud?

**Reasons to do it:**
- **Avoid Vendor Lock-in**: Don't put all eggs in one basket.
- **Best-of-Breed**: Use AWS for compute, GCP for ML, Azure for Active Directory.
- **Compliance**: Some countries require data to stay in-country.

**Reasons NOT to do it:**
- **Complexity**: You now have 3 IAM systems, 3 CLIs, 3 billing dashboards.
- **Cost**: Data transfer between clouds is expensive ($0.09/GB).
- **Talent**: Finding engineers who know all 3 clouds is hard.

### 2. Cloud-Agnostic Tools

- **Terraform**: Works with AWS, Azure, GCP, and 1000+ providers.
- **Kubernetes**: Runs anywhere (EKS, AKS, GKE, on-prem).
- **Crossplane**: Kubernetes-native IaC. Provision cloud resources using K8s CRDs.

### 3. Hybrid Cloud

On-premises + Cloud.
- **Use Case**: Legacy mainframe can't move to cloud. Connect it to AWS via Direct Connect.
- **Tools**: AWS Outposts (AWS hardware in your data center), Azure Arc (Manage on-prem servers from Azure).

### 4. Networking

Connecting clouds is non-trivial.
- **VPN**: Encrypted tunnel over the internet. Slow, cheap.
- **Direct Connect / ExpressRoute**: Dedicated fiber. Fast, expensive.
- **Transit Gateway**: Hub-and-spoke model to connect multiple VPCs/VNets.

---

## 🔧 Practical Examples

### Terraform Multi-Cloud

```hcl
# AWS Provider
provider "aws" {
  region = "us-east-1"
}

resource "aws_s3_bucket" "data" {
  bucket = "my-data-bucket"
}

# GCP Provider
provider "google" {
  project = "my-project"
  region  = "us-central1"
}

resource "google_storage_bucket" "data" {
  name     = "my-data-bucket-gcp"
  location = "US"
}
```

### Crossplane Composition

```yaml
apiVersion: apiextensions.crossplane.io/v1
kind: Composition
metadata:
  name: xpostgresqlinstances
spec:
  compositeTypeRef:
    apiVersion: database.example.org/v1alpha1
    kind: XPostgreSQLInstance
  resources:
  - name: rdsinstance
    base:
      apiVersion: database.aws.crossplane.io/v1beta1
      kind: RDSInstance
      spec:
        forProvider:
          engine: postgres
```

---

## 🎯 Hands-on Labs

- [Lab 26.1: Multi-Cloud Terraform](./labs/lab-26.1-terraform-multicloud.md)
- [Lab 26.2: Crossplane (Universal Control Plane)](./labs/lab-26.2-crossplane.md)
- [Lab 26.3: Gcp Services](./labs/lab-26.3-gcp-services.md)
- [Lab 26.4: Cloud Agnostic Tools](./labs/lab-26.4-cloud-agnostic-tools.md)
- [Lab 26.5: Hybrid Cloud](./labs/lab-26.5-hybrid-cloud.md)
- [Lab 26.6: Cloud Migration](./labs/lab-26.6-cloud-migration.md)
- [Lab 26.7: Multi Cloud Networking](./labs/lab-26.7-multi-cloud-networking.md)
- [Lab 26.8: Cost Comparison](./labs/lab-26.8-cost-comparison.md)
- [Lab 26.9: Vendor Lock In](./labs/lab-26.9-vendor-lock-in.md)
- [Lab 26.10: Multi Cloud Management](./labs/lab-26.10-multi-cloud-management.md)

---

## 📚 Additional Resources

### Official Documentation
- [Crossplane Documentation](https://crossplane.io/docs/)
- [AWS Direct Connect](https://aws.amazon.com/directconnect/)

### Tools
- [Terraform Cloud](https://cloud.hashicorp.com/products/terraform)

---

## 🔑 Key Takeaways

1.  **Multi-Cloud is Hard**: Only do it if you have a strong business reason.
2.  **Abstractions Have Limits**: Terraform can provision resources, but it can't abstract away AWS-specific features (e.g., Lambda).
3.  **Data Gravity**: Once you have 100TB in S3, moving it to GCS is painful and expensive.
4.  **Kubernetes is the Abstraction**: If you run everything in K8s, switching clouds is "just" changing the underlying nodes.

---

## ⏭️ Next Steps

1.  Complete the labs to understand multi-cloud trade-offs.
2.  Proceed to **[Module 27: Platform Engineering](../module-27-platform-engineering/README.md)** to build Internal Developer Platforms.
