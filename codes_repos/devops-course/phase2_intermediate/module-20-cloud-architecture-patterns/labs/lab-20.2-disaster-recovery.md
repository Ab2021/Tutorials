# Lab 20.2: Disaster Recovery (Cross-Region Replication)

## 🎯 Objective

Survive the meteor. If `us-east-1` goes down completely, your business must continue. You will implement **S3 Cross-Region Replication (CRR)** to automatically copy data from Virginia to Oregon.

## 📋 Prerequisites

-   AWS Account.
-   Terraform installed.

## 📚 Background

### DR Strategies
1.  **Backup & Restore**: Slowest. Cheap. (RTO: Hours/Days).
2.  **Pilot Light**: DB is running in DR region, App servers are off. (RTO: Minutes/Hours).
3.  **Warm Standby**: Scaled down version running in DR. (RTO: Minutes).
4.  **Multi-Site Active/Active**: Both regions live. (RTO: Near Zero). Expensive.

---

## 🔨 Hands-On Implementation

### Part 1: The Providers 🌍

We need to talk to two regions.

1.  **`main.tf`:**
    ```hcl
    provider "aws" {
      alias  = "primary"
      region = "us-east-1"
    }

    provider "aws" {
      alias  = "dr"
      region = "us-west-2"
    }
    ```

### Part 2: The Buckets 🪣

1.  **Create Source & Destination:**
    ```hcl
    resource "aws_s3_bucket" "source" {
      provider = aws.primary
      bucket   = "devops-lab-source-<random>"
      versioning { enabled = true }
    }

    resource "aws_s3_bucket" "dest" {
      provider = aws.dr
      bucket   = "devops-lab-dest-<random>"
      versioning { enabled = true }
    }
    ```

### Part 3: IAM Role for Replication 🔑

S3 needs permission to copy objects.

1.  **Create Role:**
    ```hcl
    resource "aws_iam_role" "replication" {
      name = "s3-replication-role"
      assume_role_policy = ... (Allow s3.amazonaws.com)
    }
    
    resource "aws_iam_policy" "replication" {
      ... (Allow GetObject on Source and PutObject on Dest)
    }
    ```
    *(Omitted full JSON for brevity, check AWS docs or solution).*

### Part 4: Replication Configuration 🔄

1.  **Enable CRR:**
    ```hcl
    resource "aws_s3_bucket_replication_configuration" "replication" {
      provider = aws.primary
      role     = aws_iam_role.replication.arn
      bucket   = aws_s3_bucket.source.id

      rule {
        id     = "replicate-all"
        status = "Enabled"

        destination {
          bucket        = aws_s3_bucket.dest.arn
          storage_class = "STANDARD"
        }
      }
    }
    ```

### Part 5: Test 🧪

1.  **Apply:**
    `terraform apply`.

2.  **Upload:**
    Go to S3 Console (Source Bucket). Upload `image.jpg`.

3.  **Verify:**
    Switch to Destination Bucket (Oregon).
    Wait a few seconds.
    `image.jpg` appears!

---

## 🎯 Challenges

### Challenge 1: Bi-Directional Replication (Difficulty: ⭐⭐⭐⭐)

**Task:**
Configure replication both ways (East -> West AND West -> East).
*Use Case:* Active/Active architecture where users upload files to the nearest region.

### Challenge 2: Delete Markers (Difficulty: ⭐⭐⭐)

**Task:**
Delete the file in Source.
Check Destination. Is it deleted?
*Answer:* By default, delete markers are NOT replicated (to prevent accidental mass deletion). You can enable it in config.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 2:**
Add `delete_marker_replication { status = "Enabled" }` to the rule.
</details>

---

## 🔑 Key Takeaways

1.  **RTO (Recovery Time Objective)**: How long can you be down?
2.  **RPO (Recovery Point Objective)**: How much data can you lose? (CRR is async, so RPO is seconds).
3.  **Compliance**: Some industries require data to stay in specific regions (GDPR).

---

## ⏭️ Next Steps

**Congratulations!** You have completed Phase 2.
You are now an Intermediate DevOps Engineer.

**Phase 3 (Advanced)** awaits:
-   **Kubernetes Operators**.
-   **Service Mesh (Istio)**.
-   **GitOps at Scale**.
-   **DevSecOps Pipelines**.

Proceed to **Phase 3: Module 21 - Advanced Kubernetes**.
