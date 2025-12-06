# Day 192: Entropy Fighters: Managing Infrastructure Drift
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 28: Platform Engineering Practices

---

> **🎯 Focus Area:** You provisioned 100 G5 instances with Terraform. Then a developer manually SSH'd in and installed a weird CUDA driver. Now the cluster is inconsistent. **Drift Detection** is the practice of continuously reconciling Reality with Code.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the Terraform Lifecycle (Init -> Plan -> Apply) and State File locking.
2.  **Deploy** **Atlantis** for Pull-Request automated infrastructure changes.
3.  **Detect** Drift by running scheduled `terraform plan` without applying.
4.  **Remediate** Drift by forcing the state back to the desired configuration.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `terraform` (or `opentofu`), `docker`.

---

## 📖 Theoretical Foundation

### 1. The State File
Terraform stores its knowledge of the world in `terraform.tfstate`.
*   **Drift:** When the Real World (AWS Console) differs from the State File.
*   **Divergence:** When the Code (`main.tf`) differs from the State File.
*   **Goal:** Code == State == Reality.

### 2. Atlantis Workflow
Developer opens PR -> Atlantis runs `terraform plan` -> Comments result on PR.
Reviewer approves -> Developer comments `atlantis apply` -> Atlantis runs apply -> Merges PR.
**Result:** No one runs Terraform from their laptop. Full audit trail.

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Terraform S3 Backend

Locking state is critical for teams.

#### 📁 `main.tf`
```hcl
terraform {
  backend "s3" {
    bucket         = "my-tf-state"
    key            = "platform/dev.tfstate"
    region         = "us-east-1"
    dynamodb_table = "tf-lock" # Prevents simultaneous writes
  }
}

provider "aws" {
  region = "us-east-1"
}

resource "aws_instance" "gpu_node" {
  ami           = "ami-12345678"
  instance_type = "g4dn.xlarge"
  tags = {
    Name = "Training-Node-1"
  }
}
```

### 👨‍💻 Infrastructure: Atlantis (Docker)

Self-hosted PR automation.

#### 📁 `docker-compose.yml`
```yaml
version: '3'
services:
  atlantis:
    image: runatlantis/atlantis:v0.25.0
    ports:
      - "4141:4141"
    environment:
      - ATLANTIS_GH_USER=mybot
      - ATLANTIS_GH_TOKEN=...
      - ATLANTIS_REPO_ALLOWLIST=github.com/myorg/*
    command: server
```

### 👨‍💻 Core Implementation: Drift Detection Script

Run this nightly via Cron/Jenkins.

#### 📁 `scripts/check_drift.sh`
```bash
#!/bin/bash
# 1. Initialize
terraform init

# 2. Plan (Check for changes)
# -detailed-exitcode: Returns 0 for no changes, 1 for error, 2 for pending changes (Drift)
terraform plan -detailed-exitcode

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo "No Drift Detected. Infrastructure is clean."
elif [ $EXIT_CODE -eq 2 ]; then
  echo "DRIFT DETECTED! Someone changed cloud resources manually."
  # Send Alert to PagerDuty / Slack
  curl -X POST -H 'Content-type: application/json' --data '{"text":"Drift Detected!"}' $SLACK_WEBHOOK
else
  echo "Terraform Error."
fi
```

---

## 🔬 Lab Exercise: "The ClickOps Crime"

### Task
Simulate manual interference.
1.  **Apply:** Run `terraform apply` to create the EC2 instance.
2.  **Verify:** Instance exists. Tag is `Training-Node-1`.
3.  **Sabotage:** Go to AWS Console. Change the Tag to `Training-Node-HACKED`.
4.  **Detect:** Run `terraform plan`.
    *   Output: `~ tags: "Name": "Training-Node-HACKED" => "Training-Node-1"`.
5.  **Remediate:** Run `terraform apply`.
    *   Result: Tag is reverted to `Training-Node-1`. Order restored.

---

## 📖 Advanced Theory: Immutable Infrastructure
Instead of using Terraform to fix a "drifted" server (e.g., someone installed a package), use **Immutable Infrastructure**.
*   **Pattern:** Bake a new AMI (Packer). Destroy the old server. Deploy the new one.
*   **Benefit:** Drift is impossible because servers are short-lived.
*   **Tools:** Karpenter (Provisions fresh nodes for Pods), Packer (Builds Images).

---

## 📝 Daily Summary

### Key Takeaways
1.  **Laptop Ban:** Never run `terraform apply` from a laptop in production. Use CI/CD (Atlantis/Github Actions). Laptops lose internet, have different versions, and leave lock files stuck.
2.  **Plan Review:** The Output of `terraform plan` is the most important artifact in Ops. Read it line by line. Deletions (`-`) are dangerous.
3.  **Crossplane:** The future? Managing Cloud Resources using Kubernetes YAML (`kubectl apply -f s3bucket.yaml`). Continuous Reconciliation is built-in to the K8s Controller loop.

### API Summary
```bash
terraform plan -out=tfplan
terraform show -json tfplan
```

---

**Day 192 Complete** ✅

*Next: Day 193 - Cost Governance Basics - FinOps.*
