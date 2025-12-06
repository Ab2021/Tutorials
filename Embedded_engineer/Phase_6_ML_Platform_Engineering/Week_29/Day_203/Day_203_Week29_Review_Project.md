# Day 203: Week 29 Review & Project - Titan Platform (Part 1)
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 29: Capstone Project Part 1

---

> **🎯 Focus Area:** We have designed the blueprint, provisioned the network, set up the training and serving engines, wired up observability, and locked down security. Today, we launch **Titan v1.0**.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Bootstrap** the entire Titan Platform using a "Seed Script" (Terraform + ArgoCD).
2.  **Execute** an End-to-End ML Workflow: Code -> Train (Ray) -> Serve (KServe) -> Monitor.
3.  **Validate** NFRs: "Does it work if I kill a node?" "Can I access it without a token?" (Security Test).
4.  **Prepare** the "Part 1 Delivery Report" for stakeholders.

---

## 📚 Week 29 Recap

| Day | Topic | The "Ah-ha" Moment |
|-----|-------|---------------------|
| 197 | Design RFC | "Writing the RFC caught the VPC CIDR overlap bug before we wrote code." |
| 198 | Data Plane | "VPC Peering enables private IP talk between US and EU." |
| 199 | Training | "RayJob allows us to spin up 10 GPUs for 1 hour, then delete them." |
| 200 | Serving | "KServe abstracts Knative/Istio complexity into a simple YAML." |
| 201 | Observability | "A/B Testing requires Propagating Context Headers properly." |
| 202 | Security | "The Zero Trust model assumes the network is already breached." |

---

## 🏗️ Final Project: "Titan Launch"

### Step 1: Bootstrap (The Big Bang)

Run this once to create the universe.

#### 📁 `titan/bootstrap.sh`
```bash
#!/bin/bash
set -e

# 1. Infrastructure (AWS)
echo "--- Provisioning AWS Network & Clusters ---"
cd iac/live
terraform init
terraform apply -auto-approve

# 2. Config Connection
aws eks update-kubeconfig --name titan-control --alias control
aws eks update-kubeconfig --name titan-gpu --alias gpu
aws eks update-kubeconfig --name titan-inf --alias inf

# 3. GitOps Seed (ArgoCD)
echo "--- Seeding ArgoCD ---"
kubectl config use-context control
kubectl apply -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
kubectl apply -f gitops/root-app.yaml
# (This Root App installs KubeRay on 'gpu', KServe on 'inf', and Thanos everywhere)
```

### Step 2: The Training Job

Submit a workload to verify the Compute Plane.

#### 📁 `titan/docs/run_training.md`
```bash
# Submit to Control Plane (Karmada/ArgoCD will route it)
kubectl apply -f manifests/jobs/bert-finetune.yaml

# Verify
kubectl get rayjob bert-finetune -w
# Wait for Status: Succeeded
```

### Step 3: Distribution (Model Registry)

Simulate CI/CD pushing the model.

```bash
# Upload Model Artifact
aws s3 cp ./model_output s3://titan-models/bert/v1 --recursive
```

### Step 4: The Serving Deployment

Deploy the model to the Inference Cluster.

#### 📁 `titan/docs/deploy_model.md`
```bash
kubectl apply -f manifests/serving/bert-v1.yaml

# Verify Endpoint
kubectl get isvc bert-v1
# URL: https://bert-v1.titan.ai
```

### Step 5: Verification (Curl)

Test Security and Latency.

```bash
# 1. No Token (Should Fail)
curl -i https://bert-v1.titan.ai
# HTTP/1.1 401 Unauthorized

# 2. With Token (Should Succeed)
TOKEN=$(oidc-login get-token)
curl -H "Authorization: Bearer $TOKEN" https://bert-v1.titan.ai/v1/models/bert:predict -d @input.json
# HTTP/1.1 200 OK
# {"predictions": [...]}
```

---

## 🔬 Lab Exercise: "The ChaosMonkey"

### Task
Kill the Brain.
1.  **Scenario:** The KubeRay Operator pod crashes on the GPU cluster.
2.  **Impact:** Existing Training Jobs continue running (Ray Cluster pods are independent). New Jobs cannot be submitted.
3.  **Observation:** ArgoCD detects "Application Degraded". K8s restarts the Operator.
4.  **Result:** System heals itself within 30 seconds.
5.  **Lesson:** Decoupling Control Plane (Operator) from Data Plane (RayWorker) increases availability.

---

## 📝 Part 1 Delivery Report
**Status:** GREEN.
**Completed:**
*   [x] Multi-Region Network (US/EU).
*   [x] GPU Training Cluster (Auto-scaling).
*   [x] Serverless Inference Cluster.
*   [x] SSO Integration (GitHub).
**Next Steps (Part 2):**
*   Load Testing (10k RPS).
*   Cost Optimization (Spot Integration).
*   Documentation (TechDocs).

---

**Week 29 Complete** ✅
**Phase 6 Capstone Part 1 Complete**

*Next: Phase 6 Capstone Part 2 - Optimization & Certification.*
