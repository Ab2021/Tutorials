# Day 156: Only what you need: RBAC for AI
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 23: Security & Governance

---

> **🎯 Focus Area:** The Data Scientist needs `read` access to S3 training data but `write` access only to the Experiment Tracking server. The Inference Pod needs `read` access to the Model Registry but NO access to the Training Data. **Least Privilege** is the law.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Map** ML Personas (Data Scientist, ML Engineer, Service Account) to Permissions.
2.  **Configure** Kubernetes Service Accounts with OIDC for AWS IAM integration (IRSA).
3.  **Implement** Row-Level Security (RLS) policies for Multi-Tenant Models.
4.  **Audit** access logs to detect privilege escalation.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine with `kubectl`.

### Software Environment
- AWS Account (Simulated via LocalStack) or real K8s cluster.

---

## 📖 Theoretical Foundation

### 1. The Persona Matrix
| Persona | S3 Data | Model Registry | Kubernetes | Feature Store |
| :--- | :--- | :--- | :--- | :--- |
| **Data Scientist** | Read-Only | Write (Training) | Read (Logs) | Read |
| **ML Engineer** | Read-Only | Write (Prod Tags) | Admin (NS) | Admin |
| **Training Job** | Read (Raw) | Write (Artifacts) | None | Read |
| **Inference Pod** | None | Read (Artifacts) | None | Read (Online) |

### 2. IAM Roles for Service Accounts (IRSA)
In the old days, we hardcoded AWS Keys in K8s Secrets. **Bad**.
Now, we map a K8s ServiceAccount (`system:serviceaccount:default:my-pod`) to an AWS IAM Role (`arn:aws:iam::...:role/MyPodRole`). The pod gets a temporary token automatically.

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Kubernetes RBAC

Allow Developers to View Logs/Describe Pods, but NOT Delete Pods.

#### 📁 `manifests/rbac-developer.yaml`
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: ml-training
  name: ml-developer
rules:
- apiGroups: [""] # Core API
  resources: ["pods", "pods/log", "services"]
  verbs: ["get", "watch", "list"]
- apiGroups: ["batch"]
  resources: ["jobs"]
  verbs: ["get", "list", "create", "delete"] # Can manage Jobs
---
kind: RoleBinding
metadata:
  name: dev-binding
  namespace: ml-training
subjects:
- kind: User
  name: "alice@example.com" # Mapped via OIDC
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: ml-developer
  apiGroup: rbac.authorization.k8s.io
```

### 👨‍💻 Infrastructure: IRSA (Terraform)

Binding S3 access to a Pod.

```hcl
# 1. IAM Policy
resource "aws_iam_policy" "model_read" {
  name = "model_read_policy"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = ["s3:GetObject"]
      Effect = "Allow"
      Resource = "arn:aws:s3:::my-model-bucket/*"
    }]
  })
}

# 2. IAM Role (Trust K8s OIDC)
resource "aws_iam_role" "inference_role" {
  name = "inference_role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Federated = "arn:aws:iam::...:oidc-provider/..."
      }
      Action = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "oidc.eks...:sub": "system:serviceaccount:prod:inference-sa"
        }
      }
    }]
  })
}
```

### 👨‍💻 Core Implementation: Row Level Security (Python logic)

If a model serves multiple tenants (Customer A and B), verify access at inference time.

#### 📁 `src/secure_inference.py`
```python
from fastapi import FastAPI, Header, HTTPException, Depends

app = FastAPI()

# Mock Auth Service
def get_current_user(x_api_key: str = Header(...)):
    # In real life, JWT verify
    if x_api_key == "key_customer_A":
        return {"id": "cust_A", "allowed_models": ["model_v1"]}
    if x_api_key == "key_customer_B":
        return {"id": "cust_B", "allowed_models": ["model_v2"]}
    raise HTTPException(401, "Invalid Key")

@app.post("/predict/{model_name}")
def predict(model_name: str, user: dict = Depends(get_current_user)):
    # 1. Check Permission
    if model_name not in user["allowed_models"]:
        # Audit Log Warning
        print(f"SECURITY: User {user['id']} tried to access {model_name}")
        raise HTTPException(403, "Access Denied")
        
    # 2. Run Inference
    return {"result": "ok"}
```

---

## 🔬 Lab Exercise: "The Break in"

### Task
Simulate Privilege Escalation.
1.  **Scenario:** You are logged in as `DataScientist`.
2.  **Action:** Try to delete the Production Model Service. `kubectl delete svc model-prod -n production`.
3.  **Result:** `Error from server (Forbidden): services "model-prod" is forbidden: User "bob" cannot delete services in namespace "production"`.
4.  **Action:** Try to read S3 production data. `aws s3 ls s3://prod-data`.
5.  **Result:** Access Denied.
6.  **Success:** The system works.

---

## 📖 Advanced Theory: ABAC (Attribute Based Access Control)
RBAC says "Scientists can Read".
ABAC says "Scientists can Read IF `resource.tag == 'public'` AND `time < 5pm`".
More granular, but harder to manage. Used for extremely sensitive data (PII/Medical).

---

## 📝 Daily Summary

### Key Takeaways
1.  **Service Accounts:** Every Pod should have its own Service Account. Never use the `default` Service Account (which often has too many or too few permissions).
2.  **Namespace Isolation:** Put Dev/Test/Prod in separate K8s Namespaces (or Clusters). Use NetworkPolicies to block traffic between them.
3.  **Audit:** Enable CloudTrail and K8s Audit Logs. If a key is stolen, you need to know what files were downloaded.

### API Summary
```yaml
kind: RoleBinding
roleRef:
  name: role-name
subjects:
- kind: ServiceAccount
```

---

**Day 156 Complete** ✅

*Next: Day 157 - Secrets Management - Rotating Keys without Downtime.*
