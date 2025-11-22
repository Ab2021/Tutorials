# Lab 26.2: Crossplane (Universal Control Plane)

## 🎯 Objective

Infrastructure as Kubernetes Data. **Crossplane** allows you to provision AWS S3 Buckets, RDS Instances, and GCP VPCs by creating Kubernetes YAML manifests (`kind: Bucket`).

## 📋 Prerequisites

-   Minikube running.
-   Helm installed.

## 📚 Background

### Concepts
-   **Provider**: The plugin for the cloud (Provider AWS).
-   **Managed Resource (MR)**: The low-level resource (`Bucket`).
-   **Composite Resource (XR)**: The high-level abstraction (`MyDatabase` -> RDS + Firewall + Subnet).
-   **Claim (XRC)**: What the developer creates (`kind: PostgresClaim`).

---

## 🔨 Hands-On Implementation

### Part 1: Install Crossplane ✈️

1.  **Install:**
    ```bash
    helm repo add crossplane-stable https://charts.crossplane.io/stable
    helm repo update
    helm install crossplane crossplane-stable/crossplane --namespace crossplane-system --create-namespace
    ```

2.  **Install AWS Provider:**
    ```bash
    kubectl crossplane install provider xpkg.upbound.io/crossplane-contrib/provider-aws:v0.33.0
    ```

### Part 2: Configure Credentials 🔑

(We will simulate this step or use a dummy secret if you don't want to provide real keys).

1.  **Create `aws-creds.conf`:**
    ```ini
    [default]
    aws_access_key_id = ASIA...
    aws_secret_access_key = ...
    ```

2.  **Create Secret:**
    ```bash
    kubectl create secret generic aws-creds -n crossplane-system --from-file=creds=./aws-creds.conf
    ```

3.  **Configure Provider:**
    ```yaml
    apiVersion: aws.crossplane.io/v1beta1
    kind: ProviderConfig
    metadata:
      name: default
    spec:
      credentials:
        source: Secret
        secretRef:
          namespace: crossplane-system
          name: aws-creds
          key: creds
    ```

### Part 3: Provision a Bucket 🪣

1.  **Create `bucket.yaml`:**
    ```yaml
    apiVersion: s3.aws.crossplane.io/v1beta1
    kind: Bucket
    metadata:
      name: crossplane-bucket-lab
    spec:
      forProvider:
        acl: private
        locationConstraint: us-east-1
      providerConfigRef:
        name: default
    ```

2.  **Apply:**
    ```bash
    kubectl apply -f bucket.yaml
    ```

3.  **Verify:**
    `kubectl get bucket`.
    Status should go from `Synced: False` to `Synced: True` and `Ready: True`.
    Check AWS Console.

### Part 4: Delete 🗑️

1.  **Delete YAML:**
    ```bash
    kubectl delete -f bucket.yaml
    ```
    *Result:* Crossplane deletes the bucket in AWS.

---

## 🎯 Challenges

### Challenge 1: Composition (Difficulty: ⭐⭐⭐⭐)

**Task:**
Create a `CompositeResourceDefinition` (XRD) called `CompositeDatabase`.
When a user creates a `CompositeDatabase`, Crossplane should create an RDS Instance AND a Security Group.
*Goal:* Create your own internal cloud platform API.

### Challenge 2: GitOps Integration (Difficulty: ⭐⭐)

**Task:**
Commit `bucket.yaml` to Git.
Let ArgoCD sync it.
*Result:* You now have GitOps for Infrastructure.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 2:**
Just standard ArgoCD setup. Point it to the folder containing `bucket.yaml`.
</details>

---

## 🔑 Key Takeaways

1.  **K8s API for Everything**: No need to learn HCL. Just use YAML.
2.  **Drift Correction**: Crossplane runs a control loop. If someone deletes the bucket in AWS console, Crossplane recreates it instantly.
3.  **Platform Building**: You can define "Golden Paths" (e.g., `kind: ProductionDB`) that hide complexity from developers.

---

## ⏭️ Next Steps

We are building platforms. Let's formalize this.

Proceed to **Module 27: Platform Engineering**.
