# Lab 13.2: Helm Charts

## 🎯 Objective

Stop Copy-Pasting YAML. **Helm** is the Package Manager for Kubernetes (like `apt` or `npm`). You will install an existing chart (Nginx) and create your own custom chart.

## 📋 Prerequisites

-   Minikube running.
-   Helm installed (`brew install helm` or `choco install kubernetes-helm`).

## 📚 Background

### What is a Chart?
A bundle of YAML templates.
-   `templates/deployment.yaml`: Contains `replicas: {{ .Values.replicaCount }}`.
-   `values.yaml`: Contains `replicaCount: 1`.
-   **Result**: Helm merges them to generate valid YAML.

---

## 🔨 Hands-On Implementation

### Part 1: Install a Chart (Bitnami) 📦

1.  **Add Repo:**
    ```bash
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo update
    ```

2.  **Search:**
    ```bash
    helm search repo nginx
    ```

3.  **Install:**
    ```bash
    helm install my-webserver bitnami/nginx
    ```
    *Result:* It creates Deployment, Service, ConfigMap, etc.

4.  **List:**
    ```bash
    helm list
    ```

5.  **Uninstall:**
    ```bash
    helm uninstall my-webserver
    ```

### Part 2: Create a Custom Chart 🎨

1.  **Create Scaffold:**
    ```bash
    helm create mychart
    ```
    *Structure:*
    -   `Chart.yaml`: Metadata.
    -   `values.yaml`: Default variables.
    -   `templates/`: The YAMLs with `{{ }}` logic.

2.  **Edit `values.yaml`:**
    Change `replicaCount: 1` to `replicaCount: 3`.
    Change `image.repository` to `nginx`.

3.  **Install Local Chart:**
    ```bash
    helm install my-custom-app ./mychart
    ```

4.  **Verify:**
    `kubectl get pods`. You should see 3 replicas.

### Part 3: Upgrade 🆙

1.  **Change Values:**
    Edit `values.yaml` -> `replicaCount: 5`.

2.  **Upgrade:**
    ```bash
    helm upgrade my-custom-app ./mychart
    ```

3.  **Verify:**
    `kubectl get pods`. Now 5 replicas.

4.  **Rollback:**
    ```bash
    helm rollback my-custom-app 1
    ```
    *Result:* Back to 3 replicas (Revision 1).

---

## 🎯 Challenges

### Challenge 1: Dry Run (Difficulty: ⭐)

**Task:**
Before installing, see what YAML will be generated.
`helm install --debug --dry-run test-release ./mychart`
*Goal:* Debug templating errors without messing up the cluster.

### Challenge 2: Templating Logic (Difficulty: ⭐⭐⭐)

**Task:**
In `templates/deployment.yaml`, add an `if` statement.
If `.Values.enableService` is true, create a Service. If false, don't.
*Hint:* `{{- if .Values.enableService }} ... {{- end }}`.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 2:**
Wrap the entire `service.yaml` content in:
```yaml
{{- if .Values.service.enabled }}
apiVersion: v1
kind: Service
...
{{- end }}
```
</details>

---

## 🔑 Key Takeaways

1.  **Reusability**: Write the YAML once, install it 100 times with different `values.yaml` (Dev, Staging, Prod).
2.  **Versioning**: Charts are versioned. You can rollback to `v1.2.0` easily.
3.  **Community**: Artifact Hub has thousands of ready-made charts (Prometheus, Grafana, MySQL).

---

## ⏭️ Next Steps

We have CI/CD and Package Management. Let's go deeper into Infrastructure.

Proceed to **Module 14: Advanced Terraform**.
