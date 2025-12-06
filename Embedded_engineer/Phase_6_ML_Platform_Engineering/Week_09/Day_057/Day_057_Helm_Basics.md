# Day 57: The Package Manager for Kubernetes: Helm
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 9: Helm, Operators & GitOps

---

> **🎯 Focus Area:** Copy-pasting 500 lines of YAML to change one Image Tag is unsustainable. Master **Helm** to template, package, and version your Kubernetes applications.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the relationship between Chart, Values, and Release.
2.  **Create** a basic Helm Chart from scratch using `helm create`.
3.  **Inject** variables into YAML using Go Template syntax (`{{ .Values.x }}`).
4.  **Install, Upgrade, and Rollback** a release.
5.  **Use** public repositories to install complex apps (e.g., Prometheus) in one command.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Minikube/K8s Cluster.

### Software Environment
- `helm` (binary installed).

---

## 📖 Theoretical Foundation

### 1. The YAML Hell
In Week 7, we wrote `backend-deployment.yaml` and `backend-service.yaml`.
If we want a Staging environment (different URL, fewer replicas), we duplicate the files. Now we have 2 sources of truth. If we fix a bug in Prod, we forget Staging.

### 2. The Helm Architecture
*   **Chart:** A directory containing Templates (`templates/*.yaml`) and Defaults (`values.yaml`).
*   **Values:** A user-supplied YAML file that overlays the defaults.
*   **Engine:** Merges *Templates* + *Values* to generate *Manifests*.
*   **Release:** The state tracking of "Chart X installed as Name Y".

### 3. Folder Structure
```text
my-chart/
├── Chart.yaml          # Metadata (Name, Version)
├── values.yaml         # Default variables
├── charts/             # Dependencies
└── templates/          # The Logic
    ├── deployment.yaml
    └── service.yaml
```

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Your First Chart

1.  **Scaffold:**
    ```bash
    helm create ml-model
    ```
    This creates a robust default chart (Nginx based). Let's clean it up.

2.  **Edit `values.yaml`:**
    ```yaml
    replicaCount: 1
    image:
      repository: python
      tag: "3.9-slim"
    service:
      type: ClusterIP
      port: 80
    model:
      version: "v1"
    ```

3.  **Edit `templates/deployment.yaml`:**
    ```yaml
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: {{ include "ml-model.fullname" . }}
      labels:
        {{- include "ml-model.labels" . | nindent 4 }}
    spec:
      replicas: {{ .Values.replicaCount }}
      selector:
        matchLabels:
          {{- include "ml-model.selectorLabels" . | nindent 6 }}
      template:
        metadata:
          labels:
            {{- include "ml-model.selectorLabels" . | nindent 8 }}
        spec:
          containers:
            - name: {{ .Chart.Name }}
              image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
              env:
                - name: MODEL_VERSION
                  value: {{ .Values.model.version | quote }}
    ```

### 👨‍💻 Lab Operations: Install & Upgrade

1.  **Dry Run (Debug):**
    See what YAML it *would* generate without installing.
    ```bash
    helm install dry-run-test ./ml-model --dry-run --debug
    ```

2.  **Install (Release 1):**
    ```bash
    helm install my-inference ./ml-model
    ```
    *Result: "NAME: my-inference, REVISION: 1"*

3.  **Upgrade (Change Config):**
    We want 3 replicas now.
    ```bash
    helm upgrade my-inference ./ml-model --set replicaCount=3
    ```
    *Result: "REVISION: 2"*

4.  **Rollback (Oops!):**
    ```bash
    helm rollback my-inference 1
    ```
    *Result: Back to 1 replica.*

### 👨‍💻 Installing Public Charts

The power of Helm is the ecosystem. Install a Redis cluster in seconds.

```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install my-redis bitnami/redis --set architecture=standalone
```

---

## 🔬 Lab Exercise: "The Dependent Chart"

### Task
Make your `ml-model` chart rely on Redis.
1.  Edit `ml-model/Chart.yaml`:
    ```yaml
    dependencies:
      - name: redis
        version: 17.3.x
        repository: https://charts.bitnami.com/bitnami
    ```
2.  Run `helm dependency build ./ml-model`.
    *   It downloads Redis into `charts/`.
3.  Install your chart.
    *   Helm installs *both* your Deployment and the Redis sub-chart automatically.

### Insight
This allows you to package an entire "Platform" (Model + DB + Prometheus + Ingress) as a single installable unit.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Values Protocol:** Always expose things that change (Replicas, Image Tag, Resources) in `values.yaml`. Never hardcode `cpu: 100m` in the template if it might differ in Prod.
2.  **Functions:** Go Templates are powerful. You can use loops `{{ range }}`, conditionals `{{ if }}`, and pipes `{{ . | b64enc }}`.
3.  **Lifecycle:** Helm tracks revisions using Secrets (`sh.helm.release.v1...`). Don't delete these secrets manually or Helm breaks.

### API Summary
```bash
helm create <name>
helm install <release> <chart>
helm upgrade <release> <chart>
helm list
helm uninstall <release>
```

---

**Day 57 Complete** ✅

*Next: Day 58 - Helm Advanced Features - Hooks, Testing, and publishing your chart to a repo.*
