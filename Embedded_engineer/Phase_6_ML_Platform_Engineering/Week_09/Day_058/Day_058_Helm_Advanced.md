# Day 58: Beyond `helm install`: Hooks and Repos
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 9: Helm, Operators & GitOps

---

> **🎯 Focus Area:** Deployment is rarely just "Start Pod". You often need to "Migrate DB -> Start Pod -> Verify Health". Master **Helm Hooks** and **Test** frameworks to orchestrate complex lifecycles.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Implement** a `pre-install` hook to run a setup job (e.g., Download Weights) before the main app starts.
2.  **Write** a `helm test` pod to validate the deployment automatically.
3.  **Package** a chart into a `.tgz` archive.
4.  **Host** a private Helm Repository using a static web server (GitHub Pages/S3).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Minikube/K8s Cluster.

### Software Environment
- `helm`.

---

## 📖 Theoretical Foundation

### 1. The Deployment Lifecycle
Standard K8s applies everything at once. Helm allows staging:
1.  **Pre-Install:** Run Jobs (DB Migration). Block until success.
2.  **Install:** Apply core Deployments/Services.
3.  **Post-Install:** Run Jobs (Notification to Slack).

### 2. Hooks Implementation
A Hook is just a normal K8s resource (Job/Pod) with a special Annotation.
`"helm.sh/hook": pre-install`

### 3. Chart Repositories
To share charts, you zip them (`chart-v1.tgz`) and generate an index (`index.yaml`).
You can host this on any HTTP server.
`index.yaml` maps `chart-name` + `version` -> `download-url`.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: The Migration Hook

We want a Job that prints "Migrating Database..." before our App starts.

#### 📁 `ml-model/templates/db-migration-hook.yaml`
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: {{ include "ml-model.fullname" . }}-migration
  annotations:
    # The Magic Annotation
    "helm.sh/hook": pre-install,pre-upgrade
    "helm.sh/hook-weight": "-5" # Run early
    "helm.sh/hook-delete-policy": hook-succeeded # Cleanup after success
spec:
  template:
    spec:
      containers:
      - name: migration-tool
        image: busybox
        command: ["/bin/sh", "-c"]
        args: ["echo Initializing DB Schema...; sleep 5; echo Done"]
      restartPolicy: Never
```

### 👨‍💻 Core Implementation: The Test

A Pod that confirms the Service is creating 200 OK responses.

#### 📁 `ml-model/templates/tests/test-connection.yaml`
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: {{ include "ml-model.fullname" . }}-test-connection
  labels:
    {{- include "ml-model.labels" . | nindent 4 }}
  annotations:
    "helm.sh/hook": test
spec:
  containers:
    - name: wget
      image: busybox
      command: ['wget']
      args: ['{{ include "ml-model.fullname" . }}:{{ .Values.service.port }}']
  restartPolicy: Never
```

### 👨‍💻 Lab Operations

1.  **Install with Hook:**
    ```bash
    helm install release-v2 ./ml-model
    ```
    *Observation:* The CLI hangs at "STATUS: deployed".
    *   In background: K8s runs the Job.
    *   Once Job succeeds, Helm applies Deployment.

2.  **Run Tests:**
    ```bash
    helm test release-v2
    ```
    *Result:* Launches the test pod. If exit code 0, prints "PASSED". Useful for CI/CD pipelines.

3.  **Package & Distribute:**
    ```bash
    helm package ./ml-model
    # Creates ml-model-0.1.0.tgz
    
    # Create Index
    helm repo index . 
    # Creates index.yaml
    ```
    
    *To Host:* Upload `index.yaml` and `.tgz` to a Github Repo `/docs` folder, enable Pages, and then:
    `helm repo add my-repo https://username.github.io/repo`

---

## 🔬 Lab Exercise: "Weight Download Barrier"

### Task
Simulate an ML use case.
1.  Create a Hook Job that "downloads" a 10GB model (simulated with `sleep 10`).
2.  Set `wait: true` in your Helm install command.
3.  **Observation:** Helm prevents the main Inference Pods from even starting until the "Download" hook is verified complete.
4.  Why? Prevents CrashLoopBackOff on inference pods if the model volume isn't populated yet.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Hooks logic is simpler than InitContainers:** InitContainers run on *every pod* (good for per-pod setup). Hooks run *once per release* (good for global state setup like DBs).
2.  **Cleanup:** Use `hook-delete-policy` to remove failed hook jobs, otherwise `helm install` fails next time saying "Job already exists".
3.  **CI Integration:** `helm lint`, `helm install --wait`, and `helm test` is the holy trinity of Chart CI.

### API Summary
```bash
helm package <dir>
helm repo index <dir>
helm test <release>
```

---

**Day 58 Complete** ✅

*Next: Day 59 - Kubernetes Operators - When Helm isn't enough, we need code that runs in the cluster.*
