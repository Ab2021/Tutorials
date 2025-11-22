# Lab 25.2: LitmusChaos

## 🎯 Objective

Chaos as a Service. **LitmusChaos** provides a centralized Chaos Center to manage experiments across multiple clusters. It uses a "Hub" of pre-defined experiments.

## 📋 Prerequisites

-   Minikube running.
-   Helm installed.

## 📚 Background

### Architecture
-   **Chaos Center**: The UI/Server.
-   **Chaos Agent**: Runs on the target cluster.
-   **Chaos Hub**: Marketplace of experiments (like Docker Hub).

---

## 🔨 Hands-On Implementation

### Part 1: Install Litmus 🧪

1.  **Install:**
    ```bash
    helm repo add litmuschaos https://litmuschaos.github.io/litmus-helm/
    helm install litmus litmuschaos/litmus --namespace litmus --create-namespace
    ```

2.  **Access UI:**
    `kubectl port-forward svc/litmus-frontend-service -n litmus 9091:9091`.
    Open `http://localhost:9091`.
    Default: `admin` / `litmus`.

### Part 2: Connect Agent 🔌

1.  **Self-Agent:**
    In the UI, go to **Targets**.
    You should see the "Self-Agent" (the cluster Litmus is running on) is already connected (or follow instructions to connect it).

### Part 3: Run a Workflow 🌊

1.  **Create Workflow:**
    -   Click **Schedule a Workflow**.
    -   Select **Self-Agent**.
    -   Choose **ChaosHub**.
    -   Search for `pod-delete`.
    -   Select the experiment.

2.  **Configure:**
    -   Target Application: `nginx` (namespace `default`).
    -   Tune: `TOTAL_CHAOS_DURATION=30`, `CHAOS_INTERVAL=10`.

3.  **Run:**
    Click **Finish**.

### Part 4: Observe 👁️

1.  **Visualizer:**
    Watch the graph in the UI.
    Green -> Running -> Pass/Fail.

2.  **Kubectl:**
    `kubectl get pods`. You see the chaos runner pod starting, then killing the nginx pod.

---

## 🎯 Challenges

### Challenge 1: Probes (Difficulty: ⭐⭐⭐)

**Task:**
Add a **Probe** to the workflow.
"Check URL `http://nginx` returns 200".
If the probe fails, the experiment is marked as "Failed".
*Goal:* Automated hypothesis testing.

### Challenge 2: GitOps (Difficulty: ⭐⭐)

**Task:**
Export the Workflow as YAML.
Commit it to Git.
Use ArgoCD to trigger the Chaos Workflow.
*Concept:* Continuous Chaos.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
In the UI, Step 4 (Probes), add an HTTP Probe.
</details>

---

## 🔑 Key Takeaways

1.  **Marketplace**: LitmusHub has experiments for AWS, Azure, Kafka, Cassandra, etc.
2.  **Resilience Score**: Litmus calculates a score (0-100%) based on how many experiments passed.
3.  **Game Days**: Use the UI to run "Game Days" where the team gathers to watch the system break.

---

## ⏭️ Next Steps

We broke one cloud. Let's use multiple clouds.

Proceed to **Module 26: Multi-Cloud & Hybrid**.
