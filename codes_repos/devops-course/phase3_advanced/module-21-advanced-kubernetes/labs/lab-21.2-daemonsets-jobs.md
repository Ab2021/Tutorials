# Lab 21.2: DaemonSets, Jobs, & CronJobs

## 🎯 Objective

Run special workloads.
-   **DaemonSet**: Run 1 pod on *every* node (e.g., Logs, Monitoring).
-   **Job**: Run to completion (e.g., Database Migration).
-   **CronJob**: Run on a schedule (e.g., Backups).

## 📋 Prerequisites

-   Minikube running.

## 📚 Background

### Use Cases
-   **DaemonSet**: Fluentd, Node Exporter, CNI Plugins.
-   **Job**: Video rendering, Batch processing.
-   **CronJob**: Daily reports, Cert renewal.

---

## 🔨 Hands-On Implementation

### Part 1: DaemonSet (Node Exporter) 😈

1.  **Create `node-exporter.yaml`:**
    ```yaml
    apiVersion: apps/v1
    kind: DaemonSet
    metadata:
      name: node-exporter
    spec:
      selector:
        matchLabels:
          app: node-exporter
      template:
        metadata:
          labels:
            app: node-exporter
        spec:
          containers:
          - name: node-exporter
            image: prom/node-exporter
            ports:
            - containerPort: 9100
              hostPort: 9100
    ```

2.  **Apply:**
    ```bash
    kubectl apply -f node-exporter.yaml
    ```

3.  **Verify:**
    If you are on Minikube (1 node), you see 1 pod.
    If you add a node (`minikube node add`), a new pod automatically starts on it.

### Part 2: Job (Pi Calculation) 🧮

1.  **Create `pi-job.yaml`:**
    ```yaml
    apiVersion: batch/v1
    kind: Job
    metadata:
      name: pi
    spec:
      template:
        spec:
          containers:
          - name: pi
            image: perl
            command: ["perl",  "-Mbignum=bpi", "-wle", "print bpi(2000)"]
          restartPolicy: Never
      backoffLimit: 4
    ```

2.  **Apply:**
    ```bash
    kubectl apply -f pi-job.yaml
    ```

3.  **Verify:**
    `kubectl get jobs`.
    Wait for `COMPLETIONS 1/1`.
    Check logs: `kubectl logs job/pi`.
    *Result:* 3.1415...

### Part 3: CronJob (The Clock) ⏰

1.  **Create `cronjob.yaml`:**
    ```yaml
    apiVersion: batch/v1
    kind: CronJob
    metadata:
      name: hello
    spec:
      schedule: "*/1 * * * *" # Every minute
      jobTemplate:
        spec:
          template:
            spec:
              containers:
              - name: hello
                image: busybox
                imagePullPolicy: IfNotPresent
                command:
                - /bin/sh
                - -c
                - date; echo Hello from the Kubernetes cluster
              restartPolicy: OnFailure
    ```

2.  **Apply:**
    ```bash
    kubectl apply -f cronjob.yaml
    ```

3.  **Verify:**
    Wait 1 minute.
    `kubectl get jobs`. You see `hello-xxxxx`.
    `kubectl get pods`.
    `kubectl logs <pod>`.

---

## 🎯 Challenges

### Challenge 1: Parallel Jobs (Difficulty: ⭐⭐)

**Task:**
Modify the Pi Job to run 5 times in parallel.
`completions: 5`
`parallelism: 5`
*Result:* 5 pods start at once.

### Challenge 2: DaemonSet Update Strategy (Difficulty: ⭐⭐⭐)

**Task:**
By default, DaemonSets use `RollingUpdate`.
Change it to `OnDelete`.
Update the image.
*Observation:* The pod does NOT update until you manually delete it. This is useful for critical node components where you want manual control.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```yaml
spec:
  completions: 5
  parallelism: 5
```
</details>

---

## 🔑 Key Takeaways

1.  **DaemonSets**: Essential for Cluster Admins (Logs/Metrics).
2.  **Jobs**: Great for "Run once" tasks. Use `initContainers` in Deployments for migrations instead of Jobs if possible (simpler).
3.  **CronJobs**: Replaces the old crontab server. Distributed and resilient.

---

## ⏭️ Next Steps

We have mastered workloads. Now let's master traffic.

Proceed to **Module 22: Service Mesh (Istio)**.
