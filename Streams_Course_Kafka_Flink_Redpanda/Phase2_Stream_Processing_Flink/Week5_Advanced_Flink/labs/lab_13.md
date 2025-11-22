# Lab 13: K8s Deployment (Session)

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Deploy Flink Session Cluster on K8s.

## Problem Statement
Use `kubectl` to deploy a JobManager and TaskManager. Submit a job to it.

## Starter Code
```yaml
# jobmanager-deployment.yaml
# taskmanager-deployment.yaml
```

## Hints
<details>
<summary>Hint 1</summary>
Use the official Flink K8s docs YAMLs.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```bash
kubectl create -f jobmanager-service.yaml
kubectl create -f jobmanager-deployment.yaml
kubectl create -f taskmanager-deployment.yaml

# Forward port
kubectl port-forward service/flink-jobmanager 8081:8081

# Submit
flink run -m localhost:8081 examples/streaming/WordCount.jar
```
</details>
