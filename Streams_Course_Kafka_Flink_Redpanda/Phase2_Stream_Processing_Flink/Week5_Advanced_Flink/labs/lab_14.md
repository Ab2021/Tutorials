# Lab 14: K8s Application Mode

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Build a custom Docker image.
-   Deploy in Application Mode.

## Problem Statement
1.  Create Dockerfile: `FROM flink`, `COPY my-job.jar`.
2.  Build image.
3.  Deploy using `flink run-application -t kubernetes-application ...`.

## Starter Code
```bash
flink run-application -t kubernetes-application   -Dkubernetes.cluster-id=my-app   -Dkubernetes.container.image=my-image:latest   local:///opt/flink/usrlib/my-job.jar
```

## Hints
<details>
<summary>Hint 1</summary>
The JAR path must be local to the container (`local:///...`).
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Dockerfile
```dockerfile
FROM flink:1.18
RUN mkdir -p /opt/flink/usrlib
COPY target/my-job.jar /opt/flink/usrlib/my-job.jar
```

### Command
```bash
flink run-application     --target kubernetes-application     -Dkubernetes.cluster-id=my-first-app-cluster     -Dkubernetes.container.image=my-custom-flink-image     local:///opt/flink/usrlib/my-job.jar
```
</details>
