# Lab 12.7: DaemonSets and Jobs

## Objective
Use DaemonSets for node-level services and Jobs for batch processing.

## Learning Objectives
- Create DaemonSets
- Run Jobs and CronJobs
- Understand use cases
- Manage job completion

---

## DaemonSet

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
      hostNetwork: true
      hostPID: true
      containers:
      - name: node-exporter
        image: prom/node-exporter:latest
        ports:
        - containerPort: 9100
```

## Job

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: data-migration
spec:
  completions: 1
  parallelism: 1
  backoffLimit: 3
  template:
    spec:
      containers:
      - name: migrate
        image: myapp:latest
        command: ["python", "migrate.py"]
      restartPolicy: Never
```

## CronJob

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: backup
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: backup:latest
            command: ["/backup.sh"]
          restartPolicy: OnFailure
```

## Success Criteria
✅ DaemonSet running on all nodes  
✅ Jobs completing successfully  
✅ CronJobs scheduled  

**Time:** 40 min
