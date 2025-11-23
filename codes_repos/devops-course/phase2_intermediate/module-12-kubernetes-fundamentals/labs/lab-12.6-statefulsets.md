# Lab 12.6: StatefulSets

## Objective
Deploy stateful applications with StatefulSets.

## Learning Objectives
- Create StatefulSets
- Understand stable network identities
- Use volumeClaimTemplates
- Manage ordered deployment/scaling

---

## StatefulSet

```yaml
apiVersion: v1
kind: Service
metadata:
  name: mysql
spec:
  clusterIP: None  # Headless service
  selector:
    app: mysql
  ports:
  - port: 3306
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mysql
spec:
  serviceName: mysql
  replicas: 3
  selector:
    matchLabels:
      app: mysql
  template:
    metadata:
      labels:
        app: mysql
    spec:
      containers:
      - name: mysql
        image: mysql:8.0
        env:
        - name: MYSQL_ROOT_PASSWORD
          value: "password"
        ports:
        - containerPort: 3306
        volumeMounts:
        - name: data
          mountPath: /var/lib/mysql
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
```

## Stable Network Identity

```bash
# Pods get predictable names
mysql-0
mysql-1
mysql-2

# DNS entries
mysql-0.mysql.default.svc.cluster.local
mysql-1.mysql.default.svc.cluster.local
```

## Ordered Operations

```bash
# Scale up (sequential)
kubectl scale statefulset mysql --replicas=5
# Creates: mysql-3, then mysql-4

# Scale down (reverse order)
kubectl scale statefulset mysql --replicas=2
# Deletes: mysql-4, then mysql-3
```

## Update Strategy

```yaml
spec:
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      partition: 2  # Only update pods >= 2
```

## Success Criteria
✅ StatefulSet deployed  
✅ Stable network identities  
✅ Persistent storage per pod  
✅ Ordered scaling working  

**Time:** 45 min
