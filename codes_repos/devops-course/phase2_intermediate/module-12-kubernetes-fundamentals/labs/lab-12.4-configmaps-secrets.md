# Lab 12.4: ConfigMaps and Secrets

## Objective
Manage application configuration and sensitive data in Kubernetes.

## Learning Objectives
- Create and use ConfigMaps
- Manage Secrets securely
- Mount configs as volumes
- Use environment variables from configs

---

## ConfigMaps

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  app.properties: |
    database.host=db.example.com
    database.port=5432
  log.level: "INFO"
  max.connections: "100"
```

```bash
# Create from file
kubectl create configmap app-config --from-file=app.properties

# Create from literal
kubectl create configmap app-config \
  --from-literal=log.level=INFO \
  --from-literal=max.connections=100
```

## Using ConfigMaps

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: myapp
spec:
  containers:
  - name: app
    image: myapp:latest
    env:
    - name: LOG_LEVEL
      valueFrom:
        configMapKeyRef:
          name: app-config
          key: log.level
    volumeMounts:
    - name: config
      mountPath: /etc/config
  volumes:
  - name: config
    configMap:
      name: app-config
```

## Secrets

```yaml
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: db-secret
type: Opaque
data:
  username: YWRtaW4=  # base64 encoded
  password: cGFzc3dvcmQ=
```

```bash
# Create from literal
kubectl create secret generic db-secret \
  --from-literal=username=admin \
  --from-literal=password=secret123
```

## Using Secrets

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: myapp
spec:
  containers:
  - name: app
    image: myapp:latest
    env:
    - name: DB_USERNAME
      valueFrom:
        secretKeyRef:
          name: db-secret
          key: username
    - name: DB_PASSWORD
      valueFrom:
        secretKeyRef:
          name: db-secret
          key: password
```

## Success Criteria
✅ ConfigMaps created and used  
✅ Secrets managed securely  
✅ Configs mounted as volumes  
✅ Env vars from configs working  

**Time:** 40 min
