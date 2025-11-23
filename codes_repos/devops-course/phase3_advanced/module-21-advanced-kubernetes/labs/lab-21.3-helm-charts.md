# Lab 21.3: Helm Charts

## Objective
Package and deploy applications using Helm.

## Learning Objectives
- Create Helm charts
- Use templates and values
- Deploy with Helm
- Manage releases

---

## Create Chart

```bash
helm create myapp

# Structure:
myapp/
├── Chart.yaml
├── values.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── ingress.yaml
```

## Chart.yaml

```yaml
apiVersion: v2
name: myapp
version: 1.0.0
appVersion: "1.0"
description: My application Helm chart
```

## values.yaml

```yaml
replicaCount: 3
image:
  repository: nginx
  tag: "1.21"
service:
  type: ClusterIP
  port: 80
```

## Template

```yaml
# templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Release.Name }}
spec:
  replicas: {{ .Values.replicaCount }}
  template:
    spec:
      containers:
      - name: {{ .Chart.Name }}
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
```

## Deploy

```bash
# Install
helm install myrelease ./myapp

# Upgrade
helm upgrade myrelease ./myapp

# Rollback
helm rollback myrelease 1

# Uninstall
helm uninstall myrelease
```

## Success Criteria
✅ Helm chart created  
✅ Application deployed  
✅ Values customized  
✅ Upgrades working  

**Time:** 45 min
