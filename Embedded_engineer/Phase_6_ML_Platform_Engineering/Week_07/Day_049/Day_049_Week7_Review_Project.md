# Day 49: Week 7 Review & Project - The ML-Ops Base Layer
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 7: Kubernetes Fundamentals

---

> **🎯 Focus Area:** Capstone Project. Synthesize all Kubernetes primitives (Deployments, Services, Ingress, PVC, Secrets, RBAC) to deploy a generic **3-Tier ML Application** stack locally.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Architect** a multi-service application (Frontend -> Backend -> DB) in K8s.
2.  **Write** a monolithic YAML (or set of YAMLs) deploying the entire stack.
3.  **Debug** inter-service communication issues using DNS and Environment variables.
4.  **Demonstrate** persistence by killing the DB and verifying data survival.

---

## 📚 Week 7 Recap

| Day | Topic | The "Ah-ha" Moment |
|-----|-------|---------------------|
| 43 | Architecture | "K8s is not a tool; it's an Operating System." |
| 44 | Workloads | "Pods are cattle. Deployments are the shepherds." |
| 45 | Networking | "I don't need to know IPs. I just call `http://backend`." |
| 46 | Storage | "State needs a PVC, or it vanishes." |
| 47 | Config | "Build once, deploy anywhere (Dev/Prod) just by swapping the ConfigMap." |
| 48 | RBAC | "Don't run as Root. Don't give Admin rights to a script." |

---

## 🏗️ Final Project: "ModelStore" Platform

### Architecture
1.  **Redis (Data Tier):** Stores model metadata. Needs Persistence (PVC) and Authenticated access (Secret).
2.  **API (Backend Tier):** Python FastAPI. Reads from Redis. Internal Service only.
3.  **Dashboard (Frontend Tier):** Streamlit UI. Talks to API. Exposed to User via Ingress.

### File Structure
```
project/
├── 00-namespace.yaml
├── 01-redis.yaml       (StatefulSet + PVC + Secret)
├── 02-backend.yaml     (Deployment + Service + CM)
├── 03-frontend.yaml    (Deployment + Service + Ingress)
└── 04-rbac.yaml        (Minimal permissions for Backend)
```

---

### Step 1: Namespace & Secrets

#### 📁 `project/00-common.yaml`
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: modelstore
---
apiVersion: v1
kind: Secret
metadata:
  name: redis-creds
  namespace: modelstore
stringData:
  password: "SecureRedisPassword123"
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: modelstore
data:
  REDIS_HOST: "redis-svc"
  BACKEND_URL: "http://backend-svc:80"
```

### Step 2: The Data Tier (Redis)

#### 📁 `project/01-redis.yaml`
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis
  namespace: modelstore
spec:
  serviceName: "redis-svc"
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7.0-alpine
        command: ["redis-server", "--requirepass", "$(REDIS_PASS)", "--appendonly", "yes"]
        env:
          - name: REDIS_PASS
            valueFrom:
              secretKeyRef:
                name: redis-creds
                key: password
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: redis-data
          mountPath: /data
  volumeClaimTemplates:
  - metadata:
      name: redis-data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 1Gi
---
apiVersion: v1
kind: Service
metadata:
  name: redis-svc
  namespace: modelstore
spec:
  ports:
  - port: 6379
    name: redis
  clusterIP: None # Headless Service for StatefulSet
  selector:
    app: redis
```

### Step 3: The Backend (Mock API)

#### 📁 `project/02-backend.yaml`
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
  namespace: modelstore
spec:
  replicas: 2
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
      - name: api
        image: python:3.9-slim
        command: ["/bin/sh", "-c"]
        # Simulate an API that checks Redis connection
        args:
          - |
            pip install redis flask &&
            python -c '
            from flask import Flask
            import redis, os
            app = Flask(__name__)
            r = redis.Redis(host=os.environ["REDIS_HOST"], port=6379, password=os.environ["REDIS_PASS"])
            
            @app.route("/")
            def health():
                try:
                    r.ping()
                    return "Backend OK. Redis Connected."
                except Exception as e:
                    return str(e), 500
            
            app.run(host="0.0.0.0", port=8080)
            '
        env:
          - name: REDIS_HOST
            valueFrom: { configMapKeyRef: { name: app-config, key: REDIS_HOST } }
          - name: REDIS_PASS
            valueFrom: { secretKeyRef: { name: redis-creds, key: password } }
        ports:
        - containerPort: 8080
---
apiVersion: v1
kind: Service
metadata:
  name: backend-svc
  namespace: modelstore
spec:
  selector:
    app: backend
  ports:
    - port: 80
      targetPort: 8080
```

### Step 4: The Frontend (Mock Dashboard)

#### 📁 `project/03-frontend.yaml`
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
  namespace: modelstore
spec:
  replicas: 1
  selector:
    matchLabels:
      app: frontend
  template:
    metadata:
      labels:
        app: frontend
    spec:
      containers:
      - name: web
        image: python:3.9-slim
        command: ["/bin/sh", "-c"]
        # Simulate UI calling backend
        args:
          - |
            pip install requests flask &&
            python -c '
            from flask import Flask
            import requests, os
            app = Flask(__name__)
            
            @app.route("/")
            def index():
                url = os.environ["BACKEND_URL"]
                try:
                    res = requests.get(url, timeout=2).text
                    return f"<h1>Frontend</h1><p>Backend says: {res}</p>"
                except Exception as e:
                    return f"Error contacting backend: {e}"
            
            app.run(host="0.0.0.0", port=80)
            '
        env:
          - name: BACKEND_URL
            valueFrom: { configMapKeyRef: { name: app-config, key: BACKEND_URL } }
        ports:
        - containerPort: 80
---
apiVersion: v1
kind: Service
metadata:
  name: frontend-svc
  namespace: modelstore
spec:
  selector:
    app: frontend
  ports:
    - port: 80
---
# Ingress (Requires Minikube Ingress Addon)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: frontend-ingress
  namespace: modelstore
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: modelstore.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend-svc
            port:
              number: 80
```

### Step 5: Verification Script

#### 📁 `verify_deployment.sh`
```bash
#!/bin/bash
# 1. Apply All
kubectl apply -f project/

# 2. Wait
echo "Waiting for Pods..."
kubectl wait --for=condition=Ready pods --all -n modelstore --timeout=120s

# 3. Port Forward (If Ingress is tricky on your local setup)
echo "Port Forwarding: http://localhost:8880"
kubectl port-forward svc/frontend-svc 8880:80 -n modelstore &
PF_PID=$!

sleep 5
# 4. Curl
echo "Checking Deployment..."
curl http://localhost:8880

# 5. Cleanup
kill $PF_PID
# kubectl delete -f project/
```

---

## 📝 Success Criteria
If run correctly, `curl` returns:
```html
<h1>Frontend</h1><p>Backend says: Backend OK. Redis Connected.</p>
```
This confirms:
1.  Frontend -> Backend Service DNS resolution works.
2.  Backend -> Redis Service DNS resolution works.
3.  Redis Password authentication works.
4.  Pods are schedulable.

---

**Week 7 Complete** ✅

*Next Week: Week 8 - Kubernetes Scheduling & GPUs - Getting access to the Tensor Cores inside your cluster.*
