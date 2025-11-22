# Lab 12.4: Kubernetes Capstone Project

## 🎯 Objective

Deploy a **Guestbook Application**. This is the classic Kubernetes example. It consists of a **Redis Master** (Write), **Redis Slaves** (Read), and a **PHP Frontend**. You will write YAML for all of them.

## 📋 Prerequisites

-   Completed Module 12.
-   Minikube running.

## 📚 Background

### Architecture
1.  **Redis Master**: Deployment (1 replica) + Service.
2.  **Redis Slave**: Deployment (2 replicas) + Service.
3.  **Frontend**: Deployment (3 replicas) + Service (LoadBalancer).

---

## 🔨 Hands-On Implementation

### Step 1: Redis Master 🔴

1.  **`redis-master.yaml`:**
    ```yaml
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: redis-master
    spec:
      replicas: 1
      selector:
        matchLabels:
          app: redis
          role: master
      template:
        metadata:
          labels:
            app: redis
            role: master
        spec:
          containers:
          - name: master
            image: k8s.gcr.io/redis:e2e
            ports:
            - containerPort: 6379
    ---
    apiVersion: v1
    kind: Service
    metadata:
      name: redis-master
    spec:
      ports:
      - port: 6379
        targetPort: 6379
      selector:
        app: redis
        role: master
    ```
    *Note:* We used `---` to put Deployment and Service in one file.

### Step 2: Redis Slave 🔵

1.  **`redis-slave.yaml`:**
    ```yaml
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: redis-slave
    spec:
      replicas: 2
      selector:
        matchLabels:
          app: redis
          role: slave
      template:
        metadata:
          labels:
            app: redis
            role: slave
        spec:
          containers:
          - name: slave
            image: gcr.io/google_samples/gb-redisslave:v3
            env:
            - name: GET_HOSTS_FROM
              value: dns
            ports:
            - containerPort: 6379
    ---
    apiVersion: v1
    kind: Service
    metadata:
      name: redis-slave
    spec:
      ports:
      - port: 6379
      selector:
        app: redis
        role: slave
    ```

### Step 3: Frontend (PHP) 🟢

1.  **`frontend.yaml`:**
    ```yaml
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: frontend
    spec:
      replicas: 3
      selector:
        matchLabels:
          app: guestbook
          tier: frontend
      template:
        metadata:
          labels:
            app: guestbook
            tier: frontend
        spec:
          containers:
          - name: php-redis
            image: gcr.io/google-samples/gb-frontend:v4
            ports:
            - containerPort: 80
    ---
    apiVersion: v1
    kind: Service
    metadata:
      name: frontend
    spec:
      type: LoadBalancer
      ports:
      - port: 80
      selector:
        app: guestbook
        tier: frontend
    ```

### Step 4: Deploy & Verify 🚀

1.  **Apply All:**
    ```bash
    kubectl apply -f redis-master.yaml
    kubectl apply -f redis-slave.yaml
    kubectl apply -f frontend.yaml
    ```

2.  **Check Status:**
    ```bash
    kubectl get pods
    kubectl get svc
    ```
    Wait until all pods are `Running`.

3.  **Access:**
    Run `minikube tunnel` (if not running).
    Get External IP of `frontend`.
    Open in browser.
    Type a message "Hello K8s". Click Submit.
    *Result:* It persists!

---

## 🎯 Challenges

### Challenge 1: Clean Up (Difficulty: ⭐)

**Task:**
Delete everything using labels.
`kubectl delete all -l app=redis`
`kubectl delete all -l app=guestbook`

### Challenge 2: Debugging (Difficulty: ⭐⭐⭐)

**Task:**
Intentionally break the `redis-master` service name in `redis-slave.yaml` (if configurable) or change the label selector in the Service.
Observe how the Frontend fails to write data (but might still read).
Fix it.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
The `-l` flag selects resources by label. It's a fast way to clean up specific apps without deleting the whole cluster.
</details>

---

## 🔑 Key Takeaways

1.  **Microservices**: We deployed 3 tiers (Master, Slave, Frontend) that talk to each other via Services.
2.  **Environment Variables**: The Slaves find the Master via DNS (or Env Vars).
3.  **Stateless vs Stateful**: The Frontend is stateless. Redis is stateful (handling state in K8s is harder - see StatefulSets in Module 19).

---

## ⏭️ Next Steps

**Congratulations!** You have deployed a real app on Kubernetes.

Proceed to **Module 13: Advanced CI/CD (GitOps)**.
