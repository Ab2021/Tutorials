# Lab 21.1: StatefulSets & Headless Services

## 🎯 Objective

Deploy a Database on Kubernetes. Unlike stateless apps (Deployments), databases care about their identity (`db-0` vs `db-1`) and their storage. You will deploy a **StatefulSet** with persistent storage.

## 📋 Prerequisites

-   Minikube running.

## 📚 Background

### Deployment vs StatefulSet
-   **Deployment**: Pods are interchangeable (`web-xc92j`). If one dies, a new random one replaces it.
-   **StatefulSet**: Pods have sticky identity (`web-0`, `web-1`). If `web-0` dies, it is recreated as `web-0` and reattached to the *same* disk.

### Headless Service
A Service with `ClusterIP: None`. It doesn't load balance. Instead, it returns the IPs of *all* pods. Used for peer discovery (e.g., "I am db-0, looking for db-1").

---

## 🔨 Hands-On Implementation

### Part 1: The Headless Service 🗣️

1.  **Create `nginx-headless.yaml`:**
    ```yaml
    apiVersion: v1
    kind: Service
    metadata:
      name: nginx
      labels:
        app: nginx
    spec:
      ports:
      - port: 80
        name: web
      clusterIP: None
      selector:
        app: nginx
    ```

### Part 2: The StatefulSet 🏛️

1.  **Create `web-statefulset.yaml`:**
    ```yaml
    apiVersion: apps/v1
    kind: StatefulSet
    metadata:
      name: web
    spec:
      selector:
        matchLabels:
          app: nginx
      serviceName: "nginx"
      replicas: 3
      template:
        metadata:
          labels:
            app: nginx
        spec:
          containers:
          - name: nginx
            image: k8s.gcr.io/nginx-slim:0.8
            ports:
            - containerPort: 80
              name: web
            volumeMounts:
            - name: www
              mountPath: /usr/share/nginx/html
      volumeClaimTemplates:
      - metadata:
          name: www
        spec:
          accessModes: [ "ReadWriteOnce" ]
          resources:
            requests:
              storage: 1Gi
    ```

2.  **Apply:**
    ```bash
    kubectl apply -f nginx-headless.yaml
    kubectl apply -f web-statefulset.yaml
    ```

### Part 3: Verify Identity 🆔

1.  **Watch Creation:**
    ```bash
    kubectl get pods -w
    ```
    *Observation:* They start sequentially. `web-0` -> `web-1` -> `web-2`. (Deployments start in parallel).

2.  **Check Hostnames:**
    ```bash
    kubectl exec web-0 -- sh -c 'hostname'
    ```
    *Result:* `web-0`.

3.  **Check DNS:**
    ```bash
    kubectl run -i --tty --image busybox:1.28 dns-test --restart=Never --rm
    nslookup nginx
    ```
    *Result:* Returns IPs of `web-0`, `web-1`, `web-2`.

### Part 4: Persistence Test 💾

1.  **Write Data to web-0:**
    ```bash
    kubectl exec web-0 -- sh -c 'echo "I am web-0" > /usr/share/nginx/html/index.html'
    ```

2.  **Delete web-0:**
    ```bash
    kubectl delete pod web-0
    ```

3.  **Wait for Resurrection:**
    `web-0` comes back.

4.  **Read Data:**
    ```bash
    kubectl exec web-0 -- cat /usr/share/nginx/html/index.html
    ```
    *Result:* "I am web-0". The data survived!

---

## 🎯 Challenges

### Challenge 1: Scaling Down (Difficulty: ⭐⭐)

**Task:**
Scale replicas to 2.
Check PVCs (`kubectl get pvc`).
*Observation:* The PVC for `web-2` is NOT deleted. This is safety. You must delete it manually if you want to save money.

### Challenge 2: Cassandra Ring (Difficulty: ⭐⭐⭐⭐)

**Task:**
Deploy a Cassandra cluster using StatefulSet.
Use the Headless Service for the nodes to discover each other and form a ring.
*Note:* This is a classic advanced K8s interview question.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
`kubectl scale statefulset web --replicas=2`
`kubectl get pvc` -> `www-web-2` still exists.
</details>

---

## 🔑 Key Takeaways

1.  **Ordered Deployment**: 0, then 1, then 2.
2.  **Ordered Termination**: 2, then 1, then 0.
3.  **Stable Network ID**: `web-0.nginx.default.svc.cluster.local`.

---

## ⏭️ Next Steps

We managed state. Now let's manage system tasks.

Proceed to **Lab 21.2: DaemonSets & Jobs**.
