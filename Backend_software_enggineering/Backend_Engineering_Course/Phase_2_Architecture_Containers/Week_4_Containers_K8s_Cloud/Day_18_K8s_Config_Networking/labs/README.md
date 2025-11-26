# Lab: Day 18 - Configs & Secrets

## Goal
Inject configuration and secrets into a Pod.

## Directory Structure
```
day18/
├── configmap.yaml
├── secret.yaml
├── pod.yaml
└── README.md
```

## Step 1: ConfigMap (`configmap.yaml`)

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  GREETING: "Hello from K8s Config!"
  LOG_LEVEL: "DEBUG"
```

## Step 2: Secret (`secret.yaml`)

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: app-secret
type: Opaque
data:
  # "password123" base64 encoded
  DB_PASSWORD: cGFzc3dvcmQxMjM=
```

## Step 3: The Pod (`pod.yaml`)

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: config-test-pod
spec:
  containers:
  - name: test-container
    image: busybox
    command: ["sh", "-c", "echo $GREETING && echo My Password is $DB_PASSWORD && sleep 3600"]
    env:
      # Load from ConfigMap
      - name: GREETING
        valueFrom:
          configMapKeyRef:
            name: app-config
            key: GREETING
      # Load from Secret
      - name: DB_PASSWORD
        valueFrom:
          secretKeyRef:
            name: app-secret
            key: DB_PASSWORD
```

## Step 4: Apply & Verify

1.  **Apply**:
    ```bash
    kubectl apply -f configmap.yaml
    kubectl apply -f secret.yaml
    kubectl apply -f pod.yaml
    ```

2.  **Verify**:
    ```bash
    kubectl logs config-test-pod
    ```
    *Output*:
    ```
    Hello from K8s Config!
    My Password is password123
    ```

## Challenge
Modify `pod.yaml` to mount the ConfigMap as a volume at `/etc/config`.
Exec into the pod (`kubectl exec -it config-test-pod -- sh`) and `cat /etc/config/GREETING`.
