# Day 164: Elasticity: Autoscaling Strategies with KEDA
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 24: Cost Optimization & FinOps

---

> **🎯 Focus Area:** Standard Kubernetes HPA scales on CPU. But an Inference Service might have low CPU usage even when the GPU is overloaded or the Request Queue is 10,000 items deep. **KEDA (Kubernetes Event-driven Autoscaling)** solves this.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** between HPA (Resource-based) and KEDA (Event-based) scaling.
2.  **Deploy** a ScaledObject that scales pods based on AWS SQS Queue depth.
3.  **Implement** "Scale-to-Zero" for idle services to save costs.
4.  **Tune** Hysteresis (cooldown periods) to prevent "Thrashing" (Rapid scale up/down).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine with `kubectl`.

### Software Environment
- KEDA installed on Cluster.

---

## 📖 Theoretical Foundation

### 1. The Lag Problem
*   **CPU Autoscaling:** Users hit API -> CPU spikes 3 mins later -> Pod scales. (Too slow).
*   **Metrics Autoscaling (Prometheus):** Users hit API -> Generic Metric spikes -> Pod scales. (Better).
*   **Event Autoscaling (KEDA):** Queue has 100 items -> Scale 10 Pods immediately. (Proactive).

### 2. KEDA Architecture
*   **Agent:** Activates deployments from 0 to 1.
*   **Metrics Server:** Exposes queue length to HPA for scaling 1 to N.
*   **Scalers:** Connectors for Kafka, SQS, Redis, Postgres, etc.

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Install KEDA

```bash
helm repo add kedacore https://kedacore.github.io/charts
helm repo update
helm install keda kedacore/keda --namespace keda --create-namespace
```

### 👨‍💻 Infrastructure: Scaling on SQS

Scenario: Asynchronous Batch Inference. Users upload images to S3, triggering SQS messages. Workers process images.

#### 📁 `manifests/scaled-object.yaml`
```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: image-processor-scaler
  namespace: prod
spec:
  scaleTargetRef:
    name: image-processor # Deployment Name
  minReplicaCount: 0      # Scale to Zero enabled!
  maxReplicaCount: 50
  pollingInterval: 30     # Check queue every 30s
  cooldownPeriod: 300     # Wait 5m before scaling down (prevents thrashing)
  
  triggers:
  - type: aws-sqs-queue
    metadata:
      queueURL: https://sqs.us-east-1.amazonaws.com/123/my-queue
      queueLength: "5"    # Target: 5 messages per Pod
      awsRegion: "us-east-1"
    authenticationRef:
      name: keda-trigger-auth
---
apiVersion: keda.sh/v1alpha1
kind: TriggerAuthentication
metadata:
  name: keda-trigger-auth
spec:
  podIdentity:
    provider: aws-eks # Use IRSA
```

### 👨‍💻 Infrastructure: Scaling on GPU Metrics (Prometheus)

Scenario: Synchronous Real-time Inference. Scale if GPU > 80%.

#### 📁 `manifests/scaled-object-prom.yaml`
```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: gpu-inference-scaler
spec:
  scaleTargetRef:
    name: diff-diffusion-model
  minReplicaCount: 1 # Don't scale to zero for real-time (cold start)
  maxReplicaCount: 10
  
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus-server.monitoring.svc.cluster.local
      metricName: dcgm_gpu_utilization # Just a name for KEDA
      threshold: "80"
      query: |
        avg(DCGM_FI_DEV_GPU_UTIL{app="diff-diffusion"})
```

### 👨‍💻 Core Implementation: Worker (Python)

The worker needs to gracefully handle shutdown when KEDA scales down.

#### 📁 `src/worker.py`
```python
import boto3
import signal
import sys
import time

stop_requested = False

def handle_sigterm(signum, frame):
    global stop_requested
    print("Received SIGTERM from K8s. Finishing current job...")
    stop_requested = True

signal.signal(signal.SIGTERM, handle_sigterm)

sqs = boto3.client('sqs')
queue_url = "..."

def process_loop():
    while not stop_requested:
        # Long polling
        resp = sqs.receive_message(
            QueueUrl=queue_url, 
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20 
        )
        
        if 'Messages' in resp:
            for msg in resp['Messages']:
                process_image(msg['Body'])
                sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=msg['ReceiptHandle'])
        
        if stop_requested:
            print("Exiting loop.")
            break

def process_image(body):
    print("Processing...")
    time.sleep(5) # Simulate GPU work

if __name__ == "__main__":
    process_loop()
```

---

## 🔬 Lab Exercise: "The Burst"

### Task
Scale to Zero and Back.
1.  Apply `minReplicaCount: 0`. Wait 5 mins. Deployment scales to 0 pods.
2.  Send 100 messages to SQS.
    *   `aws sqs send-message-batch ...`
3.  **Observation:**
    *   KEDA operator sees queue > 0.
    *   Scales Deployment to 1.
    *   HPA sees queue=100 with target=5.
    *   Scales Deployment to $100/5 = 20$ pods.
    *   Cluster Autoscaler adds nodes.
4.  **Result:** Queue drained in seconds.
5.  **Cleanup:** Queue empty. Wait 5m (Cooldown). Deployment scales to 0. Cost = $0.

---

## 📖 Advanced Theory: Predictive Scaling
Reactive scaling is always late.
**Predictive Scaling:** Use historical traffic patterns (e.g., "Mondays at 9 AM spike").
AWS Auto Scaling has "Predictive Scaling" policies.
Custom: Run a CronJob that sets `minReplicas=10` at 8:55 AM, and reset to `1` at 10 AM.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Scale to Zero:** Essential for Dev environments and sporadic Batch/Asynchronous workloads. Not for critical user-facing APIs (Cold Start latency is ~2-5 mins for GPU pods).
2.  **Target Value:** If `queueLength: 5`, KEDA tries to maintain 5 messages per pod. If queue has 500 items, KEDA asks for 100 pods. Ensure you set `maxReplicaCount`.
3.  **HPA Conflict:** Do not use `HorizontalPodAutoscaler` and `ScaledObject` on the same deployment. KEDA creates its own HPA.

### API Summary
```yaml
triggers:
- type: aws-sqs-queue
- type: prometheus
```

---

**Day 164 Complete** ✅

*Next: Day 165 - Multi-Tenancy - Sharing GPUs.*
