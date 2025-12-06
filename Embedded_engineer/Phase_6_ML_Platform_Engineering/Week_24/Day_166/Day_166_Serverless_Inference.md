# Day 166: Pay for Usage, Not Idle: Serverless Inference
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 24: Cost Optimization & FinOps

---

> **🎯 Focus Area:** Your model is accessed once an hour. Running a constantly active G4 instance (\$400/month) is wasteful. **Serverless Inference** scales to zero effectively, but be wary of the "Cold Start".

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Deploy** a PyTorch model to AWS Lambda using Container Image Support.
2.  **Optimize** Cold Start latency (< 5s) by trimming dependencies.
3.  **Evaluate** SageMaker Serverless Inference vs Lambda.
4.  **Architect** an Async Inference architecture (API Gateway -> SQS -> Lambda) for heavy models.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `pip install aws-sam-cli`.

---

## 📖 Theoretical Foundation

### 1. Lambda Constraints
*   **Memory:** Max 10GB.
*   **Time:** Max 15 minutes.
*   **Storage:** 512MB - 10GB Ephemeral (/tmp).
*   **GPU:** NONE. (CPU Only).

### 2. Cold Start
When traffic arrives, AWS spins up a microVM (Firecracker).
*   **Init Phase:** Download Image -> Start Python -> Import TensorFlow (Slow).
*   **Invoke Phase:** `handler(event)` (Fast).
*   **Optimization:** Use ONNX Runtime (Small) instead of PyTorch (Huge). Use `provisioned concurrency` (No longer strictly serverless cost-wise).

### 3. SageMaker Serverless
*   Supports GPU (sometimes, depending on region/preview).
*   Higher memory limits.
*   Managed Endpoint abstraction.

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Dockerfile for Lambda

Standard method to bypass the 250MB size limit of Lambda Zip files.

#### 📁 `lambda/Dockerfile`
```dockerfile
FROM public.ecr.aws/lambda/python:3.9

# 1. Install Dependencies
# Trick: Install CPU-only torch to save space
RUN pip install torch==1.13.0+cpu torchvision==0.14.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --no-cache-dir

COPY requirements.txt .
RUN pip install -r requirements.txt

# 2. Copy Code & Model
COPY app.py ${LAMBDA_TASK_ROOT}
COPY model_quantized.onnx ${LAMBDA_TASK_ROOT}

# 3. CMD
CMD [ "app.handler" ]
```

### 👨‍💻 Core Implementation: The Handler

#### 📁 `lambda/app.py`
```python
import json
import onnxruntime as ort
import numpy as np
import time

# Global Init (Runs only on Cold Start)
print("Init: Loading Model...")
start_init = time.time()
session = ort.InferenceSession("model_quantized.onnx")
end_init = time.time()
print(f"Init: Model loaded in {end_init - start_init:.2f}s")

def handler(event, context):
    try:
        # 1. Parse Input
        body = json.loads(event['body'])
        data = np.array(body['inputs'], dtype=np.float32)
        
        # 2. Inference
        start_inf = time.time()
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: data})
        latency = time.time() - start_inf
        
        # 3. Response
        return {
            "statusCode": 200,
            "body": json.dumps({
                "prediction": outputs[0].tolist(),
                "latency_ms": latency * 1000,
                "cold_start": False # Logic to detect cold start via container ID usually
            })
        }
    except Exception as e:
        return {"statusCode": 500, "body": str(e)}
```

### 👨‍💻 Infrastructure: Deploy via SAM (Serverless Application Model)

#### 📁 `template.yaml`
```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Resources:
  InferenceFunction:
    Type: AWS::Serverless::Function
    Properties:
      PackageType: Image
      MemorySize: 4096 # 4GB RAM (gives proportional CPU)
      Timeout: 30
      Events:
        Api:
          Type: HttpApi
    Metadata:
      DockerTag: v1
      DockerContext: ./lambda
      Dockerfile: Dockerfile
```

---

## 🔬 Lab Exercise: "Slim Fast"

### Task
Reduce docker image size.
1.  **Baseline:** `pip install pandas matplotlib tensorflow`. Image size: 1.5GB. Cold Start: 8 seconds.
2.  **Optimization 1:** Use `tensorflow-cpu`. Save 400MB.
3.  **Optimization 2:** Remove `pandas` (use `json`). Remove `matplotlib` (not needed for inference).
4.  **Optimization 3:** Use ONNX Runtime instead of TF. Image size: 200MB.
5.  **Result:** Cold Start: 1.5 seconds.
6.  **Lesson:** "General Purpose" containers are bad for Serverless.

---

## 📖 Advanced Theory: Async Inference Pattern
If your model takes 30s to run (e.g., Stable Diffusion), API Gateway will timeout (29s limit).
**Architecture:**
1.  User `POST /task` -> Gateway -> Lambda (Producer).
2.  Lambda pushes Job ID to SQS. Returns `202 Accepted` to User.
3.  SQS triggers Lambda (Consumer, highly configured CPU/Memory).
4.  Consumer runs inference (up to 15m), writes result to DynamoDB/S3.
5.  User polls `GET /task/{id}` to see status (Pending -> Completed).

---

## 📝 Daily Summary

### Key Takeaways
1.  **CPU Only:** Lambda is CPU only. Use quantization (INT8) to make models fast enough. If you NEED GPU, look at SageMaker Serverless or KScale.
2.  **Provisioned Concurrency:** You can pay to keep N instances "Warm". This eliminates cold start but reintroduces a baseline cost (defeats purpose of scale-to-zero?). Use sparingly.
3.  **Cost Trap:** High Traffic + Serverless = Expensive. If you have constant 100 RPS, EC2/Fargate is cheaper than Lambda. Lambda is for bursty/sporadic traffic.

### API Summary
```python
ort.InferenceSession("model.onnx")
```

---

**Day 166 Complete** ✅

*Next: Day 167 - FinOps Dashboards & Reporting.*
