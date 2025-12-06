# Day 163: Living on the Edge: Spot Instances
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 24: Cost Optimization & FinOps

---

> **🎯 Focus Area:** You can get a \$30/hr `p3.16xlarge` for \$3/hr if you are willing to die at any moment. **Spot Instances** are the secret weapon of efficient ML training, provided you can handle the "2-minute warning".

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Architect** a fault-tolerant training loop that saves checkpoints every epoch.
2.  **Configure** Kubernetes NodeGroups to mix Spot and On-Demand instances.
3.  **Handle** the AWS Spot Interruption Warning using a Termination Handler.
4.  **Use** `ray.train` to automatically recover from worker node loss.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `pip install boto3 ray[train]`.

---

## 📖 Theoretical Foundation

### 1. The Spot Market
AWS has spare capacity. They sell it at 90% discount.
If a paying customer arrives, AWS reclaims the instance.
You get a **Rebalance Recommendation** or **Spot Instance Interruption Notice** (2 minutes before death).

### 2. Capacity Pools
Each Instance Type + Zone is a pool.
*   `us-east-1a` + `g4dn.xlarge` = Pool A.
*   `us-east-1b` + `g4dn.xlarge` = Pool B.
*   **Strategy:** Diversify. Don't request just `g4dn.xlarge`. Request "Any GPU with > 16GB RAM".

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Mixed Instance Policy (Terraform/ASG)

Tell K8s/AWS: "Try to get Spot. If unavailable, fall back to On-Demand."

#### 📁 `infra/spot_asg.tf`
```hcl
resource "aws_autoscaling_group" "gpu_training" {
  mixed_instances_policy {
    instances_distribution {
      on_demand_base_capacity                  = 0 # 100% Spot
      on_demand_percentage_above_base_capacity = 0
      spot_allocation_strategy                 = "capacity-optimized" # Pick pool least likely to die
    }

    launch_template {
      launch_template_specification {
        launch_template_id = aws_launch_template.gpu_worker.id
        version            = "$Latest"
      }

      # Diversify: If g4dn.xlarge unavailable, try g4dn.2xlarge
      override { instance_type = "g4dn.xlarge" }
      override { instance_type = "g4dn.2xlarge" }
      override { instance_type = "g5.xlarge" }
    }
  }
}
```

### 👨‍💻 Core Implementation: Handling Termination (Python)

Poll the Instance Metadata Service (IMDS) to see if we are doomed.

#### 📁 `src/termination_handler.py`
```python
import requests
import time
import os
import signal

def monitor_spot_interruption():
    """
    Run in a background thread.
    """
    while True:
        try:
            # AWS IMDS v2
            token = requests.put(
                "http://169.254.169.254/latest/api/token",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
                timeout=1
            ).text
            
            resp = requests.get(
                "http://169.254.169.254/latest/meta-data/spot/instance-action",
                headers={"X-aws-ec2-metadata-token": token},
                timeout=1
            )
            
            if resp.status_code == 200:
                print("⚠️ SPOT INTERRUPTION DETECTED! We have 2 minutes to save state.")
                action = resp.json()
                handle_termination(action)
                break
                
        except Exception:
            pass
            
        time.sleep(5)

def handle_termination(action):
    print(f"Action: {action['action']} at {action['time']}")
    # 1. Trigger graceful Checkpoint
    checkpoint_everything()
    # 2. Cordon Node (If K8s)
    # 3. Exit safely
    os._exit(0)

def checkpoint_everything():
    print("Saving model.pt to S3...")
    # s3.upload_file(...)
    print("Done.")

# Start monitoring
# threading.Thread(target=monitor_spot_interruption).start()
```

### 👨‍💻 Core Implementation: Fault Tolerant Ray Training

Ray Train handles this automatically if configured correctly.

#### 📁 `src/ray_spot_train.py`
```python
import ray
from ray.train import ScalingConfig, CheckpointConfig, RunConfig
from ray.train.torch import TorchTrainer

def train_func():
    # ... Standard PyTorch loop ...
    # Recover from checkpoint if exists
    start_epoch = 0
    checkpoint = ray.train.get_checkpoint()
    if checkpoint:
        # Load state
        start_epoch = checkpoint.to_dict()["epoch"]
    
    for epoch in range(start_epoch, 100):
        # ... train ...
        
        # Save EVERY epoch
        ray.train.report(
            {"loss": 0.5},
            checkpoint=ray.train.Checkpoint.from_dict({"epoch": epoch})
        )

trainer = TorchTrainer(
    train_func,
    scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
    run_config=RunConfig(
        storage_path="s3://my-bucket/ray-results", # Durable storage
        checkpoint_config=CheckpointConfig(num_to_keep=2),
        failure_config=ray.train.FailureConfig(max_failures=-1) # Retry forever
    )
)

print("Starting Spot Training...")
trainer.fit()
```

---

## 🔬 Lab Exercise: "Chaos Monkey"

### Task
Simulate Interruption using `aws ec2 terminate-instances`.
1.  Start a Ray Connect cluster with 1 Head (On-Demand) and 4 Workers (Spot).
2.  Submit the `ray_spot_train.py` job.
3.  While running (Epoch 10), kill 1 Worker instance from AWS Console.
4.  **Observation:**
    *   Ray detects node failure.
    *   Job pauses.
    *   AutoScaler provisions a new Spot instance replacement.
    *   Job resumes from Epoch 10 (Last checkpoint).
    *   **Loss:** ~5 mins of time. **Gain:** 90% cost savings.

---

## 📖 Advanced Theory: Checkpoint Frequency
How often to save?
*   Too often (Every batch): S3 upload costs + Slowdown.
*   Too rare (Every 10 epochs): Lose 10 epochs of work (Costly on large clusters).
*   **Formula:** $T_{checkpoint} \approx \sqrt{2 \times T_{MTBF} \times T_{save}}$.
    *   Essentially: Every epoch is usually fine for Deep Learning.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Head Node:** The Head Node (Orchestrator) should ALWAYS be On-Demand. If the Head dies, the whole cluster state is lost. Only Workers should be Spot.
2.  **Grace Period:** You have 120 seconds. Do not try to finish the epoch. Stop immediately, save RAM to Disk/S3, and die.
3.  **Capacity-Optimized:** Use this allocation strategy. AWS gives you the instance types that are *least likely* to be interrupted, even if they are slightly more expensive than the absolute cheapest.

### API Summary
```python
ray.train.FailureConfig(max_failures=-1)
```

---

**Day 163 Complete** ✅

*Next: Day 164 - Autoscaling Strategies - KEDA.*
