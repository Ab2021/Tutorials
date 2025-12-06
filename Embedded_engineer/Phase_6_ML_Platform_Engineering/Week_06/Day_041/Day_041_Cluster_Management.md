# Day 41: Cluster Management with Slurm & Kubernetes
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 6: Distributed Training & Large Scale Systems

---

> **🎯 Focus Area:** Learn how to submit and manage training jobs on High Performance Computing (HPC) clusters using **Slurm**, and understand the cloud-native equivalent using **Kubernetes (K8s)**.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Write** an SBATCH script to request GPU resources across multiple nodes.
2.  **Determine** the Master IP Address dynamically within a Slurm job.
3.  **Launch** a Multi-Node PyTorch training job using `srun`.
4.  **Describe** how Kubeflow's `PyTorchJob` CRD (Custom Resource Definition) translates to pods and services in K8s.
5.  **Differentiate** between Slurm (Batch/Queue based) and K8s (Service/Container based) philosophies.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Access to a Slurm Cluster (or `slurmd` simulator in Docker).
- *Alternatively:* Minikube for K8s demo.

### Software Environment
- Slurm Workload Manager.
- Kubernetes `kubectl`.

### Prior Knowledge
- Bash scripting.
- SSH/Networking.
- Day 36: `torchrun` arguments (`--master_addr`).

---

## 📖 Theoretical Foundation

### 1. Slurm Workload Manager
The de-facto standard for Supercomputers (Jean Zay, Summit, etc.).
*   **Philosophy:** Users submit "Jobs". Jobs sit in a "Queue" until resources (Nodes, GPUs) are available.
*   **SBATCH:** Directives at the top of the script (`#SBATCH`) tell the scheduler what you need ("2 Nodes, 4 GPUs each, 24h limit").
*   **srun:** Launches a process in parallel across the allocated nodes.

### 2. Kubernetes (K8s)
The standard for Enterprise/Cloud.
*   **Philosophy:** Users submit "Manifests" (YAML). The Control Plane ensures the desired state (Pod running) matches actual state.
*   **Operators:** K8s doesn't know "Distributed Training". We install the **Training Operator** (formerly `tf-operator`/`pytorch-operator`) which adds a `PyTorchJob` resource type.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: The Universal Slurm Script

This script is the "Gold Standard" for launching PyTorch DDP on Slurm. It handles the tricky part: Finding who is the Master Node.

#### 📁 `src/train_ddp.slurm`
```bash
#!/bin/bash
#SBATCH --job-name=pytorch-ddp
#SBATCH --nodes=2              # Request 2 Nodes
#SBATCH --ntasks-per-node=1    # Run 1 task (process) per node (The manager process)
#SBATCH --cpus-per-task=16     # CPU cores
#SBATCH --gres=gpu:4           # 4 GPUs per node
#SBATCH --time=01:00:00        # 1 Hour limit
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# 1. Environment Logic
echo "Starting Job $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"

# 2. Find the Master Node IP
# Slurm gives a list like "node[01-02]". We need the first one.
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Head Node: $head_node with IP: $head_node_ip"

export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29500
export LOGLEVEL=INFO

# 3. Launch PyTorch DDP
# We use 'srun' to run the script on ALL nodes.
# But inside the script, we use torchrun?
# Actually, convenient pattern: Use torchrun on each node.
# So we launch 'torchrun' N times (once per node).

srun torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=$RANDOM \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    src/ddp_training.py
```

### 👨‍💻 Conceptual: Kubernetes Manifest

We don't "run" K8s scripts; we "apply" YAMLs. The Operator creates the pods.

#### 📁 `src/pytorch_job.yaml`
```yaml
apiVersion: "kubeflow.org/v1"
kind: PyTorchJob
metadata:
  name: pytorch-dist-mnist
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: my-docker-repo/pytorch-training:latest
              command: ["python", "src/ddp_training.py"]
              resources:
                limits:
                  nvidia.com/gpu: 4
    Worker:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: my-docker-repo/pytorch-training:latest
              command: ["python", "src/ddp_training.py"]
              resources:
                limits:
                  nvidia.com/gpu: 4
```
*Note: The Operator automatically injects `MASTER_ADDR` and `RANK` env vars into the containers!*

---

## 🔬 Lab Exercise: "Slurm Simulation"

### Task
If you don't have a cluster, simulate the Environment Variables manually on one machine to understand what Slurm does.

1.  Open Terminal A (Node 0):
    ```bash
    export MASTER_ADDR=localhost
    export MASTER_PORT=29500
    export WORLD_SIZE=2
    export RANK=0
    python src/ddp_manual.py # (A script using dist.init_process_group)
    ```
2.  Open Terminal B (Node 1):
    ```bash
    export MASTER_ADDR=localhost # Same IP
    export MASTER_PORT=29500
    export WORLD_SIZE=2
    export RANK=1
    python src/ddp_manual.py
    ```
3.  **Observation:** Terminal A waits... Terminal B starts... They connect and train. This is all `srun` does: it sets these variables on remote machines and starts the process.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Slurm is Rigid:** It expects a fixed number of nodes for a fixed time. If a node dies, the job often crashes (unless using advanced fault tolerance).
2.  **K8s is Elastic:** If a generic pod dies, K8s restarts it. But for DDP (Rank 0 talks to Rank 1), losing Rank 1 is catastrophic. Elastic Training (TorchElastic) is required to handle dynamic node joining/leaving.
3.  **Head Node resolution:** The most common bug in Multi-Node is failing to resolve the correct IP address for `MASTER_ADDR` (e.g., getting the Docker bridge IP instead of the InfiniBand IP).

### API Summary
```bash
sbatch script.slurm       # Submit
squeue -u user            # Check status
scancel jobid             # Kill
srun --pty bash           # Interactive shell on compute node
```

---

**Day 41 Complete** ✅

*Next: Day 42 - Phase 6 Review & Final Project - Training a GPT-2 model from scratch on Multi-Node simulated setup.*
