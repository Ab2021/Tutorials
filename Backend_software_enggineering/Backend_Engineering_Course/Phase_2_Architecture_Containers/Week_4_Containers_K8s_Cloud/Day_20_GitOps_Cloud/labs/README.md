# Lab: Day 20 - GitOps Simulator

## Goal
Understand the "Pull Model". We will write a tiny Python script that acts like ArgoCD. It will watch a "Git Repo" (folder) and sync changes to a "Cluster" (another folder).

## Directory Structure
```
day20/
├── git_repo/ (Source of Truth)
│   └── deployment.yaml
├── cluster_state/ (The Live Environment)
│   └── deployment.yaml
├── argocd_sim.py
└── README.md
```

## Step 1: Setup
Create the folders and initial file.

`git_repo/deployment.yaml`:
```yaml
replicas: 3
image: nginx:1.19
```

`cluster_state/deployment.yaml`:
```yaml
replicas: 3
image: nginx:1.19
```

## Step 2: The Agent (`argocd_sim.py`)

```python
import time
import shutil
import filecmp
import os

GIT_REPO = "./git_repo/deployment.yaml"
CLUSTER = "./cluster_state/deployment.yaml"

def sync():
    # 1. Check for Drift
    if not os.path.exists(CLUSTER) or not filecmp.cmp(GIT_REPO, CLUSTER):
        print("⚠️  Drift Detected! Syncing...")
        shutil.copy2(GIT_REPO, CLUSTER)
        print("✅ Synced: Cluster now matches Git.")
        
        # Simulate "Applying"
        with open(CLUSTER, 'r') as f:
            print(f"   Current State: {f.read().strip()}")
    else:
        print("zzz... Cluster is in sync.")

if __name__ == "__main__":
    print("🚀 ArgoCD Simulator Started...")
    while True:
        sync()
        time.sleep(2)
```

## Step 3: Run It

1.  **Start the Simulator**:
    `python argocd_sim.py`

2.  **Simulate a Git Commit**:
    Open `git_repo/deployment.yaml` and change `replicas: 3` to `replicas: 10`. Save.

3.  **Watch the Simulator**:
    It should detect the change and update `cluster_state/deployment.yaml`.

4.  **Simulate Manual Interference (Drift)**:
    Manually edit `cluster_state/deployment.yaml` (change image to `nginx:hacked`). Save.
    
    *Result*: The Simulator will overwrite your manual change with the version from `git_repo`. This is **Self-Healing**.

## Challenge
Add a "Pre-Sync Hook". Modify the script to run a validation check (e.g., "replicas cannot be > 20") before syncing. If validation fails, refuse to sync.
