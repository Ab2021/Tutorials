# Days 78-84: Week 12 - Networking for Distributed ML Labs
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## Day 78: InfiniBand & RDMA

```bash
# Check IB status
ibstat
ibstatus

# Test bandwidth
ib_write_bw -d mlx5_0
```

---

## Day 79: RoCE

```bash
# RoCE v2 verification
show_gids
rdma link show

# Test RoCE performance
perftest -d mlx5_0 --rdma_cm
```

---

## Day 80: AWS EFA

```bash
# EFA check
fi_info -p efa

# NCCL with EFA
export FI_PROVIDER=efa
export NCCL_DEBUG=INFO
```

---

## Day 81: GPUDirect

```bash
# Verify GPUDirect RDMA
nvidia-smi nvlink -s

# Check peer access
nvidia-smi topo -p
```

---

## Day 82: NCCL Tuning

```bash
# NCCL environment variables
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0

# Run all-reduce benchmark
python -m torch.distributed.run --nproc_per_node=8 allreduce_bench.py
```

---

## Day 83: Network Troubleshooting

```bash
# Check for packet loss
ping -c 1000 <target>

# Monitor bandwidth
iftop -i eth0

# NCCL debug
export NCCL_DEBUG=TRACE
```

---

## Day 84: Week 12 Project

```yaml
# HPC Training Pod
spec:
  hostNetwork: true
  containers:
  - name: train
    image: pytorch/pytorch:2.0-cuda12.0
    env:
    - name: NCCL_DEBUG
      value: INFO
    - name: NCCL_IB_DISABLE
      value: "0"
    resources:
      limits:
        nvidia.com/gpu: 8
        rdma/hca: 1
```

---

## 📝 Week 12 Summary
| Day | Topic | Technology |
|-----|-------|------------|
| 78 | InfiniBand | IB verbs |
| 79 | RoCE | RDMA over Ethernet |
| 80 | EFA | AWS fabric |
| 81 | GPUDirect | RDMA, Storage |
| 82 | NCCL | Collectives |
| 83 | Debug | Troubleshooting |
| 84 | Project | HPC cluster |

---

## 🎓 Phase 6B Complete!
Kubernetes & Cloud Infrastructure (Weeks 7-12)
