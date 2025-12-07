# Days 155-161: Week 23 - GPU Operations Labs
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## GPU Ops Quick Reference

### DCGM Metrics
```bash
# Install DCGM exporter
helm install dcgm-exporter nvidia/dcgm-exporter

# Key metrics
DCGM_FI_DEV_GPU_UTIL       # GPU utilization %
DCGM_FI_DEV_FB_USED        # Memory used (bytes)
DCGM_FI_DEV_FB_FREE        # Memory free (bytes)
DCGM_FI_DEV_POWER_USAGE    # Power (watts)
DCGM_FI_DEV_GPU_TEMP       # Temperature (C)
```

### MIG Configuration
```bash
# Enable MIG mode
sudo nvidia-smi -i 0 -mig 1

# Create GPU instances
sudo nvidia-smi mig -cgi 9,9,9,9,9,9,9 -C

# List instances
nvidia-smi mig -lgi
nvidia-smi mig -lci
```

### MIG in Kubernetes
```yaml
resources:
  limits:
    nvidia.com/mig-1g.5gb: 1  # Request MIG slice
```

### Health Checks
```bash
# Run GPU diagnostics
dcgmi diag -r 1  # Quick test
dcgmi diag -r 3  # Full test

# Check Xid errors
dmesg | grep -i nvrm
```

### Grafana Dashboard Queries
```promql
# GPU utilization by pod
avg(DCGM_FI_DEV_GPU_UTIL) by (pod)

# Memory usage %
DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_FREE * 100
```

---

## 📝 Week 23 Summary
| Day | Topic |
|-----|-------|
| 155 | DCGM |
| 156 | MIG |
| 157 | Monitoring |
| 158 | Alerts |
| 159 | Debugging |
| 160 | Diagnostics |
| 161 | Project |
