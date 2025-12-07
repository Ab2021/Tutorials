# Days 183-189: Week 27 - Advanced Troubleshooting Labs
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## Day 183-189: Troubleshooting Quick Reference

### Network Debugging (Day 184)
```bash
# Capture packets
kubectl exec -it debug-pod -- tcpdump -i eth0 -w /tmp/capture.pcap

# DNS debugging
kubectl exec -it debug-pod -- nslookup kubernetes.default
```

### GPU Debugging (Day 185)
```bash
# Check GPU errors
nvidia-smi -q -d PERFORMANCE

# Xid errors
dmesg | grep -i nvrm
```

### Memory Leak Hunting (Day 186)
```python
import torch
print(torch.cuda.memory_summary())

# Find leaks
import gc
gc.collect()
torch.cuda.empty_cache()
```

### Log Analytics (Day 187)
```promql
# LogQL query
{app="ml-api"} |= "error" | json | latency_ms > 1000
```

### Distributed Tracing (Day 188)
```python
from opentelemetry import trace
tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("predict"):
    result = model.predict(data)
```

---

## 📝 Week 27 Summary
| Day | Topic | Tool |
|-----|-------|------|
| 183 | eBPF | bpftrace |
| 184 | Network | tcpdump |
| 185 | GPU | nvidia-smi |
| 186 | Memory | torch.cuda |
| 187 | Logs | Loki |
| 188 | Traces | Tempo |
| 189 | Project | War Room |
