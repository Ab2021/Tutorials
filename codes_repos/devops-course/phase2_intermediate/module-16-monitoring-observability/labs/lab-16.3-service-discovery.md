# Lab 16.3: Service Discovery

## Objective
Configure Prometheus service discovery for dynamic infrastructure.

## Learning Objectives
- Use Kubernetes service discovery
- Configure EC2 service discovery
- Implement file-based discovery
- Use relabeling

---

## Kubernetes Service Discovery

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
```

## EC2 Service Discovery

```yaml
scrape_configs:
  - job_name: 'ec2'
    ec2_sd_configs:
      - region: us-east-1
        port: 9100
    relabel_configs:
      - source_labels: [__meta_ec2_tag_Name]
        target_label: instance
```

## File-Based Discovery

```yaml
scrape_configs:
  - job_name: 'file'
    file_sd_configs:
      - files:
        - '/etc/prometheus/targets/*.json'
        refresh_interval: 30s
```

```json
[
  {
    "targets": ["host1:9100", "host2:9100"],
    "labels": {
      "env": "production"
    }
  }
]
```

## Success Criteria
✅ K8s service discovery working  
✅ EC2 instances auto-discovered  
✅ File-based discovery configured  
✅ Relabeling applied correctly  

**Time:** 40 min
