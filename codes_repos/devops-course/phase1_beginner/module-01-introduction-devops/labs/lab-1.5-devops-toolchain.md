# Lab 1.5: DevOps Toolchain

## Objective
Build a complete DevOps toolchain for a project.

## Learning Objectives
- Select appropriate tools
- Integrate tools together
- Automate workflows
- Measure toolchain effectiveness

---

## Toolchain Components

```yaml
# DevOps Toolchain
toolchain:
  version_control:
    tool: Git
    platform: GitHub
    
  ci_cd:
    tool: GitHub Actions
    alternative: Jenkins
    
  containerization:
    tool: Docker
    registry: Docker Hub
    
  orchestration:
    tool: Kubernetes
    platform: EKS
    
  infrastructure:
    tool: Terraform
    state: S3
    
  monitoring:
    metrics: Prometheus
    visualization: Grafana
    logging: Loki
    
  security:
    scanning: Trivy
    secrets: Vault
```

## Tool Integration

```yaml
# GitHub Actions with full toolchain
name: Complete Pipeline

on: [push]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build with Docker
        run: docker build -t myapp:${{ github.sha }} .
      
      - name: Security scan
        run: trivy image myapp:${{ github.sha }}
      
      - name: Push to registry
        run: docker push myapp:${{ github.sha }}
  
  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy with Terraform
        run: |
          terraform init
          terraform apply -auto-approve
      
      - name: Deploy to Kubernetes
        run: kubectl apply -f k8s/
```

## Success Criteria
✅ Toolchain designed  
✅ Tools integrated  
✅ Workflows automated  
✅ Effectiveness measured  

**Time:** 45 min
