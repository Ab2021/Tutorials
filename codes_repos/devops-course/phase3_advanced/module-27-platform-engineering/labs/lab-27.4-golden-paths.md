# Lab 27.4: Golden Paths

## Objective
Create golden paths for common development workflows.

## Learning Objectives
- Define golden paths
- Create templates
- Document best practices
- Measure adoption

---

## Golden Path: New Service

```yaml
# .github/workflows/new-service.yaml
name: Create New Service

on:
  workflow_dispatch:
    inputs:
      service_name:
        required: true
      language:
        type: choice
        options: [python, nodejs, go]

jobs:
  create:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Create from template
        run: |
          cookiecutter gh:myorg/service-template-${{ inputs.language }} \
            service_name=${{ inputs.service_name }}
      
      - name: Create repo
        run: |
          gh repo create myorg/${{ inputs.service_name }} --private
          cd ${{ inputs.service_name }}
          git init
          git add .
          git commit -m "Initial commit"
          git push
      
      - name: Setup CI/CD
        run: |
          # Auto-configure GitHub Actions
          # Setup monitoring
          # Configure alerts
```

## Service Template

```
service-template/
├── {{cookiecutter.service_name}}/
│   ├── src/
│   ├── tests/
│   ├── Dockerfile
│   ├── .github/workflows/ci.yaml
│   ├── k8s/
│   │   ├── deployment.yaml
│   │   └── service.yaml
│   └── README.md
```

## Success Criteria
✅ Golden paths defined  
✅ Templates created  
✅ Automation working  
✅ Adoption tracked  

**Time:** 40 min
