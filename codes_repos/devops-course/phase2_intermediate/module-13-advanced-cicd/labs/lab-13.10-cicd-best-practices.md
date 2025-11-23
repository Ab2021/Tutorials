# Lab 13.10: CI/CD Best Practices

## Objective
Implement CI/CD best practices for production pipelines.

## Learning Objectives
- Follow pipeline design patterns
- Implement security best practices
- Ensure reliability and observability
- Document pipeline processes

---

## Pipeline Design

```yaml
name: Production Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  # Stage 1: Validate
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: npm run lint
      - run: npm run type-check
  
  # Stage 2: Test
  test:
    needs: validate
    strategy:
      matrix:
        node: [16, 18]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: npm test
  
  # Stage 3: Build
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: docker build -t myapp:${{ github.sha }} .
  
  # Stage 4: Security
  security:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - run: trivy image myapp:${{ github.sha }}
  
  # Stage 5: Deploy
  deploy:
    needs: security
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: production
    steps:
      - run: kubectl apply -f k8s/
```

## Best Practices Checklist

✅ **Security**
- Scan code and containers
- Use secrets management
- Implement RBAC
- Enable audit logging

✅ **Reliability**
- Automated testing
- Rollback on failure
- Health checks
- Monitoring and alerts

✅ **Performance**
- Caching dependencies
- Parallel execution
- Incremental builds
- Optimized Docker images

✅ **Maintainability**
- Clear documentation
- Reusable workflows
- Consistent naming
- Version control

## Success Criteria
✅ Pipeline follows best practices  
✅ Security integrated  
✅ Reliability ensured  
✅ Documentation complete  

**Time:** 40 min
