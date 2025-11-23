# Lab 06.6: Deployment Automation

## Objective
Automate deployments to different environments.

## Learning Objectives
- Deploy to staging/production
- Use environment-specific configs
- Implement approval gates
- Rollback on failure

---

## Multi-Environment Deployment

```yaml
name: Deploy

on:
  push:
    branches: [main, develop]

jobs:
  deploy-staging:
    if: github.ref == 'refs/heads/develop'
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to staging
        run: |
          kubectl set image deployment/myapp \
            myapp=myapp:${{ github.sha }} \
            -n staging
  
  deploy-production:
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment:
      name: production
      url: https://myapp.com
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to production
        run: |
          kubectl set image deployment/myapp \
            myapp=myapp:${{ github.sha }} \
            -n production
      - name: Verify deployment
        run: |
          kubectl rollout status deployment/myapp -n production
```

## Approval Gates

```yaml
  deploy-production:
    needs: deploy-staging
    environment:
      name: production
    steps:
      # Manual approval required before this runs
      - name: Deploy
        run: ./deploy.sh
```

## Rollback on Failure

```yaml
  - name: Deploy
    id: deploy
    run: kubectl apply -f k8s/
  
  - name: Verify
    id: verify
    run: ./verify-deployment.sh
  
  - name: Rollback on failure
    if: failure()
    run: kubectl rollout undo deployment/myapp
```

## Success Criteria
✅ Staging auto-deploys  
✅ Production requires approval  
✅ Rollback on failure  
✅ Deployment verified  

**Time:** 40 min
