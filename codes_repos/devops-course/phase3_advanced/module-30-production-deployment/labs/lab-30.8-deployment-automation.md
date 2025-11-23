# Lab 30.8: Deployment Automation

## Objective
Fully automate deployment pipelines from commit to production.

## Learning Objectives
- Automate entire deployment flow
- Implement approval gates
- Configure notifications
- Track deployment metrics

---

## Complete Deployment Pipeline

```yaml
name: Production Deployment

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build -t myapp:${{ github.sha }} .
      - name: Push to registry
        run: docker push myapp:${{ github.sha }}
  
  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to staging
        run: |
          kubectl set image deployment/myapp myapp=myapp:${{ github.sha }} -n staging
          kubectl rollout status deployment/myapp -n staging
      - name: Run smoke tests
        run: pytest tests/smoke/
  
  approve:
    needs: deploy-staging
    runs-on: ubuntu-latest
    steps:
      - uses: trstringer/manual-approval@v1
        with:
          approvers: platform-team
          minimum-approvals: 1
  
  deploy-production:
    needs: approve
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: |
          kubectl set image deployment/myapp myapp=myapp:${{ github.sha }} -n production
          kubectl rollout status deployment/myapp -n production
      - name: Verify deployment
        run: ./scripts/verify-deployment.sh
      - name: Notify Slack
        run: |
          curl -X POST ${{ secrets.SLACK_WEBHOOK }} \
            -d '{"text":"Deployed myapp:${{ github.sha }} to production"}'
```

## Success Criteria
✅ Fully automated pipeline  
✅ Approval gates working  
✅ Notifications sent  
✅ Zero manual steps  

**Time:** 45 min
