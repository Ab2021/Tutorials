# Lab 13.4: Deployment Strategies in CI/CD

## Objective
Implement advanced deployment strategies in CI/CD pipelines.

## Learning Objectives
- Implement blue-green deployments
- Configure canary releases
- Use feature flags
- Automate rollbacks

---

## Blue-Green Deployment

```yaml
name: Blue-Green Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy to green
        run: |
          kubectl apply -f k8s/green/
          kubectl wait --for=condition=ready pod -l version=green
      
      - name: Run smoke tests
        run: ./smoke-tests.sh green
      
      - name: Switch traffic to green
        run: |
          kubectl patch service myapp -p '{"spec":{"selector":{"version":"green"}}}'
      
      - name: Delete blue
        run: kubectl delete -f k8s/blue/
```

## Canary Deployment

```yaml
  canary-deploy:
    steps:
      - name: Deploy canary (10%)
        run: |
          kubectl apply -f k8s/canary/
          kubectl scale deployment myapp-canary --replicas=1
          kubectl scale deployment myapp-stable --replicas=9
      
      - name: Monitor metrics
        run: |
          sleep 300  # Monitor for 5 minutes
          ERROR_RATE=$(prometheus-query 'rate(errors[5m])')
          if [ $ERROR_RATE -gt 0.01 ]; then
            kubectl delete deployment myapp-canary
            exit 1
          fi
      
      - name: Promote canary
        run: |
          kubectl scale deployment myapp-canary --replicas=10
          kubectl delete deployment myapp-stable
```

## Feature Flags

```yaml
  deploy-with-flags:
    steps:
      - name: Deploy
        run: kubectl apply -f k8s/
      
      - name: Enable feature for 10%
        run: |
          curl -X POST https://api.launchdarkly.com/api/v2/flags/my-project/new-feature \
            -H "Authorization: ${{ secrets.LD_API_KEY }}" \
            -d '{"variations": [{"value": true}], "rollout": {"percentage": 10}}'
```

## Automated Rollback

```yaml
  deploy-with-rollback:
    steps:
      - name: Deploy
        id: deploy
        run: kubectl apply -f k8s/
      
      - name: Verify deployment
        id: verify
        run: |
          kubectl rollout status deployment/myapp
          ./verify-health.sh
      
      - name: Rollback on failure
        if: failure()
        run: kubectl rollout undo deployment/myapp
```

## Success Criteria
✅ Blue-green deployment working  
✅ Canary release automated  
✅ Feature flags integrated  
✅ Rollback on failure  

**Time:** 50 min
