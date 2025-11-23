# Lab 06.10: Pipeline Best Practices

## Objective
Implement CI/CD pipeline best practices.

## Learning Objectives
- Optimize pipeline performance
- Implement security best practices
- Handle failures gracefully
- Monitor pipeline health

---

## Performance Optimization

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node: [14, 16, 18]
      fail-fast: false  # Don't cancel other jobs
    steps:
      - uses: actions/checkout@v3
      
      - uses: actions/setup-node@v3
        with:
          node-version: ${{ matrix.node }}
          cache: 'npm'  # Cache dependencies
      
      - run: npm ci  # Faster than npm install
      - run: npm test
```

## Security Best Practices

```yaml
  security:
    steps:
      # Scan for secrets
      - uses: trufflesecurity/trufflehog@main
      
      # Dependency scanning
      - run: npm audit
      
      # Container scanning
      - uses: aquasecurity/trivy-action@master
        with:
          image-ref: myapp:latest
          severity: 'CRITICAL,HIGH'
```

## Error Handling

```yaml
  deploy:
    steps:
      - name: Deploy
        id: deploy
        continue-on-error: true
        run: ./deploy.sh
      
      - name: Rollback on failure
        if: steps.deploy.outcome == 'failure'
        run: ./rollback.sh
      
      - name: Notify team
        if: failure()
        run: |
          curl -X POST ${{ secrets.SLACK_WEBHOOK }} \
            -d '{"text":"Pipeline failed!"}'
```

## Monitoring

```yaml
  - name: Report metrics
    run: |
      echo "build_duration_seconds ${{ job.duration }}" | \
        curl --data-binary @- http://pushgateway:9091/metrics/job/ci
```

## Best Practices Checklist

✅ Use caching for dependencies  
✅ Run tests in parallel  
✅ Scan for security vulnerabilities  
✅ Implement proper error handling  
✅ Monitor pipeline performance  
✅ Use secrets for sensitive data  
✅ Fail fast on critical errors  
✅ Notify team on failures  

**Time:** 40 min
