# Lab 30.6: Deployment Verification

## Objective
Verify deployments are successful before completing rollout.

## Learning Objectives
- Implement smoke tests
- Verify health endpoints
- Check metrics
- Validate functionality

---

## Smoke Tests

```bash
#!/bin/bash
# smoke-test.sh

API_URL="https://api.example.com"

# Test health endpoint
if ! curl -f "$API_URL/health"; then
  echo "Health check failed"
  exit 1
fi

# Test critical endpoints
if ! curl -f "$API_URL/api/users"; then
  echo "Users endpoint failed"
  exit 1
fi

# Check response time
RESPONSE_TIME=$(curl -o /dev/null -s -w '%{time_total}' "$API_URL")
if (( $(echo "$RESPONSE_TIME > 1.0" | bc -l) )); then
  echo "Response time too slow: ${RESPONSE_TIME}s"
  exit 1
fi

echo "All smoke tests passed"
```

## Kubernetes Job

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: deployment-verification
spec:
  template:
    spec:
      containers:
      - name: verify
        image: curlimages/curl
        command: ["/bin/sh"]
        args:
        - -c
        - |
          curl -f http://myapp/health || exit 1
          curl -f http://myapp/api/status || exit 1
      restartPolicy: Never
```

## Success Criteria
✅ Smoke tests implemented  
✅ Health checks passing  
✅ Metrics validated  

**Time:** 35 min
