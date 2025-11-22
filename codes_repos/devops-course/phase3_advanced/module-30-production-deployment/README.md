# Production Deployment Strategies

## 🎯 Learning Objectives

By the end of this module, you will have a comprehensive understanding of Production Deployment, including:
- **Deployment Strategies**: Blue/Green, Canary, and Rolling deployments.
- **Feature Flags**: Decoupling deployment from release with tools like **LaunchDarkly**.
- **Zero Downtime**: Techniques to deploy without user impact.
- **Database Migrations**: Safely changing schemas in production.
- **Verification**: Smoke tests and automated rollback.

---

## 📖 Theoretical Concepts

### 1. Deployment Strategies

- **Rolling Deployment**: Replace instances one at a time. Default in K8s. Risk: If v2 is broken, some users see errors.
- **Blue/Green**: Run v1 (Blue) and v2 (Green) side-by-side. Switch traffic 100% when ready. Instant rollback.
- **Canary**: Gradually shift traffic (5% -> 25% -> 100%). Monitor metrics. Rollback if error rate spikes.

### 2. Feature Flags

Decouple **Deploy** (code goes to production) from **Release** (users see the feature).
- **Kill Switch**: Turn off a broken feature without redeploying.
- **A/B Testing**: Show Feature X to 50% of users.
- **Gradual Rollout**: Enable for internal users first, then 1%, then 10%, then 100%.

### 3. Zero Downtime Deployments

- **Health Checks**: K8s waits for `/health` to return 200 before sending traffic.
- **Graceful Shutdown**: On SIGTERM, stop accepting new requests, finish in-flight requests, then exit.
- **Connection Draining**: Load balancer waits for active connections to close before removing the instance.

### 4. Database Migrations (Expand/Contract Pattern)

**Problem**: Renaming a column breaks the old code.

**Solution**:
1.  **Expand**: Add new column. Old code ignores it.
2.  **Migrate**: Write to both columns.
3.  **Contract**: Remove old column after all code is updated.

---

## 🔧 Practical Examples

### Blue/Green with Kubernetes

```yaml
# Blue (Current)
apiVersion: v1
kind: Service
metadata:
  name: my-app
spec:
  selector:
    app: my-app
    version: blue

# Green (New)
# Deploy green pods
# Test them
# Update Service selector to version: green
```

### Feature Flag (LaunchDarkly SDK)

```python
import ldclient
from ldclient.config import Config

ldclient.set_config(Config("sdk-key"))
client = ldclient.get()

user = {"key": "user@example.com"}

if client.variation("new-checkout-flow", user, False):
    # New code
    render_new_checkout()
else:
    # Old code
    render_old_checkout()
```

### Graceful Shutdown (Go)

```go
func main() {
    srv := &http.Server{Addr: ":8080"}
    
    go func() {
        if err := srv.ListenAndServe(); err != nil {
            log.Fatal(err)
        }
    }()
    
    // Wait for interrupt signal
    quit := make(chan os.Signal, 1)
    signal.Notify(quit, syscall.SIGTERM)
    <-quit
    
    // Graceful shutdown (30s timeout)
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()
    srv.Shutdown(ctx)
}
```

---

## 🎯 Hands-on Labs

- [Lab 30.1: Feature Flags (Decoupling Deploy from Release)](./labs/lab-30.1-feature-flags.md)
- [Lab 30.2: Zero Downtime Database Migrations](./labs/lab-30.2-db-migrations.md)
- [Lab 30.3: Feature Flags](./labs/lab-30.3-feature-flags.md)
- [Lab 30.4: Progressive Rollout](./labs/lab-30.4-progressive-rollout.md)
- [Lab 30.5: Rollback Strategies](./labs/lab-30.5-rollback-strategies.md)
- [Lab 30.6: Deployment Verification](./labs/lab-30.6-deployment-verification.md)
- [Lab 30.7: Smoke Testing](./labs/lab-30.7-smoke-testing.md)
- [Lab 30.8: Deployment Automation](./labs/lab-30.8-deployment-automation.md)
- [Lab 30.9: Zero Downtime](./labs/lab-30.9-zero-downtime.md)
- [Lab 30.10: Deployment Best Practices](./labs/lab-30.10-deployment-best-practices.md)

---

## 📚 Additional Resources

### Official Documentation
- [LaunchDarkly Documentation](https://docs.launchdarkly.com/)
- [Kubernetes Rolling Updates](https://kubernetes.io/docs/tutorials/kubernetes-basics/update/update-intro/)

### Tools
- [Flagger](https://flagger.app/) - Progressive delivery for K8s.

---

## 🔑 Key Takeaways

1.  **Deploy != Release**: Use feature flags to control when users see changes.
2.  **Monitor During Rollout**: Watch error rates, latency, and business metrics.
3.  **Automate Rollback**: If error rate > threshold, rollback automatically.
4.  **Test in Production**: Canary deployments are a form of testing in production with real traffic.

---

## ⏭️ Next Steps

1.  Complete the labs to master production deployments.
2.  **Congratulations!** You have completed the entire DevOps course (30 modules). You are now ready to build and operate production systems at scale.
