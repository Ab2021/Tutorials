# Lab 12.10: Kubernetes Troubleshooting

## Objective
Diagnose and fix common Kubernetes issues.

## Learning Objectives
- Debug pod failures
- Investigate networking issues
- Analyze logs and events
- Use kubectl debugging tools

---

## Debug Pod Issues

```bash
# Check pod status
kubectl get pods
kubectl describe pod myapp

# View logs
kubectl logs myapp
kubectl logs myapp --previous  # Previous container
kubectl logs myapp -c container-name  # Specific container

# Execute commands
kubectl exec -it myapp -- /bin/sh

# Debug with ephemeral container
kubectl debug myapp -it --image=busybox
```

## Network Debugging

```bash
# Test connectivity
kubectl run test --image=busybox --rm -it -- wget -O- http://myservice

# Check DNS
kubectl run test --image=busybox --rm -it -- nslookup myservice

# Port forward for testing
kubectl port-forward pod/myapp 8080:80
```

## Events and Logs

```bash
# View events
kubectl get events --sort-by='.lastTimestamp'

# Describe resources
kubectl describe deployment myapp
kubectl describe service myapp

# Check resource usage
kubectl top pods
kubectl top nodes
```

## Common Issues

```bash
# ImagePullBackOff
kubectl describe pod myapp | grep -A 5 Events

# CrashLoopBackOff
kubectl logs myapp --previous

# Pending pods
kubectl describe pod myapp | grep -A 10 Events

# Node issues
kubectl describe node mynode
```

## Success Criteria
✅ Pod issues diagnosed  
✅ Network problems resolved  
✅ Logs analyzed  
✅ Common issues fixed  

**Time:** 45 min
