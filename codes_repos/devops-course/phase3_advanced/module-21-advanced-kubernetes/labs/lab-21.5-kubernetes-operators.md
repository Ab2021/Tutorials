# Lab 21.5: Kubernetes Operators

## Objective
Create custom Kubernetes operators for application management.

## Learning Objectives
- Understand operator pattern
- Create custom operators
- Manage custom resources
- Implement reconciliation loops

---

## Operator SDK

```bash
# Install Operator SDK
brew install operator-sdk

# Create operator
operator-sdk init --domain=example.com --repo=github.com/myorg/myapp-operator

# Create API
operator-sdk create api --group=apps --version=v1 --kind=MyApp --resource --controller
```

## Custom Resource

```yaml
apiVersion: apps.example.com/v1
kind: MyApp
metadata:
  name: myapp-sample
spec:
  size: 3
  image: myapp:latest
  port: 8080
```

## Controller Logic

```go
func (r *MyAppReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
    // Fetch MyApp instance
    myapp := &appsv1.MyApp{}
    err := r.Get(ctx, req.NamespacedName, myapp)
    if err != nil {
        return ctrl.Result{}, client.IgnoreNotFound(err)
    }
    
    // Create Deployment
    deployment := &appsv1.Deployment{
        ObjectMeta: metav1.ObjectMeta{
            Name:      myapp.Name,
            Namespace: myapp.Namespace,
        },
        Spec: appsv1.DeploymentSpec{
            Replicas: &myapp.Spec.Size,
            Template: corev1.PodTemplateSpec{
                Spec: corev1.PodSpec{
                    Containers: []corev1.Container{{
                        Name:  "myapp",
                        Image: myapp.Spec.Image,
                    }},
                },
            },
        },
    }
    
    return ctrl.Result{}, r.Create(ctx, deployment)
}
```

## Success Criteria
✅ Operator created  
✅ Custom resources managed  
✅ Reconciliation working  
✅ Operator deployed  

**Time:** 50 min
