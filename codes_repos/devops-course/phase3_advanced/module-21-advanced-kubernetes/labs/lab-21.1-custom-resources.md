# Lab 21.1: Custom Resources (CRDs)

## Objective
Create and manage Kubernetes Custom Resource Definitions.

## Learning Objectives
- Define CRDs
- Create custom resources
- Use kubectl with CRDs
- Implement controllers

---

## Define CRD

```yaml
# myapp-crd.yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: myapps.example.com
spec:
  group: example.com
  versions:
    - name: v1
      served: true
      storage: true
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              properties:
                replicas:
                  type: integer
                image:
                  type: string
  scope: Namespaced
  names:
    plural: myapps
    singular: myapp
    kind: MyApp
```

## Create CRD

```bash
kubectl apply -f myapp-crd.yaml
kubectl get crds
```

## Use Custom Resource

```yaml
# myapp-instance.yaml
apiVersion: example.com/v1
kind: MyApp
metadata:
  name: my-application
spec:
  replicas: 3
  image: nginx:latest
```

```bash
kubectl apply -f myapp-instance.yaml
kubectl get myapps
kubectl describe myapp my-application
```

## Success Criteria
✅ CRD created  
✅ Custom resources deployed  
✅ kubectl commands working  

**Time:** 40 min
