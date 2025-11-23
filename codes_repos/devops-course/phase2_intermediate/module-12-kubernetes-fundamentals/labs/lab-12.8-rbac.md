# Lab 12.8: RBAC (Role-Based Access Control)

## Objective
Implement Kubernetes RBAC for security and access control.

## Learning Objectives
- Create Roles and ClusterRoles
- Bind roles to users/service accounts
- Implement least privilege
- Test permissions

---

## Role

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: pod-reader
  namespace: default
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
```

## RoleBinding

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: read-pods
  namespace: default
subjects:
- kind: User
  name: jane
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io
```

## ClusterRole

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: cluster-admin-custom
rules:
- apiGroups: ["*"]
  resources: ["*"]
  verbs: ["*"]
```

## ServiceAccount

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: myapp-sa
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: myapp-binding
subjects:
- kind: ServiceAccount
  name: myapp-sa
roleRef:
  kind: Role
  name: pod-reader
```

## Test Permissions

```bash
# Check permissions
kubectl auth can-i get pods --as=jane
kubectl auth can-i delete pods --as=jane

# Use service account
kubectl run test --image=nginx --serviceaccount=myapp-sa
```

## Success Criteria
✅ Roles created  
✅ RoleBindings working  
✅ Permissions enforced  
✅ Service accounts configured  

**Time:** 45 min
