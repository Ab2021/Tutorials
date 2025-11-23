# Lab 13.7: Compliance and Policy Gates

## Objective
Implement compliance checks and policy gates in CI/CD.

## Learning Objectives
- Use Open Policy Agent (OPA)
- Implement policy as code
- Enforce compliance rules
- Audit policy violations

---

## OPA Policy

```rego
# policy.rego
package kubernetes.admission

deny[msg] {
  input.request.kind.kind == "Pod"
  not input.request.object.spec.securityContext.runAsNonRoot
  msg = "Pods must run as non-root user"
}

deny[msg] {
  input.request.kind.kind == "Deployment"
  not input.request.object.spec.template.spec.containers[_].resources.limits
  msg = "Containers must have resource limits"
}
```

## Conftest

```yaml
name: Policy Check

on: [pull_request]

jobs:
  policy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install Conftest
        run: |
          wget https://github.com/open-policy-agent/conftest/releases/download/v0.45.0/conftest_0.45.0_Linux_x86_64.tar.gz
          tar xzf conftest_0.45.0_Linux_x86_64.tar.gz
          sudo mv conftest /usr/local/bin
      
      - name: Test policies
        run: conftest test k8s/*.yaml -p policy/
```

## Kyverno Policies

```yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-labels
spec:
  validationFailureAction: enforce
  rules:
  - name: check-labels
    match:
      resources:
        kinds:
        - Pod
    validate:
      message: "Labels 'app' and 'env' are required"
      pattern:
        metadata:
          labels:
            app: "?*"
            env: "?*"
```

## Success Criteria
✅ OPA policies enforced  
✅ Non-compliant resources blocked  
✅ Policy violations audited  

**Time:** 45 min
