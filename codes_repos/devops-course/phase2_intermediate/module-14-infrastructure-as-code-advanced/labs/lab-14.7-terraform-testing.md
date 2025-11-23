# Lab 14.7: Terraform Testing

## Objective
Test Terraform configurations for correctness and compliance.

## Learning Objectives
- Write Terraform tests
- Use Terratest
- Validate configurations
- Implement compliance checks

---

## Terraform Validate

```bash
terraform init
terraform validate
terraform fmt -check
```

## Terratest

```go
// test/terraform_test.go
package test

import (
    "testing"
    "github.com/gruntwork-io/terratest/modules/terraform"
    "github.com/stretchr/testify/assert"
)

func TestTerraformVPC(t *testing.T) {
    terraformOptions := &terraform.Options{
        TerraformDir: "../",
    }
    
    defer terraform.Destroy(t, terraformOptions)
    terraform.InitAndApply(t, terraformOptions)
    
    vpcId := terraform.Output(t, terraformOptions, "vpc_id")
    assert.NotEmpty(t, vpcId)
}
```

## Checkov

```bash
# Scan for security issues
checkov -d .

# Specific checks
checkov -d . --check CKV_AWS_20
```

## Success Criteria
✅ Tests passing  
✅ Configurations validated  
✅ Security checks passed  

**Time:** 40 min
