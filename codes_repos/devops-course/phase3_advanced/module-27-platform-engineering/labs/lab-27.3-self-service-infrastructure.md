# Lab 27.3: Self-Service Infrastructure

## Objective
Implement self-service infrastructure provisioning.

## Learning Objectives
- Create service catalogs
- Implement approval workflows
- Automate provisioning
- Track resource usage

---

## Terraform Cloud Workspaces

```hcl
# workspace-template/main.tf
variable "environment" {
  type = string
}

variable "owner" {
  type = string
}

resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
  tags = {
    Environment = var.environment
    Owner       = var.owner
  }
}
```

## Self-Service Portal

```python
# Flask API for provisioning
from flask import Flask, request
import subprocess

app = Flask(__name__)

@app.route('/provision', methods=['POST'])
def provision():
    data = request.json
    env = data['environment']
    owner = data['owner']
    
    # Create workspace
    subprocess.run([
        'terraform', 'workspace', 'new', f"{owner}-{env}"
    ])
    
    # Apply
    subprocess.run([
        'terraform', 'apply',
        '-var', f'environment={env}',
        '-var', f'owner={owner}',
        '-auto-approve'
    ])
    
    return {"status": "provisioned"}
```

## Resource Quotas

```yaml
# quotas.yaml
teams:
  - name: frontend
    quotas:
      ec2_instances: 10
      rds_instances: 2
      s3_buckets: 5
  - name: backend
    quotas:
      ec2_instances: 20
      rds_instances: 5
```

## Success Criteria
✅ Service catalog created  
✅ Self-service provisioning working  
✅ Approval workflows implemented  
✅ Quotas enforced  

**Time:** 45 min
