# Lab 07.8: Terraform Data Sources

## Objective
Use data sources to reference existing infrastructure.

## Learning Objectives
- Query existing resources
- Use data sources in configurations
- Filter and select resources
- Combine with resource creation

---

## Data Sources

```hcl
# Query existing VPC
data "aws_vpc" "existing" {
  filter {
    name   = "tag:Name"
    values = ["production-vpc"]
  }
}

# Query AMI
data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]
  
  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }
}

# Use data sources
resource "aws_instance" "web" {
  ami           = data.aws_ami.amazon_linux.id
  subnet_id     = data.aws_vpc.existing.id
  instance_type = "t2.micro"
}
```

## Remote State Data Source

```hcl
data "terraform_remote_state" "network" {
  backend = "s3"
  config = {
    bucket = "terraform-state"
    key    = "network/terraform.tfstate"
    region = "us-east-1"
  }
}

resource "aws_instance" "app" {
  subnet_id = data.terraform_remote_state.network.outputs.subnet_id
}
```

## Success Criteria
✅ Data sources querying resources  
✅ Existing infrastructure referenced  
✅ Remote state accessed  

**Time:** 35 min
