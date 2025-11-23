# Lab 07.7: Terraform Modules

## Objective
Create and use Terraform modules for code reusability.

## Learning Objectives
- Create custom modules
- Use module inputs/outputs
- Version modules
- Use module registry

---

## Create Module

```hcl
# modules/vpc/main.tf
variable "cidr_block" {
  type = string
}

variable "name" {
  type = string
}

resource "aws_vpc" "main" {
  cidr_block           = var.cidr_block
  enable_dns_hostnames = true
  
  tags = {
    Name = var.name
  }
}

resource "aws_subnet" "public" {
  vpc_id     = aws_vpc.main.id
  cidr_block = cidrsubnet(var.cidr_block, 8, 1)
  
  tags = {
    Name = "${var.name}-public"
  }
}

output "vpc_id" {
  value = aws_vpc.main.id
}

output "subnet_id" {
  value = aws_subnet.public.id
}
```

## Use Module

```hcl
# main.tf
module "vpc" {
  source = "./modules/vpc"
  
  cidr_block = "10.0.0.0/16"
  name       = "production-vpc"
}

resource "aws_instance" "web" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"
  subnet_id     = module.vpc.subnet_id
}
```

## Module Registry

```hcl
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.0.0"
  
  name = "my-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["us-east-1a", "us-east-1b"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]
}
```

## Module Composition

```hcl
module "network" {
  source = "./modules/network"
  
  environment = var.environment
}

module "compute" {
  source = "./modules/compute"
  
  vpc_id    = module.network.vpc_id
  subnet_id = module.network.subnet_id
}
```

## Success Criteria
✅ Custom module created  
✅ Module used in root config  
✅ Registry module used  
✅ Module composition working  

**Time:** 45 min
