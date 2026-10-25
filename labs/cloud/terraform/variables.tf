variable "project" {
  description = "Short project name used in resource names."
  type        = string
  default     = "biomed"
}

variable "env" {
  description = "Environment name."
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.env)
    error_message = "env must be dev, staging, or prod."
  }
}

variable "aws_region" {
  description = "AWS region."
  type        = string
  default     = "us-east-1"
}

variable "vpc_cidr" {
  description = "VPC CIDR."
  type        = string
  default     = "10.42.0.0/16"
}

variable "cluster_version" {
  description = "EKS Kubernetes version."
  type        = string
  default     = "1.30"
}

variable "node_instance_types" {
  description = "Managed node group instance types."
  type        = list(string)
  default     = ["t3.medium"]
}

variable "node_desired_size" {
  description = "Desired worker count."
  type        = number
  default     = 2
}

variable "node_min_size" {
  description = "Minimum worker count."
  type        = number
  default     = 1
}

variable "node_max_size" {
  description = "Maximum worker count."
  type        = number
  default     = 4
}
