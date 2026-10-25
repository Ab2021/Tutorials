output "vpc_id" {
  value = module.network.vpc_id
}

output "raw_bucket" {
  value = module.storage.raw_bucket
}

output "kms_key_arn" {
  value = module.storage.kms_key_arn
}

output "ecr_repository_url" {
  value = module.storage.ecr_repository_url
}

output "eks_cluster_name" {
  value = module.eks.cluster_name
}

output "eks_cluster_endpoint" {
  value     = module.eks.cluster_endpoint
  sensitive = true
}
