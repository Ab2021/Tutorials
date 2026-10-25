module "network" {
  source = "./modules/network"

  project  = var.project
  env      = var.env
  vpc_cidr = var.vpc_cidr
}

module "storage" {
  source = "./modules/storage"

  project = var.project
  env     = var.env
}

module "eks" {
  source = "./modules/eks"

  project            = var.project
  env                = var.env
  cluster_version    = var.cluster_version
  vpc_id             = module.network.vpc_id
  private_subnet_ids = module.network.private_subnet_ids
  node_instance_types = var.node_instance_types
  node_desired_size   = var.node_desired_size
  node_min_size       = var.node_min_size
  node_max_size       = var.node_max_size
}
