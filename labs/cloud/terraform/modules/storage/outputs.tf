output "raw_bucket" { value = aws_s3_bucket.raw.bucket }
output "kms_key_arn" { value = aws_kms_key.data.arn }
output "ecr_repository_url" { value = aws_ecr_repository.app.repository_url }
