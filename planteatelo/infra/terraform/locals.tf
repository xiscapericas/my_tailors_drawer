locals {
  name_prefix = "${var.project_name}-${var.environment}"

  s3_images_prefix = "${local.name_prefix}-images"

  common_tags = {
    project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}
