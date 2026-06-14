resource "aws_s3_bucket" "planteatelo" {
  bucket = var.project_name

  tags = local.common_tags
}

resource "aws_s3_bucket_public_access_block" "planteatelo" {
  bucket = aws_s3_bucket.planteatelo.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_object" "images_folder" {
  bucket  = aws_s3_bucket.planteatelo.id
  key     = "${local.s3_images_prefix}/"
  content = ""

  tags = local.common_tags
}
