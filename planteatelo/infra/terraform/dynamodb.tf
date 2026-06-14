resource "aws_dynamodb_table" "readings" {
  name         = "${local.name_prefix}-readings"
  billing_mode = "PAY_PER_REQUEST"

  hash_key  = "plant_id"
  range_key = "timestamp"

  attribute {
    name = "plant_id"
    type = "S"
  }

  attribute {
    name = "timestamp"
    type = "S"
  }

  tags = local.common_tags
}

resource "aws_dynamodb_table" "analyses" {
  name         = "${local.name_prefix}-analyses"
  billing_mode = "PAY_PER_REQUEST"

  hash_key  = "plant_id"
  range_key = "timestamp"

  attribute {
    name = "plant_id"
    type = "S"
  }

  attribute {
    name = "timestamp"
    type = "S"
  }

  tags = local.common_tags
}
