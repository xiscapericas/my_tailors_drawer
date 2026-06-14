resource "aws_iot_thing" "raspberry" {
  name = "${local.name_prefix}-raspberry-001"

  attributes = {
    plant_id = var.plant_id
  }
}

resource "aws_iot_policy" "raspberry_publish" {
  name = "${local.name_prefix}-raspberry-publish"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "iot:Connect"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "iot:Publish"
        ]
        Resource = [
          "arn:aws:iot:${var.aws_region}:*:topic/${var.project_name}/${var.plant_id}/readings"
        ]
      }
    ]
  })
}
