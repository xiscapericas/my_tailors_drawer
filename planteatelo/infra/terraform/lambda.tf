data "archive_file" "process_reading" {
  type        = "zip"
  source_dir  = "${path.module}/../../backend/lambdas/process_reading"
  output_path = "${path.module}/build/process_reading.zip"
}

resource "aws_lambda_function" "process_reading" {
  function_name = "${local.name_prefix}-process-reading"
  role          = aws_iam_role.lambda_process_reading.arn
  handler       = "handler.lambda_handler"
  runtime       = "python3.12"

  filename         = data.archive_file.process_reading.output_path
  source_code_hash = data.archive_file.process_reading.output_base64sha256

  environment {
    variables = {
      READINGS_TABLE = aws_dynamodb_table.readings.name
      ANALYSES_TABLE = aws_dynamodb_table.analyses.name
      IMAGES_BUCKET  = aws_s3_bucket.planteatelo.bucket
      IMAGES_PREFIX  = local.s3_images_prefix
    }
  }

  tags = local.common_tags
}

resource "aws_lambda_permission" "allow_iot" {
  statement_id  = "AllowExecutionFromIoT"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.process_reading.function_name
  principal     = "iot.amazonaws.com"
}

resource "aws_iot_topic_rule" "readings_to_lambda" {
  name        = "${replace(local.name_prefix, "-", "_")}_readings_to_lambda"
  description = "Route Planteatelo readings to Lambda"
  enabled     = true
  sql         = "SELECT * FROM '${var.project_name}/${var.plant_id}/readings'"
  sql_version = "2016-03-23"

  lambda {
    function_arn = aws_lambda_function.process_reading.arn
  }

  tags = local.common_tags
}
