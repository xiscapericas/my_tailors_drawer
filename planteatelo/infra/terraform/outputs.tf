output "images_bucket" {
  value = aws_s3_bucket.planteatelo.bucket
}

output "images_prefix" {
  value = local.s3_images_prefix
}

output "readings_table" {
  value = aws_dynamodb_table.readings.name
}

output "analyses_table" {
  value = aws_dynamodb_table.analyses.name
}

output "iot_thing_name" {
  value = aws_iot_thing.raspberry.name
}

output "iot_topic" {
  value = "${var.project_name}/${var.plant_id}/readings"
}

output "process_reading_lambda" {
  value = aws_lambda_function.process_reading.function_name
}
