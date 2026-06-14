variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "eu-west-1"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "planteatelo"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "plant_id" {
  description = "Plant identifier"
  type        = string
  default     = "plant_001"
}
