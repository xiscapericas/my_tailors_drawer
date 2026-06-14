import json
import os
from datetime import datetime, timezone
from decimal import Decimal

import boto3

dynamodb = boto3.resource("dynamodb")

READINGS_TABLE = os.environ["READINGS_TABLE"]


def convert_floats(value):
    if isinstance(value, float):
        return Decimal(str(value))

    if isinstance(value, dict):
        return {k: convert_floats(v) for k, v in value.items()}

    if isinstance(value, list):
        return [convert_floats(v) for v in value]

    return value


def lambda_handler(event, context):
    table = dynamodb.Table(READINGS_TABLE)

    safe_event = convert_floats(event)

    item = {
        "plant_id": safe_event.get("plant_id", "unknown"),
        "timestamp": safe_event.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "payload": safe_event,
    }

    table.put_item(Item=item)

    return {
        "statusCode": 200,
        "body": json.dumps({"message": "reading stored"}),
    }