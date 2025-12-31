"""
Example usage of the RADCURE Processor package.

This script demonstrates how to use the CaseProcessor class to process
RADCURE DICOM cases for tumor detection model training.

Before running, set up your environment variables:
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
Or use a .env file (see .env.example)
"""

import os
from radcure_processor import CaseProcessor

# Try to load from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Note: python-dotenv not installed. Using environment variables directly.")

# Configuration - Load from environment variables
MAIN_PATH = os.getenv('MAIN_PATH', '/path/to/your/dataset/')
AWS_BUCKET_NAME = os.getenv('AWS_BUCKET_NAME', 'your-bucket-name')
AWS_FOLDER = os.getenv('AWS_FOLDER', 'RADCURE/all_cases/')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'eu-west-1')

# Validate AWS credentials
if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
    raise ValueError(
        "AWS credentials not found. Please set AWS_ACCESS_KEY_ID and "
        "AWS_SECRET_ACCESS_KEY environment variables or use a .env file."
    )

ORGAN_DICTIONARY_PATH = os.getenv(
    'ORGAN_DICTIONARY_PATH',
    os.path.join(MAIN_PATH, 'radcure_dictionary.json')
)
PROCESSED_CASES_FILE = os.getenv(
    'PROCESSED_CASES_FILE',
    os.path.join(MAIN_PATH, 'processed_cases.txt')
)
FAILED_CASES_FILE = os.getenv(
    'FAILED_CASES_FILE',
    os.path.join(MAIN_PATH, 'processed_cases_failed.txt')
)

# TotalSegmentator tasks to run
TASKS_TO_RUN = [
    'head_glands_cavities',
    'head_muscles',
    'headneck_bones_vessels',
    'headneck_muscles',
    'oculomotor_muscles',
    'craniofacial_structures'
]

# Initialize processor
processor = CaseProcessor(
    main_path=MAIN_PATH,
    aws_bucket_name=AWS_BUCKET_NAME,
    aws_folder=AWS_FOLDER,
    organ_dictionary_path=ORGAN_DICTIONARY_PATH,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    aws_region=AWS_REGION,
    total_segmentator_tasks=TASKS_TO_RUN,
    reverse_slices=False,
    slice_expansion=5
)

# Option 1: Process a single case
# case_id = 'RADCURE-0005'
# result = processor.process_case(case_id)
# print(f"Processing complete: {result}")

# Option 2: Process multiple cases (all available cases)
processor.process_multiple_cases(
    case_ids=None,  # None = process all available cases
    processed_cases_file=PROCESSED_CASES_FILE,
    failed_cases_file=FAILED_CASES_FILE
)

# Option 3: Process specific cases
# specific_cases = ['RADCURE-0005', 'RADCURE-0006', 'RADCURE-0007']
# processor.process_multiple_cases(
#     case_ids=specific_cases,
#     processed_cases_file=PROCESSED_CASES_FILE,
#     failed_cases_file=FAILED_CASES_FILE
# )
