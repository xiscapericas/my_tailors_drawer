"""
Process all RADCURE cases for TotalSegmentator retraining.

This script processes all available RADCURE cases from AWS S3, generating
the necessary outputs for training. After completion, run split_dataset.py
to split the processed cases into train/validation/test sets.

Usage:
    python process_all_cases.py

Before running:
    1. Copy env.example to .env and fill in your credentials
    2. Ensure you have sufficient disk space
    3. Make sure TotalSegmentator is installed and configured

The script will:
    - Download all cases from AWS S3
    - Process each case through the full pipeline
    - Generate NIfTI files, masks, and visualizations
    - Log successful and failed cases
"""

import os
import sys
from datetime import datetime
from radcure_processor import CaseProcessor

# Try to load from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded environment variables from .env file")
except ImportError:
    print("Note: python-dotenv not installed. Using environment variables directly.")

# Configuration - Load from environment variables
MAIN_PATH = os.getenv('MAIN_PATH')
AWS_BUCKET_NAME = os.getenv('AWS_BUCKET_NAME')
AWS_FOLDER = os.getenv('AWS_FOLDER', 'RADCURE/all_cases/')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'eu-west-1')

# Validate required environment variables
if not MAIN_PATH:
    raise ValueError(
        "MAIN_PATH not set. Please set it in your .env file or environment variables."
    )

if not AWS_BUCKET_NAME:
    raise ValueError(
        "AWS_BUCKET_NAME not set. Please set it in your .env file or environment variables."
    )

if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
    raise ValueError(
        "AWS credentials not found. Please set AWS_ACCESS_KEY_ID and "
        "AWS_SECRET_ACCESS_KEY in your .env file or environment variables."
    )

# Optional configuration with defaults
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

# Processing configuration
REVERSE_SLICES = os.getenv('REVERSE_SLICES', 'False').lower() == 'true'
SLICE_EXPANSION = int(os.getenv('SLICE_EXPANSION', '5'))


def print_header():
    """Print script header with configuration information."""
    print("=" * 70)
    print("RADCURE Case Processor - Process All Cases")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Main path: {MAIN_PATH}")
    print(f"AWS bucket: {AWS_BUCKET_NAME}")
    print(f"AWS folder: {AWS_FOLDER}")
    print(f"Organ dictionary: {ORGAN_DICTIONARY_PATH}")
    print(f"Processed cases log: {PROCESSED_CASES_FILE}")
    print(f"Failed cases log: {FAILED_CASES_FILE}")
    print(f"Reverse slices: {REVERSE_SLICES}")
    print(f"Slice expansion: {SLICE_EXPANSION}")
    print(f"TotalSegmentator tasks: {len(TASKS_TO_RUN)}")
    print("=" * 70)
    print()


def main():
    """Main processing function."""
    print_header()
    
    # Initialize processor
    print("Initializing CaseProcessor...")
    try:
        processor = CaseProcessor(
            main_path=MAIN_PATH,
            aws_bucket_name=AWS_BUCKET_NAME,
            aws_folder=AWS_FOLDER,
            organ_dictionary_path=ORGAN_DICTIONARY_PATH,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            aws_region=AWS_REGION,
            total_segmentator_tasks=TASKS_TO_RUN,
            reverse_slices=REVERSE_SLICES,
            slice_expansion=SLICE_EXPANSION
        )
        print("✓ CaseProcessor initialized successfully\n")
    except Exception as e:
        print(f"✗ Failed to initialize CaseProcessor: {e}")
        sys.exit(1)
    
    # Process all cases
    print("Starting to process all available cases...")
    print("This may take a long time depending on the number of cases.\n")
    
    try:
        processor.process_multiple_cases(
            case_ids=None,  # None = process all available cases
            processed_cases_file=PROCESSED_CASES_FILE,
            failed_cases_file=FAILED_CASES_FILE
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user (Ctrl+C)")
        print("Progress has been saved. You can resume by running this script again.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Fatal error during processing: {e}")
        sys.exit(1)
    
    # Final summary
    print("\n" + "=" * 70)
    print("Processing Complete!")
    print("=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Count processed and failed cases
    processed_count = 0
    failed_count = 0
    
    if os.path.exists(PROCESSED_CASES_FILE):
        with open(PROCESSED_CASES_FILE, 'r') as f:
            processed_count = len([line for line in f if line.strip()])
    
    if os.path.exists(FAILED_CASES_FILE):
        with open(FAILED_CASES_FILE, 'r') as f:
            failed_count = len([line for line in f if line.strip()])
    
    print(f"Successfully processed: {processed_count} cases")
    print(f"Failed: {failed_count} cases")
    print(f"Total: {processed_count + failed_count} cases")
    print()
    print("Next steps:")
    print("  1. Review failed cases in:", FAILED_CASES_FILE)
    print("  2. Run split_dataset.py to split cases into train/val/test sets:")
    print(f"     python split_dataset.py --main_path {MAIN_PATH} --output_path <output_path>")
    print("=" * 70)


if __name__ == '__main__':
    main()

