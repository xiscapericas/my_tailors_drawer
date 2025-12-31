"""
Main script for nnUNet training pipeline.

This script orchestrates the complete training workflow:
1. Prepare dataset
2. Train model
3. Predict and evaluate

Usage:
    python train_nnunet.py --step prepare
    python train_nnunet.py --step train
    python train_nnunet.py --step evaluate
    python train_nnunet.py --step all  # Run all steps
"""

import argparse
import sys

# Try to load from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded environment variables from .env file")
except ImportError:
    print("Note: python-dotenv not installed. Using environment variables directly.")

from nnunet_training import prepare_dataset, train_model, predict_and_evaluate
from nnunet_training.config import TrainingConfig


def main():
    parser = argparse.ArgumentParser(
        description='nnUNet training pipeline for RADCURE dataset'
    )
    parser.add_argument(
        '--step',
        type=str,
        choices=['prepare', 'train', 'evaluate', 'all'],
        required=True,
        help='Which step to run: prepare, train, evaluate, or all'
    )
    
    args = parser.parse_args()
    
    # Load configuration (will validate environment variables)
    try:
        config = TrainingConfig()
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        print("\nRequired environment variables:")
        print("  - DATASET_FOLDER: Path to DatasetXXX_TotalSegmentator folder")
        print("  - ORGAN_DICTIONARY_PATH: Path to radcure_dictionary.json")
        print("  - NNUNET_PATH: Path to nnUNet installation (optional)")
        print("  - NNUNET_RETRAIN_PATH: Path for nnUNet raw/preprocessed/results (optional)")
        sys.exit(1)
    
    print("=" * 70)
    print("nnUNet Training Pipeline")
    print("=" * 70)
    print(f"Dataset: {config.dataset_name} (ID: {config.dataset_id})")
    print(f"Step: {args.step}")
    print("=" * 70)
    print()
    
    if args.step == 'prepare' or args.step == 'all':
        print(">>> STEP 1: Preparing Dataset")
        print("-" * 70)
        try:
            prepare_dataset.main()
        except Exception as e:
            print(f"✗ Dataset preparation failed: {e}")
            sys.exit(1)
        print()
    
    if args.step == 'train' or args.step == 'all':
        print(">>> STEP 2: Training Model")
        print("-" * 70)
        try:
            train_model.main()
        except Exception as e:
            print(f"✗ Training failed: {e}")
            sys.exit(1)
        print()
    
    if args.step == 'evaluate' or args.step == 'all':
        print(">>> STEP 3: Prediction and Evaluation")
        print("-" * 70)
        try:
            predict_and_evaluate.main()
        except Exception as e:
            print(f"✗ Evaluation failed: {e}")
            sys.exit(1)
        print()
    
    print("=" * 70)
    print("Pipeline complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()

