"""
Train nnUNet model.

This module handles:
- Planning and preprocessing
- Model training
"""

import os
import subprocess
import sys

# Try to load from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from nnunet_training.config import TrainingConfig


def add_nnunet_to_path(nnunet_path: str):
    """Add nnUNet to Python path."""
    if nnunet_path not in sys.path:
        sys.path.append(nnunet_path)


def plan_and_preprocess(config: TrainingConfig, log_file: str = None):
    """
    Plan and preprocess dataset for nnUNet.
    
    Parameters
    ----------
    config : TrainingConfig
        Configuration object
    log_file : str, optional
        Path to log file. If None, uses default.
    """
    if log_file is None:
        log_file = os.path.join(config.log_dir, f'preprocess_d{config.dataset_id}.log')
    
    print(f"Planning and preprocessing dataset {config.dataset_id}...")
    print(f"  Configuration: {config.configuration}")
    print(f"  Number of processes: {config.num_processes}")
    print(f"  Log file: {log_file}")
    
    cmd = [
        'nnUNetv2_plan_and_preprocess',
        '-d', str(config.dataset_id),
        '-pl', 'ExperimentPlanner',
        '-c', config.configuration,
        '-np', str(config.num_processes)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    with open(log_file, 'w') as log:
        result = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True
        )
    
    if result.returncode == 0:
        print(f"✓ Planning and preprocessing completed successfully")
        print(f"  Log saved to: {log_file}")
    else:
        print(f"✗ Planning and preprocessing failed (exit code: {result.returncode})")
        print(f"  Check log file: {log_file}")
        raise RuntimeError(f"Preprocessing failed. Check {log_file} for details.")


def train_model(config: TrainingConfig, log_file: str = None):
    """
    Train nnUNet model.
    
    Parameters
    ----------
    config : TrainingConfig
        Configuration object
    log_file : str, optional
        Path to log file. If None, uses default.
    """
    if log_file is None:
        log_file = os.path.join(config.log_dir, f'train_d{config.dataset_id}_f{config.fold}.log')
    
    print(f"Training model for dataset {config.dataset_id}...")
    print(f"  Configuration: {config.configuration}")
    print(f"  Trainer: {config.trainer}")
    print(f"  Fold: {config.fold}")
    print(f"  Log file: {log_file}")
    
    cmd = [
        'nnUNetv2_train',
        str(config.dataset_id),
        config.configuration,
        str(config.fold),
        '-tr', config.trainer
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print("This may take a long time. Training in background...")
    
    with open(log_file, 'w') as log:
        result = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True
        )
    
    if result.returncode == 0:
        print(f"✓ Training completed successfully")
        print(f"  Log saved to: {log_file}")
    else:
        print(f"✗ Training failed (exit code: {result.returncode})")
        print(f"  Check log file: {log_file}")
        raise RuntimeError(f"Training failed. Check {log_file} for details.")


def main_plan():
    """Main function for planning and preprocessing."""
    print("=" * 70)
    print("nnUNet Planning and Preprocessing")
    print("=" * 70)
    
    # Load configuration
    config = TrainingConfig()
    
    # Add nnUNet to path
    add_nnunet_to_path(config.nnunet_path)
    
    # Setup environment
    config.setup_nnunet_environment()
    
    # Plan and preprocess
    plan_and_preprocess(config)
    
    print("\n" + "=" * 70)
    print("Planning and preprocessing complete!")
    print("=" * 70)
    print("\nNext step: Run train_nnunet.py --step train to start training")


def main_train():
    """Main function for model training."""
    print("=" * 70)
    print("nnUNet Model Training")
    print("=" * 70)
    
    # Load configuration
    config = TrainingConfig()
    
    # Add nnUNet to path
    add_nnunet_to_path(config.nnunet_path)
    
    # Setup environment
    config.setup_nnunet_environment()
    
    # Train model
    train_model(config)
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"Model saved in: {os.environ['nnUNet_results']}")
    print("\nNext step: Run train_nnunet.py --step evaluate to evaluate the model")


def main():
    """Main function that runs both plan and train (for backward compatibility)."""
    print("=" * 70)
    print("nnUNet Model Training (Plan + Train)")
    print("=" * 70)
    
    # Load configuration
    config = TrainingConfig()
    
    # Add nnUNet to path
    add_nnunet_to_path(config.nnunet_path)
    
    # Setup environment
    config.setup_nnunet_environment()
    
    # Plan and preprocess
    print("\nStep 1: Planning and preprocessing...")
    plan_and_preprocess(config)
    
    # Train model
    print("\nStep 2: Training model...")
    train_model(config)
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"Model saved in: {os.environ['nnUNet_results']}")
    print("\nNext step: Run train_nnunet.py --step evaluate to evaluate the model")


if __name__ == '__main__':
    main()

