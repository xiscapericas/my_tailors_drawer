"""
Train nnUNet model.

This module handles:
- Planning and preprocessing
- Model training
"""

import os
import subprocess
import sys
import json

# Try to load from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from nnunet_training.config import TrainingConfig
from nnunet_training.install_trainer_variants import ensure_trainer_installed
from nnunet_training.splits_utils import ensure_splits_final


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
    compile_setting = os.environ.get("nnUNet_compile", "(not set — nnUNet default, usually compile ON)")
    print(f"  nnUNet_compile: {compile_setting}")
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "(not set — all GPUs visible)")
    print(f"  CUDA_VISIBLE_DEVICES: {cuda_devices}")
    if os.environ.get("nnUNet_preprocessed"):
        print(f"  nnUNet_preprocessed: {os.environ['nnUNet_preprocessed']}")
    
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


def verify_dataset_in_nnunet_raw(config: TrainingConfig):
    """
    Verify that dataset exists in nnUNet_raw folder and mapping is correct.
    
    Parameters
    ----------
    config : TrainingConfig
        Configuration object
    
    Raises
    ------
    FileNotFoundError
        If dataset is not found in nnUNet_raw
    """
    nnunet_raw_path = os.environ["nnUNet_raw"]
    expected_path = os.path.join(nnunet_raw_path, config.dataset_name)
    
    # Check if nnUNet_raw exists
    if not os.path.exists(nnunet_raw_path):
        raise FileNotFoundError(
            f"nnUNet_raw folder not found: {nnunet_raw_path}\n"
            f"Please check your NNUNET_RETRAIN_PATH environment variable."
        )
    
    # Check if dataset folder exists
    if not os.path.exists(expected_path):
        # List available datasets for debugging
        available = []
        if os.path.exists(nnunet_raw_path):
            available = [d for d in os.listdir(nnunet_raw_path) 
                        if os.path.isdir(os.path.join(nnunet_raw_path, d)) and d.startswith('Dataset')]
        
        error_msg = (
            f"Dataset not found in nnUNet_raw:\n"
            f"  Expected: {expected_path}\n"
            f"  nnUNet_raw: {nnunet_raw_path}\n"
            f"  Dataset ID: {config.dataset_id}\n"
            f"  Dataset name: {config.dataset_name}\n"
        )
        
        if available:
            error_msg += f"\n  Available datasets in nnUNet_raw: {', '.join(available)}\n"
        else:
            error_msg += f"\n  No datasets found in nnUNet_raw folder.\n"
        
        error_msg += f"\nPlease run 'python train_nnunet.py --step prepare' first to copy the dataset."
        
        raise FileNotFoundError(error_msg)
    
    # Verify dataset.json exists
    dataset_json_path = os.path.join(expected_path, 'dataset.json')
    if not os.path.exists(dataset_json_path):
        raise FileNotFoundError(
            f"dataset.json not found in {expected_path}\n"
            f"Please run 'python train_nnunet.py --step prepare' first."
        )
    
    # Verify mapping file
    mapping_file = os.path.join(
        config.nnunet_path,
        'nnunetv2/dataset_conversion/dataset_id_to_name_mapping.json'
    )
    if os.path.exists(mapping_file):
        import json
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
        if str(config.dataset_id) in mapping:
            print(f"✓ Dataset mapping verified: {config.dataset_id} -> {mapping[str(config.dataset_id)]}")
        else:
            print(f"⚠️  Warning: Dataset ID {config.dataset_id} not found in mapping file")
            print(f"  Mapping file: {mapping_file}")
            print(f"  Available IDs: {list(mapping.keys())}")
    
    print(f"✓ Dataset verified in nnUNet_raw: {expected_path}")


def verify_preprocessed_for_training(config: TrainingConfig):
    """
    Fail fast if preprocessed .b2nd files are missing for the training fold.

    Common cause: NNUNET_PREPROCESSED_PATH still points at an older retrain run
    (e.g. Test1/Test3) while DATASET_FOLDER holds a new dataset (Test4).
    """
    preproc_root = os.environ["nnUNet_preprocessed"]
    dataset_preproc = os.path.join(preproc_root, config.dataset_name)
    plans_dir = os.path.join(dataset_preproc, f"nnUNetPlans_{config.configuration}")
    splits_path = os.path.join(dataset_preproc, "splits_final.json")

    if config.preprocessed_path:
        expected_local = os.path.join(config.main_retrain_path, "nnUNet_preprocessed")
        if os.path.abspath(config.preprocessed_path) != os.path.abspath(expected_local):
            print(
                "⚠️  NNUNET_PREPROCESSED_PATH is set to an external folder:\n"
                f"     {config.preprocessed_path}\n"
                f"   NNUNET_RETRAIN_PATH: {config.main_retrain_path}\n"
                "   Reusing old preprocess is OK only when labels are unchanged "
                "(e.g. Test3 reusing Test2). For Test4, unset NNUNET_PREPROCESSED_PATH."
            )

    if not os.path.isdir(plans_dir):
        raise FileNotFoundError(
            f"Preprocessed plans folder not found:\n  {plans_dir}\n"
            "Run: python train_nnunet.py --step plan"
        )

    if not os.path.isfile(splits_path):
        raise FileNotFoundError(
            f"splits_final.json not found:\n  {splits_path}\n"
            "This should have been created automatically before training. "
            "Re-run: python train_nnunet.py --step train"
        )

    with open(splits_path) as f:
        splits = json.load(f)

    if config.fold >= len(splits):
        raise ValueError(
            f"Fold {config.fold} out of range; splits_final.json has {len(splits)} folds"
        )

    fold_cases = splits[config.fold]["train"] + splits[config.fold]["val"]
    missing = [
        case
        for case in fold_cases
        if not os.path.isfile(os.path.join(plans_dir, f"{case}.b2nd"))
    ]
    if missing:
        sample = ", ".join(missing[:5])
        extra = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        raise FileNotFoundError(
            f"{len(missing)} preprocessed case(s) missing under:\n  {plans_dir}\n"
            f"  Examples: {sample}{extra}\n"
            f"  nnUNet_preprocessed in use: {preproc_root}\n\n"
            "Likely fix:\n"
            "  1. export NNUNET_USE_LOCAL_PREPROCESS=1   # overrides .env reuse path\n"
            "     (or comment out NNUNET_PREPROCESSED_PATH in .env)\n"
            "  2. export NNUNET_RETRAIN_PATH to this experiment's retrain folder\n"
            "  3. python train_nnunet.py --step prepare --link-raw\n"
            "  4. python train_nnunet.py --step plan\n"
            "  5. python train_nnunet.py --step train"
        )

    print(
        f"✓ Preprocessed data verified: {len(fold_cases)} cases for fold {config.fold} "
        f"in {plans_dir}"
    )


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
    
    # Verify dataset exists in nnUNet_raw
    print("\nVerifying dataset in nnUNet_raw...")
    verify_dataset_in_nnunet_raw(config)
    
    # Plan and preprocess
    plan_and_preprocess(config)

    print("\nEnsuring splits_final.json exists...")
    ensure_splits_final(
        config.dataset_folder,
        os.environ["nnUNet_preprocessed"],
        config.dataset_name,
    )
    
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

    if config.trainer.startswith("nnUNetTrainer_") and config.trainer.endswith("_NoMirroring"):
        print("\nEnsuring custom trainer variants are installed in nnUNet...")
        ensure_trainer_installed(config.trainer, config.nnunet_path)
    
    # Setup environment
    config.setup_nnunet_environment()

    print("\nEnsuring splits_final.json exists...")
    ensure_splits_final(
        config.dataset_folder,
        os.environ["nnUNet_preprocessed"],
        config.dataset_name,
    )

    print("\nVerifying preprocessed data before training...")
    verify_preprocessed_for_training(config)
    
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

