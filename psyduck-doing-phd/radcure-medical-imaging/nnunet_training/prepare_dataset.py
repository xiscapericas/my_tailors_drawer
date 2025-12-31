"""
Prepare dataset for nnUNet training.

This module handles:
- Generating dataset.json file
- Creating dataset ID to name mapping
- Setting up dataset structure in nnUNet_raw folder
"""

import os
import sys
import json
import shutil
from pathlib import Path
from nnunet_training.config import TrainingConfig


def add_nnunet_to_path(nnunet_path: str):
    """Add nnUNet to Python path."""
    if nnunet_path not in sys.path:
        sys.path.append(nnunet_path)


def generate_dataset_json(config: TrainingConfig):
    """
    Generate dataset.json file for nnUNet.
    
    Parameters
    ----------
    config : TrainingConfig
        Configuration object
    """
    from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
    
    paths = config.get_dataset_paths()
    imagesTr_dir = paths['imagesTr']
    
    if not os.path.exists(imagesTr_dir):
        raise FileNotFoundError(f"Training images folder not found: {imagesTr_dir}")
    
    num_training_cases = len([f for f in os.listdir(imagesTr_dir) if f.endswith('.nii.gz')])
    
    if num_training_cases == 0:
        raise ValueError(f"No training cases found in {imagesTr_dir}")
    
    print(f"Generating dataset.json for {config.dataset_name}...")
    print(f"  Training cases: {num_training_cases}")
    print(f"  Labels: {len(config.labels)}")
    
    generate_dataset_json(
        output_folder=config.dataset_folder,
        channel_names={0: "CT"},
        labels=config.labels,
        num_training_cases=num_training_cases,
        file_ending=".nii.gz",
        dataset_name=config.dataset_name,
        converted_by='Xisca Pe'
    )
    
    print(f"✓ dataset.json created in {config.dataset_folder}")


def create_dataset_mapping(config: TrainingConfig):
    """
    Create or update dataset ID to name mapping.
    
    Parameters
    ----------
    config : TrainingConfig
        Configuration object
    """
    mapping_file = os.path.join(
        config.nnunet_path,
        'nnunetv2/dataset_conversion/dataset_id_to_name_mapping.json'
    )
    
    # Load existing mapping if it exists
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
    else:
        mapping = {}
    
    # Add or update mapping
    mapping[str(config.dataset_id)] = config.dataset_name
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(mapping_file), exist_ok=True)
    
    # Write mapping
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=4)
    
    print(f"✓ Dataset mapping updated: {config.dataset_id} -> {config.dataset_name}")
    print(f"  Mapping file: {mapping_file}")


def copy_dataset_to_nnunet_raw(config: TrainingConfig):
    """
    Copy dataset to nnUNet_raw folder.
    
    Parameters
    ----------
    config : TrainingConfig
        Configuration object
    """
    nnunet_raw_path = os.environ["nnUNet_raw"]
    target_path = os.path.join(nnunet_raw_path, config.dataset_name)
    
    if os.path.exists(target_path):
        print(f"⚠️  Dataset already exists in nnUNet_raw: {target_path}")
        response = input("Do you want to overwrite? (yes/no): ")
        if response.lower() != 'yes':
            print("Skipping dataset copy.")
            return
        shutil.rmtree(target_path)
    
    print(f"Copying dataset to nnUNet_raw...")
    print(f"  From: {config.dataset_folder}")
    print(f"  To: {target_path}")
    
    shutil.copytree(config.dataset_folder, target_path)
    print(f"✓ Dataset copied to {target_path}")


def main():
    """Main function for dataset preparation."""
    print("=" * 70)
    print("nnUNet Dataset Preparation")
    print("=" * 70)
    
    # Load configuration
    config = TrainingConfig()
    
    # Add nnUNet to path
    add_nnunet_to_path(config.nnunet_path)
    
    # Setup environment
    config.setup_nnunet_environment()
    
    # Generate dataset.json
    generate_dataset_json(config)
    
    # Create dataset mapping
    create_dataset_mapping(config)
    
    # Copy dataset to nnUNet_raw (optional - can be done manually)
    copy_choice = input("\nCopy dataset to nnUNet_raw folder? (yes/no, default: yes): ").lower()
    if copy_choice != 'no':
        copy_dataset_to_nnunet_raw(config)
    
    print("\n" + "=" * 70)
    print("Dataset preparation complete!")
    print("=" * 70)
    print(f"Dataset ID: {config.dataset_id}")
    print(f"Dataset name: {config.dataset_name}")
    print(f"Dataset folder: {config.dataset_folder}")
    print("\nNext step: Run train_model.py to start training")


if __name__ == '__main__':
    main()

