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
    
    # Update license field in dataset.json
    dataset_json_path = os.path.join(config.dataset_folder, 'dataset.json')
    if os.path.exists(dataset_json_path):
        with open(dataset_json_path, 'r') as f:
            dataset_json = json.load(f)
        
        dataset_json['licence'] = "Of course I checked! I'm not lazy"
        
        with open(dataset_json_path, 'w') as f:
            json.dump(dataset_json, f, indent=2)
    
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
    
    # Verify the mapping was written correctly
    with open(mapping_file, 'r') as f:
        verify_mapping = json.load(f)
    
    if str(config.dataset_id) not in verify_mapping:
        raise RuntimeError(f"Failed to create dataset mapping for ID {config.dataset_id}")
    
    if verify_mapping[str(config.dataset_id)] != config.dataset_name:
        raise RuntimeError(
            f"Mapping mismatch: ID {config.dataset_id} maps to '{verify_mapping[str(config.dataset_id)]}' "
            f"but expected '{config.dataset_name}'"
        )
    
    print(f"✓ Dataset mapping updated: {config.dataset_id} -> {config.dataset_name}")
    print(f"  Mapping file: {mapping_file}")
    print(f"  Verified: Mapping contains {len(verify_mapping)} dataset(s)")


def copy_dataset_to_nnunet_raw(config: TrainingConfig, overwrite: bool = False):
    """
    Copy dataset to nnUNet_raw folder.
    
    Parameters
    ----------
    config : TrainingConfig
        Configuration object
    overwrite : bool
        If True, overwrite existing dataset without asking
    """
    nnunet_raw_path = os.environ["nnUNet_raw"]
    target_path = os.path.join(nnunet_raw_path, config.dataset_name)
    
    if os.path.exists(target_path):
        if not overwrite:
            print(f"⚠️  Dataset already exists in nnUNet_raw: {target_path}")
            response = input("Do you want to overwrite? (yes/no): ")
            if response.lower() != 'yes':
                print("Skipping dataset copy.")
                return
        print(f"Removing existing dataset at {target_path}...")
        shutil.rmtree(target_path)
    
    print(f"Copying dataset to nnUNet_raw...")
    print(f"  From: {config.dataset_folder}")
    print(f"  To: {target_path}")
    
    if not os.path.exists(config.dataset_folder):
        raise FileNotFoundError(
            f"Source dataset folder not found: {config.dataset_folder}\n"
            f"Please check your DATASET_FOLDER environment variable."
        )
    
    shutil.copytree(config.dataset_folder, target_path)
    
    # Verify the copy was successful
    if not os.path.exists(os.path.join(target_path, 'dataset.json')):
        raise RuntimeError(
            f"Dataset copy incomplete: dataset.json not found in {target_path}\n"
            f"Please check that the source dataset folder contains dataset.json"
        )
    
    print(f"✓ Dataset copied to {target_path}")
    print(f"✓ Dataset ID {config.dataset_id} -> {config.dataset_name}")


def verify_and_fix_file_naming(config: TrainingConfig):
    """
    Verify and fix file naming convention for nnUNet.
    
    nnUNet expects:
    - Images: case_XXXX_0000.nii.gz (with _0000 suffix)
    - Labels: case_XXXX.nii.gz (without _0000 suffix)
    
    Parameters
    ----------
    config : TrainingConfig
        Configuration object
    """
    paths = config.get_dataset_paths()
    
    print("\nVerifying file naming convention...")
    
    # Check images
    for folder_name, folder_path in [
        ('imagesTr', paths['imagesTr']),
        ('imagesTs', paths['imagesTs']),
        ('imagesVa', paths['imagesVa'])
    ]:
        if not os.path.exists(folder_path):
            continue
        
        files = [f for f in os.listdir(folder_path) if f.endswith('.nii.gz')]
        for file in files:
            if not file.endswith('_0000.nii.gz'):
                print(f"⚠️  Warning: Image file {file} in {folder_name} should end with '_0000.nii.gz'")
    
    # Check and fix labels
    renamed_count = 0
    for folder_name, folder_path in [
        ('labelsTr', paths['labelsTr']),
        ('labelsTs', paths['labelsTs']),
        ('labelsVa', paths['labelsVa'])
    ]:
        if not os.path.exists(folder_path):
            continue
        
        files = [f for f in os.listdir(folder_path) if f.endswith('.nii.gz')]
        for file in files:
            if file.endswith('_0000.nii.gz'):
                # Rename to remove _0000 suffix
                old_path = os.path.join(folder_path, file)
                new_name = file.replace('_0000.nii.gz', '.nii.gz')
                new_path = os.path.join(folder_path, new_name)
                
                if os.path.exists(new_path):
                    print(f"⚠️  Warning: Both {file} and {new_name} exist in {folder_name}, skipping rename")
                else:
                    os.rename(old_path, new_path)
                    print(f"✓ Renamed: {file} -> {new_name} in {folder_name}")
                    renamed_count += 1
    
    if renamed_count > 0:
        print(f"\n✓ Fixed {renamed_count} label file(s) by removing '_0000' suffix")
    else:
        print("✓ All files have correct naming convention")


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
    
    # Verify and fix file naming
    verify_and_fix_file_naming(config)
    
    # Generate dataset.json
    generate_dataset_json(config)
    
    # Create dataset mapping
    create_dataset_mapping(config)
    
    # Copy dataset to nnUNet_raw (required for nnUNet to find the dataset)
    print("\nCopying dataset to nnUNet_raw folder (required)...")
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

