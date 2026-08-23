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
from typing import Optional

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


def _nnunet_channel_index(filename: str) -> Optional[int]:
    """Return channel index from ``stem_0001.nii.gz``, or None."""
    if not filename.endswith(".nii.gz"):
        return None
    stem = filename[: -len(".nii.gz")]
    if "_" not in stem:
        return None
    suffix = stem.rsplit("_", 1)[-1]
    if suffix.isdigit() and len(suffix) == 4:
        return int(suffix)
    return None


def _count_training_cases(imagesTr_dir: str) -> int:
    """Unique nnUNet cases (``*_0000.nii.gz``), not one count per channel file."""
    names = [f for f in os.listdir(imagesTr_dir) if f.endswith(".nii.gz")]
    n0000 = sum(1 for f in names if f.endswith("_0000.nii.gz"))
    if n0000 > 0:
        return n0000
    # Fallback: unique stem without channel suffix
    stems = set()
    for f in names:
        ch = _nnunet_channel_index(f)
        if ch is None:
            stems.add(f[: -len(".nii.gz")])
        else:
            stems.add(f[: -len(f"_{ch:04d}.nii.gz")])
    return len(stems)


def _channel_names_from_images(imagesTr_dir: str) -> dict:
    channels = set()
    for f in os.listdir(imagesTr_dir):
        ch = _nnunet_channel_index(f)
        if ch is not None:
            channels.add(ch)
    default = {0: "CT", 1: "PET"}
    if not channels:
        return {0: "CT"}
    return {i: default.get(i, f"channel_{i}") for i in sorted(channels)}


def _case_channels_in_images_dir(images_dir: str) -> dict:
    """Map nnUNet case stem → set of channel indices present (``_0000``, ``_0001``, …)."""
    by_stem = {}
    if not os.path.isdir(images_dir):
        return by_stem
    for f in os.listdir(images_dir):
        ch = _nnunet_channel_index(f)
        if ch is None:
            continue
        stem = f[: -len(f"_{ch:04d}.nii.gz")]
        by_stem.setdefault(stem, set()).add(ch)
    return by_stem


def find_incomplete_nnunet_channel_cases(
    images_dir: str,
    required_channels=None,
):
    """
    Stems that lack a channel file every other case (or ``required_channels``) has.

    nnUNet fingerprint extraction IndexErrors if some cases have CT-only and
    others have CT+PET (``r[2][i]`` length differs).
    """
    by_stem = _case_channels_in_images_dir(images_dir)
    if not by_stem:
        return []
    if required_channels is None:
        required = sorted(set().union(*by_stem.values()))
    else:
        required = list(required_channels)
    missing = []
    for stem in sorted(by_stem):
        miss = [c for c in required if c not in by_stem[stem]]
        if miss:
            missing.append((stem, miss))
    return missing


def assert_nnunet_channels_complete(images_dir: str, required_channels=None) -> None:
    missing = find_incomplete_nnunet_channel_cases(images_dir, required_channels)
    if not missing:
        return
    n = len(missing)
    preview = "\n".join(
        f"  {stem}: missing " + ", ".join(f"_{c:04d}" for c in miss)
        for stem, miss in missing[:25]
    )
    more = f"\n  … and {n - 25} more" if n > 25 else ""
    raise ValueError(
        f"{n} case(s) in {images_dir} do not have every channel file "
        f"(this causes nnUNet fingerprint IndexError).\n{preview}{more}\n"
        "For Test 8.0: python -m pipelines.test8_0.verify_channels\n"
        "Then: python -m pipelines.test8_0.build_dataset --only-missing-pet"
    )


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
    
    num_training_cases = _count_training_cases(imagesTr_dir)
    
    if num_training_cases == 0:
        raise ValueError(f"No training cases found in {imagesTr_dir}")

    channel_names = _channel_names_from_images(imagesTr_dir)
    assert_nnunet_channels_complete(imagesTr_dir)
    
    print(f"Generating dataset.json for {config.dataset_name}...")
    print(f"  Training cases: {num_training_cases}")
    print(f"  Channels: {channel_names}")
    print(f"  Labels: {len(config.labels)}")
    
    generate_dataset_json(
        output_folder=config.dataset_folder,
        channel_names=channel_names,
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


def copy_dataset_to_nnunet_raw(
    config: TrainingConfig,
    overwrite: bool = False,
    use_symlink: bool = False,
):
    """
    Register dataset in nnUNet_raw (copy or symlink).

    Parameters
    ----------
    config : TrainingConfig
        Configuration object
    overwrite : bool
        If True, overwrite existing dataset without asking
    use_symlink : bool
        If True, symlink DATASET_FOLDER into nnUNet_raw (set env NNUNET_LINK_RAW=1).
    """
    use_symlink = use_symlink or os.getenv("NNUNET_LINK_RAW", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    nnunet_raw_path = os.environ["nnUNet_raw"]
    target_path = os.path.join(nnunet_raw_path, config.dataset_name)
    src = os.path.abspath(config.dataset_folder)

    if not os.path.exists(config.dataset_folder):
        raise FileNotFoundError(
            f"Source dataset folder not found: {config.dataset_folder}\n"
            f"Please check your DATASET_FOLDER environment variable."
        )

    if os.path.lexists(target_path):
        if os.path.islink(target_path) and os.path.realpath(target_path) == src:
            print(f"✓ Dataset already linked in nnUNet_raw: {target_path} -> {src}")
            return
        if not overwrite:
            print(f"⚠️  Dataset already exists in nnUNet_raw: {target_path}")
            response = input("Do you want to overwrite? (yes/no): ")
            if response.lower() != "yes":
                print("Skipping dataset registration in nnUNet_raw.")
                return
        print(f"Removing existing entry at {target_path}...")
        if os.path.islink(target_path):
            os.unlink(target_path)
        else:
            shutil.rmtree(target_path)

    if use_symlink:
        print("Linking dataset into nnUNet_raw (no copy)...")
        print(f"  From: {src}")
        print(f"  To:   {target_path}")
        os.symlink(src, target_path)
    else:
        print("Copying dataset to nnUNet_raw...")
        print(f"  From: {config.dataset_folder}")
        print(f"  To: {target_path}")
        shutil.copytree(config.dataset_folder, target_path)

    if not os.path.exists(os.path.join(target_path, "dataset.json")):
        raise RuntimeError(
            f"dataset.json not found in {target_path}\n"
            f"Please check that the source dataset folder contains dataset.json"
        )

    action = "linked" if use_symlink else "copied"
    print(f"✓ Dataset {action} to {target_path}")
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
            if _nnunet_channel_index(file) is None:
                print(f"⚠️  Warning: Image file {file} in {folder_name} should end with '_XXXX.nii.gz'")
    
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

