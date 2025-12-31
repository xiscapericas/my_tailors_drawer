"""
Split processed RADCURE cases into Train/Validation/Test sets for TotalSegmentator retraining.

This script:
1. Finds all processed cases in the output directory
2. Splits cases randomly: 80% Train/Test, then 80/20 Train/Val within Train
3. Creates 6 folders: imagesTr, imagesTs, imagesVa, labelsTr, labelsTs, labelsVa
4. Moves files from case output folders to the appropriate split folders
"""

import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple
import argparse


def find_processed_cases(main_path: str) -> List[str]:
    """
    Find all processed cases that have output files.
    
    Parameters
    ----------
    main_path : str
        Main path containing TotalSegmentatorRetrain folder
    
    Returns
    -------
    List[str]
        List of case IDs that have been processed
    """
    retrain_path = os.path.join(main_path, 'TotalSegmentatorRetrain')
    
    if not os.path.exists(retrain_path):
        raise ValueError(f"TotalSegmentatorRetrain folder not found at {retrain_path}")
    
    processed_cases = []
    
    for case_folder in os.listdir(retrain_path):
        case_path = os.path.join(retrain_path, case_folder)
        
        # Check if it's a directory and has output folder
        if os.path.isdir(case_path):
            output_path = os.path.join(case_path, 'output')
            image_path = os.path.join(output_path, 'image')
            labels_path = os.path.join(output_path, 'labels')
            
            # Check if both image and labels folders exist and have files
            if (os.path.exists(image_path) and os.path.exists(labels_path)):
                image_files = [f for f in os.listdir(image_path) if f.endswith('.nii.gz')]
                label_files = [f for f in os.listdir(labels_path) if f.endswith('.nii.gz')]
                
                if image_files and label_files:
                    processed_cases.append(case_folder)
    
    return sorted(processed_cases)


def split_cases(
    cases: List[str],
    train_test_split: float = 0.8,
    train_val_split: float = 0.8,
    random_seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split cases into Train, Validation, and Test sets.
    
    Parameters
    ----------
    cases : List[str]
        List of case IDs
    train_test_split : float
        Ratio for train/test split (default: 0.8 = 80% train, 20% test)
    train_val_split : float
        Ratio for train/val split within train set (default: 0.8 = 80% train, 20% val)
    random_seed : int
        Random seed for reproducibility
    
    Returns
    -------
    Tuple[List[str], List[str], List[str]]
        (train_cases, val_cases, test_cases)
    """
    # Set random seed
    random.seed(random_seed)
    
    # Shuffle cases
    shuffled_cases = cases.copy()
    random.shuffle(shuffled_cases)
    
    # First split: Train/Test (80/20)
    split_idx = int(len(shuffled_cases) * train_test_split)
    train_val_cases = shuffled_cases[:split_idx]
    test_cases = shuffled_cases[split_idx:]
    
    # Second split: Train/Val within train set (80/20)
    val_split_idx = int(len(train_val_cases) * train_val_split)
    train_cases = train_val_cases[:val_split_idx]
    val_cases = train_val_cases[val_split_idx:]
    
    return train_cases, val_cases, test_cases


def create_split_folders(output_path: str) -> dict:
    """
    Create folders for train/val/test splits.
    
    Parameters
    ----------
    output_path : str
        Base path for split folders
    
    Returns
    -------
    dict
        Dictionary with paths to all split folders
    """
    folders = {
        'imagesTr': os.path.join(output_path, 'imagesTr'),
        'imagesTs': os.path.join(output_path, 'imagesTs'),
        'imagesVa': os.path.join(output_path, 'imagesVa'),
        'labelsTr': os.path.join(output_path, 'labelsTr'),
        'labelsTs': os.path.join(output_path, 'labelsTs'),
        'labelsVa': os.path.join(output_path, 'labelsVa'),
    }
    
    for folder_path in folders.values():
        os.makedirs(folder_path, exist_ok=True)
        print(f"Created folder: {folder_path}")
    
    return folders


def move_case_files(
    case_id: str,
    main_path: str,
    target_image_folder: str,
    target_label_folder: str,
    copy_files: bool = False
) -> bool:
    """
    Move or copy case files to target folders.
    
    Parameters
    ----------
    case_id : str
        Case ID (e.g., 'RADCURE-0005')
    main_path : str
        Main path containing TotalSegmentatorRetrain folder
    target_image_folder : str
        Target folder for image files
    target_label_folder : str
        Target folder for label files
    copy_files : bool
        If True, copy files instead of moving
    
    Returns
    -------
    bool
        True if successful
    """
    retrain_path = os.path.join(main_path, 'TotalSegmentatorRetrain')
    case_path = os.path.join(retrain_path, case_id)
    output_path = os.path.join(case_path, 'output')
    
    image_source = os.path.join(output_path, 'image')
    label_source = os.path.join(output_path, 'labels')
    
    if not os.path.exists(image_source) or not os.path.exists(label_source):
        print(f"Warning: Case {case_id} missing output folders")
        return False
    
    # Find image and label files
    image_files = [f for f in os.listdir(image_source) if f.endswith('.nii.gz')]
    label_files = [f for f in os.listdir(label_source) if f.endswith('.nii.gz')]
    
    if not image_files or not label_files:
        print(f"Warning: Case {case_id} has no files in output folders")
        return False
    
    # Move/copy image file
    # Images should have _0000 suffix (e.g., case_0006_0000.nii.gz)
    image_file = image_files[0]  # Should be only one file
    image_source_path = os.path.join(image_source, image_file)
    image_target_path = os.path.join(target_image_folder, image_file)
    
    if copy_files:
        shutil.copy2(image_source_path, image_target_path)
    else:
        shutil.move(image_source_path, image_target_path)
    
    # Move/copy label file
    # Labels should NOT have _0000 suffix (e.g., case_0006.nii.gz)
    # nnUNet expects labels without channel suffix
    label_file = label_files[0]  # Should be only one file
    label_source_path = os.path.join(label_source, label_file)
    
    # Remove _0000 suffix from label filename if present
    if label_file.endswith('_0000.nii.gz'):
        label_target_name = label_file.replace('_0000.nii.gz', '.nii.gz')
    else:
        label_target_name = label_file
    
    label_target_path = os.path.join(target_label_folder, label_target_name)
    
    if copy_files:
        shutil.copy2(label_source_path, label_target_path)
    else:
        shutil.move(label_source_path, label_target_path)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Split processed RADCURE cases into Train/Validation/Test sets'
    )
    parser.add_argument(
        '--main_path',
        type=str,
        required=True,
        help='Main path containing TotalSegmentatorRetrain folder'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Output path where DatasetXXX_TotalSegmentator folder will be created'
    )
    parser.add_argument(
        '--train_test_split',
        type=float,
        default=0.8,
        help='Ratio for train/test split (default: 0.8)'
    )
    parser.add_argument(
        '--train_val_split',
        type=float,
        default=0.8,
        help='Ratio for train/val split within train (default: 0.8)'
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--copy',
        action='store_true',
        help='Copy files instead of moving them'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Show what would be done without actually moving files'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("RADCURE Dataset Splitter")
    print("=" * 60)
    print(f"Main path: {args.main_path}")
    print(f"Output path: {args.output_path}")
    print(f"Train/Test split: {args.train_test_split}")
    print(f"Train/Val split: {args.train_val_split}")
    print(f"Random seed: {args.random_seed}")
    print(f"Mode: {'COPY' if args.copy else 'MOVE'}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 60)
    
    # Find processed cases
    print("\nFinding processed cases...")
    cases = find_processed_cases(args.main_path)
    print(f"Found {len(cases)} processed cases")
    
    if len(cases) == 0:
        print("No processed cases found. Exiting.")
        return
    
    # Create dataset folder with case count
    dataset_folder_name = f"Dataset{len(cases)}_TotalSegmentator"
    dataset_path = os.path.join(args.output_path, dataset_folder_name)
    
    # Split cases
    print("\nSplitting cases...")
    train_cases, val_cases, test_cases = split_cases(
        cases,
        train_test_split=args.train_test_split,
        train_val_split=args.train_val_split,
        random_seed=args.random_seed
    )
    
    print(f"Train cases: {len(train_cases)} ({100*len(train_cases)/len(cases):.1f}%)")
    print(f"Validation cases: {len(val_cases)} ({100*len(val_cases)/len(cases):.1f}%)")
    print(f"Test cases: {len(test_cases)} ({100*len(test_cases)/len(cases):.1f}%)")
    
    if args.dry_run:
        print("\n=== DRY RUN - No files will be moved ===")
        print(f"\nDataset folder would be created: {dataset_path}")
        print(f"\nTrain cases ({len(train_cases)}):")
        for case in train_cases[:10]:
            print(f"  - {case}")
        if len(train_cases) > 10:
            print(f"  ... and {len(train_cases) - 10} more")
        
        print(f"\nValidation cases ({len(val_cases)}):")
        for case in val_cases[:10]:
            print(f"  - {case}")
        if len(val_cases) > 10:
            print(f"  ... and {len(val_cases) - 10} more")
        
        print(f"\nTest cases ({len(test_cases)}):")
        for case in test_cases[:10]:
            print(f"  - {case}")
        if len(test_cases) > 10:
            print(f"  ... and {len(test_cases) - 10} more")
        return
    
    # Create dataset folder
    print(f"\nCreating dataset folder: {dataset_path}")
    os.makedirs(dataset_path, exist_ok=True)
    
    # Create split folders
    print("\nCreating split folders...")
    folders = create_split_folders(dataset_path)
    
    # Move/copy files
    print("\nMoving/copying files...")
    
    # Train cases
    print(f"\nProcessing {len(train_cases)} train cases...")
    train_success = 0
    for case_id in train_cases:
        if move_case_files(
            case_id,
            args.main_path,
            folders['imagesTr'],
            folders['labelsTr'],
            copy_files=args.copy
        ):
            train_success += 1
    
    # Validation cases
    print(f"\nProcessing {len(val_cases)} validation cases...")
    val_success = 0
    for case_id in val_cases:
        if move_case_files(
            case_id,
            args.main_path,
            folders['imagesVa'],
            folders['labelsVa'],
            copy_files=args.copy
        ):
            val_success += 1
    
    # Test cases
    print(f"\nProcessing {len(test_cases)} test cases...")
    test_success = 0
    for case_id in test_cases:
        if move_case_files(
            case_id,
            args.main_path,
            folders['imagesTs'],
            folders['labelsTs'],
            copy_files=args.copy
        ):
            test_success += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Train: {train_success}/{len(train_cases)} cases processed")
    print(f"Validation: {val_success}/{len(val_cases)} cases processed")
    print(f"Test: {test_success}/{len(test_cases)} cases processed")
    print(f"Total: {train_success + val_success + test_success}/{len(cases)} cases processed")
    print(f"\nDataset folder: {dataset_path}")
    print("\nSplit folders created at:")
    for name, path in folders.items():
        print(f"  {name}: {path}")
    print("=" * 60)


if __name__ == '__main__':
    main()

