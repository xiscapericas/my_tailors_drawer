"""
One-time script to restore files from split dataset folders back to original case output folders.

This script moves files from:
- DatasetXXX_TotalSegmentator/imagesTr, imagesTs, imagesVa
- DatasetXXX_TotalSegmentator/labelsTr, labelsTs, labelsVa

Back to:
- TotalSegmentatorRetrain/RADCURE-XXXX/output/image/
- TotalSegmentatorRetrain/RADCURE-XXXX/output/labels/

Note: Labels will have the _0000 suffix restored.
"""

import os
import shutil
import re
import argparse
from pathlib import Path


def extract_case_id_from_filename(filename: str) -> str:
    """
    Extract RADCURE case ID from filename.
    
    Parameters
    ----------
    filename : str
        Filename like 'case_0006_0000.nii.gz' or 'case_0006.nii.gz'
    
    Returns
    -------
    str
        Case ID like 'RADCURE-0006'
    """
    # Match pattern: case_XXXX or case_XXXX_0000
    match = re.match(r'case_(\d+)', filename)
    if match:
        case_num = match.group(1)
        # Pad with zeros if needed (should already be 4 digits)
        case_num = case_num.zfill(4)
        return f'RADCURE-{case_num}'
    else:
        raise ValueError(f"Could not extract case ID from filename: {filename}")


def restore_files_from_split(
    dataset_folder: str,
    main_path: str,
    dry_run: bool = False
) -> dict:
    """
    Restore files from split dataset folders back to original case output folders.
    
    Parameters
    ----------
    dataset_folder : str
        Path to DatasetXXX_TotalSegmentator folder
    main_path : str
        Main path containing TotalSegmentatorRetrain folder
    dry_run : bool
        If True, only show what would be done without actually moving files
    
    Returns
    -------
    dict
        Statistics about restored files
    """
    stats = {
        'images_restored': 0,
        'labels_restored': 0,
        'images_failed': 0,
        'labels_failed': 0,
        'cases_not_found': []
    }
    
    # Define split folders
    split_folders = {
        'imagesTr': 'imagesTr',
        'imagesTs': 'imagesTs',
        'imagesVa': 'imagesVa',
        'labelsTr': 'labelsTr',
        'labelsTs': 'labelsTs',
        'labelsVa': 'labelsVa',
    }
    
    retrain_path = os.path.join(main_path, 'TotalSegmentatorRetrain')
    
    if not os.path.exists(retrain_path):
        raise ValueError(f"TotalSegmentatorRetrain folder not found at {retrain_path}")
    
    print("=" * 70)
    print("Restoring Files from Split Dataset")
    print("=" * 70)
    print(f"Dataset folder: {dataset_folder}")
    print(f"Main path: {main_path}")
    print(f"Mode: {'DRY RUN' if dry_run else 'RESTORE'}")
    print("=" * 70)
    
    # Process each split folder
    for folder_key, folder_name in split_folders.items():
        split_folder = os.path.join(dataset_folder, folder_name)
        
        if not os.path.exists(split_folder):
            print(f"\n⚠️  Folder not found: {split_folder}, skipping...")
            continue
        
        files = [f for f in os.listdir(split_folder) if f.endswith('.nii.gz')]
        
        if not files:
            print(f"\n✓ No files in {folder_name}")
            continue
        
        print(f"\nProcessing {folder_name} ({len(files)} files)...")
        
        # Determine if it's an image or label folder
        is_image = folder_key.startswith('images')
        
        for filename in files:
            try:
                # Extract case ID
                case_id = extract_case_id_from_filename(filename)
                
                # Build original paths
                case_path = os.path.join(retrain_path, case_id)
                output_path = os.path.join(case_path, 'output')
                
                if not os.path.exists(output_path):
                    print(f"  ⚠️  Case folder not found: {case_path}")
                    if case_id not in stats['cases_not_found']:
                        stats['cases_not_found'].append(case_id)
                    if is_image:
                        stats['images_failed'] += 1
                    else:
                        stats['labels_failed'] += 1
                    continue
                
                # Determine target folder and filename
                if is_image:
                    target_folder = os.path.join(output_path, 'image')
                    target_filename = filename  # Keep _0000 suffix
                else:
                    target_folder = os.path.join(output_path, 'labels')
                    # Restore _0000 suffix for labels
                    if not filename.endswith('_0000.nii.gz'):
                        target_filename = filename.replace('.nii.gz', '_0000.nii.gz')
                    else:
                        target_filename = filename
                
                source_path = os.path.join(split_folder, filename)
                target_path = os.path.join(target_folder, target_filename)
                
                if not os.path.exists(target_folder):
                    if not dry_run:
                        os.makedirs(target_folder, exist_ok=True)
                    print(f"  Created folder: {target_folder}")
                
                if os.path.exists(target_path):
                    print(f"  ⚠️  Target already exists: {target_path}, skipping...")
                    if is_image:
                        stats['images_failed'] += 1
                    else:
                        stats['labels_failed'] += 1
                    continue
                
                if dry_run:
                    print(f"  Would restore: {filename} -> {case_id}/{target_filename}")
                else:
                    shutil.move(source_path, target_path)
                    print(f"  ✓ Restored: {filename} -> {case_id}/{target_filename}")
                
                if is_image:
                    stats['images_restored'] += 1
                else:
                    stats['labels_restored'] += 1
                    
            except Exception as e:
                print(f"  ✗ Error processing {filename}: {e}")
                if is_image:
                    stats['images_failed'] += 1
                else:
                    stats['labels_failed'] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Restore files from split dataset folders back to original case output folders'
    )
    parser.add_argument(
        '--dataset_folder',
        type=str,
        required=True,
        help='Path to DatasetXXX_TotalSegmentator folder'
    )
    parser.add_argument(
        '--main_path',
        type=str,
        required=True,
        help='Main path containing TotalSegmentatorRetrain folder'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Show what would be done without actually moving files'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_folder):
        print(f"Error: Dataset folder not found: {args.dataset_folder}")
        return
    
    try:
        stats = restore_files_from_split(
            args.dataset_folder,
            args.main_path,
            dry_run=args.dry_run
        )
        
        # Print summary
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        print(f"Images restored: {stats['images_restored']}")
        print(f"Labels restored: {stats['labels_restored']}")
        print(f"Images failed: {stats['images_failed']}")
        print(f"Labels failed: {stats['labels_failed']}")
        
        if stats['cases_not_found']:
            print(f"\nCases not found ({len(stats['cases_not_found'])}):")
            for case_id in sorted(stats['cases_not_found']):
                print(f"  - {case_id}")
        
        total_restored = stats['images_restored'] + stats['labels_restored']
        total_failed = stats['images_failed'] + stats['labels_failed']
        print(f"\nTotal: {total_restored} restored, {total_failed} failed")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise


if __name__ == '__main__':
    main()

