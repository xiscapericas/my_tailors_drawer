"""
Interactive DICOM slice browser.

This script allows you to easily browse through all slices of a DICOM CT series.
Use arrow keys or mouse wheel to navigate through slices.

Usage:
    python browse_dicom_slices.py --dicom_folder /path/to/dicom/folder
    python browse_dicom_slices.py --case_id RADCURE-0005 --main_path /path/to/dataset
"""

import argparse
import os
import sys
from pathlib import Path

# Try to load from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from image_processor.visualization import MedicalImageVisualizer
from image_processor.io.file_handler import FileHandler


class DicomSliceBrowser:
    """Interactive browser for DICOM slices."""
    
    def __init__(self, dicom_folder_path: str, window_level: tuple = None):
        """
        Initialize the browser.
        
        Parameters
        ----------
        dicom_folder_path : str
            Path to folder containing DICOM CT series
        window_level : tuple, optional
            (window, level) for windowing. If None, uses automatic windowing.
            Common values: (400, 50) for soft tissue, (1500, -500) for bone
        """
        self.dicom_folder_path = dicom_folder_path
        self.window_level = window_level
        
        # Load DICOM series
        print(f"Loading DICOM series from: {dicom_folder_path}")
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(dicom_folder_path)
        
        if not series_ids:
            raise ValueError(f"No DICOM series found in {dicom_folder_path}")
        
        series_file_names = reader.GetGDCMSeriesFileNames(dicom_folder_path, series_ids[0])
        reader.SetFileNames(series_file_names)
        image = reader.Execute()
        
        # Convert to numpy array
        self.ct_array = sitk.GetArrayFromImage(image)
        print(f"Loaded DICOM series: shape {self.ct_array.shape}")
        print(f"Number of slices: {self.ct_array.shape[0]}")
        
        # Apply windowing if specified
        if window_level is not None:
            window, level = window_level
            self.ct_array = np.clip(
                self.ct_array,
                level - window // 2,
                level + window // 2
            )
            self.ct_array = (self.ct_array - (level - window // 2)) / window
        
        # Initialize slice index
        self.current_slice = self.ct_array.shape[0] // 2
        self.num_slices = self.ct_array.shape[0]
        
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        # Display first slice
        self.update_display()
        
    def get_slice(self, idx: int) -> np.ndarray:
        """Get a slice at the given index."""
        if idx < 0:
            idx = 0
        elif idx >= self.num_slices:
            idx = self.num_slices - 1
        return np.rot90(self.ct_array[idx, :, :])
    
    def update_display(self):
        """Update the displayed slice."""
        slice_data = self.get_slice(self.current_slice)
        self.ax.clear()
        self.ax.imshow(slice_data, cmap='gray')
        self.ax.set_title(
            f'Slice {self.current_slice + 1}/{self.num_slices} '
            f'(Use ↑↓ arrows or mouse wheel to navigate, Q to quit)',
            fontsize=12
        )
        self.ax.axis('off')
        self.fig.canvas.draw()
    
    def on_key(self, event):
        """Handle keyboard events."""
        if event.key == 'up' or event.key == 'right':
            if self.current_slice < self.num_slices - 1:
                self.current_slice += 1
                self.update_display()
        elif event.key == 'down' or event.key == 'left':
            if self.current_slice > 0:
                self.current_slice -= 1
                self.update_display()
        elif event.key == 'q' or event.key == 'escape':
            plt.close(self.fig)
            sys.exit(0)
        elif event.key == 'home':
            self.current_slice = 0
            self.update_display()
        elif event.key == 'end':
            self.current_slice = self.num_slices - 1
            self.update_display()
    
    def on_scroll(self, event):
        """Handle mouse scroll events."""
        if event.button == 'up':
            if self.current_slice < self.num_slices - 1:
                self.current_slice += 1
                self.update_display()
        elif event.button == 'down':
            if self.current_slice > 0:
                self.current_slice -= 1
                self.update_display()
    
    def show(self):
        """Show the interactive browser."""
        plt.show()


def get_dicom_path_from_case(case_id: str, main_path: str) -> str:
    """
    Get DICOM folder path from case ID.
    
    Parameters
    ----------
    case_id : str
        Case ID (e.g., 'RADCURE-0005')
    main_path : str
        Main path containing the case folders
    
    Returns
    -------
    str
        Path to DICOM CT folder
    """
    # Try to find the case in the retrain folder
    retrain_path = os.path.join(main_path, 'TotalSegmentatorRetrain', case_id)
    if os.path.exists(retrain_path):
        # Look for DICOM folder structure
        for root, dirs, files in os.walk(retrain_path):
            # Check if this looks like a DICOM folder (has .dcm files)
            if any(f.endswith('.dcm') or f.endswith('.DCM') for f in files):
                return root
    
    # Try original download location
    download_path = os.path.join(main_path, 'RADCURE', case_id)
    if os.path.exists(download_path):
        file_handler = FileHandler()
        try:
            dicom_path = file_handler.get_dicom_path(main_path, case_id)
            ct_paths = file_handler.get_ct_and_mask_paths(dicom_path)
            return ct_paths['ct_path']
        except:
            pass
    
    raise FileNotFoundError(
        f"Could not find DICOM folder for case {case_id} in {main_path}. "
        "Please provide the DICOM folder path directly using --dicom_folder"
    )


def main():
    parser = argparse.ArgumentParser(
        description='Interactive DICOM slice browser'
    )
    parser.add_argument(
        '--dicom_folder',
        type=str,
        help='Path to folder containing DICOM CT series'
    )
    parser.add_argument(
        '--case_id',
        type=str,
        help='Case ID (e.g., RADCURE-0005). Requires --main_path'
    )
    parser.add_argument(
        '--main_path',
        type=str,
        default=os.getenv('MAIN_PATH', ''),
        help='Main path containing case folders (default: from MAIN_PATH env var)'
    )
    parser.add_argument(
        '--window',
        type=int,
        default=None,
        help='Window width for windowing (e.g., 400 for soft tissue, 1500 for bone)'
    )
    parser.add_argument(
        '--level',
        type=int,
        default=None,
        help='Window level for windowing (e.g., 50 for soft tissue, -500 for bone)'
    )
    
    args = parser.parse_args()
    
    # Determine DICOM folder path
    if args.dicom_folder:
        dicom_folder = args.dicom_folder
    elif args.case_id:
        if not args.main_path:
            print("Error: --main_path is required when using --case_id")
            sys.exit(1)
        try:
            dicom_folder = get_dicom_path_from_case(args.case_id, args.main_path)
            print(f"Found DICOM folder for {args.case_id}: {dicom_folder}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        print("Error: Either --dicom_folder or --case_id must be provided")
        parser.print_help()
        sys.exit(1)
    
    # Set up windowing
    window_level = None
    if args.window is not None and args.level is not None:
        window_level = (args.window, args.level)
        print(f"Using windowing: window={args.window}, level={args.level}")
    
    # Create and show browser
    try:
        browser = DicomSliceBrowser(dicom_folder, window_level=window_level)
        browser.show()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
