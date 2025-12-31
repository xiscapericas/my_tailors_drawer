"""
Configuration for nnUNet training pipeline.

This module handles all configuration settings for dataset preparation,
training, and evaluation.
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional


class TrainingConfig:
    """Configuration class for nnUNet training."""
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        # nnUNet environment path
        self.nnunet_path = os.getenv('NNUNET_PATH', '/path/to/nnUNet')
        
        # Dataset paths
        self.dataset_folder = os.getenv('DATASET_FOLDER')  # Path to DatasetXXX_TotalSegmentator folder
        self.main_retrain_path = os.getenv('NNUNET_RETRAIN_PATH', '/path/to/nnunet_retrain')
        
        # Organ dictionary path (from radcure_processor)
        self.organ_dictionary_path = os.getenv(
            'ORGAN_DICTIONARY_PATH',
            os.path.join(os.getenv('MAIN_PATH', ''), 'radcure_dictionary.json')
        )
        
        # Dataset ID (will be extracted from folder name or set manually)
        self.dataset_id = os.getenv('DATASET_ID')  # Optional: override auto-detection
        
        # Training configuration
        self.configuration = os.getenv('NNUNET_CONFIGURATION', '3d_fullres')
        self.trainer = os.getenv('NNUNET_TRAINER', 'nnUNetTrainerNoMirroring')
        self.fold = int(os.getenv('NNUNET_FOLD', '0'))
        self.num_processes = int(os.getenv('NNUNET_NUM_PROCESSES', '1'))
        
        # Prediction configuration
        self.disable_tta = os.getenv('NNUNET_DISABLE_TTA', 'True').lower() == 'true'
        self.prediction_fold = int(os.getenv('NNUNET_PREDICTION_FOLD', '0'))
        
        # Evaluation configuration
        self.evaluation_num_cpus = int(os.getenv('EVALUATION_NUM_CPUS', '8'))
        self.surface_distance_tolerance = float(os.getenv('SURFACE_DICE_TOLERANCE', '3.0'))
        
        # Logging
        self.log_dir = os.getenv('LOG_DIR', './logs')
        
        # Validate required paths
        if not self.dataset_folder:
            raise ValueError("DATASET_FOLDER environment variable is required")
        
        if not os.path.exists(self.dataset_folder):
            raise ValueError(f"Dataset folder not found: {self.dataset_folder}")
        
        # Auto-detect dataset ID and name from folder
        self._detect_dataset_info()
        
        # Load organ dictionary
        self.labels = self._load_organ_dictionary()
    
    def _detect_dataset_info(self):
        """Auto-detect dataset ID and name from folder path."""
        folder_name = Path(self.dataset_folder).name
        
        # Extract dataset ID from folder name (e.g., Dataset150_TotalSegmentator -> 150)
        if 'Dataset' in folder_name and '_TotalSegmentator' in folder_name:
            try:
                # Extract number between Dataset and _TotalSegmentator
                parts = folder_name.replace('Dataset', '').replace('_TotalSegmentator', '')
                if not self.dataset_id:
                    self.dataset_id = parts
                self.dataset_name = folder_name
            except:
                raise ValueError(f"Could not extract dataset ID from folder name: {folder_name}")
        else:
            raise ValueError(f"Folder name must be in format DatasetXXX_TotalSegmentator, got: {folder_name}")
        
        if not self.dataset_id:
            raise ValueError("Could not determine dataset ID. Set DATASET_ID environment variable.")
    
    def _load_organ_dictionary(self) -> Dict[str, int]:
        """
        Load organ dictionary from radcure_processor package.
        
        Returns
        -------
        Dict[str, int]
            Dictionary mapping organ names to indices
        """
        if not os.path.exists(self.organ_dictionary_path):
            raise FileNotFoundError(
                f"Organ dictionary not found at {self.organ_dictionary_path}. "
                "Please ensure you have processed cases and the dictionary was created."
            )
        
        with open(self.organ_dictionary_path, 'r') as f:
            dictionary = json.load(f)
        
        # Invert dictionary: {organ_name: index} -> {index: organ_name} for nnUNet
        # But nnUNet expects {organ_name: index}, so we keep it as is
        print(f"✓ Loaded organ dictionary with {len(dictionary)} labels from {self.organ_dictionary_path}")
        return dictionary
    
    def setup_nnunet_environment(self):
        """Set up nnUNet environment variables."""
        os.environ["nnUNet_raw"] = self.main_retrain_path
        os.environ["nnUNet_preprocessed"] = f"{self.main_retrain_path}/nnUNet_preprocessed"
        os.environ["nnUNet_results"] = f"{self.main_retrain_path}/nnUNet_results"
        
        # Create directories
        os.makedirs(os.environ["nnUNet_raw"], exist_ok=True)
        os.makedirs(os.environ["nnUNet_preprocessed"], exist_ok=True)
        os.makedirs(os.environ["nnUNet_results"], exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        print(f"✓ nnUNet environment configured:")
        print(f"  nnUNet_raw: {os.environ['nnUNet_raw']}")
        print(f"  nnUNet_preprocessed: {os.environ['nnUNet_preprocessed']}")
        print(f"  nnUNet_results: {os.environ['nnUNet_results']}")
    
    def get_dataset_paths(self) -> Dict[str, str]:
        """Get paths to dataset folders."""
        return {
            'imagesTr': os.path.join(self.dataset_folder, 'imagesTr'),
            'imagesTs': os.path.join(self.dataset_folder, 'imagesTs'),
            'imagesVa': os.path.join(self.dataset_folder, 'imagesVa'),
            'labelsTr': os.path.join(self.dataset_folder, 'labelsTr'),
            'labelsTs': os.path.join(self.dataset_folder, 'labelsTs'),
            'labelsVa': os.path.join(self.dataset_folder, 'labelsVa'),
        }

