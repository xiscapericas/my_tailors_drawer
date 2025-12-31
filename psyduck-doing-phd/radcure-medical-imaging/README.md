# RADCURE Medical Imaging Processor

A Python package for processing RADCURE DICOM cases with TotalSegmentator for tumor detection model training.

## Overview

This package provides a structured, class-based approach to processing medical imaging data from the RADCURE dataset. It handles the entire pipeline from downloading cases from AWS S3, processing DICOM files, running TotalSegmentator segmentation, generating combined masks, and saving results in NIfTI format.

## Features

- 🔄 **Complete Pipeline**: End-to-end processing from AWS S3 to NIfTI outputs
- 🏗️ **Modular Design**: Class-based architecture for easy extension and maintenance
- 🧠 **TotalSegmentator Integration**: Automated organ segmentation
- 🎯 **Tumor Detection**: Specialized handling for GTVp (tumor) annotations with automatic alignment
- 📊 **Visualization**: Automatic PDF generation and DICOM visualization tools
- 🔐 **Secure**: Environment variable support for AWS credentials
- 📦 **Dataset Preparation**: Scripts for splitting processed cases into train/val/test sets

## Installation

### Prerequisites

- Python 3.7 or higher
- AWS credentials with access to your S3 bucket
- TotalSegmentator installed (see [TotalSegmentator documentation](https://github.com/wasserth/TotalSegmentator))

### Install Package

```bash
# Clone the repository
git clone https://github.com/yourusername/radcure-medical-imaging.git
cd radcure-medical-imaging

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Quick Start

### 1. Set Up Environment Variables

Create a `.env` file in the project root (copy from `env.example`):

```bash
cp env.example .env
```

Edit `.env` with your credentials:

```env
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_REGION=eu-west-1
MAIN_PATH=/path/to/your/dataset/
AWS_BUCKET_NAME=your-bucket-name
AWS_FOLDER=RADCURE/all_cases/
```

**Important**: Never commit your `.env` file to version control!

### 2. Basic Usage

```python
from radcure_processor import CaseProcessor
import os

# Load from environment variables
processor = CaseProcessor(
    main_path=os.getenv('MAIN_PATH'),
    aws_bucket_name=os.getenv('AWS_BUCKET_NAME'),
    aws_folder=os.getenv('AWS_FOLDER'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
)

# Process a single case
result = processor.process_case('RADCURE-0005')
print(f"Processing complete: {result}")

# Process multiple cases
processor.process_multiple_cases(
    case_ids=None,  # None = process all available
    processed_cases_file='processed_cases.txt',
    failed_cases_file='failed_cases.txt'
)
```

### 3. Process All Cases

For production use, run the dedicated script to process all cases:

```bash
python process_all_cases.py
```

This script will:
- Process all available cases from AWS S3
- Log successful cases to `processed_cases.txt`
- Log failed cases with error messages to `processed_cases_failed.txt`
- Handle interruptions gracefully
- Provide progress updates and final statistics

### 4. Split Dataset for Training

After processing all cases, split them into train/validation/test sets:

```bash
python split_dataset.py \
    --main_path /path/to/your/dataset/ \
    --output_path /path/to/split/output/
```

This creates a `DatasetXXX_TotalSegmentator` folder with:
- `imagesTr`, `imagesTs`, `imagesVa` (CT images)
- `labelsTr`, `labelsTs`, `labelsVa` (masks)

See [Dataset Splitting](#dataset-splitting) section for more details.

## Package Structure

```
radcure_processor/
├── core/                    # Core processing modules
│   ├── case_processor.py    # Main CaseProcessor class
│   ├── dicom_handler.py     # DICOM file operations
│   ├── segmentator.py       # TotalSegmentator wrapper
│   └── mask_generator.py    # Mask generation and combination
├── io/                      # I/O operations
│   ├── aws_handler.py       # AWS S3 operations
│   ├── file_handler.py      # File operations (zip, paths, etc.)
│   └── nifti_handler.py     # NIfTI file operations
├── visualization/           # Visualization utilities
│   └── visualizer.py        # Medical image visualization
└── utils/                   # Utility modules
    ├── image_processing.py  # Image processing utilities
    └── organ_dictionary.py  # Organ dictionary management
```

## Main Classes

### CaseProcessor

The main class that orchestrates the entire processing pipeline:

```python
processor = CaseProcessor(
    main_path='/path/to/dataset/',
    aws_bucket_name='your-bucket',
    aws_folder='RADCURE/all_cases/',
    organ_dictionary_path='dictionary.json',
    total_segmentator_tasks=['head_glands_cavities', 'head_muscles'],
    slice_expansion=5
)
```

### Individual Components

You can also use components independently:

```python
from radcure_processor.io import AWSHandler, FileHandler, NIfTIHandler
from radcure_processor.core import DICOMHandler, TotalSegmentatorWrapper
from radcure_processor.utils import OrganDictionary, ImageProcessor
from radcure_processor.visualization import MedicalImageVisualizer

# Download a case
aws_handler = AWSHandler(bucket_name='bucket', aws_folder='folder/')
aws_handler.download_case('RADCURE-0005', '/local/path')

# Get DICOM paths
file_handler = FileHandler()
dicom_path = file_handler.get_dicom_path('/local/path', 'RADCURE-0005')
ct_mask_paths = file_handler.get_ct_and_mask_paths(dicom_path)
mask_file_path = file_handler.get_dicom_mask_file_path(ct_mask_paths['mask_path'])

# Load mask
dicom_handler = DICOMHandler()
mask = dicom_handler.load_tumor_mask(ct_mask_paths['ct_path'], ct_mask_paths['mask_path'])

# Get non-zero slices
non_zero_slices = ImageProcessor.get_non_zero_slices(mask)

# Align and save mask
nifti_handler = NIfTIHandler()
nifti_handler.save_and_align_mask_with_ct(
    mask,
    'ct.nii.gz',
    'mask_aligned.nii.gz'
)
```

## Configuration Options

The `CaseProcessor` accepts several configuration parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `main_path` | Base path for storing processed cases | Required |
| `aws_bucket_name` | AWS S3 bucket name | Required |
| `aws_folder` | AWS S3 folder prefix | Required |
| `organ_dictionary_path` | Path to organ dictionary JSON file | None |
| `total_segmentator_tasks` | List of TotalSegmentator tasks | Head/neck tasks |
| `reverse_slices` | Whether to reverse slice order | False |
| `slice_expansion` | Slices to add above/below tumor | 5 |

## Output Structure

For each processed case, the following structure is created:

```
TotalSegmentatorRetrain/
└── RADCURE-XXXX/
    ├── RADCURE-XXXX.nii.gz                    # Converted CT
    ├── RADCURE-XXXX_tumor_mask_aligned.nii.gz # Aligned tumor mask
    ├── total_segmentator_output/               # TotalSegmentator outputs
    └── output/
        ├── image/
        │   └── case_XXXX_0000.nii.gz          # CT slices
        ├── labels/
        │   └── case_XXXX_0000.nii.gz          # Combined masks
        └── RADCURE-XXXX.pdf                    # Visualization PDF
```

**Note**: The tumor mask is automatically aligned with the CT geometry before combining with TotalSegmentator results to ensure proper registration.

## TotalSegmentator Tasks

The package supports various TotalSegmentator tasks. Default tasks include:

- `head_glands_cavities`
- `head_muscles`
- `headneck_bones_vessels`
- `headneck_muscles`
- `oculomotor_muscles`
- `craniofacial_structures`

See [TotalSegmentator documentation](https://github.com/wasserth/TotalSegmentator) for all available tasks.

## Visualization

The package includes several visualization utilities:

### Visualize DICOM Series Directly

```python
from radcure_processor.visualization import MedicalImageVisualizer

visualizer = MedicalImageVisualizer()

# Visualize DICOM CT from folder
ct_array = visualizer.visualize_dicom_series(
    dicom_folder_path='/path/to/dicom/folder',
    slice_indices=[50, 100, 150],  # Optional: specific slices
    save_path='preview.png',
    show=True
)
```

### Download and Visualize a Case

```python
from radcure_processor import CaseProcessor

processor = CaseProcessor(...)

# Download and visualize a case
result = processor.download_and_visualize_case(
    'RADCURE-0005',
    slice_indices=[50, 100, 150],
    save_path='preview.png',
    show=True
)
```

### Visualize NIfTI Files with Masks

```python
# Visualize CT and mask side by side
visualizer.show_two_niis_side_by_side(
    path_image='ct.nii.gz',
    path_mask='mask.nii.gz',
    axis=2,  # axial slices
    slice_indices=[50, 100, 150],  # Optional: specific slices
    save_pdf_path='visualization.pdf',
    show=True
)
```

## Dataset Splitting

After processing all cases, use `split_dataset.py` to prepare data for training:

### Basic Usage

```bash
python split_dataset.py \
    --main_path /path/to/dataset/ \
    --output_path /path/to/split/output/
```

### Options

- `--main_path`: Path containing `TotalSegmentatorRetrain` folder (required)
- `--output_path`: Where to create the split folders (required)
- `--train_test_split`: Train/Test ratio (default: 0.8)
- `--train_val_split`: Train/Val ratio within train (default: 0.8)
- `--random_seed`: Random seed for reproducibility (default: 42)
- `--copy`: Copy files instead of moving them
- `--dry_run`: Preview without moving files

### Output Structure

The script creates a `DatasetXXX_TotalSegmentator` folder (where XXX is the total number of cases) with:

```
DatasetXXX_TotalSegmentator/
├── imagesTr/    # Training images (~64%)
├── imagesTs/    # Test images (~20%)
├── imagesVa/    # Validation images (~16%)
├── labelsTr/    # Training labels
├── labelsTs/    # Test labels
└── labelsVa/    # Validation labels
```

The split is: 80% Train/Test, then 80/20 Train/Val within the training set.

## Error Handling

Failed cases are logged to `processed_cases_failed.txt` with both the case ID and error message:

```
RADCURE-1330: GTVp structure not found in RTSTRUCT file
RADCURE-1450: No DICOM series found in folder
```

This makes it easy to identify and troubleshoot problematic cases.

## Dependencies

### Core Dependencies

- `boto3` - AWS S3 operations
- `numpy`, `pandas` - Data manipulation
- `SimpleITK` - Medical image processing
- `pydicom` - DICOM file handling
- `totalsegmentator` - Organ segmentation
- `rt-utils` - RTSTRUCT handling
- `nibabel` - NIfTI file operations
- `matplotlib` - Visualization
- `scikit-image`, `scipy` - Image processing
- `opencv-python` - Computer vision
- `python-dotenv` - Environment variable management

### nnUNet Training Dependencies

- `p_tqdm` - Parallel progress bars
- `surface-distance` - Surface distance metrics (install from: `git+https://github.com/google-deepmind/surface-distance.git`)
- `nnUNet` - Medical image segmentation framework (see [nnUNet documentation](https://github.com/MIC-DKFZ/nnUNet))

## nnUNet Training

After processing and splitting your dataset, you can train an nnUNet model using the provided training pipeline.

### Setup

1. **Install additional dependencies**:
   ```bash
   pip install p_tqdm
   pip install git+https://github.com/google-deepmind/surface-distance.git
   ```

2. **Configure environment variables**:
   ```bash
   cp nnunet_training.env.example .env
   # Edit .env with your paths
   ```

   Required variables:
   - `DATASET_FOLDER`: Path to your `DatasetXXX_TotalSegmentator` folder
   - `ORGAN_DICTIONARY_PATH`: Path to `radcure_dictionary.json`

   Optional variables (see `nnunet_training.env.example` for all options):
   - `NNUNET_PATH`: Path to nnUNet installation
   - `NNUNET_RETRAIN_PATH`: Base path for nnUNet folders
   - `NNUNET_CONFIGURATION`: Model configuration (default: `3d_fullres`)
   - `NNUNET_TRAINER`: Trainer class (default: `nnUNetTrainerNoMirroring`)

### Usage

**Run all steps sequentially:**
```bash
python train_nnunet.py --step all
```

**Or run individual steps:**

1. **Prepare dataset**:
   ```bash
   python train_nnunet.py --step prepare
   ```
   - Generates `dataset.json` file
   - Creates dataset ID to name mapping
   - Optionally copies dataset to nnUNet_raw folder

2. **Train model**:
   ```bash
   python train_nnunet.py --step train
   ```
   - Plans and preprocesses the dataset
   - Trains the model

3. **Evaluate model**:
   ```bash
   python train_nnunet.py --step evaluate
   ```
   - Runs predictions on test set
   - Computes Dice and Surface Dice metrics
   - Saves detailed results to CSV

### Output

- **Logs**: Saved in `LOG_DIR` (default: `./logs/`)
  - `preprocess_d{ID}.log`: Preprocessing log
  - `train_d{ID}_f{fold}.log`: Training log
  - `prediction_d{ID}.log`: Prediction log
  - `evaluation_d{ID}.csv`: Detailed evaluation results

- **Model**: Saved in `{NNUNET_RETRAIN_PATH}/nnUNet_results/`

- **Predictions**: Saved in `{DATASET_FOLDER}/labelsTs_predicted/`

For more details, see `nnunet_training/README.md`.

## Workflow

### Complete Processing Workflow

1. **Set up environment**:
   ```bash
   cp env.example .env
   # Edit .env with your credentials
   ```

2. **Process all cases**:
   ```bash
   python process_all_cases.py
   ```

3. **Review failed cases** (if any):
   ```bash
   cat processed_cases_failed.txt
   ```

4. **Split dataset for training**:
   ```bash
   python split_dataset.py \
       --main_path /path/to/dataset/ \
       --output_path /path/to/split/output/
   ```

5. **Train nnUNet model**:
   ```bash
   # Configure nnUNet training environment
   cp nnunet_training.env.example .env
   # Edit .env with nnUNet paths
   
   # Run training pipeline
   python train_nnunet.py --step all
   ```

## Security

**Never commit sensitive credentials to version control!**

- Use environment variables or `.env` files
- The `.env` file is already in `.gitignore`
- Always use `.env.example` as a template

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license here]

## Citation

If you use this package in your research, please cite:

```bibtex
@software{radcure_processor,
  title = {RADCURE Medical Imaging Processor},
  author = {Xisca Pericàs},
  year = {2025},
  url = {https://github.com/xiscapericas/my_tailors_drawer/tree/main/psyduck-doing-phd/radcure-medical-imaging}
}
```

## Acknowledgments

- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) for organ segmentation
- RADCURE dataset contributors

## Support

For issues and questions, please open an issue on GitHub.

## Author

Xisca Pericàs

---

**Note**: This package is designed for research purposes. Ensure you have proper data access permissions and comply with all relevant regulations when processing medical imaging data.

