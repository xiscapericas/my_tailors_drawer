# RADCURE Medical Imaging Processor

A Python package for processing RADCURE DICOM cases with TotalSegmentator for tumor detection model training.

## Overview

This package provides a structured, class-based approach to processing medical imaging data from the RADCURE dataset. It handles the entire pipeline from downloading cases from AWS S3, processing DICOM files, running TotalSegmentator segmentation, generating combined masks, and saving results in NIfTI format.

## Features

- 🔄 **Complete Pipeline**: End-to-end processing from AWS S3 to NIfTI outputs
- 🏗️ **Modular Design**: Class-based architecture for easy extension and maintenance
- 🧠 **TotalSegmentator Integration**: Automated organ segmentation
- 🎯 **Tumor Detection**: Specialized handling for GTVp (tumor) annotations
- 📊 **Visualization**: Automatic PDF generation for quality control
- 🔐 **Secure**: Environment variable support for AWS credentials

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

### 3. Run Example Script

```bash
python example_usage.py
```

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
from radcure_processor.io import AWSHandler, FileHandler
from radcure_processor.core import DICOMHandler, TotalSegmentatorWrapper
from radcure_processor.utils import OrganDictionary

# Use individual components as needed
aws_handler = AWSHandler(bucket_name='bucket', aws_folder='folder/')
cases = aws_handler.list_cases()
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
    ├── RADCURE-XXXX.nii.gz          # Converted CT
    ├── total_segmentator_output/     # TotalSegmentator outputs
    └── output/
        ├── image/
        │   └── case_XXXX_0000.nii.gz # CT slices
        ├── labels/
        │   └── case_XXXX_0000.nii.gz # Combined masks
        └── RADCURE-XXXX.pdf          # Visualization PDF
```

## TotalSegmentator Tasks

The package supports various TotalSegmentator tasks. Default tasks include:

- `head_glands_cavities`
- `head_muscles`
- `headneck_bones_vessels`
- `headneck_muscles`
- `oculomotor_muscles`
- `craniofacial_structures`

See [TotalSegmentator documentation](https://github.com/wasserth/TotalSegmentator) for all available tasks.

## Dependencies

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
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/radcure-medical-imaging}
}
```

## Acknowledgments

- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) for organ segmentation
- RADCURE dataset contributors

## Support

For issues and questions, please open an issue on GitHub.

## Author

[Your Name]

---

**Note**: This package is designed for research purposes. Ensure you have proper data access permissions and comply with all relevant regulations when processing medical imaging data.

