# nnUNet Training Pipeline

This module provides a complete pipeline for training nnUNet models on RADCURE datasets.

## Structure

```
nnunet_training/
├── config.py              # Configuration management
├── prepare_dataset.py     # Dataset preparation
├── train_model.py         # Model training
├── predict_and_evaluate.py # Prediction and evaluation
└── README.md             # This file
```

## Setup

1. **Install dependencies**:
   ```bash
   pip install p_tqdm
   pip install git+https://github.com/google-deepmind/surface-distance.git
   ```

2. **Set up environment variables**:
   ```bash
   cp env.example .env
   # Edit .env with your paths
   ```
   
   **Note**: Use the main `env.example` file which includes both processing and training variables.
   If you already have a `.env` file for processing, just add the nnUNet training variables to it.

   Required variables for training:
   - `DATASET_FOLDER`: Path to your `DatasetXXX_TotalSegmentator` folder
   - `ORGAN_DICTIONARY_PATH`: Path to `radcure_dictionary.json` (from radcure_processor)

   Optional variables:
   - `NNUNET_PATH`: Path to nnUNet installation (optional, has default)
   - `NNUNET_RETRAIN_PATH`: Base path for nnUNet folders (optional, has default)
   - `NNUNET_CONFIGURATION`: Model configuration (default: `3d_fullres`)
   - `NNUNET_TRAINER`: Trainer class (default: `nnUNetTrainerNoMirroring`)
   - `NNUNET_FOLD`: Fold to train (default: `0`)
   - And more... (see `env.example` in project root for complete list)

## Usage

### Option 1: Run all steps sequentially

```bash
python train_nnunet.py --step all
```

### Option 2: Run individual steps

**1. Prepare dataset:**
```bash
python train_nnunet.py --step prepare
```

This will:
- Generate `dataset.json` file
- Create dataset ID to name mapping
- Optionally copy dataset to nnUNet_raw folder

**2. Plan and preprocess:**
```bash
python train_nnunet.py --step plan
```

This will:
- Plan the dataset
- Preprocess the data for training

**3. Train model:**
```bash
python train_nnunet.py --step train
```

This will:
- Train the model
- Note: Requires planning and preprocessing to be completed first

**4. Evaluate model:**
```bash
python train_nnunet.py --step evaluate
```

This will:
- Run predictions on test set
- Compute Dice and Surface Dice metrics
- Save detailed results to CSV

### Option 3: Run modules directly

You can also run the individual modules:

```bash
python -m nnunet_training.prepare_dataset
python -m nnunet_training.train_model
python -m nnunet_training.predict_and_evaluate
```

## Workflow

1. **Process cases** (using `process_all_cases.py`):
   ```bash
   python process_all_cases.py
   ```

2. **Split dataset** (using `split_dataset.py`):
   ```bash
   python split_dataset.py --main_path /path/to/dataset --output_path /path/to/output
   ```

3. **Train model** (using this module):
   ```bash
   python train_nnunet.py --step all
   ```

## Output

- **Logs**: Saved in `LOG_DIR` (default: `./logs/`)
  - `preprocess_d{ID}.log`: Preprocessing log
  - `train_d{ID}_f{fold}.log`: Training log
  - `prediction_d{ID}.log`: Prediction log
  - `evaluation_d{ID}.csv`: Detailed evaluation results

- **Model**: Saved in `{NNUNET_RETRAIN_PATH}/nnUNet_results/`

- **Predictions**: Saved in `{DATASET_FOLDER}/labelsTs_predicted/`

## Notes

- The organ dictionary is automatically loaded from `radcure_dictionary.json`
- Dataset ID is auto-detected from folder name (e.g., `Dataset150_TotalSegmentator` → ID `150`)
- All paths are configurable via environment variables
- The pipeline handles errors gracefully and provides clear error messages

