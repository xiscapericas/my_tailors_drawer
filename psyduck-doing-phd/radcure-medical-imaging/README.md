# RADCURE / HECKTOR medical image segmentation

Dataset-agnostic preprocessing (TotalSegmentator + GTVp) and nnUNet training/evaluation for head-and-neck tumour segmentation on **RADCURE** and **HECKTOR**.

**Documentation:** start at [`docs/README.md`](docs/README.md) — purpose, Cursor setup, and full doc index.

## Install

```bash
pip install -e .
# or: pip install -r requirements.txt
cp env.example .env   # edit with your paths; never commit .env
```

Prerequisites: Python 3.10+, TotalSegmentator, nnUNet v2 (for training). See [`env.example`](env.example).

## Quick start

**1. Preprocess RADCURE**

```bash
python process_all_cases.py
python split_dataset.py --main_path /path/to/dataset --output_path /path/to/output
```

**2. Train nnUNet**

```bash
python train_nnunet.py --step prepare
python train_nnunet.py --step plan
python train_nnunet.py --step train
python train_nnunet.py --step evaluate
```

**3. RADHECK experiments (RADCURE + HECKTOR combined)**

See [`experiments/registry.yaml`](experiments/registry.yaml) and [`docs/cursor-setup.md`](docs/cursor-setup.md).

## Main entry points

| Script | Purpose |
|--------|---------|
| [`process_all_cases.py`](process_all_cases.py) | Batch RADCURE preprocessing |
| [`split_dataset.py`](split_dataset.py) | Train/val/test → `DatasetXXX_TotalSegmentator` |
| [`train_nnunet.py`](train_nnunet.py) | nnUNet pipeline (prepare / plan / train / evaluate / viz) |
| [`pipelines/`](pipelines/) | RADHECK dataset build + HECKTOR test eval (`python -m pipelines.*`) |
| `python -m pipelines.hecktor.test_pipeline` | HECKTOR test predict/eval (`--predict-only`, `--eval-only`) |

## Python API (minimal)

```python
from image_processor import CaseProcessor

processor = CaseProcessor(
    main_path="/path/to/dataset",
    aws_bucket_name="your-bucket",
    aws_folder="RADCURE/all_cases/",
)
processor.process_case("RADCURE-0005")
```

For HECKTOR, set `convention="hecktor"` and `cases_root=...`. More examples: [`example_usage.py`](example_usage.py).

## Security

Never commit `.env`, AWS keys, or patient data. Large artifacts (`.nii.gz`, predictions) stay on the server — see [`.gitignore`](.gitignore).

## Citation

```bibtex
@software{radcure_medical_imaging,
  title = {RADCURE / HECKTOR Medical Imaging Pipeline},
  author = {Xisca Pericàs},
  year = {2025},
  url = {https://github.com/xiscapericas/my_tailors_drawer/tree/main/psyduck-doing-phd/radcure-medical-imaging}
}
```

## Author

Xisca Pericàs
