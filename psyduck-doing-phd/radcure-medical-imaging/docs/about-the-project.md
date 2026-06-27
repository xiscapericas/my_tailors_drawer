# What this project is for

[← Documentation hub](README.md)

## Purpose

This repository supports a **PhD study** on the best approach to **head-and-neck tumour (GTVp) segmentation** using CT, combining:

- **RADCURE** — DICOM from S3, GTVp from RTSTRUCT  
- **HECKTOR** — NIfTI on disk, GTVp/GTVn merged in training masks  

Long-term goals:

1. **Experiment quickly** — test hypotheses (data mix, training schedule, etc.) with a clear record of each run.  
2. **Ship reproducible code** — implementation others can run and validate.  
3. **Publish** — methods, results, and analysis that support a scientific paper.

## Two-stage pipeline

| Stage | What happens | Code |
|-------|----------------|------|
| **1. Preprocessing** | TotalSegmentator organs + head/body/background separation + **tumour label** → nnUNet-ready NIfTI | `image_processor/`, `process_all_cases.py`, RADHECK build scripts |
| **2. nnUNet** | Train segmenter, predict test sets, **GTVp Dice** (+ Surface Dice, slice comparison PDFs) | `nnunet_training/`, `train_nnunet.py`, `pipelines.hecktor.test_pipeline` |

**Primary metric:** GTVp Dice on two held-out test pools:

- **RADCURE test** — cases in combined `Dataset650/imagesTs`  
- **HECKTOR test** — cases in `Dataset152/imagesTs` (never used in training)

## Research vs implementation

| | Research | Implementation |
|---|----------|----------------|
| **Purpose** | Explore ideas, write the paper | Run on server, reproduce, share on GitHub |
| **Location** | `research_notebooks/` (notebooks, study markdown) | `image_processor/`, `nnunet_training/`, root CLIs |
| **Experiments record** | Narrative in `retrain_epoch_study/` | **Canonical:** `experiments/registry.yaml` |

Stable logic should live in **implementation**; notebooks and study markdown interpret results — not replace the registry.

## Experiments so far (summary)

| ID | Training | Epochs | RADCURE GTVp Dice | HECKTOR GTVp Dice |
|----|----------|--------|-------------------|-------------------|
| test1 | RADCURE only (366) | 1000 | 0.377 | 0.330 |
| test2 | RADCURE + HECKTOR (650) | 1000 | 0.383 | 0.486 |
| test3 | RADCURE + HECKTOR (650) | 700 | 0.383 | 0.545 |

Full record: [experiments/registry.yaml](../experiments/registry.yaml).  
Interpretation: [retrain_epoch_study.md](../research_notebooks/retrain_epoch_study/retrain_epoch_study.md).

## What lives outside Git

- Patient imaging (`.nii.gz`, DICOM)  
- Trained weights and predictions on server  
- `.env`, `experiments/configs/local.yaml`, `radheck_server_paths.json`  

See [documentation-index.md](documentation-index.md) for path templates.

## Next steps (project direction)

1. **Phase 1 (done):** experiment registry + configs + Cursor skill  
2. **Tier A (done):** doc hub and de-duplicated entry README  
3. **Tier B (done):** pipeline scripts in `pipelines/`  
4. **Publication (planned):** `docs/METHODS.md`, `docs/REPRODUCE.md`, `tests/`

[← Documentation hub](README.md) · [Cursor setup →](cursor-setup.md) · [Main docs →](documentation-index.md)
