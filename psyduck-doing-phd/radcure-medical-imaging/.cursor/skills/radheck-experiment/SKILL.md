---
name: radheck-experiment
description: >-
  Run and record RADCURE/HECKTOR (RADHECK) nnUNet segmentation experiments:
  registry, YAML configs, train, dual-cohort GTVp Dice evaluation. Use when
  starting a new test, reproducing Test1–3, evaluating on Dataset650 and
  Dataset152, or updating experiments/registry.yaml.
---

# RADHECK experiment workflow

## Before any work

1. Read `docs/README.md` — doc hub (purpose, Cursor setup, main index).
2. Read `experiments/registry.yaml` — existing tests, metrics, path keys.
3. Read the experiment YAML: `experiments/configs/testN_*.yaml`.
4. On the server, maintain `experiments/configs/local.yaml` (from `local.example.yaml`; gitignored).
5. Preserve **research vs implementation**: configs/registry in Git; paths and data on server only.

## Pipeline (do not skip steps without reason)

| Stage | Scripts |
|-------|---------|
| Preprocess RADCURE | `process_all_cases.py`, `split_dataset.py` |
| Build RADHECK Dataset650 | `python -m pipelines.radheck.build_nnunet_dataset` |
| **Test5 from scratch** | `clean_workspace` → `transform_cases` → `build_datasets` — see `Retrain-Radheck-Test5.md` |
| **Test6 STU-Net FT** | `setup_stunet` → `link_test5_dataset` → `train_finetune` → `evaluate` — see `Retrain-Radheck-Test6.md` |
| **Test7 probability outputs** | `link_dataset` → `predict_probabilities` (slim crop) → curves → `probability_visualisation` — see `Retrain-Radheck-Test7.md` |
| nnUNet train/eval | `train_nnunet.py` |
| HECKTOR test eval | `python -m pipelines.hecktor.test_pipeline` (Test5 unified Ts preferred) |

**Primary metric:** GTVp Dice on **both** RADCURE test (650 `imagesTs`) and HECKTOR test (152 `imagesTs`).

## Path rules (common failures)

| Variable | Meaning |
|----------|---------|
| `DATASET_FOLDER` | Folder with test images/labels — **not** `NNUNET_RETRAIN_PATH` |
| `NNUNET_RETRAIN_PATH` | `nnUNet_raw` / `nnUNet_results` / logs root |
| `DATASET_ID` | **Model** id (366 Test1, 650 Test2/3) when predicting on 152 |
| `NNUNET_PREPROCESSED_PATH` | Reuse preprocess (Test3 → point at Test2) |
| `HECKTOR_EVAL_OUTPUT_DIR` | Separate HECKTOR outputs (e.g. `{retrain}/hecktor_validation`) |

## Naming convention (required for Test4+)

Full rules: [experiments/README.md](../../experiments/README.md#naming-convention).

**Before creating files**, complete:

> Test{N} changes ___ compared to {changed_from}; everything else is identical.

| Layer | Pattern | Example |
|-------|---------|---------|
| Config file | `test{N}_{cohort}_{variable}.yaml` | `test4_radheck_500epochs.yaml` |
| Cohort | `radcure` (366) or `radheck` (650) | |
| Variable slug | the one changed knob | `500epochs`, `cosine_lr` |
| Path key | `RETRAIN_RADHECK_TEST{N}` | `RETRAIN_RADHECK_TEST4` |
| Server folder | `nnunet_radheck_test_{N}_retrain` | must match test id (test2 is legacy) |

When starting a new test, agent must:

1. Propose filename + hypothesis + `changed_from` before editing YAML.
2. Add registry block with matching `id`, `name`, `config_file`, `changed_from`.
3. Add path key to `local.example.yaml` comment pattern (user fills `local.yaml` on server).

## New experiment checklist

```
- [ ] One-line rule written (one variable vs changed_from)
- [ ] Copy _template.yaml → configs/test{N}_{cohort}_{variable}.yaml
- [ ] Add entry under experiments: in registry.yaml
- [ ] RETRAIN_RADHECK_TEST{N} in local.yaml (folder number = N)
- [ ] status: planned → running → completed
- [ ] After run: fill results.radcure / results.hecktor mean_dice in registry + YAML
```

## Run commands (Test2/3 pattern — adjust paths from local.yaml)

### Train (RADHECK 650)

```bash
cd /path/to/radcure-medical-imaging
source .venv/bin/activate

export DATASET_FOLDER=/path/to/Dataset650_TotalSegmentator
export NNUNET_RETRAIN_PATH=/path/to/nnunet_radheck_test_N_retrain
export NNUNET_PREPROCESSED_PATH=/path/to/previous/nnUNet_preprocessed  # if reusing
export ORGAN_DICTIONARY_PATH=/path/to/radcure_dictionary.json
export NNUNET_TRAINER=nnUNetTrainerNoMirroring  # or nnUNetTrainer_700epochs_NoMirroring
export nnUNet_compile=false
export CUDA_VISIBLE_DEVICES=1

python -m nnunet_training.install_trainer_variants  # if custom *_NoMirroring trainer
python train_nnunet.py --step prepare --link-raw   # skip if raw link exists
python train_nnunet.py --step plan                 # skip if reusing preprocess
python train_nnunet.py --step train
```

### Evaluate RADCURE test

```bash
export DATASET_FOLDER=/path/to/Dataset650_TotalSegmentator
export NNUNET_RETRAIN_PATH=/path/to/nnunet_radheck_test_N_retrain
export NNUNET_TRAINER=...  # must match trained model

python train_nnunet.py --step evaluate
python train_nnunet.py --step evaluation_visualization
```

### Evaluate HECKTOR test (model 650 on Dataset152)

```bash
export DATASET_ID=650
export DATASET_FOLDER=/path/to/Dataset152_TotalSegmentator
export NNUNET_RETRAIN_PATH=/path/to/nnunet_radheck_test_N_retrain
export HECKTOR_EVAL_OUTPUT_DIR=/path/to/nnunet_radheck_test_N_retrain/hecktor_validation
export NNUNET_TRAINER=...
export nnUNet_compile=false

python -m pipelines.hecktor.test_pipeline --predict-only
python -m pipelines.hecktor.test_pipeline --eval-only   # if predictions already exist
```

## Completed experiments (GTVp Dice)

| ID | Train | Epochs | RADCURE | HECKTOR |
|----|-------|--------|---------|---------|
| test1 | RADCURE 366 | 1000 | 0.377 | 0.330 |
| test2 | RADHECK 650 | 1000 | 0.383 | 0.486 |
| test3 | RADHECK 650 | 700 | 0.383 | 0.545 |

Config files: `experiments/configs/test1_radcure_only.yaml`, `test2_radheck_1000epochs.yaml`, `test3_radheck_700epochs.yaml`.

## After completing a run

1. Update `experiments/registry.yaml` → `results` and `status: completed`.
2. Update the experiment YAML `results` block.
3. Add findings to `research_notebooks/retrain_epoch_study/retrain_epoch_study.md` or a new study note.
4. Never commit `local.yaml`, `.env`, or `radheck_server_paths.json`.

## Additional reference

- Full repo map: [docs/documentation-index.md](../../docs/documentation-index.md)
- Registry and path templates: [experiments/registry.yaml](../../experiments/registry.yaml)
- Human-readable guide: [experiments/README.md](../../experiments/README.md)
- Leak checks before training: `python -m pipelines.radheck.verify_radheck_no_leak`
