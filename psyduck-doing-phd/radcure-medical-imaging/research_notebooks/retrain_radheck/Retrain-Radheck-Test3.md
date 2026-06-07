# RADCURE + HECKTOR retrain — Test3 (700 epochs)

Test3 retrains **Dataset650** with **700 epochs** instead of the default **1000** (`nnUNetTrainerNoMirroring`). Everything else matches Test1/Test2: same combined dataset, same preprocessing, no mirroring.

| | Test1 / Test2 | Test3 |
|---|---------------|-------|
| Dataset | `Dataset650_TotalSegmentator` | same |
| Preprocessing | full plan + preprocess | **reuse** (no `--step plan`) |
| Trainer | `nnUNetTrainerNoMirroring` (1000 ep) | `nnUNetTrainer_700epochs_NoMirroring` |
| Results path | `nnunet_radheck_test_1_retrain` (or Test2) | **`nnunet_radheck_test_3_retrain`** |

## One-time: install custom trainer

From the repo root (requires `DATASET_FOLDER` in `.env` or env, same as training):

```bash
python -m nnunet_training.install_trainer_variants
```

This copies `nnUNetTrainer_700epochs_NoMirroring.py` into the **active nnunetv2 package** (pip/venv site-packages). `NNUNET_PATH` is only used as an extra target when it points to a separate source tree.

`train_nnunet.py --step train` runs this automatically when that trainer is selected.

## Server setup (no reprocessing)

Replace paths with yours. Assumes Test1 already ran plan/preprocess under `nnunet_radheck_test_1_retrain`.

```bash
cd /path/to/radcure-medical-imaging

# Combined dataset (unchanged)
export DATASET_FOLDER=/media/HDD_8TB/xisca/work/nnunet_radheck_test_1/Dataset650_TotalSegmentator
export ORGAN_DICTIONARY_PATH=/media/HDD_8TB/xisca/dataset/RadcureComplete/radcure_dictionary.json
export NNUNET_PATH=/path/to/nnUNet

# New retrain root — only nnUNet_results will be new; raw + preprocessed are reused
export NNUNET_RETRAIN_PATH=/media/HDD_8TB/xisca/work/nnunet_radheck_test_3_retrain
export NNUNET_PREPROCESSED_PATH=/media/HDD_8TB/xisca/work/nnunet_radheck_test_1_retrain/nnUNet_preprocessed

# 700 epochs, no mirroring
export NNUNET_TRAINER=nnUNetTrainer_700epochs_NoMirroring
export NNUNET_CONFIGURATION=3d_fullres
export NNUNET_FOLD=0
export LOG_DIR=${NNUNET_RETRAIN_PATH}/logs

# Reduce GPU memory use (recommended if you hit OOM on epoch 0)
export nnUNet_compile=false
# Use physical GPU 1 only (GPU 0 may be busy with other jobs)
export CUDA_VISIBLE_DEVICES=1
# Optional: fewer data-augmentation workers (helps CPU RAM, sometimes GPU pressure)
# export nnUNet_n_proc_DA=4

mkdir -p "$NNUNET_RETRAIN_PATH" "$LOG_DIR"
```

Register the dataset in the new raw folder (symlink only — no copy):

```bash
python train_nnunet.py --step prepare --link-raw
```

Train (skip `plan`):

```bash
python train_nnunet.py --step train
```

Model output:

```text
${NNUNET_RETRAIN_PATH}/nnUNet_results/Dataset650_TotalSegmentator/
  nnUNetTrainer_700epochs_NoMirroring__nnUNetPlans__3d_fullres/fold_0/
```

## Evaluation

**RADCURE test** (combined `imagesTs`):

```bash
export DATASET_FOLDER=/media/HDD_8TB/xisca/work/nnunet_radheck_test_1/Dataset650_TotalSegmentator
export NNUNET_RETRAIN_PATH=/media/HDD_8TB/xisca/work/nnunet_radheck_test_3_retrain
export NNUNET_TRAINER=nnUNetTrainer_700epochs_NoMirroring
export CUDA_VISIBLE_DEVICES=1
export nnUNet_compile=false

python train_nnunet.py --step evaluate
python train_nnunet.py --step evaluation_visualization
```

**HECKTOR test** (Dataset152), same pattern as Test1 but point retrain path at Test3:

```bash
export DATASET_ID=152
export DATASET_FOLDER=/media/HDD_8TB/xisca/work/nnunet_hecktor_test1/Dataset152_TotalSegmentator
export NNUNET_RETRAIN_PATH=/media/HDD_8TB/xisca/work/nnunet_radheck_test_3_retrain
export NNUNET_TRAINER=nnUNetTrainer_700epochs_NoMirroring
export CUDA_VISIBLE_DEVICES=1
export nnUNet_compile=false

python run_hecktor_test1_pipeline.py --predict-only
```

## Alternative: symlink preprocessed

Instead of `NNUNET_PREPROCESSED_PATH`, you can symlink inside the Test3 retrain folder:

```bash
ln -s /media/HDD_8TB/xisca/work/nnunet_radheck_test_1_retrain/nnUNet_preprocessed \
  /media/HDD_8TB/xisca/work/nnunet_radheck_test_3_retrain/nnUNet_preprocessed
```

Then omit `NNUNET_PREPROCESSED_PATH`.

## CUDA out of memory (OOM)

Your log shows `Using torch.compile...` and only **~170 MiB free** on a **24 GB** GPU. Test1 used the same `batch_size: 2` plans, so this is usually one of:

1. **Another process on the GPU** — check and stop it first:

   ```bash
   nvidia-smi
   # kill stale training/predict jobs, then retry
   ```

2. **`torch.compile` extra memory on epoch 0** — disable it (no replanning needed):

   ```bash
   export nnUNet_compile=false
   export CUDA_VISIBLE_DEVICES=1
   python train_nnunet.py --step train
   ```

3. **Still OOM with batch 2** — add a batch-1 configuration to the existing plans file (still **no** re-preprocessing). Edit:

   `.../nnUNet_preprocessed/Dataset650_TotalSegmentator/nnUNetPlans.json`

   Inside `"configurations"`, add:

   ```json
   "3d_fullres_bs1": {
     "inherits_from": "3d_fullres",
     "batch_size": 1
   }
   ```

   Then train with:

   ```bash
   export NNUNET_CONFIGURATION=3d_fullres_bs1
   export nnUNet_compile=false
   python train_nnunet.py --step train
   ```

   Results will be under `...__nnUNetPlans__3d_fullres_bs1/` instead of `...__3d_fullres/`.

## Results (fill in after run)

| Dataset | Average Dice (GTVp) | vs Test2 |
|---------|---------------------|----------|
| HECKTOR | | |
| RADCURE | | |
| Overall | | |

Notes:

- 
