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

**HECKTOR test** (Dataset152 images, **Dataset650 model**):

Use `DATASET_ID=650` so nnUNet loads the Test3 checkpoint; `DATASET_FOLDER` points at the HECKTOR test set (152).

```bash
export DATASET_ID=650
export DATASET_FOLDER=/media/HDD_8TB/xisca/work/nnunet_hecktor_test1/Dataset152_TotalSegmentator
export NNUNET_RETRAIN_PATH=/media/HDD_8TB/xisca/work/nnunet_radheck_test_3_retrain
export NNUNET_PREPROCESSED_PATH=/media/HDD_8TB/xisca/work/nnunet_radheck_test_1_retrain/nnUNet_preprocessed
export ORGAN_DICTIONARY_PATH=/media/HDD_8TB/xisca/dataset/RadcureComplete/radcure_dictionary.json
export NNUNET_TRAINER=nnUNetTrainer_700epochs_NoMirroring
export NNUNET_CONFIGURATION=3d_fullres
export NNUNET_WORK_DIR=/media/HDD_8TB/xisca/work/nnunet_hecktor_test1
export HECKTOR_EVAL_OUTPUT_DIR=/media/HDD_8TB/xisca/work/nnunet_radheck_test_3_retrain/hecktor_validation
export CUDA_VISIBLE_DEVICES=1
export nnUNet_compile=false

python run_hecktor_test1_pipeline.py --predict-only
```

Outputs (under `hecktor_validation/`, **not** in Dataset152):

| What | Path |
|------|------|
| Predictions | `.../hecktor_validation/labelsTs_predicted/` |
| Dice + PDFs | `.../hecktor_validation/labelsTs_dice_and_viz/` |
| Logs | `.../hecktor_validation/logs/` |

Dataset152 `labelsTs_predicted` from earlier runs is left unchanged.

## Verify HECKTOR was in Test3 training

Training used whatever is in **Dataset650** (`imagesTr` + `imagesVa`). Check the build manifest:

```bash
DATASET=/media/HDD_8TB/xisca/work/nnunet_radheck_test_1/Dataset650_TotalSegmentator

python - <<PY
import json
p = "$DATASET/split_manifest.json"
m = json.load(open(p))
h_tr = m.get("hecktor_train_cases") or []
h_va = m.get("hecktor_val_cases") or []
rc = m.get("radcure_counts") or {}
print("split_manifest:", p)
print(f"  HECKTOR train cases: {len(h_tr)}")
print(f"  HECKTOR val cases:   {len(h_va)}")
print(f"  HECKTOR total in Tr+Va: {len(h_tr) + len(h_va)}")
print(f"  RADCURE counts (366 source): Tr={rc.get('Tr')} Va={rc.get('Va')} Ts={rc.get('Ts')}")
print(f"  HECKTOR test excluded: {len(m.get('hecktor_excluded_case_folders') or [])} cases (Dataset152)")
PY
```

On-disk counts (after any dedupe):

```bash
echo "imagesTr: $(ls $DATASET/imagesTr/*_0000.nii.gz 2>/dev/null | wc -l)"
echo "imagesVa: $(ls $DATASET/imagesVa/*_0000.nii.gz 2>/dev/null | wc -l)"
echo "imagesTs (RADCURE test only): $(ls $DATASET/imagesTs/*_0000.nii.gz 2>/dev/null | wc -l)"
```

nnUNet fold split during training (from your log: 288 train + 73 val per fold) comes from `splits_final.json` under preprocessed Dataset650 — that pool is **all** RADCURE Tr+Va + HECKTOR Tr+Va combined.

Optional leak / overlap audit:

```bash
python research_notebooks/retrain_radheck/verify_radheck_no_leak.py \
  --combined-dataset "$DATASET" \
  --radcure-dataset /media/HDD_8TB/xisca/work/nnunet_retrain_radcure366/Dataset366_TotalSegmentator \
  --hecktor-test-dataset /media/HDD_8TB/xisca/work/nnunet_hecktor_test1/Dataset152_TotalSegmentator
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
