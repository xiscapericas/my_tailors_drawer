# Test 8.0 — PET as a second nnUNet channel (HECKTOR-only Test5 split)

**One-line rule:** Test **8.0** changes the input from CT-only to **CT + PET**
compared to Test5, and keeps only **HECKTOR** cases from the Test5 Tr/Va/Ts
split (RADCURE has no PET). Labels, TotalSegmentator organs, and trainer
`nnUNetTrainer_700epochs_NoMirroring` (700 epochs) stay the same.

- **8** = PET-channel family
- **0** = first attempt

Phase 1 explore (format / load / overlay):  
[`research_notebooks/test8_0_pet/README.md`](../../research_notebooks/test8_0_pet/README.md)

Config: [`experiments/configs/test8_0_hecktor_pet_channel.yaml`](../../experiments/configs/test8_0_hecktor_pet_channel.yaml)

| Role | Path |
|------|------|
| Work root | `/media/HDD_8TB/xisca/work/retrain_test8_0` |
| Test5 Dataset650 (read-only) | `/media/HDD_8TB/xisca/work/retrain_test5/Dataset650_TotalSegmentator` |
| Test 8.0 Dataset650 | `${TEST8_0_WORK_ROOT}/Dataset650_TotalSegmentator` |
| nnUNet retrain | `${TEST8_0_WORK_ROOT}/nnunet_retrain` |

Do **not** reuse Test5 `nnUNet_preprocessed`. Do **not** re-run TotalSegmentator.

Eval PDFs are **four panels** when `_0001` PET exists: CT, PET, GT mask, predicted mask (GTVp still red).

---

## Step 0 — Env

```bash
cd /path/to/radcure-medical-imaging
source .venv/bin/activate
set -a && source .env && set +a

export TEST8_0_WORK_ROOT=/media/HDD_8TB/xisca/work/retrain_test8_0
export TEST5_WORK_ROOT=/media/HDD_8TB/xisca/work/retrain_test5
export TEST8_0_DATASET650=/media/HDD_8TB/xisca/work/retrain_test5/Dataset650_TotalSegmentator
export TEST5_HECKTOR_TRAIN_SOURCE=/media/HDD_8TB/xisca/dataset/hecktor/HECKTOR2025_task1_training/unzipped/task1
export TEST5_HECKTOR_TEST_SOURCE=/media/HDD_8TB/xisca/dataset/hecktor/test1/unzipped/test1
export CUDA_VISIBLE_DEVICES=1
export nnUNet_compile=false
mkdir -p "$TEST8_0_WORK_ROOT"
```

In `experiments/configs/local.yaml`:

- `RETRAIN_RADHECK_TEST8_0` → `${TEST8_0_WORK_ROOT}/nnunet_retrain`
- `RADHECK_DATASET_TEST8_0` → `${TEST8_0_WORK_ROOT}/Dataset650_TotalSegmentator`
- `TEST8_0_WORK_ROOT` / `TEST8_0_DATASET650`

---

## Step 1 — Build CT + PET Dataset650 (HECKTOR-only)

Hardlinks Test5 `_0000` CT + labels; writes `_0001` PET (SUV, resampled to CT, same slice crop). Fails if any kept HECKTOR case is missing `__PT.nii.gz`.

```bash
python -m pipelines.test8_0.build_dataset --dry-run
python -m pipelines.test8_0.build_dataset --max-cases 2   # smoke
python -m pipelines.test8_0.build_dataset

source ${TEST8_0_WORK_ROOT}/TEST8_0_ENV.sh

echo "Tr=$(ls ${TEST8_0_WORK_ROOT}/Dataset650_TotalSegmentator/imagesTr/*_0000.nii.gz | wc -l)"
echo "PET=$(ls ${TEST8_0_WORK_ROOT}/Dataset650_TotalSegmentator/imagesTr/*_0001.nii.gz | wc -l)"
echo "Ts=$(ls ${TEST8_0_WORK_ROOT}/Dataset650_TotalSegmentator/imagesTs/*_0000.nii.gz | wc -l)"
```

`dataset.json` must list `"channel_names": {"0": "CT", "1": "PET"}`.

---

## Step 2 — Install custom trainer (once)

```bash
python -m nnunet_training.install_trainer_variants
```

---

## Step 3 — Prepare, plan, train (700 epochs, from scratch)

```bash
source ${TEST8_0_WORK_ROOT}/TEST8_0_ENV.sh
export LOG_DIR=${NNUNET_RETRAIN_PATH}/logs
mkdir -p "$NNUNET_RETRAIN_PATH" "$LOG_DIR"
unset DATASET_ID
export DATASET_ID=650
unset NNUNET_PREPROCESSED_PATH

python train_nnunet.py --step prepare --link-raw
python train_nnunet.py --step plan
python train_nnunet.py --step train
```

---

## Step 4 — Evaluate (HECKTOR Ts only)

```bash
source ${TEST8_0_WORK_ROOT}/TEST8_0_ENV.sh
unset DATASET_ID
export DATASET_ID=650

python train_nnunet.py --step evaluate
python train_nnunet.py --step evaluation_visualization
```

Dice CSV: `${LOG_DIR}/evaluation_d650.csv`.  
PDFs: `${DATASET_FOLDER}/labelsTs_dice_and_viz/visualizations/` (CT | PET | GT | pred).

Compare GTVp Dice only to the **Test5 HECKTOR** test subset, not to Test5 RADCURE Ts.

---

## After the run

1. Fill `experiments/registry.yaml` → `test8.0.results`
2. Fill `experiments/configs/test8_0_hecktor_pet_channel.yaml` → `results`
3. Short note under `research_notebooks/test8_0_pet/` if useful
