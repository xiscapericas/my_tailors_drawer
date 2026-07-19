# Test5 — train nnUNet after improved preprocessing (vs Test4)

**One-line rule:** Test5 changes **preprocessing** (improved anatomical background +
anatomy QC + canonical organ dictionary) compared to Test4; everything else
(separate GTVp/GTVn, 700 epochs, NoMirroring, Dataset650, same Tr/Va/Ts family)
is identical, except QC-discarded cases are dropped from the splits.

**Prerequisite:** Existing TotalSegmentator outputs on disk (same sources as Test4).

---

## Step 0 — Paths

```bash
cd /path/to/radcure-medical-imaging
source .venv/bin/activate
set -a && source .env && set +a

export TEST5_WORK_ROOT=/media/HDD_8TB/xisca/work/retrain_test5
# Prefer Test2/Test3 Dataset650 (has split_manifest.json). Test4's Dataset650
# often lacks the manifest — do not use it as reference unless the file exists:
export TEST5_REFERENCE_DATASET650=/media/HDD_8TB/xisca/work/nnunet_radheck_test_1/Dataset650_TotalSegmentator
# Same sources as Test4 Phase 2:
export TEST5_RADCURE_SOURCE_MAIN_PATH=/media/HDD_8TB/xisca/dataset/RadcureComplete
export TEST5_HECKTOR_SOURCE_CASES_ROOT=/media/HDD_8TB/xisca/dataset/hecktor/.../cases

export ORGAN_DICTIONARY_PATH=${TEST5_WORK_ROOT}/radcure_dictionary_test5.json
```

Also set in `experiments/configs/local.yaml`:

- `RETRAIN_RADHECK_TEST5`
- `RADHECK_DATASET_TEST5`
- `TEST5_WORK_ROOT` / `TEST5_REFERENCE_DATASET650`

---

## Step 1 — Phase 2: relabel (improved bg + QC)

Reuses `total_segmentator_output/`. Seeds canonical organ dict. Applies anatomy QC
(threshold **0.70**). Background = **improved**.

```bash
python -m pipelines.test5.relabel_tumor_batch --dry-run
python -m pipelines.test5.relabel_tumor_batch
```

Outputs:

- `${TEST5_WORK_ROOT}/TotalSegmentatorRetrain/.../output/`
- `${TEST5_WORK_ROOT}/hecktor/.../output/`
- `${TEST5_WORK_ROOT}/logs/anatomy_qc/anatomy_qc_decisions.jsonl`
- `${TEST5_WORK_ROOT}/anatomy_qc_discarded.csv`
- `${TEST5_WORK_ROOT}/radcure_dictionary_test5.json`

---

## Step 2 — Phase 3: build Dataset650

Same reference Tr/Va/Ts as Test4, **minus** QC discards.

```bash
python -m pipelines.test5.build_dataset650 --dry-run
python -m pipelines.test5.build_dataset650
```

Output: `${TEST5_WORK_ROOT}/Dataset650_TotalSegmentator/`

Verify `dataset.json` contains **GTVp** and **GTVn**. Check `split_manifest.json`
for `anatomy_qc_discarded_case_ids` and `counts_built`.

---

## Step 3 — Install custom trainer (once)

```bash
python -m nnunet_training.install_trainer_variants
```

---

## Step 4 — Prepare, plan, train

**Do not** reuse Test4 preprocess (new bg + possibly different label indices).

```bash
export DATASET_FOLDER=${TEST5_WORK_ROOT}/Dataset650_TotalSegmentator
export ORGAN_DICTIONARY_PATH=${TEST5_WORK_ROOT}/radcure_dictionary_test5.json
export NNUNET_RETRAIN_PATH=${TEST5_WORK_ROOT}/nnunet_retrain
export NNUNET_TRAINER=nnUNetTrainer_700epochs_NoMirroring
export NNUNET_CONFIGURATION=3d_fullres
export NNUNET_FOLD=0
export LOG_DIR=${NNUNET_RETRAIN_PATH}/logs
export NNUNET_USE_LOCAL_PREPROCESS=1
export nnUNet_compile=false
export CUDA_VISIBLE_DEVICES=1

mkdir -p "$NNUNET_RETRAIN_PATH" "$LOG_DIR"

python train_nnunet.py --step prepare --link-raw
python train_nnunet.py --step plan
python train_nnunet.py --step train
```

Optional — same fold assignment as Test3/Test4:

```bash
export NNUNET_SPLITS_REFERENCE=/path/to/nnunet_radheck_test_1_retrain/nnUNet_preprocessed
```

---

## Step 5 — Evaluate RADCURE test (Dataset650 Ts)

```bash
export DATASET_FOLDER=${TEST5_WORK_ROOT}/Dataset650_TotalSegmentator
export NNUNET_RETRAIN_PATH=${TEST5_WORK_ROOT}/nnunet_retrain
export NNUNET_TRAINER=nnUNetTrainer_700epochs_NoMirroring

python train_nnunet.py --step evaluate
python train_nnunet.py --step evaluation_visualization
```

Compare **GTVp Dice** (and GTVn) to Test4 on the overlapping Ts cases.

---

## Step 6 — Evaluate HECKTOR test (Dataset152)

Same as Test4: model 650 on held-out HECKTOR test set.

```bash
export DATASET_ID=650
export DATASET_FOLDER=/path/to/Dataset152_TotalSegmentator
export NNUNET_RETRAIN_PATH=${TEST5_WORK_ROOT}/nnunet_retrain
export HECKTOR_EVAL_OUTPUT_DIR=${NNUNET_RETRAIN_PATH}/hecktor_validation
export NNUNET_TRAINER=nnUNetTrainer_700epochs_NoMirroring
export nnUNet_compile=false

python -m pipelines.hecktor.test_pipeline --predict-only
python -m pipelines.hecktor.test_pipeline --eval-only
```

---

## After the run

1. Fill `experiments/registry.yaml` → `test5.results`
2. Fill `experiments/configs/test5_radheck_improved_preprocess.yaml` → `results`
3. Short note under `research_notebooks/` comparing Test4 vs Test5 Dice
