# Test5 — improved preprocessing, then 700-epoch retrain (from scratch)

**One-line rule:** Test5 changes **preprocessing** (improved anatomical
background + separate GTVp/GTVn + canonical organ dictionary; **no anatomy QC**)
compared to Test4; trainer stays `nnUNetTrainer_700epochs_NoMirroring`.
**Ts stays the same 74 cases** for comparison; **Tr uses all other ready cases**
(not the old manifesto Tr=361).

**Layout (unified cohort — do not split RADCURE / HECKTOR trees):**

```
${TEST5_WORK_ROOT}/
  RADHECK_{N}/
    cases/                 # RADCURE-* and HECKTOR centers together
    organ_dictionary_test5.json
    STATUS.json
    transform_ok.txt
  RADHECK_CURRENT -> RADHECK_{N}
  split_manifest.json      # Test1 recovery (copied from repo artifact)
  Dataset650_TotalSegmentator/
  Dataset152_TotalSegmentator/
  nnunet_retrain/
```

| Role | Path |
|------|------|
| Work root | `/media/HDD_8TB/xisca/work/retrain_test5` |
| RADCURE originals | `/media/HDD_8TB/xisca/dataset/RadcureComplete/TotalSegmentatorRetrain` |
| HECKTOR train/val | `/media/HDD_8TB/xisca/dataset/hecktor/HECKTOR2025_task1_training/unzipped/task1` |
| HECKTOR held-out test | `/media/HDD_8TB/xisca/dataset/hecktor/test1/unzipped/test1` |

> **Important:** `test1` alone is **not** enough for Dataset650. The manifesto
> needs ~227 train + ~57 val HECKTOR cases from the **training** zip, plus the
> held-out test folders from `test1` for Dataset152.

---

## Step 0 — Clean the messy work root

```bash
cd /path/to/radcure-medical-imaging
source .venv/bin/activate
set -a && source .env && set +a

export TEST5_WORK_ROOT=/media/HDD_8TB/xisca/work/retrain_test5

python -m pipelines.test5.clean_workspace --dry-run
# removes legacy TotalSegmentatorRetrain/, hecktor/, Dataset*, nnunet_retrain, …
python -m pipelines.test5.clean_workspace --yes --also-radheck
```

Does **not** touch the original dataset sources.

---

## Step 1 — Paths

```bash
export TEST5_WORK_ROOT=/media/HDD_8TB/xisca/work/retrain_test5
mkdir -p "$TEST5_WORK_ROOT"

export TEST5_RADCURE_SOURCE=/media/HDD_8TB/xisca/dataset/RadcureComplete/TotalSegmentatorRetrain
export TEST5_HECKTOR_TRAIN_SOURCE=/media/HDD_8TB/xisca/dataset/hecktor/HECKTOR2025_task1_training/unzipped/task1
export TEST5_HECKTOR_TEST_SOURCE=/media/HDD_8TB/xisca/dataset/hecktor/test1/unzipped/test1
# equivalent: export TEST5_HECKTOR_SOURCES="${TEST5_HECKTOR_TRAIN_SOURCE}:${TEST5_HECKTOR_TEST_SOURCE}"

export TEST5_REFERENCE_DATASET650=${TEST5_WORK_ROOT}/Dataset650_TotalSegmentator
export TEST5_RADCURE_DATASET366=/media/HDD_8TB/xisca/work/nnunet_retrain_radcure366/Dataset366_TotalSegmentator
export ORGAN_DICTIONARY_PATH=${TEST5_WORK_ROOT}/organ_dictionary_test5.json

export HECKTOR_CLEANUP_INTERMEDIATES=1
export HECKTOR_TS_NR_THR_SAVING=1
```

In `experiments/configs/local.yaml`:

- `RETRAIN_RADHECK_TEST5` → `${TEST5_WORK_ROOT}/nnunet_retrain`
- `RADHECK_DATASET_TEST5` → `${TEST5_WORK_ROOT}/Dataset650_TotalSegmentator`
- `HECKTOR_TEST_DATASET` → `${TEST5_WORK_ROOT}/Dataset152_TotalSegmentator`
- `TEST5_WORK_ROOT`, sources, `TEST5_RADCURE_DATASET366`

---

## Step 2 — Phase 2: transform → `RADHECK_{N}/cases/`

Improved background, separate GTVp/GTVn, **no anatomy QC**. All cases land in one
folder (`RADHECK_{N}` where `N` = RADCURE + unique HECKTOR count).

```bash
python -m pipelines.test5.transform_cases --dry-run
python -m pipelines.test5.transform_cases --max-cases 2   # smoke
python -m pipelines.test5.transform_cases
```

Outputs under `${TEST5_WORK_ROOT}/RADHECK_{N}/cases/{id}/output/`.

Check status:

```bash
cat ${TEST5_WORK_ROOT}/RADHECK_CURRENT/STATUS.json
# or: ls -la ${TEST5_WORK_ROOT}/RADHECK_*
```

---

## Step 3 — Phase 3: build Dataset650 + Dataset152

**Default train mode:** keep the **same 74 Ts** as Test1 (RADCURE comparison), put
**all other ready cases** into `imagesTr`, leave `imagesVa` empty, and keep
HECKTOR held-out test **only** in Dataset152 (not in 650 Tr).

That is why you may see ~1047 cases under `RADHECK_{N}` but only ~361 in the old
manifesto Tr — the manifesto was a fixed subset; max-train uses the full pool.

```bash
python -m pipelines.test5.build_datasets --dry-run
python -m pipelines.test5.build_datasets --link hardlink
# old manifesto Tr≈361 / Va≈71 / Ts≈74:
# python -m pipelines.test5.build_datasets --link hardlink --manifest-splits
```

Expect roughly:

- `imagesTs` ≈ **74** (same as Test1–4)
- `imagesTr` ≈ ready − 74 − HECKTOR Dataset152 − stem collisions
- `imagesVa` = **0** (nnUNet fold val comes from `splits_final.json` inside Tr)

Verify:

- `Dataset650_TotalSegmentator/dataset.json` has **GTVp** and **GTVn**
- `split_manifest.json` → `counts_built`, `train_all_except_ts: true`
- `Dataset152_TotalSegmentator/imagesTs` only held-out HECKTOR test folders

Leak check (optional):

```bash
python -m pipelines.radheck.verify_radheck_no_leak \
  --dataset650 ${TEST5_WORK_ROOT}/Dataset650_TotalSegmentator \
  --hecktor-test-dataset ${TEST5_WORK_ROOT}/Dataset152_TotalSegmentator \
  --split-manifest ${TEST5_WORK_ROOT}/Dataset650_TotalSegmentator/split_manifest.json
```

---

## Step 4 — Install custom trainer (once)

```bash
python -m nnunet_training.install_trainer_variants
```

---

## Step 5 — Prepare, plan, train (700 epochs)

**Do not** reuse older preprocess trees (labels / bg differ).

```bash
export DATASET_FOLDER=${TEST5_WORK_ROOT}/Dataset650_TotalSegmentator
export ORGAN_DICTIONARY_PATH=${TEST5_WORK_ROOT}/organ_dictionary_test5.json
export NNUNET_RETRAIN_PATH=${TEST5_WORK_ROOT}/nnunet_retrain
export NNUNET_TRAINER=nnUNetTrainer_700epochs_NoMirroring
export NNUNET_CONFIGURATION=3d_fullres
export NNUNET_FOLD=0
export LOG_DIR=${NNUNET_RETRAIN_PATH}/logs
export NNUNET_USE_LOCAL_PREPROCESS=1
export nnUNet_compile=false
export CUDA_VISIBLE_DEVICES=1
# Critical: do not leave DATASET_ID=152 from HECKTOR eval in the shell / .env
unset DATASET_ID
export DATASET_ID=650

mkdir -p "$NNUNET_RETRAIN_PATH" "$LOG_DIR"

python train_nnunet.py --step prepare --link-raw
python train_nnunet.py --step plan
python train_nnunet.py --step train
```

Confirm before plan:

```bash
echo "FOLDER=$DATASET_FOLDER"
echo "ID=${DATASET_ID:-"(auto)"}"
ls -la ${NNUNET_RETRAIN_PATH}/Dataset650_TotalSegmentator | head
# should exist (symlink or copy from prepare)
```

Optional — same fold assignment as Test3:

```bash
export NNUNET_SPLITS_REFERENCE=/media/HDD_8TB/xisca/work/nnunet_radheck_test_1_retrain/nnUNet_preprocessed
```

---

## Step 6 — Evaluate RADCURE test (Dataset650 Ts)

```bash
export DATASET_FOLDER=${TEST5_WORK_ROOT}/Dataset650_TotalSegmentator
export NNUNET_RETRAIN_PATH=${TEST5_WORK_ROOT}/nnunet_retrain
export NNUNET_TRAINER=nnUNetTrainer_700epochs_NoMirroring

python train_nnunet.py --step evaluate
python train_nnunet.py --step evaluation_visualization
```

---

## Step 7 — Evaluate HECKTOR test (Dataset152)

`DATASET_ID=650` is the **trained model**; `DATASET_FOLDER` is Dataset152.

```bash
export DATASET_ID=650
export DATASET_FOLDER=${TEST5_WORK_ROOT}/Dataset152_TotalSegmentator
export NNUNET_RETRAIN_PATH=${TEST5_WORK_ROOT}/nnunet_retrain
export NNUNET_WORK_DIR=${TEST5_WORK_ROOT}
export HECKTOR_EVAL_OUTPUT_DIR=${NNUNET_RETRAIN_PATH}/hecktor_validation
export NNUNET_TRAINER=nnUNetTrainer_700epochs_NoMirroring
export NNUNET_CONFIGURATION=3d_fullres
export NNUNET_FOLD=0
export nnUNet_compile=false
export CUDA_VISIBLE_DEVICES=1
export ORGAN_DICTIONARY_PATH=${TEST5_WORK_ROOT}/organ_dictionary_test5.json

mkdir -p "$HECKTOR_EVAL_OUTPUT_DIR"

python -m pipelines.hecktor.test_pipeline --predict-only
python -m pipelines.hecktor.test_pipeline --eval-only
```

Dice / viz land under `$HECKTOR_EVAL_OUTPUT_DIR`.

---

## After the run

1. Fill `experiments/registry.yaml` → `test5.results`
2. Fill `experiments/configs/test5_radheck_improved_preprocess.yaml` → `results`
3. Short note under `research_notebooks/` comparing Test3/Test4 vs Test5 Dice

## Legacy CLIs

- `python -m pipelines.test5.relabel_tumor_batch` → redirects to `transform_cases`
- `python -m pipelines.test5.build_dataset650` → Dataset650 only (still OK)
