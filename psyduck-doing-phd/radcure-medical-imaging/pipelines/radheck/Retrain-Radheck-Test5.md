# Test5 — train nnUNet after improved preprocessing (vs Test4)

**One-line rule:** Test5 changes **preprocessing** (improved anatomical background +
anatomy QC + canonical organ dictionary) compared to Test4; everything else
(separate GTVp/GTVn, 700 epochs, NoMirroring, Dataset650, same Tr/Va/Ts family)
is identical, except QC-discarded cases are dropped from the splits.

**Original sources (do not write into these trees):**

| Cohort | Path |
|--------|------|
| RADCURE | `/media/HDD_8TB/xisca/dataset/RadcureComplete/TotalSegmentatorRetrain/` |
| HECKTOR | `/media/HDD_8TB/xisca/dataset/hecktor/test1/unzipped/test1/` |

Outputs go only under `TEST5_WORK_ROOT` (relabel / full process + Dataset650).

**Memory / disk:** Phase 2 cleans HECKTOR TS intermediates; Phase 3 defaults to
**hardlinks** into Dataset650. Prefer a fresh empty `TEST5_WORK_ROOT` after
deleting old work trees.

---

## Step 0 — Paths (reset)

```bash
cd /path/to/radcure-medical-imaging
source .venv/bin/activate
set -a && source .env && set +a

export TEST5_WORK_ROOT=/media/HDD_8TB/xisca/work/retrain_test5
mkdir -p "$TEST5_WORK_ROOT"

# Output Dataset650 (also used as reference target after Test1 folder was deleted)
export TEST5_REFERENCE_DATASET650=${TEST5_WORK_ROOT}/Dataset650_TotalSegmentator

# Needed to reconstruct RADCURE Tr/Va/Ts when Dataset650 images* are gone
export TEST5_RADCURE_DATASET366=/media/HDD_8TB/xisca/work/nnunet_retrain_radcure366/Dataset366_TotalSegmentator

# Original sources (read-only)
export TEST5_RADCURE_SOURCE=/media/HDD_8TB/xisca/dataset/RadcureComplete/TotalSegmentatorRetrain
export TEST5_HECKTOR_SOURCE_CASES_ROOT=/media/HDD_8TB/xisca/dataset/hecktor/test1/unzipped/test1

export ORGAN_DICTIONARY_PATH=${TEST5_WORK_ROOT}/radcure_dictionary_test5.json

# Low-memory HECKTOR full process (if raw cases have no total_segmentator_output/)
export HECKTOR_CLEANUP_INTERMEDIATES=1
export HECKTOR_TS_NR_THR_SAVING=1

# Restore recovered Test1 split_manifest.json (repo artifact)
python -m pipelines.test5.restore_split_reference
```

Also set in `experiments/configs/local.yaml`:

- `RETRAIN_RADHECK_TEST5` → `${TEST5_WORK_ROOT}/nnunet_retrain`
- `RADHECK_DATASET_TEST5` → `${TEST5_WORK_ROOT}/Dataset650_TotalSegmentator`
- `TEST5_WORK_ROOT` / `TEST5_REFERENCE_DATASET650`
- `TEST5_RADCURE_DATASET366`
- `TEST5_RADCURE_SOURCE` / `TEST5_HECKTOR_SOURCE_CASES_ROOT`

Optional aliases still accepted: `TEST5_RADCURE_SOURCE_MAIN_PATH` (parent
`RadcureComplete` **or** the `TotalSegmentatorRetrain` folder).

---

## Step 1 — Phase 2: transform sources

- **RADCURE:** relabel from existing `total_segmentator_output/` (no TS re-run)
- **HECKTOR:** relabel if TS exists; otherwise stage CT+mask into work root and
  run full TotalSegmentator + improved bg (intermediates deleted after each case)

```bash
python -m pipelines.test5.relabel_tumor_batch --dry-run

# Fresh QC logs if restarting after a bad run:
rm -f ${TEST5_WORK_ROOT}/logs/anatomy_qc/anatomy_qc_decisions.jsonl
rm -f ${TEST5_WORK_ROOT}/anatomy_qc_discarded.csv

python -m pipelines.test5.relabel_tumor_batch --force --anatomy-qc-threshold 0.50
```

Smoke-test a few cases first:

```bash
python -m pipelines.test5.relabel_tumor_batch --max-cases 2
```

Outputs:

- `${TEST5_WORK_ROOT}/TotalSegmentatorRetrain/.../output/`
- `${TEST5_WORK_ROOT}/hecktor/.../output/`
- `${TEST5_WORK_ROOT}/logs/anatomy_qc/anatomy_qc_decisions.jsonl`
- `${TEST5_WORK_ROOT}/anatomy_qc_discarded.csv`
- `${TEST5_WORK_ROOT}/radcure_dictionary_test5.json`

---

## Step 2 — Phase 3: build Dataset650

Same Tr/Va/Ts membership as Test1 (recovered `split_manifest.json` + Dataset366
for RADCURE stems + HECKTOR train/val lists), **minus** QC discards.
Default `--link` hardlinks files to save disk.

```bash
# if you have not run restore yet:
python -m pipelines.test5.restore_split_reference

python -m pipelines.test5.build_dataset650 --dry-run
python -m pipelines.test5.build_dataset650 --link hardlink
# if Phase 2 is incomplete:
# python -m pipelines.test5.build_dataset650 --link hardlink --skip-missing
```

If some HECKTOR stems are not transformed yet, use `--skip-missing` or finish
Phase 2 first (old reference NIfTIs are gone, so there is no label fallback).

Output: `${TEST5_WORK_ROOT}/Dataset650_TotalSegmentator/`

Verify `dataset.json` contains **GTVp** and **GTVn**. Check `split_manifest.json`
for `anatomy_qc_discarded_case_ids`, `copy_source_counts`, and `counts_built`.

---

## Step 3 — Install custom trainer (once)

```bash
python -m nnunet_training.install_trainer_variants
```

---

## Step 4 — Prepare, plan, train

**Do not** reuse Test4 preprocess (deleted / different labels).

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

Optional — same fold assignment as Test3:

```bash
export NNUNET_SPLITS_REFERENCE=/media/HDD_8TB/xisca/work/nnunet_radheck_test_1_retrain/nnUNet_preprocessed
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

---

## Step 6 — Evaluate HECKTOR test (Dataset152)

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
