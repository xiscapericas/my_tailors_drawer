# Test4 Phase 3 — Dataset650 + 700-epoch retrain

Build combined **Dataset650** from Phase 2 relabels, using **the same Tr/Va/Ts splits as Test3**, then train and evaluate.

## Step 1 — Build dataset

```bash
cd /path/to/radcure-medical-imaging
source .venv/bin/activate
set -a && source .env && set +a

export TEST4_WORK_ROOT=/media/HDD_8TB/xisca/work/retrain_test4
export TEST4_REFERENCE_DATASET650=/media/HDD_8TB/xisca/work/nnunet_radheck_test_1/Dataset650_TotalSegmentator
export ORGAN_DICTIONARY_PATH=${TEST4_WORK_ROOT}/radcure_dictionary_test4.json

python -m pipelines.test4.build_dataset650 --dry-run
python -m pipelines.test4.build_dataset650
```

Output: `${TEST4_WORK_ROOT}/Dataset650_TotalSegmentator/`

Check `dataset.json` lists both **GTVp** and **GTVn**.

## Step 2 — Train (700 epochs)

Full server runbook: [`pipelines/radheck/Retrain-Radheck-Test4.md`](../../pipelines/radheck/Retrain-Radheck-Test4.md)

```bash
export DATASET_FOLDER=${TEST4_WORK_ROOT}/Dataset650_TotalSegmentator
export ORGAN_DICTIONARY_PATH=${TEST4_WORK_ROOT}/radcure_dictionary_test4.json
export NNUNET_RETRAIN_PATH=${TEST4_WORK_ROOT}/nnunet_retrain
export NNUNET_TRAINER=nnUNetTrainer_700epochs_NoMirroring

export NNUNET_USE_LOCAL_PREPROCESS=1   # required if .env still has Test3 reuse path

python -m nnunet_training.install_trainer_variants
python train_nnunet.py --step prepare --link-raw
python train_nnunet.py --step plan
python train_nnunet.py --step train
```

**Do not** reuse Test3 `nnUNet_preprocessed` — label set changed (GTVp + GTVn).

## Step 3 — Evaluate

- RADCURE test: `python train_nnunet.py --step evaluate` (+ visualization)
- HECKTOR test (152): `python -m pipelines.hecktor.test_pipeline --predict-only --eval-only` with `DATASET_ID=650`

Record **GTVp Dice** and **GTVn Dice** in `experiments/registry.yaml`.

## Prerequisites

- Phase 2 complete (`relabel_ok.txt`, relabeled `TotalSegmentatorRetrain/` + `hecktor/`)
- Test3 (or Test2) `Dataset650_TotalSegmentator/split_manifest.json` for split pinning
