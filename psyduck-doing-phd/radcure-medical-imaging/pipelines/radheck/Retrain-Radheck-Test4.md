# Test4 Phase 3 — train nnUNet (700 epochs, separate GTVp/GTVn)

Build Dataset650 under `work/retrain_test4`, then train and evaluate like Test3.

**Prerequisite:** Phase 2 complete (`relabel_tumor_batch`).

---

## Step 1 — Build Dataset650 (same splits as Test3)

Uses `split_manifest.json` from your **Test3** (or Test2) `Dataset650_TotalSegmentator`.

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

Verify `dataset.json` contains both **GTVp** and **GTVn** labels.

---

## Step 2 — Install custom 700-epoch trainer (once)

```bash
python -m nnunet_training.install_trainer_variants
```

---

## Step 3 — Prepare, plan, train

All under **`work/retrain_test4`** (separate from Test3 weights).

**Important:** Test4 has new labels (GTVp + GTVn). Do **not** reuse Test1/Test3 preprocess.

If your `.env` still sets `NNUNET_PREPROCESSED_PATH` (common after Test3), shell `unset`
is **not enough** — Python reloads it from `.env`. Use either:

```bash
export NNUNET_USE_LOCAL_PREPROCESS=1
```

or comment out `NNUNET_PREPROCESSED_PATH` in `.env`.

```bash
export DATASET_FOLDER=${TEST4_WORK_ROOT}/Dataset650_TotalSegmentator
export ORGAN_DICTIONARY_PATH=${TEST4_WORK_ROOT}/radcure_dictionary_test4.json
export NNUNET_RETRAIN_PATH=${TEST4_WORK_ROOT}/nnunet_retrain
export NNUNET_PATH=/path/to/nnUNet

export NNUNET_TRAINER=nnUNetTrainer_700epochs_NoMirroring
export NNUNET_CONFIGURATION=3d_fullres
export NNUNET_FOLD=0
export LOG_DIR=${NNUNET_RETRAIN_PATH}/logs
export NNUNET_USE_LOCAL_PREPROCESS=1

export nnUNet_compile=false
export CUDA_VISIBLE_DEVICES=1

mkdir -p "$NNUNET_RETRAIN_PATH" "$LOG_DIR"

# Sanity check — train must NOT point at test_1_retrain
echo "DATASET_FOLDER=$DATASET_FOLDER"
echo "NNUNET_RETRAIN_PATH=$NNUNET_RETRAIN_PATH"
echo "NNUNET_USE_LOCAL_PREPROCESS=$NNUNET_USE_LOCAL_PREPROCESS"

python train_nnunet.py --step prepare --link-raw
python train_nnunet.py --step plan
python train_nnunet.py --step train
```

After `plan`, confirm preprocessed files exist, e.g.:

```bash
ls ${NNUNET_RETRAIN_PATH}/nnUNet_preprocessed/Dataset650_TotalSegmentator/nnUNetPlans_3d_fullres/*.b2nd | wc -l
```

Expect **466** (= 2 × 233 `imagesTr` cases). `splits_final.json` is created automatically
at train time (or copied from Test3 if you set `NNUNET_SPLITS_REFERENCE`).

Optional — reuse the same fold assignment as Test3:

```bash
export NNUNET_SPLITS_REFERENCE=/media/HDD_8TB/xisca/work/nnunet_radheck_test_1_retrain/nnUNet_preprocessed
```

Model output:

```text
${NNUNET_RETRAIN_PATH}/nnUNet_results/Dataset650_TotalSegmentator/
  nnUNetTrainer_700epochs_NoMirroring__nnUNetPlans__3d_fullres/fold_0/
```

---

## Step 4 — Evaluate

### RADCURE test (650 imagesTs)

```bash
export DATASET_FOLDER=${TEST4_WORK_ROOT}/Dataset650_TotalSegmentator
export NNUNET_RETRAIN_PATH=${TEST4_WORK_ROOT}/nnunet_retrain
export NNUNET_TRAINER=nnUNetTrainer_700epochs_NoMirroring
export CUDA_VISIBLE_DEVICES=1
export nnUNet_compile=false

python train_nnunet.py --step evaluate
python train_nnunet.py --step evaluation_visualization
```

Check Dice CSV for **GTVp** and **GTVn** separately.

### HECKTOR test (Dataset152, model 650)

```bash
export DATASET_ID=650
export DATASET_FOLDER=/media/HDD_8TB/xisca/work/nnunet_hecktor_test1/Dataset152_TotalSegmentator
export NNUNET_RETRAIN_PATH=${TEST4_WORK_ROOT}/nnunet_retrain
export ORGAN_DICTIONARY_PATH=${TEST4_WORK_ROOT}/radcure_dictionary_test4.json
export NNUNET_TRAINER=nnUNetTrainer_700epochs_NoMirroring
export HECKTOR_EVAL_OUTPUT_DIR=${TEST4_WORK_ROOT}/nnunet_retrain/hecktor_validation
export CUDA_VISIBLE_DEVICES=1
export nnUNet_compile=false

python -m pipelines.hecktor.test_pipeline --predict-only
python -m pipelines.hecktor.test_pipeline --eval-only
```

---

## Step 5 — Record results

Update `experiments/registry.yaml` → `test4.results` (GTVp and GTVn Dice for RADCURE + HECKTOR).

---

## Layout after Phase 3

```
work/retrain_test4/
├── TotalSegmentatorRetrain/     # Phase 2 RADCURE relabels
├── hecktor/                     # Phase 2 HECKTOR relabels
├── Dataset650_TotalSegmentator/ # Phase 3 combined dataset
├── radcure_dictionary_test4.json
└── nnunet_retrain/              # Phase 3 weights + preprocess + results
    ├── nnUNet_preprocessed/
    ├── nnUNet_results/
    └── hecktor_validation/
```
