# Test6 — STU-Net fine-tune on Test5 RADHECK Dataset650 (GTVp/GTVn)

**One-line rule:** Test6 changes the **model** to **STU-Net fine-tuning**
(TotalSegmentator-pretrained → our labels) compared to Test5; Dataset650,
separate GTVp/GTVn, improved preprocess, and the unified Tr/Ts split are
**identical** (no TotalSegmentator reprocess).

Prior research (inference-only explore):  
[`research_notebooks/test6_stunet/README.md`](../../research_notebooks/test6_stunet/README.md)

| Role | Path |
|------|------|
| Work root | `/media/HDD_8TB/xisca/work/retrain_test6_stunet` |
| Dataset650 (reuse) | `/media/HDD_8TB/xisca/work/retrain_test5/Dataset650_TotalSegmentator` |
| STU-Net clone | `${TEST6_WORK_ROOT}/STU-Net` |
| nnUNet retrain | `${TEST6_WORK_ROOT}/nnunet_retrain` |

Default variant: **STU-Net-S** (`small`). Override with `TEST6_STU_VARIANT=base|large|huge`.

---

## Step 0 — Env

```bash
cd /path/to/radcure-medical-imaging
source .venv/bin/activate
set -a && source .env && set +a

export TEST6_WORK_ROOT=/media/HDD_8TB/xisca/work/retrain_test6_stunet
export TEST6_DATASET650=/media/HDD_8TB/xisca/work/retrain_test5/Dataset650_TotalSegmentator
export TEST6_STU_VARIANT=small
export CUDA_VISIBLE_DEVICES=1
export nnUNet_compile=false
mkdir -p "$TEST6_WORK_ROOT"

df -h /media/HDD_8TB   # confirm free space before plan/train
```

In `experiments/configs/local.yaml`:

- `RETRAIN_RADHECK_TEST6` → `${TEST6_WORK_ROOT}/nnunet_retrain`
- `RADHECK_DATASET_TEST6` → `${TEST6_WORK_ROOT}/Dataset650_TotalSegmentator` (**clean** dir: linked `images*`/`labels*` only)
- `TEST6_WORK_ROOT` / `TEST6_DATASET650` / `TEST6_STU_VARIANT`
- Organ dict from Test5 `RADHECK_CURRENT` / `RADHECK_*` (e.g. `RADHECK_1047`)

---

## Step 1 — Install STU-Net + weights

Clones [uni-medical/STU-Net](https://github.com/uni-medical/STU-Net), `pip install -e` its
**nnUNet-2.2**, installs `torchinfo` / `gdown`, downloads pretrained weights.

```bash
python -m pipelines.test6.setup_stunet
# writes ${TEST6_WORK_ROOT}/TEST6_ENV.sh
source ${TEST6_WORK_ROOT}/TEST6_ENV.sh
```

---

## Step 2 — Link Test5 Dataset650 (no reprocess)

Same Tr / Ts as Test5, but **not** a whole-folder symlink into Test5 (that would
expose Test5 `labelsTs_predicted` / dice viz and write Test6 outputs back).

Creates a clean `${TEST6_WORK_ROOT}/Dataset650_TotalSegmentator` with:
- symlinks: `imagesTr`, `labelsTr`, `imagesTs`, `labelsTs` (+ Va if present)
- copies: `dataset.json`, `case_map.json`, `ts_case_map.json`
- organ dict copied from Test5 `RADHECK_*/organ_dictionary_test5.json`

```bash
export TEST5_WORK_ROOT=/media/HDD_8TB/xisca/work/retrain_test5
rm -f ${TEST6_WORK_ROOT}/organ_dictionary_test5.json
# if an old whole-folder symlink exists:
rm -rf ${TEST6_WORK_ROOT}/Dataset650_TotalSegmentator

python -m pipelines.test6.link_test5_dataset

ls -la ${TEST6_WORK_ROOT}/Dataset650_TotalSegmentator
# should be a directory, not a symlink to retrain_test5
echo "Tr=$(ls ${TEST6_WORK_ROOT}/Dataset650_TotalSegmentator/imagesTr/*_0000.nii.gz | wc -l)"
echo "Ts=$(ls ${TEST6_WORK_ROOT}/Dataset650_TotalSegmentator/imagesTs/*_0000.nii.gz | wc -l)"
```

---

## Step 3 — Prepare, plan, fine-tune

STU-Net needs its **own** plan/preprocess under `TEST6` (do not point
`NNUNET_PREPROCESSED_PATH` at Test5 — different trainer/architecture).

```bash
source ${TEST6_WORK_ROOT}/TEST6_ENV.sh
export CUDA_VISIBLE_DEVICES=1

python -m pipelines.test6.train_finetune --step prepare
python -m pipelines.test6.train_finetune --step plan
python -m pipelines.test6.train_finetune --step train
```

Train uses:

```text
run_finetuning_stunet.py 650 3d_fullres 0 \
  -tr STUNetTrainer_small_ft \
  -pretrained_weights …/weights/small_ep4k.model
```

Encoder/decoder load from TotalSegmentator pretrain; **seg head** is new for our
GTVp/GTVn (+ organs) dictionary. Default FT epochs = **1000** (STU-Net trainer).

If weight loading fails with a shape mismatch on non-`seg_outputs` keys, the
Dataset650 plans’ pooling may differ from TotalSegmentator — open an issue in
chat with the traceback; we may need to align pool strides.

Optional: reuse Test5 `splits_final.json` if present:

```bash
export TEST6_SPLITS_REFERENCE=${TEST5_WORK_ROOT}/nnunet_retrain/nnUNet_preprocessed
```

---

## Step 4 — Evaluate (one run, unified Ts)

Same as Test5: RADCURE manifesto Ts + HECKTOR held-out in `imagesTs`.

```bash
source ${TEST6_WORK_ROOT}/TEST6_ENV.sh
export CUDA_VISIBLE_DEVICES=1

python -m pipelines.test6.evaluate
python -m pipelines.test6.evaluate --skip-predict --viz
```

Dice CSV under `${NNUNET_RETRAIN_PATH}/logs/`; cohort columns when
`ts_case_map.json` is present.

---

## After the run

1. Fill `experiments/registry.yaml` → `test6.results` (GTVp/GTVn Dice)
2. Fill `experiments/configs/test6_radheck_stunet_finetune.yaml` → `results`
3. Compare to Test5 (same Ts) in a short research note

## Notes

- Disk: plan/preprocess for ~full Tr is large — free space on `/media/HDD_8TB` first.
- Explore notebook stays research-only; this runbook is the **implementation** path.
- Do **not** re-run `transform_cases` / TotalSegmentator for Test6.

### Troubleshooting: plan looks under `nnunet_radheck_test_1`

A leftover `DATASET_FOLDER` from Test1 in the shell/`.env` used to win.
Test6 now prefers `${TEST6_WORK_ROOT}/Dataset650_TotalSegmentator`. Still safest:

```bash
unset DATASET_FOLDER
source ${TEST6_WORK_ROOT}/TEST6_ENV.sh
python -m pipelines.test6.link_test5_dataset   # if Dataset650 symlink missing
python -m pipelines.test6.train_finetune --step plan
```

### Troubleshooting: unexpected labels (e.g. Found `92`)

You do **not** need to re-run TotalSegmentator. Masks already contain
`GTVp=91` / `GTVn=92`; the failure means `dataset.json` only listed labels
`0–90`. `train_finetune` now rewrites `dataset.json` from the Test5 organ
dictionary before plan. Re-run:

```bash
python -m pipelines.test6.train_finetune --step plan
```

### Troubleshooting: `numpy.dtype size changed` / blosc2

Plan/predict must use the **same** Python as `python -m pipelines.test6…`,
not `~/.local/bin/nnUNetv2_*` (often a different Python 3.9 user install).

```bash
# Prefer project venv
source /path/to/radcure-medical-imaging/.venv/bin/activate   # if you use one
which python
python -c "import sys; print(sys.executable)"

# If blosc2/numpy still clash in *this* env:
pip install --force-reinstall --no-cache-dir numpy blosc2

# Re-run (train_finetune now invokes nnUNet via this Python, not PATH)
python -m pipelines.test6.train_finetune --step plan
```
