# Test7 — probability outputs (Test5 model, Test6 / Test5 splits)

**One-line rule:** Test7 changes **inference/evaluation** to **per-class
probability outputs** (plus region-vs-GTVp curves and alpha visualization)
compared to Test5; Dataset650 / Tr/Va/Ts (same as Test6), Test5 nnUNet
weights, and preprocess are **identical** — **no** TotalSegmentator
reprocess, **no** retrain.

Config: [`experiments/configs/test7_radheck_probability_outputs.yaml`](../../experiments/configs/test7_radheck_probability_outputs.yaml)

| Role | Path |
|------|------|
| Work root | `/media/HDD_8TB/xisca/work/retrain_test7_prob` |
| Dataset650 (reuse) | `/media/HDD_8TB/xisca/work/retrain_test5/Dataset650_TotalSegmentator` |
| Model weights | `${RETRAIN_RADHECK_TEST5}/nnUNet_results` (Test5) |
| Predictions | `${TEST7_WORK_ROOT}/predictions/` |

Trainer: **`nnUNetTrainer_700epochs_NoMirroring`** (Test5 — not STU-Net).

---

## Step 0 — Env

```bash
cd /path/to/radcure-medical-imaging
source .venv/bin/activate          # MUST be .venv, not system / ~/.local
set -a && source .env && set +a

# Confirm you are NOT on ~/.local (this caused the blosc2/numpy failure):
which python
which nnUNetv2_predict
# both should be under …/radcure-medical-imaging/.venv/bin/

export TEST7_WORK_ROOT=/media/HDD_8TB/xisca/work/retrain_test7_prob
export TEST5_WORK_ROOT=/media/HDD_8TB/xisca/work/retrain_test5
export TEST7_DATASET650=/media/HDD_8TB/xisca/work/retrain_test5/Dataset650_TotalSegmentator
export RETRAIN_RADHECK_TEST5=/media/HDD_8TB/xisca/work/retrain_test5/nnunet_retrain
export CUDA_VISIBLE_DEVICES=1
export nnUNet_compile=false
mkdir -p "$TEST7_WORK_ROOT"

df -h /media/HDD_8TB   # raw .npz is large briefly; slim crops are much smaller

# Once per env (custom 700-epoch trainer):
python -m nnunet_training.install_trainer_variants
```

If you still see `numpy.dtype size changed` / blosc2 errors **inside** `.venv`:

```bash
pip install --force-reinstall --no-cache-dir numpy blosc2
```

In `experiments/configs/local.yaml` (optional but recommended):

- `TEST7_WORK_ROOT`, `TEST7_DATASET650`, `RETRAIN_RADHECK_TEST5`
- `RETRAIN_RADHECK_TEST7` → `${TEST7_WORK_ROOT}/nnunet_retrain`
- `RADHECK_DATASET_TEST7` → `${TEST7_WORK_ROOT}/Dataset650_TotalSegmentator`

---

## Step 1 — Link Dataset650 (no reprocess)

Same Tr / Va / Ts as Test5 / Test6. Clean selective links (not a whole-folder
symlink into Test5).

```bash
python -m pipelines.test7.link_dataset
source ${TEST7_WORK_ROOT}/TEST7_ENV.sh

ls -la ${TEST7_WORK_ROOT}/Dataset650_TotalSegmentator
echo "Tr=$(ls ${TEST7_WORK_ROOT}/Dataset650_TotalSegmentator/imagesTr/*_0000.nii.gz | wc -l)"
echo "Ts=$(ls ${TEST7_WORK_ROOT}/Dataset650_TotalSegmentator/imagesTs/*_0000.nii.gz | wc -l)"
```

---

## Step 2 — Predict + slim probabilities (Test5 model)

Hard masks → `predictions/labelsTs_predicted/`  

nnUNet writes large raw `.npz` softmax dumps, then Test7 **immediately
converts** them to cropped float16 slim archives and **deletes** the raw files
(default):

`predictions/labelsTs_probabilities/{case}.slim.npz`

Each slim file keeps:

- bbox around dilated GT GTVp (fallback: high `P(GTVp)` / argmax)
- `p_gtvp` + overlapping organs + GTVn + top-k classes by mean P in the crop
- float16 channels only inside the crop

```bash
source ${TEST7_WORK_ROOT}/TEST7_ENV.sh
python -m pipelines.test7.predict_probabilities
# equivalent separate step if you used --skip-slim:
# python -m pipelines.test7.slim_probabilities
```

Confirm:

```bash
ls ${TEST7_WORK_ROOT}/predictions/labelsTs_predicted/*.nii.gz | wc -l
ls ${TEST7_WORK_ROOT}/predictions/labelsTs_probabilities/*.slim.npz | wc -l
# raw .npz should be gone unless --keep-raw
ls ${TEST7_WORK_ROOT}/predictions/labelsTs_probabilities/*.npz 2>/dev/null | wc -l
cat ${TEST7_WORK_ROOT}/predictions/labelsTs_probabilities/slim_STATUS.json | head
```

Optional flags:

```bash
python -m pipelines.test7.predict_probabilities --slim-margin 12 --slim-top-k 8
python -m pipelines.test7.slim_probabilities --keep-raw   # debug only
```

---

## Step 3 — region_tumor_probabilities_vs_dice_curves

For each organ overlapping dilated GTVp GT:

- **X** = `P(GTVp) − P(region)`
- **Y** = 1 if GT label is GTVp, else 0

Also aggregates the **P(GTVp) ≥ 0.80** hypothesis (even when another class has
higher probability) with a binomial p-value vs 0.5.

```bash
python -m pipelines.test7.region_tumor_probabilities_vs_dice_curves
# optional smoke: --max-cases 5
```

Outputs under `${TEST7_WORK_ROOT}/region_tumor_probabilities_vs_dice_curves/`:

| File | Content |
|------|---------|
| `summary.json` | Aggregate hypothesis + definitions |
| `pooled_region_curves.png` | Curves for all competing regions |
| `figures/{region}_curve.png` | Per-region plots |
| `hypothesis_per_case.csv` | Per-case 0.80 rule stats |
| `pooled_curves.json` | Binned rates |

---

## Step 4 — probability_visualisation

Same organ colours as hard-label viz; **alpha = P(class)** per voxel.

```bash
python -m pipelines.test7.probability_visualisation
# optional: --max-cases 3 --max-slices 12
```

PDFs → `${TEST7_WORK_ROOT}/predictions/labelsTs_probability_viz/`.

---

## What not to run

- Do **not** run Test5 `transform_cases` / `build_datasets` / plan / train.
- Do **not** run Test6 STU-Net setup / fine-tune.
- Hard **Dice is not** the primary metric for Test7 (curves + hypothesis are).

---

## After the run

1. Update `experiments/registry.yaml` → `test7.status` and `results` (hypothesis summary).
2. Mirror notes in `experiments/configs/test7_radheck_probability_outputs.yaml`.
3. Do not commit `local.yaml`, `.env`, or probability `.npz` dumps.
