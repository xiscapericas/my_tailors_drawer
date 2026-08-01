# Test6 (research) — STU-Net exploration

**Status:** inference explore **done** (chapter closed). Fine-tune / synonym map deferred until **after Test5 results**.

**One-line rule:** Test6 explores **STU-Net** pretrained on TotalSegmentator as an alternative large-scale organ segmentor compared to our nnUNet RADHECK pipeline; Dataset650 training stays unchanged until we decide to fine-tune.

Paper / code: [arXiv:2304.06716](https://arxiv.org/abs/2304.06716) · [uni-medical/STU-Net](https://github.com/uni-medical/STU-Net)

## What this folder is for

| File | Purpose |
|------|---------|
| [`test6_stunet_inference_explore.ipynb`](test6_stunet_inference_explore.ipynb) | Repo setup → AWS download → **Test5 preprocess** → STU-Net predict → Dice + viz |
| [`label_orders.json`](label_orders.json) | STU-Net TotalSegmentator class index → name (≠ our H&N dict) |
| This README | Scope, first-run findings, Colab pitfalls, next steps |

## Notebook flow

1. **Repo + setup** — Colab Drive / clone, or local `RADCURE_REPO` + editable install + Totalsegmentator  
2. **AWS download** — RADCURE zips via `AWSHandler`; HECKTOR from `HECKTOR_S3_URI`  
3. **Test5 preprocess** — `CaseProcessor` with `tumor_label_mode=separate`, `background_mode=improved`, `anatomy_qc_threshold=0.50`, canonical H&N dict  
4. **STU-Net** — clone repo, weights (default **STU-Net-S**), stage nnUNet CTs, `nnUNet_predict`  
5. **Dice + viz** — naive name-matched Dice (often empty); viz shows **raw STU labels** on best-z slice  

## Critical scope notes

1. **Pretrained STU-Net does not predict GTVp / GTVn.** Tumor Dice needs fine-tuning on RADHECK labels.  
2. **Our H&N organ set ≠ STU-Net’s 104 classes.** Naive string match found **0** synonyms on the first Colab run.  
3. **Research only.** No production `CaseProcessor` change; no Dataset650 rebuild for this notebook.

## First-run findings (Colab, STU-Net-S, 4 cases)

Samples: `RADCURE-0122`, `RADCURE-0040`, `HMR-012`, `CHUM-023` (Test5 preprocess).

| Observation | Detail |
|-------------|--------|
| Name matches | **0** / 104 STU vs ~88 H&N organs (exact `norm_name` intersection) |
| Matched organ Dice | `nan` (no pairs to score) |
| Tumor Dice | N/A — expected |
| Labels per case | Very few (≈0–3 unique non-zero IDs) — whole-body TS model on H&N FOV |
| Viz mid-slice | Often **blank** — use z with most STU foreground voxels |

**Interpretation:** Pretrained STU-S runs end-to-end on our pipeline, but it is not yet a drop-in H&N organ scorer. Closing this explore chapter; next research steps wait on Test5.

## Lessons learnt (Colab / nnUNet v1)

| Pitfall | Fix (now in notebook §4–§5 / §7) |
|---------|----------------------------------|
| `plans.pkl` assert | Upstream `plan_files/` is **flat** (`plans.pkl` + `*_ep4k.model.pkl`). Do not require `STU_VARIANT` in the path. |
| `SameFileError` copying trainers | When `NNUNET_PATH` = bundled `STU-Net/nnUNet-1.7.1`, skip copy — trainers already present. |
| GDrive “zip” extract finds no `.model` | Download **is** the checkpoint; PyTorch files are zip-format — do **not** `extractall` unless members end in `.model`. |
| `nnUNet_predict` missing | `pip install -e $TEST6_WORK_ROOT/STU-Net/nnUNet-1.7.1` (Colab has no cluster `NNUNET_PATH`). |
| `ModuleNotFoundError: torchinfo` | `pip install torchinfo` (imported by `STUNetTrainer`). |
| PyTorch ≥ 2.6 `UnpicklingError` | `torch.load(..., weights_only=False)` — §5 monkeypatches / patches `model_restore.py`. |
| Empty STU viz panel | Choose **best-z** (max STU foreground), not mid-volume; print present class names. |
| Cluster path on Colab | Default `/media/HDD_8TB/...` does not exist — fall back to bundled nnUNet. |

## How to re-run

### Colab

Open the notebook, §0 (Drive → clone → install → restart → import), then §1–7. Prefer the **repo copy** of the notebook (synced fixes); old Colab cells hide stderr and miss the `torch.load` patch.

### Cluster

```bash
cd /path/to/radcure-medical-imaging
source .venv/bin/activate

export TEST6_WORK_ROOT=/media/HDD_8TB/xisca/work/research_test6_stunet
export RADCURE_REPO=/path/to/radcure-medical-imaging
export NNUNET_PATH=/media/HDD_8TB/xisca/code/nnUNet   # or bundled STU-Net/nnUNet-1.7.1
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_BUCKET_NAME=xisca-lab

jupyter notebook research_notebooks/test6_stunet/test6_stunet_inference_explore.ipynb
```

## Outputs (under `TEST6_WORK_ROOT`)

```
research_test6_stunet/
├── preprocess/
│   ├── TotalSegmentatorRetrain/{RADCURE-ID}/output/{image,labels}/
│   └── hecktor/{ID}/output/{image,labels}/
├── organ_dictionary_test6.json
├── STU-Net/                 # cloned repo (+ bundled nnUNet-1.7.1)
├── weights/                 # downloaded checkpoint
├── results_folder/          # RESULTS_FOLDER layout for nnUNet_predict
├── inputs/                  # staged case_*_0000.nii.gz
├── predictions/             # STU-Net multilabel NIfTIs
├── dice/stunet_sample_dice.csv
└── figures/
```

## Next (after Test5 results — do not start yet)

1. **Synonym map** — STU-Net 104 names ↔ our H&N dictionary (aliases, laterality, TS naming). Re-score organ Dice on the same samples.  
2. **Fine-tune** STU-Net on Dataset650 with separate GTVp/GTVn → dual-cohort eval vs Test4/Test5.  
3. Promote to a full experiment in `pipelines/` + registry (`status: running`) only when fine-tune starts.

Config / registry: [`experiments/configs/test6_stunet_inference_explore.yaml`](../../experiments/configs/test6_stunet_inference_explore.yaml) · [`experiments/registry.yaml`](../../experiments/registry.yaml) (`test6`).
