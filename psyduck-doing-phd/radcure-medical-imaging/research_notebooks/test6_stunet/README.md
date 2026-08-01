# Test6 (research) — STU-Net exploration

**Status:** research / inference-only (not a full train–eval experiment yet)

**One-line rule (planned):** Test6 explores **STU-Net** pretrained on TotalSegmentator as an alternative large-scale organ segmentor compared to our nnUNet RADHECK pipeline; everything about Dataset650 training stays unchanged until we decide to fine-tune.

Paper / code: [arXiv:2304.06716](https://arxiv.org/abs/2304.06716) · [uni-medical/STU-Net](https://github.com/uni-medical/STU-Net)

## What this folder is for

| File | Purpose |
|------|---------|
| [`test6_stunet_inference_explore.ipynb`](test6_stunet_inference_explore.ipynb) | Repo setup → AWS download → **Test5 preprocess** → STU-Net predict → Dice + viz |
| [`label_orders.json`](label_orders.json) | STU-Net TotalSegmentator class index → name (differs from our H&N dict) |
| This README | Scope, expectations, next steps |

## Notebook flow

1. **Repo + setup** — Colab Drive / clone, or local `RADCURE_REPO` + editable install + Totalsegmentator  
2. **AWS download** — RADCURE zips via `AWSHandler`; HECKTOR from `HECKTOR_S3_URI`  
3. **Test5 preprocess** — `CaseProcessor` with `tumor_label_mode=separate`, `background_mode=improved`, `anatomy_qc_threshold=0.50`, canonical H&N dict  
4. **STU-Net** — clone repo, download weights (default **STU-Net-S**), stage nnUNet CTs, `nnUNet_predict`  
5. **Dice + viz** — name-matched organ Dice vs our GT labels; tumor overlay for context only  

## Critical scope notes

1. **Pretrained STU-Net does not predict GTVp / GTVn.**  
   It was trained on TotalSegmentator’s **104 anatomical structures**. Tumor Dice is **out of scope** until we fine-tune on RADHECK labels.

2. **Our H&N organ set ≠ STU-Net’s 104 classes.**  
   Overlap is partial. The notebook reports Dice **only on name-matched organs**.

3. **Research only.** No production `CaseProcessor` change; no Dataset650 rebuild required for this notebook.

## Suggested first run

### Colab

Open the notebook, run §0 (Drive → clone → install → restart → import), then §1–7 with AWS secrets in Colab userdata or prompts.

### Cluster

```bash
cd /path/to/radcure-medical-imaging
source .venv/bin/activate

export TEST6_WORK_ROOT=/media/HDD_8TB/xisca/work/research_test6_stunet
export RADCURE_REPO=/path/to/radcure-medical-imaging
export NNUNET_PATH=/media/HDD_8TB/xisca/code/nnUNet   # nnUNet v1 + STU trainers
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_BUCKET_NAME=xisca-lab
# optional: TEST6_STU_VARIANT=base  TEST6_ANATOMY_QC=0.50

jupyter notebook research_notebooks/test6_stunet/test6_stunet_inference_explore.ipynb
```

Default sample IDs in the notebook: `RADCURE-0122`, `RADCURE-0040`, `HMR-012`, `CHUM-023` (HECKTOR centers may fall back by numeric suffix).

## Outputs (under `TEST6_WORK_ROOT`)

```
research_test6_stunet/
├── preprocess/
│   ├── TotalSegmentatorRetrain/{RADCURE-ID}/output/{image,labels}/
│   └── hecktor/{ID}/output/{image,labels}/
├── organ_dictionary_test6.json
├── STU-Net/                 # cloned repo
├── weights/                 # downloaded checkpoint
├── results_folder/          # RESULTS_FOLDER layout for nnUNet_predict
├── inputs/                  # staged case_*_0000.nii.gz
├── predictions/             # STU-Net multilabel NIfTIs
├── dice/stunet_sample_dice.csv
└── figures/
```

## STU-Net layout notes (common Colab pitfalls)

Upstream [STU-Net `plan_files/`](https://github.com/uni-medical/STU-Net/tree/main/plan_files) is **flat**:

- `plans.pkl` (shared by all variants)
- `small_ep4k.model.pkl`, `base_ep4k.model.pkl`, …

The nested `STUNetTrainer_*__nnUNetPlansv2.1/` tree is what we **build** under `results_folder/`, not what the clone ships.

On Colab, the cluster default `NNUNET_PATH=/media/HDD_8TB/...` will not exist. Section 4 falls back to `STU-Net/nnUNet-1.7.1`. Install it once so `nnUNet_predict` is available:

```bash
pip install -e "$TEST6_WORK_ROOT/STU-Net/nnUNet-1.7.1"
```

(or set `NNUNET_PATH` to that bundled folder before section 4).

## Later (not this notebook)

- Fine-tune STU-Net on Dataset650 with separate GTVp/GTVn  
- Full dual-cohort eval vs Test4/Test5  
- Registry status `planned` → `running` when that starts  

See [`experiments/configs/test6_stunet_inference_explore.yaml`](../../experiments/configs/test6_stunet_inference_explore.yaml).
