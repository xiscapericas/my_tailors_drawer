# Test6 (research) — STU-Net exploration

**Status:** research / inference-only (not a full train–eval experiment yet)

**One-line rule (planned):** Test6 explores **STU-Net** pretrained on TotalSegmentator as an alternative large-scale organ segmentor compared to our nnUNet RADHECK pipeline; everything about Dataset650 training stays unchanged until we decide to fine-tune.

Paper / code: [arXiv:2304.06716](https://arxiv.org/abs/2304.06716) · [uni-medical/STU-Net](https://github.com/uni-medical/STU-Net)

## What this folder is for

| File | Purpose |
|------|---------|
| [`test6_stunet_inference_explore.ipynb`](test6_stunet_inference_explore.ipynb) | Download weights, run inference on a few RADCURE + HECKTOR CTs, Dice + viz |
| [`label_orders.json`](label_orders.json) | STU-Net TotalSegmentator class index → name (differs from our H&N dict) |
| This README | Scope, expectations, next steps |

## Critical scope notes

1. **Pretrained STU-Net does not predict GTVp / GTVn.**  
   It was trained on TotalSegmentator’s **104 anatomical structures** (abdomen / body atlas style: liver, vertebrae, brain, …). Tumor Dice is **out of scope** until we fine-tune on RADHECK labels.

2. **Our H&N organ set ≠ STU-Net’s 104 classes.**  
   Overlap is partial (e.g. brain, trachea, esophagus, cervical vertebrae, clavicle, …). The notebook reports Dice **only on name-matched organs** and lists unmatched classes.

3. **Research only.** No production `CaseProcessor` change; no Dataset650 rebuild required for this notebook.

## Suggested first run (cluster)

Use **STU-Net-S** (14.6M) for a fast smoke test, then optionally **STU-Net-B**.

```bash
cd /path/to/radcure-medical-imaging
source .venv/bin/activate

export TEST6_WORK_ROOT=/media/HDD_8TB/xisca/work/research_test6_stunet
# CT samples: nnUNet-style case_*_0000.nii.gz
export TEST6_RADCURE_IMAGE=.../Dataset650_TotalSegmentator/imagesTs/case_0122_0000.nii.gz
export TEST6_HECKTOR_IMAGE=.../Dataset152_TotalSegmentator/imagesTs/case_XXX_0000.nii.gz
# Optional GT labels for Dice (same stem without _0000):
export TEST6_RADCURE_LABEL=.../labelsTs/case_0122.nii.gz
export TEST6_HECKTOR_LABEL=.../labelsTs/case_XXX.nii.gz

# Open and run the notebook, or follow its cells as a script
jupyter notebook research_notebooks/test6_stunet/test6_stunet_inference_explore.ipynb
```

## Outputs (under `TEST6_WORK_ROOT`)

```
research_test6_stunet/
├── weights/                 # downloaded STU-Net checkpoint + plans
├── inputs/                  # copied sample CTs (nnUNet naming)
├── predictions/             # STU-Net multilabel NIfTIs
├── dice/                    # per-case CSV + summary
└── figures/                 # CT | pred organs | GT (+ tumor highlight)
```

## Later (not this notebook)

- Fine-tune STU-Net on Dataset650 with separate GTVp/GTVn  
- Full dual-cohort eval vs Test4/Test5  
- Registry status `planned` → `running` when that starts  

See [`experiments/configs/test6_stunet_inference_explore.yaml`](../../experiments/configs/test6_stunet_inference_explore.yaml).
