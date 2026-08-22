# Test 8.0 (research) — HECKTOR PET exploration

**Status:** Phase 1 explore. **Training is Test 8.0 implementation** — see:

→ [`pipelines/radheck/Retrain-Radheck-Test8.0.md`](../../pipelines/radheck/Retrain-Radheck-Test8.0.md)

**One-line rule:** Test **8.0** adds a **PET input channel** vs Test5 and keeps only **HECKTOR** cases from the Test5 Tr/Va/Ts split (RADCURE has no PET).

## What this folder is for

| File | Purpose |
|------|---------|
| [`test8_0_pet_explore.ipynb`](test8_0_pet_explore.ipynb) | Expected PET format, load in Python, geometry vs CT, overlay, missing `__PT` inventory |
| This README | Pointers |

## Expected HECKTOR PET format

Per case folder (HECKTOR 2025 Task 1):

- `{case_id}__CT.nii.gz` — CT
- `{case_id}__PT.nii.gz` — FDG PET, **SUV**, NIfTI
- `{case_id}.nii.gz` — labels (GTVp=1, GTVn=2)

Same-session PET/CT: no deformable registration, but **voxel size often differs** — resample PET onto the CT grid (SimpleITK linear) before using it as nnUNet channel 1.

Do **not** apply the CT 1–99 percentile → [0, 1] display normalization when saving the PET training channel.

## Open in Python

Shared helpers: [`image_processor/io/pet_align.py`](../../image_processor/io/pet_align.py) and [`get_hecktor_paths`](../../image_processor/conventions.py).
