# Preprocessing pipeline review

Colab notebook to audit DICOM/NIfTI → nnUNet inputs for RADCURE and HECKTOR.

**Focus:** background / `other-tissue`, stable colours, whether bad labels explain low GTVp Dice.

| File | Purpose |
|------|---------|
| [`preprocessing_pipeline_review_colab.ipynb`](preprocessing_pipeline_review_colab.ipynb) | Drive + install + 4 RADCURE + 4 HECKTOR audit cases, step-by-step pipeline |

### Audit cases

| Cohort | IDs |
|--------|-----|
| RADCURE | `RADCURE-0122`, `0040`, `0397`, `0151` |
| HECKTOR | provisional `HMR-012`, `CHUM-023`, `CHUM-098`, `HMR-057` (stems `case_012/023/098/057`; falls back to any matching `*-NNN`) |

Open in Google Colab. Do not commit AWS credentials.

See also: [`../README.md`](../README.md) · [`../../docs/README.md`](../../docs/README.md)
