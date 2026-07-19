# Preprocessing pipeline review

Colab audit of DICOM/NIfTI → nnUNet inputs (RADCURE + HECKTOR), focused on background / `other-tissue`, stable colours, and whether bad labels explain low GTVp Dice.

| File | Purpose |
|------|---------|
| [`preprocessing_pipeline_review_colab.ipynb`](preprocessing_pipeline_review_colab.ipynb) | Drive + install + audit cases, Steps A → E |
| [`FINDINGS.md`](FINDINGS.md) | **Session handoff** — learnings, APIs, next decisions |
| [`../../image_processor/resources/organ_dictionary_hn_canonical.json`](../../image_processor/resources/organ_dictionary_hn_canonical.json) | Fixed H&N label map (TS organs + GTVp/GTVn) |

### Progress

| Step | Topic | Status |
|------|--------|--------|
| A | CT + tumor (align RTSTRUCT, GTVp/GTVn) | Done |
| A2–A3 | Anatomy QC + extra cases | Done |
| B | Anatomical background (intensity + L/R symmetry + Z continuity) | Sweet spot (research) |
| C | Canonical organ dict + TotalSegmentator | **In notebook** — run on Colab |
| D | Combined mask + `other-tissue` + tumor stamp | **In notebook** |
| E | Viz: CT \| organs \| + GTVp(red)/GTVn(pink) | **In notebook** |
| Later | Production bg wiring / DatasetXXX Dice | After C–E QA |

### Audit cases (see FINDINGS for HECKTOR fallbacks)

| Cohort | IDs |
|--------|-----|
| RADCURE | `0122`, `0040`, `0397`, `0151` (+ batch 2: `0005`, `0088`, `0250`) |
| HECKTOR | suffix-resolved CHUM/CHUS/HMR stems; batch 2: `CHUM-013`, `CHUS-016` |

Open in Google Colab. Do not commit AWS credentials.

See also: [`../README.md`](../README.md) · [`../../docs/README.md`](../../docs/README.md) · [`../../docs/documentation-index.md`](../../docs/documentation-index.md)
