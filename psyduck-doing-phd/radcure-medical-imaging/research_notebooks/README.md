# Research notebooks and studies

**Documentation hub:** [`docs/README.md`](../docs/README.md) · Research docs are §3-E in [`docs/documentation-index.md`](../docs/documentation-index.md).

Exploratory work: notebooks, analysis reports, and paper drafts. Pipeline orchestration lives in [`pipelines/`](../pipelines/).

## Notebooks

| Notebook | Purpose |
|----------|---------|
| [`preprocessing_pipeline_review/`](preprocessing_pipeline_review/) | **Colab:** Steps A–E (bg + TS + fixed organ dict + tumor viz); [`FINDINGS.md`](preprocessing_pipeline_review/FINDINGS.md) |
| [`test6_stunet/`](test6_stunet/) | **Test6 research:** AWS download → Test5 preprocess → STU-Net infer + Dice |
| `compare_dice_surface_dice.ipynb` | DICE vs Surface DICE comparison |
| `background_head_organs_debug.ipynb` | Background / head / organs mask pipeline debug |
| `hecktor_explore_colab.ipynb` | HECKTOR case load + overlay (Colab) |
| `hecktor_preprocessing_preview_colab.ipynb` | HECKTOR preprocessing preview (Colab) |

## Markdown studies

| Folder / file | Purpose |
|---------------|---------|
| [`retrain_epoch_study/`](retrain_epoch_study/retrain_epoch_study.md) | Epoch count study (Test1–3 narrative, figures in `images/`) |
| [`Surface_Dice_Analysis_Report.md`](Surface_Dice_Analysis_Report.md) | Surface Dice analysis notes |

## Related (implementation, not notebooks)

| Path | Role |
|------|------|
| [`test4_gtvp_gtvn/`](../research_notebooks/test4_gtvp_gtvn/) | **Test4 Phase 1** — separate GTVp/GTVn preprocessing preview |
| [`pipelines/radheck/`](../pipelines/radheck/) | RADHECK dataset build, leak checks, server runbooks |
| [`experiments/`](../experiments/) | Experiment registry and YAML configs (canonical Test1–3 record) |
| [`docs/PROJECT_LAYOUT.md`](../docs/PROJECT_LAYOUT.md) | Full repo map |

## Guidelines

- Keep notebooks self-contained with clear markdown cells.
- When an idea stabilises, move logic to `image_processor/` or `nnunet_training/` — do not grow permanent code only in notebooks.
- Record completed experiment metrics in [`experiments/registry.yaml`](../experiments/registry.yaml).
