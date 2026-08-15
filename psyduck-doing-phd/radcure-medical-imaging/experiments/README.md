# Experiments

Central registry and configs for RADCURE / HECKTOR (RADHECK) segmentation experiments.

**Documentation hub:** [`docs/README.md`](../docs/README.md) · This folder is §3-C in [`docs/documentation-index.md`](../docs/documentation-index.md).

## Pipeline (two stages)

| Stage | What it does | Entry points |
|-------|----------------|--------------|
| **1. Preprocessing** | DICOM/NIfTI → TotalSegmentator → combined mask (organs, head/body, background, **GTVp**) | `process_all_cases.py`, `pipelines.radheck.build_nnunet_dataset` |
| **2. nnUNet** | Train on `labelsTr`/`Va`, predict test, **GTVp Dice** (+ Surface Dice, slice PDFs) | `train_nnunet.py`, `pipelines.hecktor.test_pipeline` |

**Primary metric:** GTVp Dice on both test pools:

- **RADCURE test** — `Dataset650/imagesTs` (held-out RADCURE from 366 splits)
- **HECKTOR test** — `Dataset152/imagesTs` (never in training)

## Files

| File | Purpose |
|------|---------|
| [`registry.yaml`](registry.yaml) | All experiments, hypotheses, results, path template keys |
| [`configs/test*.yaml`](configs/) | Per-experiment runnable spec |
| [`configs/_template.yaml`](configs/_template.yaml) | Copy when starting Test4+ |
| [`configs/local.example.yaml`](configs/local.example.yaml) | Server paths → copy to `local.yaml` (gitignored) |

## Completed experiments

| ID | Training | Epochs | RADCURE GTVp Dice | HECKTOR GTVp Dice |
|----|----------|--------|-------------------|-------------------|
| test1 | RADCURE only (366) | 1000 | 0.377 | 0.330 |
| test2 | RADCURE + HECKTOR (650) | 1000 | 0.383 | 0.486 |
| test3 | RADCURE + HECKTOR (650) | 700 | 0.383 | 0.545 |
| test4 | RADCURE + HECKTOR (650), **separate GTVp/GTVn** | 700 | ~0.58 GTVp | see TEST4_RESULTS |
| test5 | Same as Test4 + **improved preprocess / QC** | 700 (planned) | — | — |
| test6 | **STU-Net** fine-tune on Test5 Dataset650 | 1000 (planned) | — | — |
| test7 | **Probability outputs** (Test5 model; curves + alpha viz) | — (no retrain) | — | — |

Details and narrative: [`research_notebooks/retrain_epoch_study/retrain_epoch_study.md`](../research_notebooks/retrain_epoch_study/retrain_epoch_study.md).

## Start a new experiment (Test4+)

1. Read [Naming convention](#naming-convention) below — fill the one-line rule before copying files.
2. Copy `configs/_template.yaml` → `configs/test{N}_{cohort}_{variable}.yaml`
3. Add a block under `experiments:` in `registry.yaml` (mirror the config)
4. Add `RETRAIN_RADHECK_TEST{N}` to `configs/local.yaml` on the server
5. Use the Cursor skill **radheck-experiment** or follow env exports in the YAML

## Naming convention

Every new experiment must answer this in one sentence **before** creating files:

> **Test{N} changes ___ compared to {changed_from}; everything else is identical.**

If you cannot finish that sentence with a single clause, split into two tests.

### Layers

| Layer | Rule | Example |
|-------|------|---------|
| **Experiment id** | `test{N}` — sequential, lowercase, never reused | `test4` |
| **Config filename** | `test{N}_{cohort}_{variable}.yaml` | `test4_radheck_500epochs.yaml` |
| **Registry `name`** | Short human title (paper / table row) | `RADHECK combined — 500 epochs` |
| **Registry `hypothesis`** | One sentence: expected outcome | required |
| **`changed_from`** | Previous experiment id | `test3` |
| **Path key** (`local.yaml`) | `RETRAIN_RADHECK_TEST{N}` or `RETRAIN_RADCURE_366` | `RETRAIN_RADHECK_TEST4` |
| **Server folder** | Align folder number with test id | `nnunet_radheck_test_4_retrain` |
| **HECKTOR eval subdir** | Under retrain path unless overridden | `hecktor_validation` |
| **Custom trainer** | `nnUNetTrainer_{variable}_NoMirroring` | `nnUNetTrainer_500epochs_NoMirroring` |
| **Runbook** (optional) | `pipelines/radheck/Retrain-Radheck-Test{N}.md` | only for non-obvious server steps |

### Cohort token (second segment of filename)

| Token | Training data |
|-------|----------------|
| `radcure` | RADCURE only — Dataset366 |
| `radheck` | Combined RADCURE + HECKTOR — Dataset650 |

### Variable slug (third+ segment) — name **the one thing that changed**

Good: `500epochs`, `700epochs`, `cosine_lr`, `hecktor_frac80`, `no_dedupe`  
Avoid: `650`, `dataset152`, `good_run`, `final_v2` (not hypotheses; dataset ids belong in YAML fields, not filenames)

### Examples

| File | `changed_from` | What changed |
|------|----------------|--------------|
| `test1_radcure_only.yaml` | — | baseline: RADCURE-only |
| `test2_radheck_1000epochs.yaml` | test1 | add HECKTOR training |
| `test3_radheck_700epochs.yaml` | test2 | epochs 1000 → 700 |
| `test4_radheck_500epochs.yaml` | test3 | epochs 700 → 500 |

### Server paths

New tests: **folder number must match experiment id.**

```yaml
# local.yaml
RETRAIN_RADHECK_TEST4: /media/.../nnunet_radheck_test_4_retrain
```

Legacy exception: `test2` uses `nnunet_radheck_test_1_retrain` — do not follow that pattern for Test4+.

Registry `path_templates` keys use snake_case: `retrain_radheck_test4` → same path as `RETRAIN_RADHECK_TEST4` in `local.yaml`.

### Checklist (naming)

```
- [ ] Filename: test{N}_{cohort}_{variable}.yaml
- [ ] registry.yaml: id, name, hypothesis, changed_from, config_file path
- [ ] local.yaml: RETRAIN_RADHECK_TEST{N} with matching server folder name
- [ ] One variable changed vs changed_from
- [ ] status: planned → running → completed
- [ ] After run: results in registry.yaml + experiment YAML
```

## Critical env vars (avoid common mistakes)

| Variable | Role |
|----------|------|
| `DATASET_FOLDER` | Data with `imagesTs` / `labelsTs` (650 for RADHECK train/eval RADCURE test) |
| `NNUNET_RETRAIN_PATH` | Model + logs (`nnUNet_results`) |
| `DATASET_ID` | **Model** dataset id (650 for Test2/3), not 152 |
| `NNUNET_PREPROCESSED_PATH` | Reuse preprocess from an earlier run (Test3) |
| `HECKTOR_EVAL_OUTPUT_DIR` | Keep HECKTOR predictions out of Dataset152 |

## Related docs

- Implementation scripts: [`pipelines/radheck/`](../pipelines/radheck/)
- Test3 runbook: [`Retrain-Radheck-Test3.md`](../pipelines/radheck/Retrain-Radheck-Test3.md)
- Agent skill: [`.cursor/skills/radheck-experiment/SKILL.md`](../.cursor/skills/radheck-experiment/SKILL.md)
