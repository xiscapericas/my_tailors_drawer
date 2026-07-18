# Main documentation index

[← Documentation hub](README.md)

Single map of **all** docs. Each topic has **one canonical file** — others link here, not duplicate.

---

## A. Get started (human)

| Doc | Canonical for |
|-----|----------------|
| [../README.md](../README.md) | Install, quick start, CLI entry points |
| [../env.example](../env.example) | Environment variable reference |

---

## B. Code layout

| Path | Role |
|------|------|
| [../image_processor/](../image_processor/) | Preprocessing library |
| [../nnunet_training/](../nnunet_training/) | nnUNet train / predict / evaluate |
| [../pipelines/](../pipelines/) | Multi-step orchestration (RADHECK build, HECKTOR test eval) |
| [../experiments/](../experiments/) | Experiment registry and YAML configs |

### CLI entry points

| Command | Doc section |
|---------|-------------|
| `process_all_cases.py` | [nnunet_training/README.md](../nnunet_training/README.md) + [README quick start](../README.md) |
| `split_dataset.py` | Same |
| `train_nnunet.py` | [nnunet_training/README.md](../nnunet_training/README.md) |
| `python -m pipelines.hecktor.test_pipeline` | [experiments/configs/test3](../experiments/configs/test3_radheck_700epochs.yaml) (HECKTOR eval) |
| `python -m pipelines.radheck.build_nnunet_dataset` | [pipelines/radheck/README.md](../pipelines/radheck/README.md) |

### Pipelines

| Package | Role |
|---------|------|
| [pipelines/README.md](../pipelines/README.md) | Overview |
| [pipelines/radheck/](../pipelines/radheck/) | Build Dataset650, leak checks, server runbooks |
| [pipelines/hecktor/](../pipelines/hecktor/) | HECKTOR test download / process / predict / eval |

| Module | Role |
|--------|------|
| `build_nnunet_dataset` | Merge RADCURE + HECKTOR |
| `verify_radheck_no_leak` | Split / leak audit |
| `deduplicate_dataset_splits` | Tr/Va/Ts dedupe |
| `test_pipeline` (hecktor) | HECKTOR test predict/eval |

---

## C. Experiments (canonical record)

| Doc | Role |
|-----|------|
| **[experiments/registry.yaml](../experiments/registry.yaml)** | **Source of truth** — Test1–3, metrics, hypotheses |
| [experiments/README.md](../experiments/README.md) | Guide, results table, **naming convention** (Test4+) |
| [experiments/configs/](../experiments/configs/) | Per-test YAML (`test1`, `test2`, `test3`, `_template`) |
| [experiments/configs/local.example.yaml](../experiments/configs/local.example.yaml) | Server path template → copy to `local.yaml` |

### Server runbooks (commands only — metrics in registry)

| Runbook | Test |
|---------|------|
| [Retrain-Radheck-Test3.md](../pipelines/radheck/Retrain-Radheck-Test3.md) | test3 (700 epochs) |
| [Retrain-Radheck-Test2.md](../pipelines/radheck/Retrain-Radheck-Test2.md) | test2 (1000 epochs, multi-organ notes) |

---

## D. nnUNet (technical reference)

| Doc | Role |
|-----|------|
| [nnunet_training/README.md](../nnunet_training/README.md) | Full train/eval steps, env vars, outputs |
| Custom 700-epoch trainer | `nnunet_training/trainer_variants/` + `install_trainer_variants.py` |

---

## E. Research & publication

| Doc | Role |
|-----|------|
| [retrain_epoch_study/retrain_epoch_study.md](../research_notebooks/retrain_epoch_study/retrain_epoch_study.md) | **Paper draft** — Test1–3 narrative + figures |
| [research_notebooks/README.md](../research_notebooks/README.md) | Notebook index |
| [preprocessing_pipeline_review/](../research_notebooks/preprocessing_pipeline_review/) | Colab audit: preprocess top→bottom (bg / other-tissue / viz) |
| [Surface_Dice_Analysis_Report.md](../research_notebooks/Surface_Dice_Analysis_Report.md) | Metric exploration notes |

---

## F. Archive

| Doc | Role |
|-----|------|
| [use-case-study/README.md](../use-case-study/README.md) | Early Dataset366 / TotalSegmentator report (superseded by RADHECK work) |

---

## G. Cursor

| Doc | Role |
|-----|------|
| [cursor-setup.md](cursor-setup.md) | Rules, skills, agent workflow |
| [.cursor/skills/radheck-experiment/SKILL.md](../.cursor/skills/radheck-experiment/SKILL.md) | Experiment run skill |

---

## H. Planned (not yet created)

| Doc | Purpose |
|-----|---------|
| `docs/METHODS.md` | Publication methods section |
| `docs/REPRODUCE.md` | Step-by-step reproduce Test N |
| `tests/` | Unit tests for split utils, naming |

When added, register them in this index under the right section.

---

## Doc graph (how pieces connect)

```
docs/README.md (hub)
    ├── about-the-project.md     … purpose, pipeline, goals
    ├── cursor-setup.md          … rules + skills
    └── documentation-index.md   … this file

experiments/registry.yaml  ←── metrics (canonical)
    ↑ linked by runbooks, epoch study, skill

retrain_epoch_study.md     ←── interpretation / paper
nnunet_training/README     ←── nnUNet how-to (deep)
README.md (root)           ←── install + quick start only
```

[← Documentation hub](README.md)
