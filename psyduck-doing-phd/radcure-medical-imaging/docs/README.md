# Documentation hub

All project documentation flows from here. Avoid adding standalone READMEs without linking back to this page.

```
docs/
├── README.md                 ← you are here
├── about-the-project.md      §1 What this is for
├── cursor-setup.md           §2 Cursor (rules, skills, agent workflow)
└── documentation-index.md    §3 Main docs (pipeline, experiments, research, archive)
```

---

## §1 What this project is for

**Read:** [about-the-project.md](about-the-project.md)

PhD head-and-neck **GTVp segmentation** on RADCURE and HECKTOR: preprocessing → nnUNet → dual-cohort evaluation → publishable experiments.

---

## §2 Cursor setup

**Read:** [cursor-setup.md](cursor-setup.md)

Rules (`.cursor/rules/`), skills (`.cursor/skills/`), and how to run Test N with the agent without rediscovering paths each session.

---

## §3 Main documentation

**Read:** [documentation-index.md](documentation-index.md)

Connected map: install, pipeline scripts, experiment registry, nnUNet reference, server runbooks, research notebooks, archive.

---

## Quick links (most used)

| I want to… | Go to |
|------------|--------|
| Install and run first pipeline | [Root README](../README.md) |
| See Test1–3 results | [experiments/registry.yaml](../experiments/registry.yaml) |
| Start Test4 | [experiments/README.md § Naming](../experiments/README.md#naming-convention) + [`configs/_template.yaml`](../experiments/configs/_template.yaml) |
| Train / evaluate nnUNet | [nnunet_training/README.md](../nnunet_training/README.md) |
| Write the paper | [retrain_epoch_study.md](../research_notebooks/retrain_epoch_study/retrain_epoch_study.md) |
| Start Test5 (improved preprocess) | [Retrain-Radheck-Test5.md](../pipelines/radheck/Retrain-Radheck-Test5.md) |
| Explore STU-Net (Test6 research) | [test6_stunet/README.md](../research_notebooks/test6_stunet/README.md) |

---

## Doc maintenance rule

When you add or change documentation:

1. Decide which section (§1, §2, or §3) it belongs to.
2. Add one line to [documentation-index.md](documentation-index.md) (§3) or the relevant §1/§2 file.
3. Do **not** duplicate content — link to the canonical file instead.
