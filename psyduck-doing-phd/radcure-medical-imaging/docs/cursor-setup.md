# Cursor setup

[← Documentation hub](README.md)

How to configure Cursor so the agent works consistently on this repo.

## Rules (always applied)

| File | Role |
|------|------|
| [`.cursor/rules/working-style.mdc`](../.cursor/rules/working-style.mdc) | Research vs implementation, minimal diffs, reuse patterns, no secrets in Git |

The agent should **extend** existing code, not redesign the pipeline unless you ask.

## Skills (invoke for specific workflows)

| Skill | When to use |
|-------|-------------|
| [**radheck-experiment**](../.cursor/skills/radheck-experiment/SKILL.md) | New Test N, reproduce Test1–3, train/eval RADHECK, HECKTOR cross-dataset eval |

**How to invoke:** mention “radheck experiment”, “start Test4”, or “evaluate on HECKTOR using Test3 model”.

### Agent checklist (every experiment task)

1. Read [`experiments/registry.yaml`](../experiments/registry.yaml)  
2. Read `experiments/configs/testN_*.yaml`  
3. Use server paths from `experiments/configs/local.yaml` (gitignored)  
4. Never confuse `DATASET_FOLDER` (data) with `NNUNET_RETRAIN_PATH` (model)  
5. HECKTOR eval: `DATASET_ID=650` (model), `DATASET_FOLDER=Dataset152` (test images)  
6. After a run: update `registry.yaml` results — do not leave metrics only in chat  

## Recommended local setup

```bash
# Repo root
cp env.example .env                    # processing + nnUNet env vars
cp experiments/configs/local.example.yaml experiments/configs/local.yaml  # server paths
cp pipelines/radheck/radheck_server_paths.example.json \
   pipelines/radheck/radheck_server_paths.json          # RADHECK build only
```

On the server, fill real `/media/...` paths in `local.yaml` and `.env`.

## Working with the agent

| Task type | What to say |
|-----------|-------------|
| New experiment | “Using radheck-experiment skill, set up Test4 for [hypothesis]” |
| Re-run eval only | “Eval-only on HECKTOR for test3, predictions in hecktor_validation” |
| Code change | Point to module (`nnunet_training/`, `image_processor/`) — agent follows working-style |
| Paper text | Edit `research_notebooks/retrain_epoch_study/` — not implementation code |

## What the agent should read first

1. [docs/README.md](README.md) — this hub  
2. [experiments/registry.yaml](../experiments/registry.yaml) — if the task mentions a test  
3. [documentation-index.md](documentation-index.md) — if the task touches pipeline or docs  

## Adding a new skill later

Follow Cursor skill format (see your global `create-skill` skill). Project skills go in:

```
.cursor/skills/<skill-name>/SKILL.md
```

Link new skills from this file and from [documentation-index.md](documentation-index.md).

[← Documentation hub](README.md) · [← About the project](about-the-project.md) · [Main docs →](documentation-index.md)
