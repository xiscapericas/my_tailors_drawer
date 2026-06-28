# Pipelines

Orchestration scripts for multi-step server workflows. Library code stays in `image_processor/` and `nnunet_training/`.

| Package | Purpose |
|---------|---------|
| [`hecktor/`](hecktor/) | HECKTOR test download, process, predict, evaluate |
| [`radheck/`](radheck/) | Combined RADCURE + HECKTOR dataset build, leak checks, server runbooks |
| [`test4/`](test4/) | Test4 relabel batch (reuse TotalSegmentator, separate GTVp/GTVn) |

Run from **repository root** (after `pip install -e .`):

```bash
python -m pipelines.hecktor.test_pipeline --predict-only
python -m pipelines.radheck.build_nnunet_dataset
```

See [`docs/documentation-index.md`](../docs/documentation-index.md) § B.
