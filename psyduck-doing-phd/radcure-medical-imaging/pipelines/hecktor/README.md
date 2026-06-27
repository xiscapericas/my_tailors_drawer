# HECKTOR test pipeline

Download, process, build nnUNet test dataset, predict, and evaluate on HECKTOR held-out test (Dataset152).

```bash
# From repo root (after pip install -e .)
python -m pipelines.hecktor.test_pipeline --predict-only
python -m pipelines.hecktor.test_pipeline --eval-only

# Or via console script
run-hecktor-test --predict-only
```

See [`docs/documentation-index.md`](../../docs/documentation-index.md) and experiment configs for env vars (`DATASET_FOLDER`, `HECKTOR_EVAL_OUTPUT_DIR`, etc.).
