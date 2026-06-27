# RADCURE + HECKTOR combined retrain (radheck)

**Documentation hub:** [`docs/README.md`](../../docs/README.md) · §3-B in [`docs/documentation-index.md`](../../docs/documentation-index.md).

**Implementation scripts** for building and validating the combined Dataset650 pipeline.  
For experiment registry and Test1–3 metrics, see [`experiments/registry.yaml`](../../experiments/registry.yaml).  
Server runbooks: [`Retrain-Radheck-Test2.md`](Retrain-Radheck-Test2.md), [`Retrain-Radheck-Test3.md`](Retrain-Radheck-Test3.md).

This folder documents how **RADCURE-366** splits are merged with **HECKTOR** training cases for a single nnUNet dataset.

## Scripts

| Module | Purpose |
|--------|---------|
| `build_nnunet_dataset` | Download/process HECKTOR, merge with RADCURE → Dataset650 |
| `verify_radheck_no_leak` | Audit train/val vs test disjointness |
| `remove_hecktor_test_leak` | Remove leaked HECKTOR test stems from Tr/Va |
| `deduplicate_dataset_splits` | Fix overlapping Tr/Va/Ts stems |
| `wipe_radheck_output_work` | Wipe combined dataset output for rebuild |
| `cleanup_retrain_artifacts` | Remove nnUNet preprocessed/results without touching case data |

```bash
cd /path/to/radcure-medical-imaging
python -m pipelines.radheck.build_nnunet_dataset
python -m pipelines.radheck.verify_radheck_no_leak --help
```

## Server paths (not in GitHub)

**All machine-specific paths and the S3 URI are read from a JSON file that you maintain on the server only.**

1. **Tracked in Git:** `radheck_server_paths.example.json` — template with placeholder paths.
2. **Not tracked in Git:** `radheck_server_paths.json` — your real paths (see `.gitignore`).

On the server:

```bash
cd /path/to/radcure-medical-imaging/pipelines/radheck
cp radheck_server_paths.example.json radheck_server_paths.json
# Edit radheck_server_paths.json with your S3 URI, Dataset366 path, Dataset152 path, etc.
```

Or:

```bash
export RADHECK_SERVER_CONFIG=/secure/path/radheck_server_paths.json
python -m pipelines.radheck.build_nnunet_dataset
```

**Priority:** CLI flags override environment variables, which override values in the JSON file.

Optional keys in JSON:

- `radheck_dataset_id` — `null` or omit to auto-set dataset folder id from case counts.
- `main_path` — if `organ_dictionary_path` is empty, find `radcure_dictionary.json` as `{main_path}/radcure_dictionary.json`.

`split_manifest.json` written next to the combined dataset includes `server_config_path`.

## Where the data lives (conceptually)

### RADCURE (same splits as RADCURE-366)

Path in config: **`radheck_radcure_dataset`** → `Dataset366_TotalSegmentator`.

| Split | Folders | Role |
|-------|---------|------|
| Training | `imagesTr`, `labelsTr` | nnUNet training |
| Validation | `imagesVa`, `labelsVa` | nnUNet validation |
| Test | `imagesTs`, `labelsTs` | Held-out RADCURE test |

### HECKTOR

| Split | Role |
|-------|------|
| Test (held-out) | **`radheck_hecktor_test_dataset`** (Dataset152). Excluded from HECKTOR train/val. |
| Train + Val | S3 zip → process → 80/20 split on remainder |

Combined output: **`radheck_output_work`** → `DatasetXXX_TotalSegmentator`.

## Next steps (not in build script)

- Register combined dataset under `nnUNet_raw` and run `train_nnunet.py`.
- **Test3 (700 epochs):** [`Retrain-Radheck-Test3.md`](Retrain-Radheck-Test3.md).
