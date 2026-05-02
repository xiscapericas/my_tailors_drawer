# RADCURE + HECKTOR combined retrain (radheck)

This folder documents how **RADCURE-366** splits are merged with **HECKTOR** training cases for a single nnUNet dataset.

## Server paths (not in GitHub)

**All machine-specific paths and the S3 URI are read from a JSON file that you maintain on the server only.** Nothing under `/media/...` or your bucket name is hardcoded in the Python script that ships in the repo.

1. **Tracked in Git:** `radheck_server_paths.example.json` — template with placeholder paths and every supported key.
2. **Not tracked in Git:** `radheck_server_paths.json` — your real paths. It is listed in the repository `.gitignore`, so it is never pushed when you commit or pull.

On the server, after cloning the repo:

```bash
cd /path/to/radcure-medical-imaging/research_notebooks/retrain_radheck
cp radheck_server_paths.example.json radheck_server_paths.json
# Edit radheck_server_paths.json with your S3 URI, Dataset366 path, Dataset152 path, etc.
```

Alternatively, keep the JSON anywhere and point the script at it:

```bash
export RADHECK_SERVER_CONFIG=/secure/path/radheck_server_paths.json
python research_notebooks/retrain_radheck/build_radheck_nnunet_dataset.py
```

Or:

```bash
python research_notebooks/retrain_radheck/build_radheck_nnunet_dataset.py --config /secure/path/radheck_server_paths.json
```

**Priority:** CLI flags (e.g. `--hecktor-test-dataset`) override environment variables, which override values in the JSON file (for the keys where both exist).

Optional keys in JSON:

- `radheck_dataset_id` — `null` or omit to auto-set dataset folder id from case counts; set to an integer string if you need a fixed nnUNet dataset id.
- `main_path` — if `organ_dictionary_path` is empty, the script can still find `radcure_dictionary.json` as `{main_path}/radcure_dictionary.json` (you can also set `ORGAN_DICTIONARY_PATH` in the shell).

`split_manifest.json` written next to the combined dataset includes `server_config_path` so you can see which file was used for that run.

## Where the data lives (conceptually)

### RADCURE (same splits as RADCURE-366)

Path in your config: **`radheck_radcure_dataset`** → the `Dataset366_TotalSegmentator` folder you already use for retraining.

| Split | Folders | Role |
|-------|---------|------|
| Training | `imagesTr`, `labelsTr` | nnUNet training |
| Validation | `imagesVa`, `labelsVa` | nnUNet validation |
| Test | `imagesTs`, `labelsTs` | Held-out RADCURE test |

These are copied (or symlinked with `--link-radcure`) into the combined dataset.

### HECKTOR

| Split | Role |
|-------|------|
| Test (held-out) | Path **`radheck_hecktor_test_dataset`** in config (your nnUNet `Dataset152_TotalSegmentator` with `imagesTs` / `labelsTs`). Cases listed there are **excluded** from HECKTOR train/val so they stay disjoint. |
| Train + Val | Zip URI **`radheck_s3_uri`** (e.g. full HECKTOR2025 task-1 training zip) → download under **`radheck_download_dir`** → unzip under **`radheck_unzipped_dir`** → `image_processor` (HECKTOR) → exclude overlaps → **80% / 20%** train/val on the remainder (fraction/seed in JSON or env). |

Combined output root: **`radheck_output_work`** → `DatasetXXX_TotalSegmentator` where `XXX` is RADCURE total cases plus HECKTOR cases kept after exclusion, unless you set `radheck_dataset_id` or `RADHECK_DATASET_ID`.

## Script

`build_radheck_nnunet_dataset.py` — loads **`radheck_server_paths.json`**, then download / unzip / process / exclude / split / merge as documented above.

Run from **repository root**:

```bash
cd /path/to/radcure-medical-imaging
python research_notebooks/retrain_radheck/build_radheck_nnunet_dataset.py
```

Common options:

```bash
python research_notebooks/retrain_radheck/build_radheck_nnunet_dataset.py \
  --config /path/to/radheck_server_paths.json \
  --skip-download --skip-process \
  --hecktor-cases-root /path/to/unzipped/training/cases
```

## Next steps (not in this script)

- Copy or register the combined dataset under `nnUNet_raw` and run `train_nnunet.py` plan/train as for RADCURE-366.
- Joint evaluation can use RADCURE test from the combined `imagesTs` and HECKTOR test from your separate `Dataset152` folder if you keep two test pools.
