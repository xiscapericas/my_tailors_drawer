# Test4 Phase 2 — batch relabel

Reuse existing **TotalSegmentator** organ masks; rebuild labels with separate **GTVp** and **GTVn**.

## Output layout

```
work/retrain_test4/                    ← TEST4_WORK_ROOT
├── radcure_dictionary_test4.json
├── TotalSegmentatorRetrain/             ← RADCURE nnUNet pairs (for split / Dataset366)
│   └── RADCURE-XXXX/output/{image,labels}/
├── hecktor/                             ← HECKTOR nnUNet pairs (for RADHECK build)
│   └── CHUM-001/output/{image,labels}/
├── relabel_ok.txt
└── relabel_failed.txt
```

**Phase 3:** build `Dataset650` here using the **same train/val/test case lists** as Test3 — see [`PHASE3.md`](PHASE3.md).

## Run (server)

```bash
cd /path/to/radcure-medical-imaging
source .venv/bin/activate
set -a && source .env && set +a

export CUDA_VISIBLE_DEVICES=0
export TEST4_WORK_ROOT=/media/HDD_8TB/xisca/work/retrain_test4
export TEST4_RADCURE_SOURCE_MAIN_PATH=/media/HDD_8TB/xisca/dataset/RadcureComplete
export TEST4_HECKTOR_SOURCE_CASES_ROOT=/media/HDD_8TB/xisca/dataset/hecktor/.../unzipped/task1

python -m pipelines.test4.relabel_tumor_batch --dry-run
python -m pipelines.test4.relabel_tumor_batch
```

Sources must already have `total_segmentator_output/` per case (Test1–3 preprocessing). No TotalSegmentator re-run.

## Options

| Flag | Purpose |
|------|---------|
| `--skip-hecktor` / `--skip-radcure` | One cohort only |
| `--max-cases N` | Smoke test |
| `--force` | Overwrite existing dest `output/` |
| `--dry-run` | List what would run |

See also: [`test4_preprocessing_preview.ipynb`](test4_preprocessing_preview.ipynb) (single-case validation).
