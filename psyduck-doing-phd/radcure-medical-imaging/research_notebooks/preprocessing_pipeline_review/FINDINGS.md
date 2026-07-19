# Preprocessing review — findings & handoff

**Status (2026-07-19):** Steps A–B done (sweet spot). Steps **C–E** added (canonical organ dict, TotalSegmentator, combined mask, CT/organs/tumor viz). Research-only; improved bg **not** wired into production `MaskGenerator` yet.

**Canonical notebook:** [`preprocessing_pipeline_review_colab.ipynb`](preprocessing_pipeline_review_colab.ipynb)  
**Implementation APIs (new):** `image_processor/utils/image_processing.py`, `image_processor/utils/anatomy_qc.py`  
**Docs hub:** [`../../docs/README.md`](../../docs/README.md)

---

## Working hypothesis (still open)

```
bad patient/background mask
  → inflated anatomical_region / other-tissue
  → noisy nnUNet labels
  → low GTVp Dice
```

Index-based colormaps also hid QA issues (same organ ≠ same colour). Fix viz with **fixed RGBA** (cyan anatomical, red GTVp, magenta GTVn), not matplotlib `cool` (1.0 → magenta).

---

## Audit cases

| Batch | RADCURE | HECKTOR |
|-------|---------|---------|
| 1 | `0122`, `0040`, `0397`, `0151` | Prefer `HMR-012`, `CHUM-023`, `CHUM-098`, `HMR-057`; zip often only CHUM/CHUS → **suffix fallback** (`*-NNN`) |
| 2 | `0005`, `0088`, `0250` | `CHUM-013`, `CHUS-016` (not MDA/CHUV if missing from zip) |

QC decisions / discards: `WORK_DIR/logs/anatomy_qc/` (JSONL + CSV).

---

## Step A — CT + tumor

| Learning | Detail |
|----------|--------|
| RADCURE tumors invisible until aligned | After RTSTRUCT load, call **`save_and_align_mask_with_ct`** (same as production). |
| Keep GTVp / GTVn separate | Use **`load_labeled_tumor_volume`** (GTVp=1, GTVn=2), not merged-only `load_tumor_mask`. |
| Viz | Show **all** selected slices; fixed colours. |

---

## Anatomy QC (`anatomy_qc.py`)

- Score: tumor presence, intensity range, **patient fill** (`head_mask`), coherence, slice extent, tumor-inside-patient.
- Non-human / empty-body cases scored high when **tumor weight** dominated; **fill** was the real signal (~0.044–0.061).
- **Hard fail** if `mean_patient_fill < 0.065`; default keep threshold **0.70**.
- Notebook: Step **A2** QC → **A3** extra cases → **B0** keep-only filter.

---

## Step B — anatomical background (sweet spot)

Production `head_mask_from_array` is **not** the research target: `keep_top_ratio=0.6` zeros a large side; watershed → empty slices.

### Improved pipeline (research)

1. **`body_mask_from_intensity`** — FOV-aware air vs tissue (not naive Otsu on full image / FOV circle), centered connected component, fill cap.
2. **`_enforce_sagittal_symmetry`** — fill contralateral holes using **body vs air contrast**.
3. **`enforce_anatomical_continuity`** — weak slices ← OR of strong neighbors within ±`z_radius`, plus Z closing.
4. Orchestrator: **`anatomical_region_masks_from_slices`**.

### Critical orientation bug (fixed)

| Wrong | Right |
|-------|--------|
| `np.fliplr` / `flip_axis=1` | **`flip_axis=0`** (`np.flip(..., axis=0)` / flipud) |
| Mirrors **A/P** with notebook `imshow(img.T)` | Mirrors **L/R (sagittal)** in display |
| Symptom: table / top bar painted as anatomy | Symptom if still wrong: missing side not filled |

Symmetry fill gates:

- inside reconstruction **FOV**
- **`tissue_candidate`** (soft body vs darker FOV air)
- padded **patient bbox ∪ L/R-flipped bbox** (do not invent table)

API knobs: `enforce_symmetry`, `sagittal_flip_axis=0`, `enforce_continuity`, `min_area`.

### Viz / continuity QA

- Compare **3 consecutive** slices (not `linspace` skips) — otherwise Z continuity is invisible.
- Rows: production · raw intensity · improved (sym + Z).

---

## Step C–E — TotalSegmentator, fixed labels, tumor viz (2026-07-19)

**Status:** Research notebook Steps C–E implemented; run on Colab after A–B.

### Fixed organ dictionary (case-independent)

| Artifact | Path |
|----------|------|
| Catalog (TS task → organ names) | `image_processor/utils/totalsegmentator_organs.py` |
| Factory | `OrganDictionary.from_hn_canonical(...)` |
| Committed JSON template | `image_processor/resources/organ_dictionary_hn_canonical.json` |
| Colab working copy | `WORK_DIR/audit_organ_dictionary.json` |

Layout: `background=0`, `anatomical_region=1`, `other-tissue=2`, then **88** unique H&N TS organs (6 tasks), then `GTVp`, `GTVn`.  
Cross-task duplicate basenames (`optic_nerve_*`, `skull`) share **one** index.

**Do not** grow the dict from case discovery order for new audits — load the canonical file first.

### Stable colours (`label_colors.py`)

- **GTVp = red**, **GTVn = pink**
- TS organs: fixed palette that **never** uses those hues
- Helpers: `rgba_by_name`, `rgba_by_index`, `paint_label_rgba`

### Notebook flow

1. **C1** — load/create canonical dict + colour maps  
2. **C2** — `TotalSegmentatorWrapper.run_tasks` on QC-kept cases  
3. **D1** — `MaskGenerator` combine (prefer Step B improved bg) → other-tissue → separate GTVp/GTVn  
4. **D2** — other-tissue fill summary  
5. **E** — 3 columns: CT | CT+organs | CT+organs+tumors  

### Still open

- Production `MaskGenerator.generate_background_array` still uses `head_mask_from_array`
- Optional: make `add_organ` strict when a canonical dict is loaded
- DatasetXXX rebuild / Dice check after approving C–E visuals

---

## Colab ops (easy to lose)

- Repo: `github.com/xiscapericas/my_tailors_drawer` → `psyduck-doing-phd/radcure-medical-imaging`.
- `pip install -e <absolute REPO_ROOT>`; pin **`numpy>=2.1,<2.3`** (not 2.0.2 — breaks imagecodecs); install **`totalsegmentator`** only for Step C (then restart).
- After pull: reinstall + **clear `sys.modules`** for `image_processor*`; B1 asserts `z_radius` + `flip_axis`/`flipud` in source.
- Drive copy of notebook goes stale — re-copy or pull from git.

---

## Production gap (do not forget)

`MaskGenerator` still calls **`head_mask_from_array`** only. Improved masks live in research notebook + library helpers; **wiring into CaseProcessor is a later decision** after Step C–E visual QA.

---

## Next after C–E QA

1. Confirm other-tissue is not FOV/table air on audit cases.
2. Decide whether to replace production background with `anatomical_region_masks_from_slices`.
3. Point production `ORGAN_DICTIONARY_PATH` at the canonical JSON (or a copy).
4. Rebuild a small DatasetXXX slice and re-check GTVp Dice vs old preprocess.

---

## Key commits (local history)

- Continuity + symmetry experiments → axis fix (`Fix axis` and predecessors on `main`).
- Anatomy QC logging / thresholds.
- Canonical H&N organ dictionary + Steps C–E notebook cells.

When extending, prefer **minimal** changes to existing modules / notebook cells over parallel implementations.
