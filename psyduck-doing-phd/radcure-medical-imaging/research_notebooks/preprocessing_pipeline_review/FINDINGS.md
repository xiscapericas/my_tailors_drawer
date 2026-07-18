# Preprocessing review — findings & handoff

**Status (2026-07-18):** Step A + anatomy QC + Step B (anatomical background) reached a **visual sweet spot**. Research-only; **not yet wired into production** `CaseProcessor` / `MaskGenerator`.

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

## Colab ops (easy to lose)

- Repo: `github.com/xiscapericas/my_tailors_drawer` → `psyduck-doing-phd/radcure-medical-imaging`.
- `pip install -e <absolute REPO_ROOT>`; pin **`numpy==2.0.2` last**; **no TotalSegmentator** until Step C.
- After pull: reinstall + **clear `sys.modules`** for `image_processor*`; B1 asserts `z_radius` + `flip_axis`/`flipud` in source.
- Drive copy of notebook goes stale — re-copy or pull from git.

---

## Production gap (do not forget)

`MaskGenerator` still calls **`head_mask_from_array`** only. Improved masks live in research notebook + library helpers; **wiring into CaseProcessor is a later decision** after Step C+ validation.

---

## Next session — suggested phase order

1. **Step C** — TotalSegmentator on kept cases; organ overlays; stable **name→colour** map (not label index).
2. **`other-tissue`** — leftover anatomical voxels after organs; check air/table no longer inflate it.
3. Decide whether to **replace** production background with `anatomical_region_masks_from_slices` (or hybrid).
4. Rebuild a small DatasetXXX slice and re-check GTVp Dice vs old preprocess.
5. Optional: unit tests for `flip_axis=0` vs `1` and fill-cap behaviour.

---

## Key commits (local history)

- Continuity + symmetry experiments → axis fix (`Fix axis` and predecessors on `main`).
- Anatomy QC logging / thresholds.

When extending, prefer **minimal** changes to `image_processing.py` / notebook cells over parallel implementations.
