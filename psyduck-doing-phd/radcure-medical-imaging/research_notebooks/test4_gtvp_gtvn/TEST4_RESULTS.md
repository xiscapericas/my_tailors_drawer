# Test 4 — Separate GTVp and GTVn labels

**Status:** completed  
**Changed from:** Test 3 (same cohorts, splits, 700 epochs; only tumor label handling differs)  
**Config:** [`experiments/configs/test4_radheck_separate_gtvp_gtvn.yaml`](../../experiments/configs/test4_radheck_separate_gtvp_gtvn.yaml)  
**Runbook:** [`pipelines/radheck/Retrain-Radheck-Test4.md`](../../pipelines/radheck/Retrain-Radheck-Test4.md)

---

## Purpose and hypothesis

In earlier experiments we observed a tendency for the model to predict a **second tumour** that may not exist clinically. A plausible cause is that **primary GTVp and nodal GTVn** were merged into a single `GTVp` label during mask generation, so nnUNet could not learn distinct semantics for primary vs nodal disease.

**Hypothesis:** Splitting tumour labels again (**GTVp** and **GTVn** as separate nnUNet classes) lets the network differentiate both structures and report **separate Dice scores**, while reducing spurious second-tumour predictions.

---

## What changed vs Test 1–3

| Aspect | Test 1–3 | Test 4 |
|--------|----------|--------|
| Tumour labels | GTVp + GTVn merged → one `GTVp` index | `GTVp` and `GTVn` as distinct labels |
| Organ dictionary | Standard merged-tumour dict | `radcure_dictionary_test4.json` |
| TotalSegmentator | Unchanged | Reused (no re-run) |
| Train / val / test case lists | Dataset650 splits | **Same splits as Test 3** |
| Trainer | Test 3: 700 epochs, no mirroring | Same |
| Preprocess | Test 3 reused Test 2 | **Fresh preprocess** (label set changed) |

---

## Process (phases)

### Phase 1 — Code and validation

- Updated preprocessing (`tumor_label_mode="separate"`) to keep **GTVp** and **GTVn** as separate indices.
- Validation notebook: [`test4_preprocessing_preview.ipynb`](test4_preprocessing_preview.ipynb) — one RADCURE + one HECKTOR case, full organ/tumour visualisation.

### Phase 2 — Batch relabel

- Reused existing **TotalSegmentator** outputs; rebuilt `output/image` + `output/labels` with separate tumour labels.
- CLI: `python -m pipelines.test4.relabel_tumor_batch` → `work/retrain_test4/`.
- See [`PHASE2.md`](PHASE2.md).

### Phase 3 — Dataset650 + retrain + evaluation

- Built Dataset650 from Phase 2 labels with Test 3 splits: `python -m pipelines.test4.build_dataset650`.
- nnUNet retrain: **700 epochs**, `nnUNetTrainer_700epochs_NoMirroring`, fold 0.
- RADCURE eval: `python train_nnunet.py --step evaluate` (+ `evaluation_visualization`).
- HECKTOR eval: `python -m pipelines.hecktor.test_pipeline --predict-only --eval-only` on Dataset152.
- See [`PHASE3.md`](PHASE3.md).

---

## Results (summary)

**Detailed numbers and per-case tables:**  
[Test 4 results spreadsheet](https://docs.google.com/spreadsheets/d/1x8HGqRiozO5-Yck17yXFP-AipUIVfhWgywo07hLRVbA/edit?gid=0#gid=0)

**Case-by-case review (Test 1 vs Test 4, n = 133):**  
[Case-by-case comparison](https://docs.google.com/spreadsheets/d/1x8HGqRiozO5-Yck17yXFP-AipUIVfhWgywo07hLRVbA/edit?gid=677623803#gid=677623803)

The case-by-case set includes **74 RADCURE** and **59 HECKTOR** cases used for visual and quantitative comparison against Test 1 (single merged tumour label).

### Comparison with earlier tests (narrative review)

Figures below come from the validation / review workflow documented in the spreadsheet (not necessarily identical to registry `imagesTs` test-pool means in [`experiments/registry.yaml`](../../experiments/registry.yaml)).

| Experiment | Training | RADCURE (review) | HECKTOR (review) | Notes |
|------------|----------|------------------|------------------|-------|
| **Test 1** | RADCURE only | Mean Dice **0.51**, median **0.58** | Mean **0.33**, median **0.31** | RADCURE segmentations mostly single primary GTVp; secondary lesions often unlabelled |
| **Test 2** | RADCURE + HECKTOR (1000 ep) | **0.50** | **0.48** | HECKTOR secondary tumour merged into primary; HECKTOR ↑, RADCURE ↓ vs Test 1 |
| **Test 3** | RADCURE + HECKTOR (700 ep) | ~0.38 (test Ts, registry) | ~0.55 (test Ts, registry) | Same merged tumour label as Test 2 |
| **Test 4** | RADCURE + HECKTOR (700 ep), **separate GTVp/GTVn** | Best RADCURE so far: **~0.58 (GTVp)**, **~0.60 (GTVn)** | **Worse than Test 2/3** | See spreadsheet for HECKTOR breakdown |

### Main findings

1. **RADCURE:** Test 4 gives the **best average and median Dice** in our review so far. Where the primary tumour is consistently annotated, separate labels appear to help the network focus on the correct target.
2. **HECKTOR:** Performance **deteriorated substantially** vs Test 2/3, with many cases showing large Dice drops.
3. **False second tumours:** Visual inspection suggests the model **predicts fewer spurious secondary lesions**, but often **over-segments the primary** lesion instead.
4. **Low Dice caveats:** Several extremely low scores on HECKTOR may **not** reflect complete segmentation failure — manual verification is recommended (see case-by-case sheet).
5. **Label semantics matter:** Tumour definition (merged vs split GTVp/GTVn) strongly affects behaviour and is a likely source of **domain shift** between RADCURE and HECKTOR.

---

## Interpretation — why HECKTOR may have worsened

- **Nodal disease definition:** GTVn boundaries are less consistent; separating labels may **widen the domain gap** between cohorts.
- **Class imbalance:** Two tumour classes plus many slices without GTVn → **sparser supervision** for GTVn.
- **Anatomical ambiguity:** Primary and nodal regions can look similar on CT → **confusion between GTVp and GTVn**.
- **Cross-dataset annotation:** GTVp/GTVn are **defined and drawn differently** on RADCURE vs HECKTOR; joint training with separate labels exposes that conflict more directly than a merged GTVp class.

---

## Known issues (processing and evaluation)

| Issue | Description |
|-------|-------------|
| **Eval visualisation colour map** | Ground truth and prediction overlays do not always use the **same colormap** → misleading visual comparison; needs fix. |
| **GTVn on RADCURE** | Nodal disease is **under-represented or missing** in RADCURE labels; GTVn mapping should be reviewed. |
| **Evaluation pipeline** | Some metrics / visual outputs may be **off**; cross-check against spreadsheet and manual review. |
| **Head / background / TotalSegmentator** | Segmentation of head, background, and soft-tissue masks may need **reprocessing** for consistency. |
| **Fixed colour map** | A **single canonical colormap** across GT, prediction, and notebooks is not yet enforced. |

---

## Case-by-case review (133 cases)

A total of **133 validation cases** (74 RADCURE, 59 HECKTOR) were visually and quantitatively compared between **Test 1** (single tumour label) and **Test 4** (separate GTVp/GTVn).

**Conclusion from review:** Separating GTVp and GTVn makes the task **more semantically precise**. That helps on RADCURE (consistent primary-tumour annotation) but hurts cross-dataset generalisation on HECKTOR (nodal disease prevalence and annotation practice differ). Test 4 supports treating **annotation semantics** as a first-class factor in RADHECK domain shift — not only dataset size or epoch count.

---

## Next steps (Test 4b / preprocessing quality)

Before another retrain with the **same nnUNet setup as Test 4**:

1. Re-run **head / background / TotalSegmentator** and remaining tissue masks where quality is uncertain.
2. **Fix and document a fixed colour map** for all visualisation and evaluation PDFs.
3. Add **teeth** mask to the label set (if clinically relevant for the pipeline).
4. Detect and **discard anomalous body regions** / failed slices in preprocessing QC.
5. **Verify GTVn mapping on RADCURE** (currently weak or absent).
6. Fix **eval visualisation** GT vs prediction colour consistency.

Then retrain with separate GTVp/GTVn labels (planned follow-up experiment).

---

## References in this repository

| Resource | Path |
|----------|------|
| Phase 1 notebook | [`test4_preprocessing_preview.ipynb`](test4_preprocessing_preview.ipynb) |
| Phase 2 runbook | [`PHASE2.md`](PHASE2.md) |
| Phase 3 runbook | [`PHASE3.md`](PHASE3.md) |
| Server retrain steps | [`pipelines/radheck/Retrain-Radheck-Test4.md`](../../pipelines/radheck/Retrain-Radheck-Test4.md) |
| Experiment registry | [`experiments/registry.yaml`](../../experiments/registry.yaml) |
| Prior epoch study (Test 1–3) | [`research_notebooks/retrain_epoch_study/retrain_epoch_study.md`](../retrain_epoch_study/retrain_epoch_study.md) |
