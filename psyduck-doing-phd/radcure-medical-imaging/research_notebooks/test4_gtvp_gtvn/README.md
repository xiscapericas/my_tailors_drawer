# Test4 Phase 1 — GTVp / GTVn preprocessing preview

Validate **separate tumor labels** (`tumor_label_mode="separate"`) on one RADCURE and one HECKTOR case before full reprocessing (Phase 2).

**Test4 changes tumor label handling compared to test3; everything else stays identical for later phases.**

- Test1–3: GTVp + GTVn merged → single `GTVp` index
- Test4: `GTVp` and `GTVn` as distinct labels in the combined mask

See [`experiments/configs/test4_radheck_separate_gtvp_gtvn.yaml`](../../experiments/configs/test4_radheck_separate_gtvp_gtvn.yaml).
