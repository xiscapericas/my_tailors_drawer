> **Runbook (server commands)** — not the canonical experiment record.  
> **Results & hypotheses:** [`experiments/registry.yaml`](../../experiments/registry.yaml) (GTVp Dice) and [`retrain_epoch_study.md`](../retrain_epoch_study/retrain_epoch_study.md).

# Test2 — RADHECK combined (1000 epochs)

## Results snapshot (multi-organ summary from this run)

### Dice Score Summary

| Dataset | Average Dice Score |
| ------- | ------------------ |
| HECKTOR | 0.635              |
| RADCURE | 0.409              |
| Overall | 0.617              |

### Performance Comparison

The retrained model achieved an overall **13.83% improvement** compared to the previous version.

However, the improvement is highly dependent on the dataset being evaluated:

| Dataset | Improvement vs Previous Model |
| ------- | ----------------------------- |
| RADCURE | +0.03%                        |
| HECKTOR | +47.22%                       |

### Key Observations

* The retrained model achieves an overall Dice score of **0.617**.
* Performance differs substantially between datasets:

  * **HECKTOR: 0.635 Dice**
  * **RADCURE: 0.409 Dice**
* The gap of approximately **0.226 Dice points** between datasets indicates a strong dependency on dataset characteristics.
* The improvement on **RADCURE is negligible**, indicating that retraining did not meaningfully improve performance on this dataset.
* The large improvement on **HECKTOR** suggests that the model successfully learned characteristics specific to HECKTOR cases.
* The performance discrepancy indicates strong **dataset dependency**.
* No evidence of improved cross-dataset generalization was observed.
* Prediction patterns between the old and new models show little correlation, suggesting that both models rely on substantially different learned representations.

---

**Note:** Tables above summarise this run (multi-organ / pipeline metrics). For **GTVp Dice** on Test1–3 (74 RADCURE + 48 HECKTOR cases), see [`experiments/registry.yaml`](../../experiments/registry.yaml).
