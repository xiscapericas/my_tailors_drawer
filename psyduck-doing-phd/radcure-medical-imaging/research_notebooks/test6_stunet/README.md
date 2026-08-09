# Test6 (research) — STU-Net exploration

**Status:** inference explore **done**. **Fine-tune is now the main Test6 experiment** — see the implementation runbook:

→ [`pipelines/radheck/Retrain-Radheck-Test6.md`](../../pipelines/radheck/Retrain-Radheck-Test6.md)

**One-line rule (fine-tune):** Test6 changes the **model** to STU-Net fine-tuning vs Test5; same Dataset650 / GTVp+GTVn / splits; no TotalSegmentator reprocess.

Paper / code: [arXiv:2304.06716](https://arxiv.org/abs/2304.06716) · [uni-medical/STU-Net](https://github.com/uni-medical/STU-Net)

## What this folder is for

| File | Purpose |
|------|---------|
| [`test6_stunet_inference_explore.ipynb`](test6_stunet_inference_explore.ipynb) | Historical Colab explore (pretrained STU has no GTVp/GTVn) |
| [`label_orders.json`](label_orders.json) | STU-Net TotalSegmentator class index → name |
| This README | Explore findings; pointer to FT runbook |

## Fine-tune (server)

```bash
export TEST6_WORK_ROOT=/media/HDD_8TB/xisca/work/retrain_test6_stunet
export TEST6_DATASET650=/media/HDD_8TB/xisca/work/retrain_test5/Dataset650_TotalSegmentator
export TEST6_STU_VARIANT=small

python -m pipelines.test6.setup_stunet
python -m pipelines.test6.link_test5_dataset
python -m pipelines.test6.train_finetune --step prepare
python -m pipelines.test6.train_finetune --step plan
python -m pipelines.test6.train_finetune --step train
python -m pipelines.test6.evaluate
```

Config / registry: [`test6_radheck_stunet_finetune.yaml`](../../experiments/configs/test6_radheck_stunet_finetune.yaml) · [`registry.yaml`](../../experiments/registry.yaml) (`test6`).

---

## Explore chapter (closed) — findings

**Pretrained STU-Net does not predict GTVp / GTVn** — that is why we fine-tune.

First Colab run (STU-Net-S, 4 cases): 0 name-matched organs vs H&N dict; few STU labels on H&N FOV. Lessons (plans.pkl layout, torch.load, bundled nnUNet) remain useful for setup.

### Legacy Colab re-run

Open the notebook §0–7 if you need inference-only explore again. Prefer the repo copy (synced fixes).
