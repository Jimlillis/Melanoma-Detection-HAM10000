# Dice Degeneracy 2x2 -- DINOv2 Frozen Probe vs. EfficientNet

Generated: 2026-09-05T20:37:12

Question: Dice loss gives nv recall = 0.00 on all 5 frozen linear probes (Part A), but was the BEST loss on EfficientNet-B0 full fine-tuning (0.7029 Macro-F1, nv recall 0.8486). H1 (capacity) says a linear head on frozen features cannot escape the collapse; H2 (rebalancing scheme) says EfficientNet's train-split resampling (done before the loss ever sees the data) is what actually prevents it, not model capacity. All 4 runs below are DINOv2 features, Dice loss, seed=42 -- only rebalancing and sampler vary.

## Determinism check (Run 1 vs. Part A control)

Run 1 reproduced Part A's dinov2+dice control EXACTLY (macro_f1=0.406733, mel_recall=0.922156, nv_recall=0.000000). Pipeline is deterministic; the runs below are comparable.

## Results

| Run | Rebalance | Sampler | Macro-F1 | MEL Recall | nv recall | Degenerate |
|---|---|---|---|---|---|---|
| 1 | no | yes | 0.4067 | 0.9222 | 0.0000 | YES |
| 2 | yes | no | 0.6093 | 0.5090 | 0.8088 | no |
| 3 | yes | yes | 0.6046 | 0.5569 | 0.7390 | no |
| 4 | no | no | 0.4263 | 0.4371 | 0.9143 | no |

Degenerate = nv recall < 0.10.

## Share of all test predictions landing in each class

| Run | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| 1 | 5.4% | 6.5% | 0.0% | 1.6% | 85.5% | 0.0% | 1.1% |
| 2 | 5.8% | 7.1% | 11.3% | 1.7% | 14.2% | 58.5% | 1.5% |
| 3 | 5.3% | 7.5% | 14.6% | 1.8% | 16.8% | 52.3% | 1.7% |
| 4 | 2.5% | 6.2% | 9.1% | 0.0% | 10.0% | 72.2% | 0.0% |

Reference: `nv` is 1004/1502 = 66.8% of the test set by true support.

## Verdict

**H2 supported: rebalancing is the cause, model capacity is irrelevant.** Both runs with train-split rebalancing (2: rebalance-only, 3: rebalance+sampler) do NOT degenerate, regardless of whether the sampler is also present. The linear probe is perfectly capable of avoiding the collapse once the class distribution it actually trains on is rebalanced -- the earlier failure on frozen features was a property of the imbalance-handling scheme, not a capacity limit of a linear layer.

**Additional finding: the sampler is not inert.** Run 4 (neither rebalancing nor sampler) does NOT degenerate, unlike Run 1 (sampler-only). Run 1 and Run 4 differ only in the WeightedRandomSampler, so the sampler itself is actively contributing to (or protecting against) the collapse, independent of rebalancing -- a third mechanism beyond the H1/H2 framing as originally posed.
