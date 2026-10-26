# EfficientNet-B0 Loss-Function Ablation -- Summary

Generated: 2026-09-04T06:28:15

Four otherwise-identical two-phase EfficientNet-B0 runs (15 epochs frozen-backbone + 15 epochs full fine-tune, seed=42, AdamW, weight_decay=1e-4, batch size 32, same lesion-level split, same train-split rebalancing), differing only in the training loss. FocalLoss/DiceLoss/CombinedLoss are the exact same implementations already used by train_dino_head.py for the DINO/BiomedCLIP linear probes -- not reimplemented here.

## 1. Headline results

| Loss | Test Macro-F1 (plain) | Test MEL Recall (plain) | Test Macro-F1 (TTA+MC) | Test MEL Recall (TTA+MC) | Best epoch | Phase | Wall-clock (train) |
|---|---|---|---|---|---|---|---|
| ce | 0.6903 | 0.7186 | 0.6871 | 0.7246 | 14 | 2 | 48.8 min |
| focal | 0.6910 | 0.6826 | 0.6882 | 0.7425 | 11 | 2 | 49.0 min |
| dice | 0.7029 | 0.7246 | 0.7005 | 0.7186 | 14 | 2 | 49.4 min |
| combined | 0.6875 | 0.7126 | 0.6968 | 0.7425 | 14 | 2 | 49.2 min |

## 2. Per-class F1 (TTA+MC evaluation)

| Loss | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| ce | 0.64 | 0.75 | 0.65 | 0.60 | 0.59 | 0.88 | 0.70 |
| focal | 0.65 | 0.73 | 0.63 | 0.61 | 0.55 | 0.86 | 0.79 |
| dice | 0.63 | 0.74 | 0.54 | 0.81 | 0.56 | 0.89 | 0.74 |
| combined | 0.67 | 0.75 | 0.63 | 0.63 | 0.58 | 0.88 | 0.74 |

## 3. Degenerate-solution check (TTA+MC evaluation)

Share of ALL test-set predictions landing in each class (column sums of the confusion matrix / total). `nv` is 1004/1502 = 66.8% of the test set by true support, so some nv-heaviness in the share column is expected -- the verdict flags actual collapse: nv recall dropping toward 0 (the failure mode Dice produced on the DINO linear probes, where nv recall hit exactly 0.00), or any single class absorbing an implausible share of all predictions.

| Loss | akiec share | bcc share | bkl share | df share | mel share | nv share | vasc share | nv recall | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| ce | 4.4% | 5.6% | 12.1% | 1.3% | 16.3% | 58.6% | 1.7% | 0.8267 | OK |
| focal | 4.5% | 5.9% | 11.7% | 1.1% | 19.0% | 56.7% | 1.1% | 0.7988 | OK |
| dice | 4.3% | 5.6% | 9.1% | 1.5% | 17.2% | 61.1% | 1.1% | 0.8486 | OK |
| combined | 4.7% | 5.6% | 10.9% | 1.2% | 17.2% | 58.7% | 1.7% | 0.8217 | OK |

## 4. Run A (weighted CE) vs. the existing saved EfficientNet reports

| | Macro-F1 (plain) | MEL Recall (plain) | Macro-F1 (TTA+MC) | MEL Recall (TTA+MC) |
|---|---|---|---|---|
| Existing saved (`efficientnet_baseline_7class` / `_improved_7class`) | 0.6903 | 0.7186 | 0.7009 | 0.7246 |
| This run (Run A, weighted CE, seed=42, fixed checkpoint format) | 0.6903 | 0.7186 | 0.6871 | 0.7246 |
| Difference (this run - existing) | +0.0000 | -0.0000 | -0.0138 | -0.0000 |

The existing saved report's checkpoint has an unrecoverable seed (never stored -- see `EXPERIMENTAL_RECORD.md`, action-list item 5), so any difference above reflects the combined effect of a (possibly different) historical seed plus ordinary run-to-run stochasticity (DataLoader shuffling, augmentation sampling, cuDNN nondeterminism) -- it is not a controlled ablation of the seed alone, since the historical run cannot be reproduced or inspected further. Run A here uses the now-fixed checkpoint format (seed=42 explicit; epoch, phase and wall-clock recorded), so this gap cannot recur for future runs of this script.
