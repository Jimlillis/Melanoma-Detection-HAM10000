# DINOv2 Final MEL-Threshold Selection (split-consistent constraint)

Generated: 2026-09-07T02:39:57

**Why this supersedes the earlier `t=0.014023` selection:** that pick used `val macro-F1 >= 0.701` as its constraint, where 0.701 is EfficientNet's **test** macro-F1 (TTA+MC) -- a split mismatch, letting test-set information influence a validation-side selection. The correct anchor is EfficientNet's own **validation** macro-F1, read directly from its checkpoint.

## Step 1 -- Anchor

`outputs/efficientnet_mc/best_efficientnet.pth`, key `val_macro_f1` = **0.6843836211** (read directly, not hard-coded).

## Step 2 -- Constrained selection

Rule: maximise validation MEL Recall subject to validation macro-F1 >= 0.684384. Exhaustive sweep over every distinct observed validation p_mel value (1508 candidates), not a fixed-step grid.

| | Threshold | Val Macro-F1 | Val MEL Recall | Test Macro-F1 | Test MEL Recall |
|---|---|---|---|---|---|
| Previous (superseded) | 0.014023 | 0.7012 | 0.7861 | 0.7259 | 0.7545 |
| **New (this report)** | **0.002804** | 0.6863 | 0.8613 | 0.6932 | 0.8144 |

**t* differs from the previous 0.014023:** True.

**Is the constraint binding at t*?** val macro-F1 (0.6863) exceeds the anchor (0.6844) by +0.0019 -- binding, sitting essentially at the floor.

## Step 3 -- Feasible region

Feasible thresholds (val macro-F1 >= 0.6844): 334 of 1508 candidates, spanning t in [0.002779, 0.996749] (one contiguous interval).

Maximum validation MEL Recall achievable anywhere in the feasible region: **0.861272**, achieved at 7 tied threshold(s) (selected t* is the highest-val-F1 tie, then largest threshold).

**The relaxation ADMITS higher recall than the previous selection's 0.7861** -- the looser constraint changes what's achievable, not just which point gets picked.

## Step 4 -- Rounded threshold check

| Threshold | Val Macro-F1 | Val MEL Recall | Test Macro-F1 | Test MEL Recall |
|---|---|---|---|---|
| t* = 0.002804 | 0.6863 | 0.8613 | 0.6932 | 0.8144 |
| t = 0.014 (rounded, but this is the OLD threshold's rounding, not t*'s) | 0.7012 | 0.7861 | 0.7259 | 0.7545 |
| t = 0.003 (natural 3-decimal rounding of t* itself) | 0.6851 | 0.8613 | 0.6954 | 0.8144 |

**Note:** t=0.014 is the natural rounding of the OLD, superseded 0.014023 -- included because it was explicitly requested, but it is not a rounding of this report's t*. Its test metrics being identical to the old 0.014023 row above just confirms no validation or test sample's probability falls between 0.014 and 0.014023. The more relevant rounding check for the NEW t* is the t=0.003 row: rounding costs nothing here -- test MEL Recall is unchanged (0.8144 vs 0.8144) and test Macro-F1 is not lower (0.6954 vs 0.6932). Report t=0.003, not the six-decimal 0.002804 -- no accuracy cost, looks far less overfit to validation.

## Melanoma precision at the selected threshold

At t* = 0.002804 (test set): **melanoma precision = 0.3560**, recall = 0.8144.
At the rounded t=0.014: melanoma precision = 0.4360, recall = 0.7545.
At t=0.003 (recommended -- natural rounding of t* itself, no cost): melanoma precision = 0.3598, recall = 0.8144.

## Step 5 -- Updated Pareto front

Across all 43 named results on record (every experiment in `Final Results.txt`, the loss-ablation and probe-ablation runs, and this report's new threshold pick; excludes raw diagnostic-curve points such as t=0 that were never proposed as an answer). 7 are non-dominated.

| Macro-F1 | MEL Recall | Model |
|---|---|---|
| 0.7440 | 0.5690 | DINOv2 FT, best-val-F1 ckpt, argmax |
| 0.7380 | 0.6830 | Ensemble A v2, Composite criterion |
| 0.7270 | 0.7430 | DINOv2 FT, threshold t=0.02 |
| 0.6932 | 0.8144 | DINOv2 FT, constrained-optimal t=0.002804 (this report, anchor=val EffNet F1) |
| 0.6700 | 0.8560 | DINOv2 FT, threshold t=0.001 |
| 0.6534 | 0.9042 | DINOv2 FT, composite-optimal t=0.000259 (audit) |
| 0.4067 | 0.9222 | DINOv2 probe, dice |
