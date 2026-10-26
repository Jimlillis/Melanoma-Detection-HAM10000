# Complementarity Analysis: EfficientNet-B0 vs DINOv2 Linear Probe

Read-only inference analysis, test split (1502 images), generated to
check the Section 4.3 claim that DINOv2 and EfficientNet-B0 "fail on
different categories of images" and that DINOv2 catches atypical cases
EfficientNet-B0 misses.

## Checkpoints used

- **EfficientNet-B0**: `outputs/efficientnet_mc/best_efficientnet.pth`,
  evaluated with 5-view TTA + MC Dropout (10 passes/view) -- the thesis's
  own reported protocol for this model.
- **DINOv2 linear probe**: `outputs/probe_loss_ablation/dinov2_combined/best_dino_head.pth`
  (combined loss). Verified byte-identical `model_state_dict` to
  `outputs/dinov2_linear/best_dino_head.pth` -- both are the same trained
  weights; the ablation-named copy was used because its saved metadata
  explicitly records `loss_name='combined'`. Evaluated with plain argmax
  on cached test features (its own reported protocol; frozen backbone,
  no TTA/MC).
- **NOTE (unrelated to this analysis, but discovered while identifying the
  checkpoint)**: `Final Results.txt`'s DINOv2 row is labeled "(default)" with
  a footnote claiming the default loss is focal -- but the checkpoint
  producing 0.545/0.689 is confirmed to be the **combined**-loss run, not
  focal (focal alone gives 0.502/0.557, see `probe_loss_dinov2_focal_results.txt`).
  That mislabel was not corrected here; flagging it for a separate fix.

## Reproduction check

| Model | Recomputed Macro-F1 / MEL-Recall | Published | Status |
|---|---|---|---|
| DINOv2 combined probe | 0.5446 / 0.6886 | 0.545 / 0.689 | Exact match |
| EfficientNet-B0 (TTA+MC) | 0.6871 / 0.7246 | 0.701 / 0.725 | MEL-Recall matches; Macro-F1 does not |

**EfficientNet Macro-F1 caveat**: the 5-view TTA + MC-Dropout evaluation is
inherently stochastic (dropout masks + non-deterministic GPU kernels), so it
is not bit-reproducible run-to-run even from the identical checkpoint with
`seed=42` fixed. A confusion-matrix diff against the published
`efficientnet_improved_7class_results.txt` showed only a handful of samples
per class flipping (concentrated in the nv/bkl/mel boundary and the rare
classes), not a systematic error -- and this run's numbers (0.687/0.725)
closely match the independent seed=42 TTA+MC control already reported in
`Final Results.txt` Section 6 (0.687/0.725), i.e. this is the same
already-documented MC-Dropout variance, not a new bug. Per explicit user
decision, this run's own numbers are used as ground truth for everything
below, rather than chasing the unreproducible 0.701 headline figure.

## 3a. Overall agreement (1502 test images)

- Both correct: **856** (57.0%)
- EfficientNet-only correct: **333** (22.2%)
- DINOv2-only correct: **74** (4.9%)
- Both wrong: **239** (15.9%)
- Identical predicted label (right or wrong): **1003** (66.8%)

## 3b. Melanoma specifically (167 mel images) -- the core claim

| Outcome | Count | Fraction of 167 mel images |
|---|---|---|
| Correct by both | 92 | 55.1% |
| Correct by EfficientNet only | 29 | 17.4% |
| **Correct by DINOv2 only** | **23** | **13.8%** |
| Missed by both | 23 | 13.8% |

DINOv2-only vs EfficientNet-only on mel: 23 vs 29 (difference: -6)

## 3c. Per-class breakdown

| Class | Support | Both correct | EffNet only | DINOv2 only | Neither |
|---|---|---|---|---|---|
| akiec | 52 | 28 | 10 | 8 | 6 |
| bcc | 71 | 47 | 11 | 6 | 7 |
| bkl | 167 | 59 | 55 | 14 | 39 |
| df | 20 | 12 | 0 | 4 | 4 |
| mel | 167 | 92 | 29 | 23 | 23 |
| nv | 1004 | 604 | 226 | 16 | 158 |
| vasc | 21 | 14 | 2 | 3 | 2 |

Totals across all 7 classes -- EffNet-only correct: **333**, DINOv2-only correct: **74**.

**Interpretation**: EfficientNet-only-correct outnumbers DINOv2-only-correct in
every class except df (0 vs 4, support only 20) and vasc (2 vs 3, support only
21) -- both negligible in absolute count. On mel specifically, EfficientNet's
unique-correct count (29) is *larger* than DINOv2's (23), not smaller. So
whatever complementarity exists is not melanoma-specific and is not
DINOv2-favoring -- it is small in magnitude everywhere and, where it leans one
way, leans toward EfficientNet, consistent with EfficientNet's higher
standalone Macro-F1 (0.687 vs 0.545).

## 3d. Oracle upper bound

Oracle = always picks whichever of the two models is correct, when at
least one is (ties don't arise since both-correct implies the same
right label). When both are wrong, the oracle's predicted label is taken
from EfficientNet arbitrarily (affects only which wrong class absorbs the
error, not the right/wrong pattern that drives this bound).

- Oracle Macro-F1: **0.7934**
- Oracle MEL Recall: **0.8623**

| Model | Macro-F1 | MEL Recall |
|---|---|---|
| EfficientNet-B0 (TTA+MC), this run | 0.6871 | 0.7246 |
| DINOv2 combined probe | 0.5446 | 0.6886 |
| **Oracle (perfect selector)** | **0.7934** | **0.8623** |

The oracle bound (0.793 Macro-F1) sits well above either model
alone, which is expected (an oracle selector is a much stronger baseline
than any real ensemble can reach) and gives context for why every actual
ensemble strategy in Final Results.txt Section 5 (max Macro-F1 = 0.738, via
Ensemble A v2 Composite) fell well short of this ceiling: no soft-vote or
learned combiner can perfectly guess which single model is right on each
image the way an oracle can.

## Step 4 -- Verdict

**PARTIAL / DOES NOT CLEANLY SUPPORT**

- DINOv2-only-correct on melanoma: **23 / 167** (13.8%).
  This is a meaningful fraction of the 167 melanoma images (threshold used: 10%).
- EfficientNet-only-correct on melanoma: **29 / 167** (17.4%).
- DINOv2-only is SMALLER than EfficientNet-only on melanoma (23 vs 29).

The data shows some DINOv2-only-correct melanoma cases, but EfficientNet
catches at least as many (or more) melanoma cases that DINOv2 misses.
The claim as written -- framed as one-directional ("DINOv2 catches what
EfficientNet misses") -- is **not supported**; the complementarity, if
any, runs in both directions or is dominated by EfficientNet. Consider
rewording Section 4.3 to describe bidirectional / class-general
complementarity (see Section 3c) rather than a DINOv2-specific
melanoma-detection advantage.

Per-sample source data: `results/per_sample_effnet.csv`, `results/per_sample_dinov2_probe.csv`.