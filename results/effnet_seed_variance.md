> **ERRATUM (2026-10-02):** the row `DINOv2 (0.5446) vs DINOv3 (0.3797), frozen probes` (+0.1649, here and in the probe-sigma restatement) used the DINOv3 probe trained on mean-pooled patch tokens, not the CLS token. With the CLS-token probe (macro-F1 0.4831, MEL Recall 0.6228) the difference is +0.0615, i.e. about 5.9 sigma (proxy sigma 0.0105) or 5.5 sigma (probe sigma 0.0111): the verdict is still **exceeds noise floor** (> 2 sigma), but the gap is much smaller. The table values below are NOT rewritten.

# EfficientNet-B0 Seed Variance (corrected)

Generated: 2026-09-05T22:24:06

Seeds 1337 and 2024 re-run here with `--split-seed 42` pinned explicitly (decoupled from the varying `--seed`); seed 42 is reused unchanged from the earlier valid run (its split-seed and training-seed already coincided at 42, so it was never affected by the leakage bug). Two gates applied per new run before its numbers are trusted: (1) split-integrity -- checkpoint's own stored seed/split_seed fields plus an independent recomputation of the seed=42 split must match the canonical test-set counts exactly; (2) leakage-threshold -- test macro-F1 > 0.80 on either protocol is flagged and excluded from statistics.

## Pre-flight check

Recomputed seed=42 split independently before any training: {'akiec': 52, 'bcc': 71, 'bkl': 167, 'df': 20, 'mel': 167, 'nv': 1004, 'vasc': 21} -- matches the canonical counts exactly.

## Results

| Seed | Status | Macro-F1 (plain) | MEL Recall (plain) | Macro-F1 (TTA+MC) | MEL Recall (TTA+MC) |
|---|---|---|---|---|---|
| 42 | OK | 0.6903 | 0.7186 | 0.6871 | 0.7246 |
| 1337 | OK | 0.6722 | 0.7365 | 0.6967 | 0.7425 |
| 2024 | OK | 0.6720 | 0.6407 | 0.6980 | 0.6407 |

## Summary statistics (OK runs only, n=3)

| Statistic | Macro-F1 (plain) | MEL Recall (plain) | Macro-F1 (TTA+MC) | MEL Recall (TTA+MC) |
|---|---|---|---|---|
| mean | 0.6782 | 0.6986 | 0.6939 | 0.7026 |
| std dev (ddof=1) | 0.0105 | 0.0509 | 0.0059 | 0.0543 |
| min | 0.6720 | 0.6407 | 0.6871 | 0.6407 |
| max | 0.6903 | 0.7365 | 0.6980 | 0.7425 |
| range (max-min) | 0.0183 | 0.0958 | 0.0108 | 0.1018 |

**plain** isolates training variance (deterministic evaluation). **TTA+MC** mixes in MC-Dropout sampling noise. Kept separate, not merged, by design.

## Verdict: do these differences exceed the noise floor?

Noise floor: std dev of test Macro-F1 across seeds on the **plain** evaluation, sigma = 0.0105 (n=3). A gap counts as real only if it exceeds 2*sigma = 0.0211; 1-2 sigma is suggestive; below 1 sigma is indistinguishable from noise.

| Comparison | Difference | Verdict |
|---|---|---|
| Dice (0.7029) vs CE (0.6903), EfficientNet, plain | +0.0126 | suggestive only (1-2 sigma) |
| DINOv1 (0.554) vs DINOv2 (0.5446), frozen probes | +0.0094 | **within noise** (< 1 sigma) |
| DINOv2 (0.5446) vs DINOv3 (0.3797), frozen probes | +0.1649 | **exceeds noise floor** (> 2 sigma) |

Caveat: sigma is measured on EfficientNet-B0 (3 seeds, weighted CE). The DINOv1/DINOv2/DINOv3 comparisons use it as a proxy noise floor -- no seed-variance estimate exists for the frozen linear probes themselves (every probe result in this repo, Part A included, is a single seed).

## Probe-specific seed variance

Generated: 2026-09-06T15:43:56

DINOv2 frozen linear probe, `--loss combined`, seeds 42/1337/2024, everything else identical to the Part A probe ablation (default `train_dino_head.py` settings -- WeightedRandomSampler on, no train-split rebalancing). Purpose: the verdicts above used EfficientNet's seed-to-seed sigma (0.0105 macro-F1) as a proxy noise floor for probe comparisons; this measures the probe's own sigma directly, since a linear head on frozen features is expected to be far less variable than 30 epochs of full CNN training.

Determinism check (seed 42 vs. Part A control): Seed 42 reproduced Part A's dinov2+combined control EXACTLY (macro_f1=0.544641, mel_recall=0.688623). Pipeline is deterministic; the three seeds below are comparable.

| Seed | Macro-F1 | MEL Recall |
|---|---|---|
| 42 | 0.5446 | 0.6886 |
| 1337 | 0.5246 | 0.7425 |
| 2024 | 0.5428 | 0.6407 |

| Statistic | Macro-F1 | MEL Recall |
|---|---|---|
| mean | 0.5373 | 0.6906 |
| std dev (ddof=1) | 0.0111 | 0.0509 |

**Probe-specific Macro-F1 sigma = 0.0111** (n=3), vs. the EfficientNet proxy of 0.0105 used above. **Probe-specific MEL Recall sigma = 0.0509** (n=3) -- reported explicitly here since it was omitted for the plain evaluation in the EfficientNet section above.

### Re-stated verdicts, using the probe-specific Macro-F1 sigma

sigma = 0.0111, 2*sigma = 0.0222. Same criterion as above: a gap counts as real only if it exceeds 2*sigma; 1-2 sigma is suggestive; below 1 sigma is indistinguishable from noise.

| Comparison | Difference | Verdict (proxy sigma=0.0105) | Verdict (probe sigma=0.0111) |
|---|---|---|---|
| DINOv1 (0.554) vs DINOv2 (0.5446), frozen probes | +0.0094 | **within noise** (< 1 sigma) | **within noise** (< 1 sigma) |
| DINOv2 (0.5446) vs DINOv3 (0.3797), frozen probes | +0.1649 | **exceeds noise floor** (> 2 sigma) | **exceeds noise floor** (> 2 sigma) |
