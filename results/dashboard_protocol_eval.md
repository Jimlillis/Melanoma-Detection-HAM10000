# Dashboard uncertainty protocol: evaluation on the test set

Produced by `src/eval_dashboard_protocol.py`, calling the dashboard's own `app/main.py::_predict()` on every test image (device: `cpu`, as on the Space). Protocol: 5 TTA views × 10 MC-Dropout passes = 50 samples, seed 42 per request. Test split: lesion-level, seed 42, **1502 images** (167 melanomas, 323 EfficientNet errors).

## Reuse of the thesis code

- `build_tta_transforms` source text identical to `src/train_efficientnet.py`: **True**
- `_NORM` identical: **True**
- Views produced by app vs thesis transforms (same seed), bitwise equal: **20/20** images
- EfficientNet prediction agrees with the thesis's own run (`results/mc_uncertainty_per_sample.csv`) on **97.4%** of images; per-image mean variance correlates with the thesis values at Spearman ρ = **0.938** (Pearson r = 0.877). Differences are MC sampling noise: the thesis run was unseeded.

## (a) Classification (EfficientNet, argmax of the mean of 50 samples)

| Metric | App | Thesis |
|---|---|---|
| macro-F1 | **0.693** | 0.701 (re-runs 0.687–0.701) |
| MEL Recall | **0.725** (121/167) | 0.725 |

Within the thesis re-run range: **yes**

## (b) Section 6.3 statistic: mean variance

| Quantity | App | Thesis |
|---|---|---|
| Mean, all images | 0.0068 | 0.0067 |
| Mean, correct predictions | 0.0048 | 0.0047 |
| Mean, incorrect predictions | 0.0138 | 0.0139 |
| Ratio incorrect / correct | 2.85× | 2.96× |
| AUROC for predicting an error | 0.802 | 0.81 |
| Errors captured, top 10% referred | 22.9% | 23.5% |
| Errors captured, top 20% referred | 43.7% | 45.4% |

## (c) Displayed statistic: std of the predicted class's probability

This is the ± users see (0.052 is shown as ± 5.2%). The thesis did not analyse this statistic, so there is no thesis value.

| Quantity | App | Thesis |
|---|---|---|
| Mean, all images | 0.1042 | – |
| Mean, correct predictions | 0.0798 | – |
| Mean, incorrect predictions | 0.1931 | – |
| Ratio incorrect / correct | 2.42× | – |
| AUROC for predicting an error | 0.800 | – |
| Errors captured, top 10% referred | 22.3% | – |
| Errors captured, top 20% referred | 44.0% | – |

## (d) Melanomas: correctly detected vs missed

| Statistic | Detected (n=121) | Missed (n=46) | Ratio | Thesis |
|---|---|---|---|---|
| Mean variance | 0.0071 | 0.0113 | 1.60× | 0.0075 / 0.0102 (1.36×) |
| Displayed std | 0.1245 | 0.1810 | 1.45× | – |

## (e) Agreement score (Grad-CAM vs DINOv2 attention)

Before = the committed backend (git HEAD: 20 passes, single view). Grad-CAM always targets EfficientNet's predicted class, so r can only change if that class changes.

| Image | Before: EffNet class | Before: r | After: EffNet class | After: r |
|---|---|---|---|---|
| ISIC_0032214 | nv | -0.140 | bcc | -0.153 |
| ISIC_0025387 | bkl | 0.675 | bkl | 0.675 |
| ISIC_0026981 | akiec | 0.614 | akiec | 0.614 |
| ISIC_0027169 | nv | 0.762 | nv | 0.762 |
| ISIC_0027488 | df | 0.546 | df | 0.546 |
| ISIC_0028238 | bcc | 0.582 | bcc | 0.582 |
| ISIC_0031201 | vasc | 0.725 | vasc | 0.725 |
| ISIC_0032550 | mel | 0.701 | mel | 0.701 |

Thesis Section 7.2 quotes −0.153 for ISIC_0032214 and 0.55–0.76 for the seven correct examples.

## (f) The seven representative samples

| Sample | Primary | Primary confidence | EfficientNet | EfficientNet ± | Flag fires |
|---|---|---|---|---|---|
| sample_mel.jpg | mel | 99.6% | mel | 98.8% ± 2.5% | no |
| sample_bcc.jpg | bcc | 100.0% | bcc | 100.0% ± 0.0% | no |
| sample_akiec.jpg | akiec | 100.0% | akiec | 100.0% ± 0.0% | no |
| sample_nv.jpg | nv | 100.0% | nv | 100.0% ± 0.0% | no |
| sample_bkl.jpg | bkl | 100.0% | bkl | 100.0% ± 0.0% | no |
| sample_df.jpg | df | 100.0% | df | 100.0% ± 0.0% | no |
| sample_vasc.jpg | vasc | 100.0% | vasc | 100.0% ± 0.0% | no |

## (g) The two challenging samples

- `challenge_ISIC_0030443.jpg`: primary **mel**, flag fires: **yes**, EfficientNet mel 89.1% ± 8.9%, agreement r = 0.846 ("High agreement"), mean variance 0.0021. Two identical requests gave byte-identical JSON: **True**
- `challenge_ISIC_0032214.jpg`: primary **nv**, flag fires: **no**, EfficientNet bcc 30.9% ± 27.1%, agreement r = -0.153 ("Models focus on different regions"), mean variance 0.0245. Two identical requests gave byte-identical JSON: **True**

## Local latency (not the Space)

`_predict()` on this machine's CPU (6 threads): median 1.76 s, 90th percentile 3.06 s per image. The Space has different hardware; its latency is measured separately after deployment.
