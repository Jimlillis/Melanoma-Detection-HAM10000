# State-of-the-art comparison: extra metrics and ISIC 2018 Task 3 test set

Generated 2026-10-15 by `src/write_sota_report.py`. Inference only: no training, no threshold re-tuning, no `.tex`, checkpoint or existing results file touched.

**Configurations**

| Key | Configuration | Probabilities |
|---|---|---|
| (a) | DINOv2 ViT-S/14 full fine-tune, rule t = 0.003 (final model) | DINOv2 softmax |
| (b) | DINOv2 ViT-S/14 full fine-tune, argmax | DINOv2 softmax |
| (c) | EfficientNet-B0, 5-view TTA + MC Dropout (10 passes/view, seed 42), argmax | mean of the 50 samples |
| (d) | EfficientNet-B0, same protocol, rule t = 0.185 | mean of the 50 samples |
| (e) | EfficientNet-B0, plain forward (eval mode), argmax | single softmax |

- **Rule:** predict mel if p(mel) > t, otherwise the argmax of the other six classes (`src/dinov2_finetune_threshold_tune.py::apply_mel_threshold`). Both thresholds were chosen on our validation split.
- **Preprocessing (both test sets):** Resize((224, 224)), ToTensor, ImageNet normalisation (`src/dataset.py::build_transforms(224, is_train=False)`).
- **Devices:** DINOv2 runs on the GPU with autocast, batch 64. EfficientNet runs on the CPU, with TTA+MC via `app/tta_mc.py::tta_mc_predict`.
- **AUROC:** computed from the probabilities, so it does not depend on the threshold. (a) shares the AUROC of (b), and (d) shares the AUROC of (c).
- **Other definitions:**
  - BACC = mean per-class recall (`sklearn.metrics.balanced_accuracy_score`), the official ISIC 2018 ranking metric.
  - MEL specificity = TN/(TN+FP) for mel vs rest.
  - macro AUROC = one-vs-rest, macro average.
  - always nv: every image predicted nv. AUROC is undefined for a constant score.

## 1. Sanity gate (our test split, 1502 images)

| Config | Expected macro-F1 / MEL Recall | Reproduced (3 dp) | Unrounded | Result |
|---|---|---|---|---|
| (a) | 0.695 / 0.814 | 0.695 / 0.814 | 0.6954 / 0.8144 | PASS |
| (b) | 0.744 / 0.569 | 0.744 / 0.569 | 0.7436 / 0.5689 | PASS |
| (c) | 0.693 / 0.725 | 0.693 / 0.725 | 0.6930 / 0.7246 | PASS |
| (d) | 0.664 / 0.814 | 0.664 / 0.814 | 0.6639 / 0.8144 | PASS |
| (e) | 0.690 / 0.719 | 0.690 / 0.719 | 0.6903 / 0.7186 | PASS |

All five rows pass; every table below uses (a)–(e).

## 2. Part B: our lesion-level test split

n = 1502; class counts: akiec 52, bcc 71, bkl 167, df 20, mel 167, nv 1004, vasc 21.

| Configuration | n | Acc | BACC | macro-F1 | MEL Recall | MEL Precision | MEL Specificity | MEL AUROC | macro AUROC |
|---|---|---|---|---|---|---|---|---|---|
| (a) DINOv2 FT, t = 0.003 (final) | 1502 | 0.769 | 0.676 | 0.695 | 0.814 | 0.360 | 0.819 | 0.910 | 0.964 |
| (b) DINOv2 FT, argmax | 1502 | 0.838 | 0.725 | 0.744 | 0.569 | 0.590 | 0.951 | 0.910 | 0.964 |
| (c) EfficientNet-B0, TTA+MC, argmax | 1502 | 0.785 | 0.739 | 0.693 | 0.725 | 0.486 | 0.904 | 0.915 | 0.963 |
| (d) EfficientNet-B0, TTA+MC, t = 0.185 | 1502 | 0.724 | 0.708 | 0.664 | 0.814 | 0.349 | 0.810 | 0.915 | 0.963 |
| (e) EfficientNet-B0, plain, argmax | 1502 | 0.784 | 0.745 | 0.690 | 0.719 | 0.500 | 0.910 | 0.898 | 0.959 |
| always nv | 1502 | 0.668 | 0.143 | 0.114 | 0.000 | 0.000 | 1.000 | n/a | n/a |

**Per-class recall (our test split)**

| Configuration | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| (a) | 0.442 | 0.775 | 0.503 | 0.750 | 0.814 | 0.826 | 0.619 |
| (b) | 0.558 | 0.859 | 0.587 | 0.800 | 0.569 | 0.940 | 0.762 |
| (c) | 0.788 | 0.803 | 0.683 | 0.600 | 0.725 | 0.815 | 0.762 |
| (d) | 0.750 | 0.803 | 0.593 | 0.600 | 0.814 | 0.728 | 0.667 |
| (e) | 0.808 | 0.817 | 0.641 | 0.700 | 0.719 | 0.819 | 0.714 |
| always nv | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |

**Per-class precision (our test split)**

| Configuration | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| (a) | 0.575 | 0.846 | 0.816 | 1.000 | 0.360 | 0.937 | 0.812 |
| (b) | 0.592 | 0.813 | 0.737 | 0.889 | 0.590 | 0.902 | 0.842 |
| (c) | 0.586 | 0.687 | 0.600 | 0.706 | 0.486 | 0.942 | 0.640 |
| (d) | 0.600 | 0.695 | 0.615 | 0.750 | 0.349 | 0.954 | 0.636 |
| (e) | 0.553 | 0.659 | 0.594 | 0.667 | 0.500 | 0.941 | 0.652 |
| always nv | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.668 | 0.000 |

**Per-class F1 (our test split)**

| Configuration | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| (a) | 0.500 | 0.809 | 0.622 | 0.857 | 0.499 | 0.878 | 0.703 |
| (b) | 0.574 | 0.836 | 0.653 | 0.842 | 0.579 | 0.921 | 0.800 |
| (c) | 0.672 | 0.740 | 0.639 | 0.649 | 0.582 | 0.874 | 0.696 |
| (d) | 0.667 | 0.745 | 0.604 | 0.667 | 0.488 | 0.826 | 0.651 |
| (e) | 0.656 | 0.730 | 0.617 | 0.683 | 0.590 | 0.875 | 0.682 |
| always nv | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.801 | 0.000 |

**Confusion matrix, (a) final model, our test split** (rows true, columns predicted)

| true \ pred | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| **akiec** | 23 | 1 | 7 | 0 | 19 | 2 | 0 |
| **bcc** | 5 | 55 | 0 | 0 | 7 | 3 | 1 |
| **bkl** | 5 | 5 | 84 | 0 | 51 | 22 | 0 |
| **df** | 3 | 0 | 0 | 15 | 2 | 0 | 0 |
| **mel** | 2 | 1 | 1 | 0 | 136 | 27 | 0 |
| **nv** | 2 | 3 | 11 | 0 | 157 | 829 | 2 |
| **vasc** | 0 | 0 | 0 | 0 | 6 | 2 | 13 |

## 3. Part C: official ISIC 2018 Task 3 test set

### 3.1 Downloads

| File | URL | sha256 | Bytes |
|---|---|---|---|
| ISIC2018_Task3_Test_Input.zip | https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Test_Input.zip | `aa90bd5308259bf5fdcc6048b09dd70416d527d1200e5148aef8ae94f8678b60` | 420586171 |
| ISIC2018_Task3_Test_GroundTruth.zip | https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Test_GroundTruth.zip | `461561de4d490b9f44bd6f2efd87046cd91ac7ee430ef8e97443c7c33b2912ce` | 12202 |
| ISIC2018_Task3_Test_GroundTruth.csv (Dataverse) | https://dataverse.harvard.edu/api/access/datafile/6924466?format=original | `40ec93c1a1264586cd1a3b636b2fbc2c46c7c4c17eb5c05115939c045ffa7896` | 108383 |

- **Source of the first two URLs:** the official ISIC challenge data page (https://challenge.isic-archive.com/data/, 2018 Task 3).
- **Source of the third file:** the HAM10000 Harvard Dataverse release (doi:10.7910/DVN/DBW86T, version 4, file id 6924466), described there as the test ground truth "in the same format as the HAM10000 metadata". It adds `lesion_id` and a `dataset` (source) column.
- **Storage:** everything is under `data/isic2018_test/`, which `.gitignore` covers through its `data/` rule.

### 3.2 Integrity checks

- **Counts:** 1512 images; 1512 ground-truth rows (1512 unique ids). Images without GT: 0; GT rows without an image: 0.
- **GT class counts:** akiec 43, bcc 93, bkl 217, df 44, mel 171, nv 909, vasc 35.
- **Image ids disjoint from HAM10000:** overlap with all 10,015 HAM10000 ids is 0. By split: train 0, val 0, test 0.
- **Dataverse file agrees with the official GT:** all 1512 ids are matched both ways ({'both': 1512, 'left_only': 0, 'right_only': 0}), and dx agrees on 1512 of them.
- **Lesion overlap cannot be verified.** The Dataverse file gives 1223 test `lesion_id`s, all with the prefix ['HAMTEST']. HAM10000 lesion ids use the `HAM_` prefix, so the literal overlap of 0 is by construction. Neither the challenge page nor the Dataverse release maps test lesions to training lesions. Whether a test lesion also appears (as another image) in our training split is therefore unknown.
- **Easter-egg image ISIC_0035068 (Abhishek et al., Sci Data 2025):** present in the official test set; GT nv. In the Dataverse file its `dataset`, age, sex and localization are empty. The Dataverse description lists its own test zip as 1511 images, against 1512 in the ISIC zip. All 1512 are used for the official comparison; the 1511 without it are a secondary line.
- **Internal/external partition:** the Dataverse `dataset` column labels 316 images `external`. That equals the 316 external images in Codella et al. 2019. The remaining 1196 have the HAM10000 sources (vidir_molemax, vidir_modern, rosendahl, vienna_dias: 1195) plus the easter egg with an empty source (1). Per-partition results below use this official column directly: external = `external`, internal = everything else. Full counts: {'vidir_molemax': 495, 'vidir_modern': 414, 'external': 316, 'rosendahl': 267, 'vienna_dias': 19, 'nan': 1}.

### 3.3 All 1512 images (official comparison)

n = 1512; class counts: akiec 43, bcc 93, bkl 217, df 44, mel 171, nv 909, vasc 35.

| Configuration | n | Acc | BACC | macro-F1 | MEL Recall | MEL Precision | MEL Specificity | MEL AUROC | macro AUROC |
|---|---|---|---|---|---|---|---|---|---|
| (a) DINOv2 FT, t = 0.003 (final) | 1512 | 0.752 | 0.673 | 0.696 | 0.842 | 0.362 | 0.811 | 0.911 | 0.959 |
| (b) DINOv2 FT, argmax | 1512 | 0.823 | 0.710 | 0.743 | 0.556 | 0.642 | 0.960 | 0.911 | 0.959 |
| (c) EfficientNet-B0, TTA+MC, argmax | 1512 | 0.782 | 0.733 | 0.705 | 0.626 | 0.455 | 0.905 | 0.901 | 0.959 |
| (d) EfficientNet-B0, TTA+MC, t = 0.185 | 1512 | 0.725 | 0.721 | 0.690 | 0.848 | 0.343 | 0.793 | 0.901 | 0.959 |
| (e) EfficientNet-B0, plain, argmax | 1512 | 0.772 | 0.727 | 0.701 | 0.614 | 0.423 | 0.893 | 0.887 | 0.953 |
| always nv | 1512 | 0.601 | 0.143 | 0.107 | 0.000 | 0.000 | 1.000 | n/a | n/a |

**Next to the published top-5 entries** (Codella et al. 2019, arXiv:1902.03368, Table 2; ranges over ranks 1–5 on the entire test set)

| | Acc | BACC | MEL AUROC |
|---|---|---|---|
| ISIC 2018 Task 3 top-5 entries | 0.827–0.896 | 0.845–0.885 | 0.945–0.959 |
| (a) DINOv2 FT, t = 0.003 (final) | 0.752 | 0.673 | 0.911 |
| (b) DINOv2 FT, argmax | 0.823 | 0.710 | 0.911 |
| (c) EfficientNet-B0, TTA+MC, argmax | 0.782 | 0.733 | 0.901 |
| (d) EfficientNet-B0, TTA+MC, t = 0.185 | 0.725 | 0.721 | 0.901 |
| (e) EfficientNet-B0, plain, argmax | 0.772 | 0.727 | 0.887 |

**Per-class recall (all 1512)**

| Configuration | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| (a) | 0.558 | 0.720 | 0.571 | 0.705 | 0.842 | 0.802 | 0.514 |
| (b) | 0.651 | 0.785 | 0.659 | 0.727 | 0.556 | 0.935 | 0.657 |
| (c) | 0.581 | 0.763 | 0.793 | 0.773 | 0.626 | 0.821 | 0.771 |
| (d) | 0.581 | 0.677 | 0.677 | 0.773 | 0.848 | 0.721 | 0.771 |
| (e) | 0.628 | 0.774 | 0.765 | 0.750 | 0.614 | 0.813 | 0.743 |
| always nv | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |

**Per-class precision (all 1512)**

| Configuration | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| (a) | 0.649 | 0.838 | 0.849 | 0.886 | 0.362 | 0.915 | 0.947 |
| (b) | 0.636 | 0.793 | 0.790 | 0.889 | 0.642 | 0.863 | 0.885 |
| (c) | 0.481 | 0.724 | 0.685 | 0.810 | 0.455 | 0.937 | 0.711 |
| (d) | 0.521 | 0.733 | 0.735 | 0.829 | 0.343 | 0.966 | 0.750 |
| (e) | 0.474 | 0.735 | 0.686 | 0.846 | 0.423 | 0.934 | 0.703 |

**Per-class F1 (all 1512)**

| Configuration | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| (a) | 0.600 | 0.775 | 0.683 | 0.785 | 0.506 | 0.855 | 0.667 |
| (b) | 0.644 | 0.789 | 0.719 | 0.800 | 0.596 | 0.898 | 0.754 |
| (c) | 0.526 | 0.743 | 0.735 | 0.791 | 0.527 | 0.875 | 0.740 |
| (d) | 0.549 | 0.704 | 0.705 | 0.800 | 0.488 | 0.825 | 0.761 |
| (e) | 0.540 | 0.754 | 0.723 | 0.795 | 0.501 | 0.869 | 0.722 |

**Confusion matrix, (a) final model, ISIC 2018 test (all 1512)**

| true \ pred | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| **akiec** | 24 | 2 | 1 | 1 | 13 | 2 | 0 |
| **bcc** | 2 | 67 | 4 | 0 | 15 | 5 | 0 |
| **bkl** | 6 | 6 | 124 | 0 | 51 | 30 | 0 |
| **df** | 1 | 0 | 1 | 31 | 7 | 4 | 0 |
| **mel** | 2 | 1 | 3 | 0 | 144 | 20 | 1 |
| **nv** | 2 | 4 | 13 | 2 | 159 | 729 | 0 |
| **vasc** | 0 | 0 | 0 | 1 | 9 | 7 | 18 |

**Share of images predicted mel** (how the fixed thresholds behave on this set)

| Config | Our test split | ISIC 2018 test |
|---|---|---|
| (a) | 0.252 | 0.263 |
| (b) | 0.107 | 0.098 |
| (c) | 0.166 | 0.155 |
| (d) | 0.260 | 0.280 |
| (e) | 0.160 | 0.164 |
| true mel prevalence | 0.111 | 0.113 |

### 3.4 Secondary: 1511 images without ISIC_0035068

n = 1511; class counts: akiec 43, bcc 93, bkl 217, df 44, mel 171, nv 908, vasc 35.

| Configuration | n | Acc | BACC | macro-F1 | MEL Recall | MEL Precision | MEL Specificity | MEL AUROC | macro AUROC |
|---|---|---|---|---|---|---|---|---|---|
| (a) DINOv2 FT, t = 0.003 (final) | 1511 | 0.752 | 0.673 | 0.696 | 0.842 | 0.362 | 0.810 | 0.911 | 0.960 |
| (b) DINOv2 FT, argmax | 1511 | 0.823 | 0.710 | 0.743 | 0.556 | 0.642 | 0.960 | 0.911 | 0.960 |
| (c) EfficientNet-B0, TTA+MC, argmax | 1511 | 0.782 | 0.733 | 0.705 | 0.626 | 0.455 | 0.904 | 0.901 | 0.959 |
| (d) EfficientNet-B0, TTA+MC, t = 0.185 | 1511 | 0.725 | 0.721 | 0.690 | 0.848 | 0.343 | 0.793 | 0.901 | 0.959 |
| (e) EfficientNet-B0, plain, argmax | 1511 | 0.772 | 0.727 | 0.701 | 0.614 | 0.423 | 0.893 | 0.887 | 0.953 |
| always nv | 1511 | 0.601 | 0.143 | 0.107 | 0.000 | 0.000 | 1.000 | n/a | n/a |

### 3.5 Internal partition (1196, sources shared with HAM10000)

n = 1196; class counts: akiec 40, bcc 60, bkl 138, df 13, mel 139, nv 789, vasc 17.

| Configuration | n | Acc | BACC | macro-F1 | MEL Recall | MEL Precision | MEL Specificity | MEL AUROC | macro AUROC |
|---|---|---|---|---|---|---|---|---|---|
| (a) DINOv2 FT, t = 0.003 (final) | 1196 | 0.763 | 0.682 | 0.690 | 0.827 | 0.373 | 0.817 | 0.914 | 0.962 |
| (b) DINOv2 FT, argmax | 1196 | 0.834 | 0.714 | 0.725 | 0.547 | 0.644 | 0.960 | 0.914 | 0.962 |
| (c) EfficientNet-B0, TTA+MC, argmax | 1196 | 0.788 | 0.737 | 0.675 | 0.604 | 0.454 | 0.904 | 0.900 | 0.961 |
| (d) EfficientNet-B0, TTA+MC, t = 0.185 | 1196 | 0.746 | 0.733 | 0.674 | 0.842 | 0.368 | 0.810 | 0.900 | 0.961 |
| (e) EfficientNet-B0, plain, argmax | 1196 | 0.779 | 0.741 | 0.685 | 0.597 | 0.423 | 0.893 | 0.888 | 0.955 |
| always nv | 1196 | 0.660 | 0.143 | 0.114 | 0.000 | 0.000 | 1.000 | n/a | n/a |

**Next to the published top-5 entries** (Codella et al. 2019, arXiv:1902.03368, Table 2; ranges over ranks 1–5 on the internal partition)

| | Acc | BACC | MEL AUROC |
|---|---|---|---|
| ISIC 2018 Task 3 top-5 entries | 0.820–0.907 | 0.841–0.875 | 0.939–0.961 |
| (a) DINOv2 FT, t = 0.003 (final) | 0.763 | 0.682 | 0.914 |
| (b) DINOv2 FT, argmax | 0.834 | 0.714 | 0.914 |
| (c) EfficientNet-B0, TTA+MC, argmax | 0.788 | 0.737 | 0.900 |
| (d) EfficientNet-B0, TTA+MC, t = 0.185 | 0.746 | 0.733 | 0.900 |
| (e) EfficientNet-B0, plain, argmax | 0.779 | 0.741 | 0.888 |

**Per-class recall (internal)**

| Configuration | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| (a) | 0.575 | 0.783 | 0.536 | 0.538 | 0.827 | 0.805 | 0.706 |
| (b) | 0.625 | 0.817 | 0.638 | 0.615 | 0.547 | 0.935 | 0.824 |
| (c) | 0.575 | 0.800 | 0.775 | 0.692 | 0.604 | 0.831 | 0.882 |
| (d) | 0.575 | 0.717 | 0.674 | 0.692 | 0.842 | 0.750 | 0.882 |
| (e) | 0.625 | 0.833 | 0.732 | 0.692 | 0.597 | 0.823 | 0.882 |

### 3.6 External partition (316, `dataset == external`)

n = 316; class counts: akiec 3, bcc 33, bkl 79, df 31, mel 32, nv 120, vasc 18.

| Configuration | n | Acc | BACC | macro-F1 | MEL Recall | MEL Precision | MEL Specificity | MEL AUROC | macro AUROC |
|---|---|---|---|---|---|---|---|---|---|
| (a) DINOv2 FT, t = 0.003 (final) | 316 | 0.709 | 0.624 | 0.645 | 0.906 | 0.322 | 0.785 | 0.900 | 0.961 |
| (b) DINOv2 FT, argmax | 316 | 0.778 | 0.746 | 0.772 | 0.594 | 0.633 | 0.961 | 0.900 | 0.961 |
| (c) EfficientNet-B0, TTA+MC, argmax | 316 | 0.759 | 0.733 | 0.704 | 0.719 | 0.460 | 0.905 | 0.917 | 0.959 |
| (d) EfficientNet-B0, TTA+MC, t = 0.185 | 316 | 0.646 | 0.690 | 0.649 | 0.875 | 0.267 | 0.729 | 0.917 | 0.959 |
| (e) EfficientNet-B0, plain, argmax | 316 | 0.747 | 0.711 | 0.687 | 0.688 | 0.423 | 0.894 | 0.899 | 0.954 |
| always nv | 316 | 0.380 | 0.143 | 0.079 | 0.000 | 0.000 | 1.000 | n/a | n/a |

**Next to the published top-5 entries** (Codella et al. 2019, arXiv:1902.03368, Table 2; ranges over ranks 1–5 on the external partition)

| | Acc | BACC | MEL AUROC |
|---|---|---|---|
| ISIC 2018 Task 3 top-5 entries | 0.848–0.886 | 0.864–0.925 | 0.966–0.984 |
| (a) DINOv2 FT, t = 0.003 (final) | 0.709 | 0.624 | 0.900 |
| (b) DINOv2 FT, argmax | 0.778 | 0.746 | 0.900 |
| (c) EfficientNet-B0, TTA+MC, argmax | 0.759 | 0.733 | 0.917 |
| (d) EfficientNet-B0, TTA+MC, t = 0.185 | 0.646 | 0.690 | 0.917 |
| (e) EfficientNet-B0, plain, argmax | 0.747 | 0.711 | 0.899 |

**Per-class recall (external)**

| Configuration | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| (a) | 0.333 | 0.606 | 0.633 | 0.774 | 0.906 | 0.783 | 0.333 |
| (b) | 1.000 | 0.727 | 0.696 | 0.774 | 0.594 | 0.933 | 0.500 |
| (c) | 0.667 | 0.697 | 0.823 | 0.806 | 0.719 | 0.750 | 0.667 |
| (d) | 0.667 | 0.606 | 0.684 | 0.806 | 0.875 | 0.525 | 0.667 |
| (e) | 0.667 | 0.667 | 0.823 | 0.774 | 0.688 | 0.750 | 0.611 |

## 4. Scripts and commands

Run from the project root:

```
python src/eval_sota_metrics.py --recompute      # Part B: inference on our test split + sanity gate
python src/eval_isic2018_test.py --recompute     # Part C: inference on the ISIC 2018 test set + integrity checks
python src/write_sota_report.py                  # this report
```

**Shared code:** `src/sota_eval_common.py` (model loading, inference, metrics).

**Outputs:**

| Path | Contents | Tracked |
|---|---|---|
| `results/sota_per_sample_ours.csv` | per-image probabilities of the three models plus the five predicted labels | yes |
| `results/sota_per_sample_isic2018.csv` | same columns, plus the Dataverse source column | yes |
| `outputs/sota_eval/*_probs.npz` | cached probabilities | no (git-ignored) |
| `outputs/sota_eval/*_metrics.json` | metrics read by this report | no (git-ignored) |

