# Melanoma detection from dermoscopic images (HAM10000)

Code and results of my diploma thesis *Algorithms for Melanoma Recognition from Skin Images*
(Department of Electrical and Computer Engineering, Democritus University of Thrace, 2026;
supervisor: Assoc. Prof. Nikolaos Mitianoudis).

The task is 7-class classification of dermoscopic images (akiec, bcc, bkl, df, mel, nv, vasc).
The goal is a high melanoma recall (MEL Recall) without a drop in macro-F1.

## What is compared

1. **EfficientNet-B0**, ImageNet-pretrained, fully fine-tuned in two phases (head first, then the
   whole network), with test-time augmentation and Monte Carlo Dropout (5 views x 10 passes).
2. **Frozen backbones + linear classifier**: DINOv1, DINOv2, DINOv2 with registers, DINOv3,
   BiomedCLIP.
3. **BiomedCLIP zero-shot**: cosine similarity between image and class-description embeddings,
   no training.
4. **Ensembles of EfficientNet-B0 and frozen DINOv2**: feature concatenation, probability
   averaging, weighted averaging.
5. **DINOv2 ViT-S/14, fully fine-tuned** with layer-wise learning-rate decay, plus a melanoma
   decision threshold: predict melanoma if p(mel) > t, otherwise the most probable of the other
   six classes.

It also covers explainability (Grad-CAM, DINO attention maps, MC Dropout uncertainty) and an
evaluation on the official ISIC 2018 Task 3 test set.

All models use the same **lesion-level** 70/15/15 split (seed 42), so that images of the same
lesion never appear in both training and test: 6,981 / 1,532 / 1,502 images.

## Main results

Test set of the lesion-level split (1,502 images, 167 melanomas):

| Configuration | macro-F1 | MEL Recall |
|---|---|---|
| EfficientNet-B0 (TTA + MC Dropout) | 0.701 | 0.725 |
| Best frozen backbone (DINOv1 + linear classifier) | 0.554 | 0.629 |
| DINOv2 fine-tuned, argmax | 0.744 | 0.569 |
| **DINOv2 fine-tuned, t = 0.003 (final model)** | **0.695** | **0.814** |
| EfficientNet-B0 with its own threshold (t = 0.185) | 0.664 | 0.814 |

- Differences below 0.02 in macro-F1 and 0.10 in MEL Recall are within the run-to-run noise
  (estimated from three training seeds).
- Most of the gain in MEL Recall comes from the decision threshold, not from the backbone.
- At t = 0.003 the melanoma precision is 0.36: about two out of three melanoma flags are false
  alarms. The model is meant as a screening aid, not a diagnostic tool.
- On the official ISIC 2018 Task 3 test set (1,512 images) the final model reaches a balanced
  accuracy of 0.673. The best of our configurations (EfficientNet-B0, argmax) reaches 0.733.
  The top 5 challenge entries reached 0.845 to 0.885.

The numbers behind every table of the thesis are in `results/`.

## Repository layout

```
src/            training, feature extraction, evaluation and explainability scripts
results/        text reports and per-sample predictions behind the thesis tables
requirements.txt
```

Data, extracted features, checkpoints and generated figures are not included (see `.gitignore`).

## Setup

```bash
pip install -r requirements.txt
# needed by specific scripts:
pip install timm open_clip_torch matplotlib seaborn opencv-python scipy
```

Run every script from the repository root, e.g. `python src/train_efficientnet.py --help`.

## Data (not included)

The images are **not** part of this repository and have to be downloaded from their original
sources.

**HAM10000** (Tschandl et al., 2018): Harvard Dataverse, doi:[10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T).
Download `HAM10000_images_part_1.zip`, `HAM10000_images_part_2.zip` and `HAM10000_metadata`
(original CSV format), and arrange them as:

```
data/
  HAM10000_metadata.csv
  HAM10000_images_part_1/   ISIC_*.jpg
  HAM10000_images_part_2/   ISIC_*.jpg
```

**ISIC 2018 Task 3 test set** (1,512 images): from the [ISIC Challenge data page](https://challenge.isic-archive.com/data/),
download the Task 3 test images and test ground truth. The split into images from
HAM10000-compatible and external sources comes from the same HAM10000 Dataverse record. The
expected folder layout is described at the top of `src/eval_isic2018_test.py`.

**License of the data: non-commercial use only.** Both datasets are distributed under the
Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC 4.0). Check the
terms on the pages above before using them, and cite:

- P. Tschandl, C. Rosendahl, H. Kittler. The HAM10000 dataset, a large collection of multi-source
  dermatoscopic images of common pigmented skin lesions. *Scientific Data* 5, 180161 (2018).
- N. Codella et al. Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the
  International Skin Imaging Collaboration (ISIC). arXiv:1902.03368 (2019).

## Pretrained weights (not included)

The scripts download pretrained weights at run time. Each model has its own license, which is
**not** covered by the license of this repository:

| Model | Loaded from | License / terms |
|---|---|---|
| EfficientNet-B0 (ImageNet-1k) | `torchvision` | torchvision code: BSD-3-Clause. The weights are trained on ImageNet and may carry terms derived from it. |
| DINO ViT-S/16 | `torch.hub`, facebookresearch/dino | Apache 2.0 |
| DINOv2 ViT-S/14 (also with registers) | `torch.hub`, facebookresearch/dinov2 | Apache 2.0 |
| DINOv3 ViT-S/16 | `timm`, `vit_small_patch16_dinov3` | DINOv3 License (Meta) |
| BiomedCLIP | Hugging Face, `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | The model card limits use to research; deployed use is out of scope. |

## Interactive demo

A web demo (prediction, Grad-CAM, attention map and MC Dropout uncertainty for an uploaded image)
runs as a Hugging Face Space: <https://dimitrislil-skin-lesion-dashboard.hf.space>. Its code is
kept in a separate repository. A few evaluation scripts (`src/eval_sota_metrics.py`,
`src/eval_isic2018_test.py`, `src/write_sota_report.py`, `src/eval_dashboard_protocol.py`) reuse
the demo's inference code and expect that repository cloned as `app/` in the repository root.

## Disclaimer

This is research code. It is not a medical device and must not be used for diagnosis.

## License

Code: MIT License (see `LICENSE`). The datasets and the pretrained weights are not covered by this
license; see the two sections above.
