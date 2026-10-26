# Live dashboard outputs for the sample images

Recorded 2026-10-03 00:5x (local time, GTB) against the live Space
`https://dimitrislil-skin-lesion-dashboard.hf.space`, through its public API (`POST /predict`, one image per request). Script: `scratch/space_probe.py`; raw JSON with base64 images stripped: `scratch/space_probe.json`.

`GET /health` at the time of recording:

```
status ok | primary_model dinov2_vits14_finetune | device cpu | dino_mel_threshold 0.003
mc_protocol: TTA + MC-Dropout (thesis Section 6.3), 5 views x 10 passes = 50 samples, seed 42
```

## Per-image outputs

Image files are the ones shipped with the app (`app/static/samples/`). The `challenge_*` files are byte-identical copies of the HAM10000 originals.

Column meanings:

- **Primary**: the fine-tuned DINOv2 prediction after the t = 0.003 rule, with its own probability.
- **Top-3 primary probabilities**: the raw DINOv2 softmax.
- **r**: the Grad-CAM vs attention agreement on the 16×16 grid.
- **Second opinion**: the EfficientNet-B0 class with its mean probability ± std over the 50 TTA+MC samples. This ± is what the UI displays.
- **Mean var.**: the Section 6.3 per-image statistic.

| Image | Primary | Primary p | Top-3 primary probabilities | DINOv2 p(mel) | r (label) | Second opinion (EfficientNet) | Mean var. | EffNet p(mel) |
|---|---|---|---|---|---|---|---|---|
| sample_akiec.jpg | akiec | 100.0% | akiec 100.0%, bkl 0.0%, df 0.0% | 0.00% | 0.614 (High agreement) | akiec 100.0% ± 0.0% | 0.0000 | 0.000 |
| sample_bcc.jpg | bcc | 100.0% | bcc 100.0%, mel 0.0%, nv 0.0% | 0.00% | 0.582 (High agreement) | bcc 100.0% ± 0.0% | 0.0000 | 0.000 |
| sample_bkl.jpg | bkl | 100.0% | bkl 100.0%, nv 0.0%, vasc 0.0% | 0.00% | 0.675 (High agreement) | bkl 100.0% ± 0.0% | 0.0000 | 0.000 |
| sample_df.jpg | df | 100.0% | df 100.0%, mel 0.0%, bcc 0.0% | 0.00% | 0.546 (High agreement) | df 100.0% ± 0.0% | 0.0000 | 0.000 |
| sample_mel.jpg | mel | 99.6% | mel 99.6%, nv 0.4%, df 0.0% | 99.65% | 0.701 (High agreement) | mel 98.5% ± 3.6% | 0.0004 | 0.985 |
| sample_nv.jpg | nv | 100.0% | nv 100.0%, mel 0.0%, vasc 0.0% | 0.00% | 0.713 (High agreement) | nv 100.0% ± 0.0% | 0.0000 | 0.000 |
| sample_vasc.jpg | vasc | 100.0% | vasc 100.0%, df 0.0%, nv 0.0% | 0.00% | 0.725 (High agreement) | vasc 100.0% ± 0.0% | 0.0000 | 0.000 |
| challenge_ISIC_0032214.jpg | nv | 39.5% | nv 39.5%, bcc 34.1%, akiec 25.8% | 0.00% | −0.153 (Models focus on different regions) | bcc 32.4% ± 28.0% | 0.0284 | 0.126 |
| challenge_ISIC_0030443.jpg | **mel (threshold flag)** | 20.0% (= p(mel)) | bkl 47.6%, nv 32.5%, mel 20.0% | 19.96% | 0.846 (High agreement) | mel 89.5% ± 7.1% | 0.0014 | 0.895 |

The flag fires for ISIC_0030443 because the primary model's top class is bkl but p(mel) = 0.1996 > 0.003. It does not fire for any of the seven class samples, whose top class is the predicted class. It also does not fire for ISIC_0032214, where p(mel) is about 0.

## Response time

The figures below are round-trip wall clock from this machine: upload of `sample_mel.jpg` (≈ the size of the shipped sample), server inference, and the JSON response, which includes three base64 PNG images. One warm-up request preceded the ten timed ones.

| | Value |
|---|---|
| Warm-up request | 4.74 s |
| 10 timed requests: median | **4.46 s** |
| 10 timed requests: range | 3.88 – 4.91 s |
| Individual timings (s) | 4.58, 4.58, 4.13, 3.88, 4.89, 4.82, 4.34, 4.03, 4.28, 4.91 |

The nine sample requests that followed took 3.83–5.40 s each.

## Comparison with the thesis text and the local evaluation

| Quantity | Thesis text | Live Space (this file) | Local CPU run of the same code (`dashboard_protocol_eval.md`, `dashboard_protocol_per_sample.csv`) |
|---|---|---|---|
| Response time | Ch7:162 and 279: «περίπου 4 δευτερόλεπτα» | median 4.46 s (3.88–4.91) | median 1.76 s (`_predict()` only, no network) |
| ISIC_0032214 primary | Ch7:196: nv 39.5%, probability spread over three classes | nv 39.5% (bcc 34.1%, akiec 25.8%) | nv |
| ISIC_0032214 agreement | Ch7:133: −0.153 | −0.153 | −0.153 |
| ISIC_0032214 second opinion | Ch7:198–199: bcc, high uncertainty | bcc 32.4% ± 28.0%, mean var. 0.0284 | bcc 30.9% ± 27.1%, mean var. 0.0245 |
| ISIC_0030443 primary | Ch7:208–209: mel 20.0%, bkl 47.6% | p(mel) 20.0%, bkl 47.6% | mel (flag) |
| ISIC_0030443 second opinion | n/a | mel 89.5% ± 7.1%, mean var. 0.0014 | mel 89.1% ± 8.9%, mean var. 0.0021 |
| sample_mel second opinion | n/a | mel 98.5% ± 3.6% | mel 98.8% ± 2.5% |
| Agreement r, six class samples | n/a | 0.546–0.725 | identical values (0.546, 0.582, 0.614, 0.675, 0.701, 0.725) for the six Grad-CAM example images of the same classes |

**Notes:**

- **DINOv2 outputs are identical** to the local run. They differ slightly from the thesis-pipeline CSV `dinov2_finetune_probs_test.csv` (ISIC_0030443: 19.64% / 48.23%; ISIC_0032214: nv 39.32%), because the app and the training pipeline load and resize the image through different code paths. The predicted labels are the same.
- **The EfficientNet second opinion differs slightly** between the Space and the local CPU run, even though both use seed 42 on CPU (e.g. ISIC_0032214 std 28.0% vs 27.1%). Different hardware or library builds produce different dropout streams for the same seed. This is consistent with Ch7:282–284, which says reproducibility holds within one installation.
- **The thesis range «0.55 έως 0.76»** (Ch7:134) refers to the seven Grad-CAM representative images listed in `dashboard_protocol_eval.md` §(e) (0.546–0.762). It is not about the seven dashboard sample images: their r range on the Space is 0.546–0.725, and `sample_nv.jpg` is a different image from the Grad-CAM nv example ISIC_0027169 (r = 0.762).
