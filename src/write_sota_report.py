"""
Writes results/sota_comparison_metrics.md from outputs/sota_eval/{ours,isic}_metrics.json
(produced by src/eval_sota_metrics.py and src/eval_isic2018_test.py).
Rows that failed the sanity gate on our split are left out of every table.

Run from the project root:  python src/write_sota_report.py
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np

from sota_eval_common import CLASSES, CONFIGS, ROOT, cm_table, f3, per_class_table, summary_table

OUT = ROOT / "outputs" / "sota_eval"
REPORT = ROOT / "results" / "sota_comparison_metrics.md"

URLS = {
    "ISIC2018_Task3_Test_Input.zip": ("https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Test_Input.zip",
                                      "aa90bd5308259bf5fdcc6048b09dd70416d527d1200e5148aef8ae94f8678b60", 420586171),
    "ISIC2018_Task3_Test_GroundTruth.zip": ("https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Test_GroundTruth.zip",
                                            "461561de4d490b9f44bd6f2efd87046cd91ac7ee430ef8e97443c7c33b2912ce", 12202),
    "ISIC2018_Task3_Test_GroundTruth.csv (Dataverse)": (
        "https://dataverse.harvard.edu/api/access/datafile/6924466?format=original",
        "40ec93c1a1264586cd1a3b636b2fbc2c46c7c4c17eb5c05115939c045ffa7896", 108383),
}


# Codella et al. 2019 (arXiv:1902.03368), Table 2, top-5 submissions to Part 3: (ACC, BACC, MEL AUC) ranges.
CODELLA = {
    "all_1512": ("0.827–0.896", "0.845–0.885", "0.945–0.959"),
    "internal_1196": ("0.820–0.907", "0.841–0.875", "0.939–0.961"),
    "external_316": ("0.848–0.886", "0.864–0.925", "0.966–0.984"),
}


def rows(block: dict, keys: list[str]) -> list[tuple[str, dict]]:
    r = [(f"({k}) {CONFIGS[k]}", block["metrics"][k]) for k in keys]
    r.append(("always nv", block["always_nv"]))
    return r


def main() -> None:
    ours = json.loads((OUT / "ours_metrics.json").read_text(encoding="utf-8"))
    isic = json.loads((OUT / "isic_metrics.json").read_text(encoding="utf-8"))
    keys = [k for k in "abcde" if ours["gate"][k]["pass"]]
    failed = [k for k in "abcde" if not ours["gate"][k]["pass"]]
    L = []
    A = L.append
    A("# State-of-the-art comparison: extra metrics and ISIC 2018 Task 3 test set")
    A("")
    A(f"Generated {date.today().isoformat()} by `src/write_sota_report.py`. Inference only: no training, no threshold "
      "re-tuning, no `.tex`, checkpoint or existing results file touched.")
    A("")
    A("**Configurations**")
    A("")
    A("| Key | Configuration | Probabilities |")
    A("|---|---|---|")
    A("| (a) | DINOv2 ViT-S/14 full fine-tune, rule t = 0.003 (final model) | DINOv2 softmax |")
    A("| (b) | DINOv2 ViT-S/14 full fine-tune, argmax | DINOv2 softmax |")
    A("| (c) | EfficientNet-B0, 5-view TTA + MC Dropout (10 passes/view, seed 42), argmax | mean of the 50 samples |")
    A("| (d) | EfficientNet-B0, same protocol, rule t = 0.185 | mean of the 50 samples |")
    A("| (e) | EfficientNet-B0, plain forward (eval mode), argmax | single softmax |")
    A("")
    A("- **Rule:** predict mel if p(mel) > t, otherwise the argmax of the other six classes "
      "(`src/dinov2_finetune_threshold_tune.py::apply_mel_threshold`). Both thresholds were chosen on our validation split.")
    A("- **Preprocessing (both test sets):** Resize((224, 224)), ToTensor, ImageNet normalisation "
      "(`src/dataset.py::build_transforms(224, is_train=False)`).")
    A("- **Devices:** DINOv2 runs on the GPU with autocast, batch 64. EfficientNet runs on the CPU, with TTA+MC via "
      "`app/tta_mc.py::tta_mc_predict`.")
    A("- **AUROC:** computed from the probabilities, so it does not depend on the threshold. (a) shares the AUROC of (b), "
      "and (d) shares the AUROC of (c).")
    A("- **Other definitions:**")
    A("  - BACC = mean per-class recall (`sklearn.metrics.balanced_accuracy_score`), the official ISIC 2018 ranking metric.")
    A("  - MEL specificity = TN/(TN+FP) for mel vs rest.")
    A("  - macro AUROC = one-vs-rest, macro average.")
    A("  - always nv: every image predicted nv. AUROC is undefined for a constant score.")
    A("")
    A("## 1. Sanity gate (our test split, 1502 images)")
    A("")
    A("| Config | Expected macro-F1 / MEL Recall | Reproduced (3 dp) | Unrounded | Result |")
    A("|---|---|---|---|---|")
    for k in "abcde":
        g = ours["gate"][k]
        A(f"| ({k}) | {g['expected'][0]:.3f} / {g['expected'][1]:.3f} | {g['got'][0]:.3f} / {g['got'][1]:.3f} | "
          f"{g['got_unrounded'][0]:.4f} / {g['got_unrounded'][1]:.4f} | {'PASS' if g['pass'] else '**FAIL**'} |")
    A("")
    A("All five rows pass; every table below uses (a)–(e)." if not failed else
      f"Rows {failed} failed and are excluded from every table below.")
    A("")
    A("## 2. Part B: our lesion-level test split")
    A("")
    cc = ours["class_counts"]
    A(f"n = {ours['n']}; class counts: " + ", ".join(f"{c} {cc[c]}" for c in CLASSES) + ".")
    A("")
    L += summary_table(rows(ours, keys))
    A("")
    for title, key in (("Per-class recall", "per_class_recall"), ("Per-class precision", "per_class_precision"),
                       ("Per-class F1", "per_class_f1")):
        A(f"**{title} (our test split)**")
        A("")
        L += per_class_table([(f"({k})", ours["metrics"][k]) for k in keys] + [("always nv", ours["always_nv"])], key)
        A("")
    if "a" in keys:
        A("**Confusion matrix, (a) final model, our test split** (rows true, columns predicted)")
        A("")
        L += cm_table(np.array(ours["metrics"]["a"]["cm"]))
        A("")
    A("## 3. Part C: official ISIC 2018 Task 3 test set")
    A("")
    A("### 3.1 Downloads")
    A("")
    A("| File | URL | sha256 | Bytes |")
    A("|---|---|---|---|")
    for name, (u, h, b) in URLS.items():
        A(f"| {name} | {u} | `{h}` | {b} |")
    A("")
    A("- **Source of the first two URLs:** the official ISIC challenge data page (https://challenge.isic-archive.com/data/, 2018 Task 3).")
    A("- **Source of the third file:** the HAM10000 Harvard Dataverse release (doi:10.7910/DVN/DBW86T, version 4, file id 6924466), "
      "described there as the test ground truth \"in the same format as the HAM10000 metadata\". It adds `lesion_id` and a "
      "`dataset` (source) column.")
    A("- **Storage:** everything is under `data/isic2018_test/`, which `.gitignore` covers through its `data/` rule.")
    A("")
    A("### 3.2 Integrity checks")
    A("")
    ic = isic["integrity"]
    A(f"- **Counts:** {ic['n_images']} images; {ic['n_gt_rows']} ground-truth rows ({ic['gt_ids_unique']} unique ids). "
      f"Images without GT: {ic['images_without_gt']}; GT rows without an image: {ic['gt_without_image']}.")
    A("- **GT class counts:** " + ", ".join(f"{c} {ic['class_counts_gt'][c]}" for c in CLASSES) + ".")
    A(f"- **Image ids disjoint from HAM10000:** overlap with all 10,015 HAM10000 ids is "
      f"{ic['overlap_image_ids_with_HAM10000_all']}. By split: train {ic['overlap_image_ids_with_train']}, "
      f"val {ic['overlap_image_ids_with_val']}, test {ic['overlap_image_ids_with_test']}.")
    A(f"- **Dataverse file agrees with the official GT:** all {ic['dataverse_rows']} ids are matched both ways "
      f"({ic['dataverse_vs_official_join']}), and dx agrees on {ic['dataverse_dx_agrees_with_official']} of them.")
    A(f"- **Lesion overlap cannot be verified.** The Dataverse file gives {ic['dataverse_lesion_ids']} test `lesion_id`s, all "
      f"with the prefix {ic['dataverse_lesion_id_prefixes']}. HAM10000 lesion ids use the `HAM_` prefix, so the literal overlap "
      f"of {ic['lesion_id_overlap_with_HAM10000']} is by construction. Neither the challenge page nor the Dataverse release "
      "maps test lesions to training lesions. Whether a test lesion also appears (as another image) in our training split is "
      "therefore unknown.")
    A("- **Easter-egg image ISIC_0035068 (Abhishek et al., Sci Data 2025):** present in the official test set; GT nv. In the "
      "Dataverse file its `dataset`, age, sex and localization are empty. The Dataverse description lists its own test zip as "
      "1511 images, against 1512 in the ISIC zip. All 1512 are used for the official comparison; the 1511 without it are a "
      "secondary line.")
    A("- **Internal/external partition:** the Dataverse `dataset` column labels "
      f"{ic['dataverse_dataset_counts'].get('external', 0)} images `external`. That equals the 316 external images in "
      "Codella et al. 2019. The remaining 1196 have the HAM10000 sources (vidir_molemax, vidir_modern, rosendahl, "
      "vienna_dias: 1195) plus the easter egg with an empty source (1). Per-partition results below use this official "
      "column directly: external = `external`, internal = everything else. Full counts: "
      f"{ic['dataverse_dataset_counts']}.")
    A("")
    for sub, title in (("all_1512", "3.3 All 1512 images (official comparison)"),
                       ("without_easter_egg_1511", "3.4 Secondary: 1511 images without ISIC_0035068"),
                       ("internal_1196", "3.5 Internal partition (1196, sources shared with HAM10000)"),
                       ("external_316", "3.6 External partition (316, `dataset == external`)")):
        blk = isic["subsets"][sub]
        A(f"### {title}")
        A("")
        A(f"n = {blk['n']}; class counts: " + ", ".join(f"{c} {blk['class_counts'][c]}" for c in CLASSES) + ".")
        A("")
        L += summary_table(rows(blk, keys))
        A("")
        if sub in CODELLA:
            acc_r, bacc_r, auc_r = CODELLA[sub]
            A("**Next to the published top-5 entries** (Codella et al. 2019, arXiv:1902.03368, Table 2; "
              f"ranges over ranks 1–5 on the {'entire test set' if sub == 'all_1512' else sub.split('_')[0] + ' partition'})")
            A("")
            A("| | Acc | BACC | MEL AUROC |")
            A("|---|---|---|---|")
            A(f"| ISIC 2018 Task 3 top-5 entries | {acc_r} | {bacc_r} | {auc_r} |")
            for k in keys:
                m = blk["metrics"][k]
                A(f"| ({k}) {CONFIGS[k]} | {f3(m['acc'])} | {f3(m['bacc'])} | {f3(m['mel_auroc'])} |")
            A("")
        if sub == "all_1512":
            A("**Per-class recall (all 1512)**")
            A("")
            L += per_class_table([(f"({k})", blk["metrics"][k]) for k in keys] + [("always nv", blk["always_nv"])],
                                 "per_class_recall")
            A("")
            A("**Per-class precision (all 1512)**")
            A("")
            L += per_class_table([(f"({k})", blk["metrics"][k]) for k in keys], "per_class_precision")
            A("")
            A("**Per-class F1 (all 1512)**")
            A("")
            L += per_class_table([(f"({k})", blk["metrics"][k]) for k in keys], "per_class_f1")
            A("")
            if "a" in keys:
                A("**Confusion matrix, (a) final model, ISIC 2018 test (all 1512)**")
                A("")
                L += cm_table(np.array(blk["metrics"]["a"]["cm"]))
                A("")
            A("**Share of images predicted mel** (how the fixed thresholds behave on this set)")
            A("")
            A("| Config | Our test split | ISIC 2018 test |")
            A("|---|---|---|")
            for k in keys:
                cm_o = np.array(ours["metrics"][k]["cm"]); cm_i = np.array(blk["metrics"][k]["cm"])
                A(f"| ({k}) | {cm_o[:, CLASSES.index('mel')].sum() / cm_o.sum():.3f} | "
                  f"{cm_i[:, CLASSES.index('mel')].sum() / cm_i.sum():.3f} |")
            A(f"| true mel prevalence | {ours['class_counts']['mel'] / ours['n']:.3f} | {blk['class_counts']['mel'] / blk['n']:.3f} |")
            A("")
        if sub in ("internal_1196", "external_316"):
            A(f"**Per-class recall ({sub.split('_')[0]})**")
            A("")
            L += per_class_table([(f"({k})", blk["metrics"][k]) for k in keys], "per_class_recall")
            A("")
    A("## 4. Scripts and commands")
    A("")
    A("Run from the project root:")
    A("")
    A("```")
    A("python src/eval_sota_metrics.py --recompute      # Part B: inference on our test split + sanity gate")
    A("python src/eval_isic2018_test.py --recompute     # Part C: inference on the ISIC 2018 test set + integrity checks")
    A("python src/write_sota_report.py                  # this report")
    A("```")
    A("")
    A("**Shared code:** `src/sota_eval_common.py` (model loading, inference, metrics).")
    A("")
    A("**Outputs:**")
    A("")
    A("| Path | Contents | Tracked |")
    A("|---|---|---|")
    A("| `results/sota_per_sample_ours.csv` | per-image probabilities of the three models plus the five predicted labels | yes |")
    A("| `results/sota_per_sample_isic2018.csv` | same columns, plus the Dataverse source column | yes |")
    A("| `outputs/sota_eval/*_probs.npz` | cached probabilities | no (git-ignored) |")
    A("| `outputs/sota_eval/*_metrics.json` | metrics read by this report | no (git-ignored) |")
    A("")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
