"""
Re-select the final DINOv2 MEL-threshold with a split-consistent constraint.

The previous constrained selection (results_audit.md, "Task 1b") used
`val macro-F1 >= 0.701` as the constraint -- but 0.701 is EfficientNet's
TEST macro-F1 (TTA+MC), applied to a VALIDATION sweep. That's a split
mismatch: it lets test-set information influence which threshold gets
picked for a different model. The correct anchor is EfficientNet's own
VALIDATION macro-F1, read directly from its checkpoint.

Inference only -- no retraining. Reuses the exact model-reconstruction and
threshold-application code from dinov2_finetune_threshold_tune.py (imported,
not reimplemented) so nothing about the decision rule or split can drift.

Writes results/dinov2_finetune_threshold_final.md. Does not modify any
existing results file.

Run from project root:
  python src/dinov2_threshold_final_selection.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

_ROOT = Path(__file__).resolve().parent.parent
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import Config
from helpers import get_device, set_seed
from train_dinov2_finetune import DINOv2Classifier, load_dinov2_backbone, make_loaders
from dinov2_finetune_threshold_tune import apply_mel_threshold, predict_probs

RESULTS_DIR = _ROOT / "results"
OUT_PATH = RESULTS_DIR / "dinov2_finetune_threshold_final.md"

EFFNET_CKPT = _ROOT / "outputs" / "efficientnet_mc" / "best_efficientnet.pth"
DINOV2_CKPT = _ROOT / "outputs" / "dinov2_finetune" / "best_dinov2_finetune.pth"
EXPECTED_ANCHOR_APPROX = 0.6844

PREVIOUS_T_STAR = 0.014023
PREVIOUS_VAL_F1, PREVIOUS_VAL_MR = 0.7012, 0.7861
PREVIOUS_TEST_F1, PREVIOUS_TEST_MR = 0.7259, 0.7545

ROUNDED_T = 0.014

NL = chr(10)


# ---------------------------------------------------------------------------
# Master results list for the Pareto front (Step 5) -- every named, adopted
# result on record in this repo (Final Results.txt sections 1-7, the
# audit's two named threshold picks, and the loss/probe-ablation runs).
# Deliberately excludes raw diagnostic-curve points (e.g. t=0, t=0.0001)
# that results_audit.md itself framed as illustrations of non-convergence,
# never as a proposed answer.
# ---------------------------------------------------------------------------
ALL_RESULTS = [
    # EfficientNet-B0 (7-class)
    ("EfficientNet-B0, baseline (plain)", 0.690, 0.719),
    ("EfficientNet-B0, improved (TTA+MC)", 0.701, 0.725),
    # DINO/BiomedCLIP frozen linear probes (Part A best-loss numbers used)
    ("DINOv1 probe, focal", 0.526, 0.623),
    ("DINOv1 probe, dice", 0.436, 0.814),
    ("DINOv1 probe, combined", 0.554, 0.629),
    ("DINOv2 probe, combined", 0.5446, 0.6886),
    ("DINOv2 probe, focal", 0.5021, 0.5569),
    ("DINOv2 probe, dice", 0.4067, 0.9222),
    ("DINOv2-reg probe, combined", 0.4626, 0.6168),
    ("DINOv2-reg probe, focal", 0.4133, 0.6527),
    ("DINOv2-reg probe, dice", 0.3544, 0.6108),
    ("DINOv3 probe, combined", 0.4831, 0.6228),  # CLS-token features; was mean-pooled 0.3797 / 0.4970
    ("DINOv3 probe, focal", 0.4446, 0.5868),  # CLS-token features; was mean-pooled 0.3521 / 0.4850
    ("DINOv3 probe, dice", 0.3939, 0.6527),  # CLS-token features; was mean-pooled 0.3276 / 0.5749
    ("BiomedCLIP probe, focal", 0.420, 0.509),
    ("BiomedCLIP probe, combined", 0.4460, 0.5389),
    ("BiomedCLIP probe, dice", 0.3640, 0.6766),
    # BiomedCLIP zero-shot
    ("BiomedCLIP zero-shot P1 Averaged", 0.201, 0.401),
    ("BiomedCLIP zero-shot P1 Max-Sim", 0.157, 0.443),
    ("BiomedCLIP zero-shot P2 Averaged", 0.151, 0.108),
    ("BiomedCLIP zero-shot P2 Max-Sim", 0.142, 0.012),
    ("BiomedCLIP zero-shot P3 Averaged", 0.193, 0.479),
    ("BiomedCLIP zero-shot P3 Max-Sim", 0.222, 0.048),
    # DINOv2 full fine-tune
    ("DINOv2 FT, best-val-F1 ckpt, argmax", 0.744, 0.569),
    ("DINOv2 FT, composite ckpt, argmax", 0.719, 0.611),
    ("DINOv2 FT, threshold t=0.10", 0.732, 0.641),
    ("DINOv2 FT, threshold t=0.02", 0.727, 0.743),
    ("DINOv2 FT, threshold t=0.001", 0.670, 0.856),
    ("DINOv2 FT, composite-optimal t=0.000259 (audit)", 0.6534, 0.9042),
    ("DINOv2 FT, constrained-optimal t=0.014023 (SUPERSEDED by this report)", 0.7259, 0.7545),
    # Ensembles
    ("Ensemble A, Feature Concat", 0.724, 0.575),
    ("Ensemble A v2, MEL-boosted loss", 0.725, 0.629),
    ("Ensemble A v2, Composite criterion", 0.738, 0.683),
    ("Ensemble A v2, threshold t=0.10", 0.688, 0.754),
    ("Ensemble B, Soft", 0.696, 0.731),
    ("Ensemble C, Weighted best-F1", 0.696, 0.719),
    ("Ensemble C, Weighted best-MEL", 0.566, 0.701),
    # EfficientNet loss ablation
    ("EfficientNet, focal, plain", 0.691, 0.683),
    ("EfficientNet, focal, TTA+MC", 0.688, 0.743),
    ("EfficientNet, dice, plain", 0.703, 0.725),
    ("EfficientNet, dice, TTA+MC", 0.701, 0.719),
    ("EfficientNet, combined, plain", 0.687, 0.713),
    ("EfficientNet, combined, TTA+MC", 0.697, 0.743),
]


def pareto_front(points: list) -> list:
    """points: list of (name, f1, mr). Returns the non-dominated subset."""
    front = []
    for name, f1, mr in points:
        dominated = False
        for name2, f2, mr2 in points:
            if name2 == name:
                continue
            if f2 >= f1 and mr2 >= mr and (f2 > f1 or mr2 > mr):
                dominated = True
                break
        if not dominated:
            front.append((name, f1, mr))
    return front


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # --- Step 1: read the anchor -------------------------------------------
    print("=" * 64)
    print("STEP 1 -- reading EfficientNet's stored val_macro_f1 (the anchor)")
    print("=" * 64)
    effnet_ckpt = torch.load(EFFNET_CKPT, map_location="cpu", weights_only=False)
    anchor = effnet_ckpt["val_macro_f1"]
    print("  outputs/efficientnet_mc/best_efficientnet.pth  val_macro_f1 = "
          + format(anchor, ".10f"))
    if abs(anchor - EXPECTED_ANCHOR_APPROX) > 0.01:
        raise SystemExit(
            "STOPPING: anchor " + format(anchor, ".6f") + " differs from the expected "
            "~" + str(EXPECTED_ANCHOR_APPROX) + " by more than 0.01 -- reporting before "
            "proceeding, as instructed."
        )
    print("  Matches the expected ~" + str(EXPECTED_ANCHOR_APPROX) + ". Proceeding.")

    # --- Load DINOv2 model, reconstruct val/test probabilities --------------
    print(NL + "=" * 64)
    print("Loading DINOv2 checkpoint and scoring val/test (inference only)")
    print("=" * 64)
    cfg = Config()
    cfg.metadata_csv = _ROOT / "data" / "HAM10000_metadata.csv"
    cfg.images_dir = _ROOT / "data"
    set_seed(cfg.seed)
    device = get_device()

    ckpt = torch.load(DINOV2_CKPT, map_location=device, weights_only=False)
    label_map: dict = ckpt["label_map"]
    num_classes = len(label_map)
    feat_dim = ckpt.get("feat_dim", 384)
    image_size = ckpt.get("image_size", 224)
    idx_to_class = {v: k for k, v in label_map.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]
    mel_idx = label_map["mel"]
    print("  DINOv2 checkpoint epoch=" + str(ckpt.get("epoch")) + "  val_macro_f1="
          + format(ckpt.get("val_macro_f1", float("nan")), ".6f"))

    model = DINOv2Classifier(load_dinov2_backbone(), num_classes, feat_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    _, val_loader, test_loader, lm2, _ = make_loaders(cfg, image_size, 64, 4)
    assert lm2 == label_map, "Label map from split disagrees with checkpoint."

    use_amp = device.type == "cuda"
    y_true_val, probs_val = predict_probs(model, val_loader, device, use_amp)
    y_true_test, probs_test = predict_probs(model, test_loader, device, use_amp)
    print("  val N=" + str(len(y_true_val)) + "  test N=" + str(len(y_true_test)))

    # Sanity: unthresholded argmax should reproduce the known 0.744/0.569.
    argmax_test = probs_test.argmax(axis=1)
    argmax_f1 = f1_score(y_true_test, argmax_test, average="macro", zero_division=0)
    argmax_mr = recall_score(y_true_test, argmax_test, labels=[mel_idx], average="micro", zero_division=0)
    print("  sanity check, plain argmax on test: macro_f1=" + "{:.4f}".format(argmax_f1)
          + " mel_recall=" + "{:.4f}".format(argmax_mr) + "  (expect ~0.744/0.569)")

    # --- Step 2/3: exhaustive sweep over every distinct observed val p_mel --
    print(NL + "=" * 64)
    print("STEP 2/3 -- exhaustive constrained sweep on VALIDATION")
    print("=" * 64)
    eps = 1e-9
    candidate_ts = sorted(set(probs_val[:, mel_idx].tolist()))
    candidate_ts = [max(0.0, t - eps) for t in candidate_ts]

    sweep = []  # (t, val_f1, val_mr)
    for t in candidate_ts:
        yv = apply_mel_threshold(probs_val, mel_idx, t)
        vf1 = f1_score(y_true_val, yv, average="macro", zero_division=0)
        vmr = recall_score(y_true_val, yv, labels=[mel_idx], average="micro", zero_division=0)
        sweep.append((t, vf1, vmr))

    feasible = [(t, f1, mr) for (t, f1, mr) in sweep if f1 >= anchor]
    print("  candidates swept: " + str(len(sweep)) + "   feasible (val_f1 >= anchor): " + str(len(feasible)))

    if not feasible:
        raise SystemExit("No feasible threshold satisfies val_macro_f1 >= " + format(anchor, ".6f"))

    max_recall = max(mr for (_, _, mr) in feasible)
    tol = 1e-9
    tied = [(t, f1, mr) for (t, f1, mr) in feasible if abs(mr - max_recall) <= tol]
    # Tie-break: among max-recall ties, prefer the highest val macro-F1;
    # further ties broken by the largest threshold (more conservative /
    # less likely to be an infinitesimal single-sample cutoff).
    tied.sort(key=lambda row: (-row[1], -row[0]))
    t_star, val_f1_star, val_mr_star = tied[0]

    feasible_ts = sorted(t for (t, _, _) in feasible)
    # Report contiguity: is the feasible region one interval in threshold-space,
    # or fragmented? Check against the full sorted candidate list.
    all_ts_sorted = sorted(t for (t, _, _) in sweep)
    feasible_set = set(round(t, 12) for t in feasible_ts)
    idx_positions = [i for i, t in enumerate(all_ts_sorted) if round(t, 12) in feasible_set]
    contiguous = (idx_positions == list(range(min(idx_positions), max(idx_positions) + 1))) if idx_positions else False

    print("  feasible threshold range: [" + "{:.6f}".format(min(feasible_ts))
          + ", " + "{:.6f}".format(max(feasible_ts)) + "]  contiguous=" + str(contiguous))
    print("  max feasible val MEL recall: " + "{:.6f}".format(max_recall)
          + "  achieved at " + str(len(tied)) + " tied threshold(s)")
    print("  selected t* = " + "{:.6f}".format(t_star) + "  (val_f1=" + "{:.6f}".format(val_f1_star)
          + ", val_mr=" + "{:.6f}".format(val_mr_star) + ")")

    binding_margin = val_f1_star - anchor
    print("  constraint binding margin at t*: " + "{:+.6f}".format(binding_margin))

    differs_from_previous = abs(t_star - PREVIOUS_T_STAR) > 1e-6
    print("  t* differs from previous (0.014023): " + str(differs_from_previous))

    higher_recall_than_previous = max_recall > PREVIOUS_VAL_MR + 1e-9
    print("  relaxation admits higher val recall than previous selection's "
          + "{:.4f}".format(PREVIOUS_VAL_MR) + ": " + str(higher_recall_than_previous))

    # --- Test metrics at t* --------------------------------------------------
    y_pred_test_star = apply_mel_threshold(probs_test, mel_idx, t_star)
    test_f1_star = f1_score(y_true_test, y_pred_test_star, average="macro", zero_division=0)
    test_mr_star = recall_score(y_true_test, y_pred_test_star, labels=[mel_idx], average="micro", zero_division=0)
    test_prec_star = precision_score(y_true_test, y_pred_test_star, labels=[mel_idx], average="micro", zero_division=0)

    # --- Step 4: rounded threshold check --------------------------------------
    print(NL + "=" * 64)
    print("STEP 4 -- rounded threshold t=" + str(ROUNDED_T))
    print("=" * 64)
    y_pred_test_rounded = apply_mel_threshold(probs_test, mel_idx, ROUNDED_T)
    test_f1_rounded = f1_score(y_true_test, y_pred_test_rounded, average="macro", zero_division=0)
    test_mr_rounded = recall_score(y_true_test, y_pred_test_rounded, labels=[mel_idx], average="micro", zero_division=0)
    test_prec_rounded = precision_score(y_true_test, y_pred_test_rounded, labels=[mel_idx], average="micro", zero_division=0)
    yv_rounded = apply_mel_threshold(probs_val, mel_idx, ROUNDED_T)
    val_f1_rounded = f1_score(y_true_val, yv_rounded, average="macro", zero_division=0)
    val_mr_rounded = recall_score(y_true_val, yv_rounded, labels=[mel_idx], average="micro", zero_division=0)
    print("  t=" + str(ROUNDED_T) + "  val: f1=" + "{:.4f}".format(val_f1_rounded)
          + " mr=" + "{:.4f}".format(val_mr_rounded) + "  test: f1=" + "{:.4f}".format(test_f1_rounded)
          + " mr=" + "{:.4f}".format(test_mr_rounded))
    near_identical = (abs(test_f1_rounded - test_f1_star) < 0.002
                      and abs(test_mr_rounded - test_mr_star) < 0.002)
    print("  near-identical test metrics to t*: " + str(near_identical))

    # t=0.014 is the natural rounding of the OLD (superseded) 0.014023, not
    # of the NEW t*. Also check the natural 3-decimal rounding of t* itself
    # (0.002804 -> 0.003), since that's what "does rounding cost anything"
    # actually means for the value this report is proposing.
    t_star_rounded = round(t_star, 3)
    y_pred_test_t_star_rounded = apply_mel_threshold(probs_test, mel_idx, t_star_rounded)
    test_f1_tsr = f1_score(y_true_test, y_pred_test_t_star_rounded, average="macro", zero_division=0)
    test_mr_tsr = recall_score(y_true_test, y_pred_test_t_star_rounded, labels=[mel_idx], average="micro", zero_division=0)
    test_prec_tsr = precision_score(y_true_test, y_pred_test_t_star_rounded, labels=[mel_idx], average="micro", zero_division=0)
    yv_tsr = apply_mel_threshold(probs_val, mel_idx, t_star_rounded)
    val_f1_tsr = f1_score(y_true_val, yv_tsr, average="macro", zero_division=0)
    val_mr_tsr = recall_score(y_true_val, yv_tsr, labels=[mel_idx], average="micro", zero_division=0)
    near_identical_tsr = (abs(test_f1_tsr - test_f1_star) < 0.002 and abs(test_mr_tsr - test_mr_star) < 0.002)
    print("  [also] natural 3-decimal rounding of t* itself: t=" + "{:.3f}".format(t_star_rounded)
          + "  val: f1=" + "{:.4f}".format(val_f1_tsr) + " mr=" + "{:.4f}".format(val_mr_tsr)
          + "  test: f1=" + "{:.4f}".format(test_f1_tsr) + " mr=" + "{:.4f}".format(test_mr_tsr)
          + "  near-identical to t*: " + str(near_identical_tsr))

    # --- Step 5: Pareto front -------------------------------------------------
    print(NL + "=" * 64)
    print("STEP 5 -- Pareto front across all named results")
    print("=" * 64)
    results_list = list(ALL_RESULTS)
    # Replace the superseded entry with the new final selection.
    results_list = [r for r in results_list if "SUPERSEDED" not in r[0]]
    final_name = "DINOv2 FT, constrained-optimal t=" + "{:.6f}".format(t_star) + " (this report, anchor=val EffNet F1)"
    results_list.append((final_name, round(test_f1_star, 4), round(test_mr_star, 4)))
    front = pareto_front(results_list)
    front.sort(key=lambda r: r[1], reverse=True)
    print("  " + str(len(front)) + " / " + str(len(results_list)) + " results are non-dominated")
    for name, f1, mr in front:
        print("    " + "{:.4f}".format(f1) + " / " + "{:.4f}".format(mr) + "  " + name)

    # --- Write report ----------------------------------------------------------
    L = []
    L.append("# DINOv2 Final MEL-Threshold Selection (split-consistent constraint)")
    L.append("")
    L.append("Generated: " + datetime.now().isoformat(timespec="seconds"))
    L.append("")
    L.append("**Why this supersedes the earlier `t=0.014023` selection:** that pick used "
             "`val macro-F1 >= 0.701` as its constraint, where 0.701 is EfficientNet's "
             "**test** macro-F1 (TTA+MC) -- a split mismatch, letting test-set information "
             "influence a validation-side selection. The correct anchor is EfficientNet's "
             "own **validation** macro-F1, read directly from its checkpoint.")
    L.append("")

    L.append("## Step 1 -- Anchor")
    L.append("")
    L.append("`outputs/efficientnet_mc/best_efficientnet.pth`, key `val_macro_f1` = **"
             + format(anchor, ".10f") + "** (read directly, not hard-coded).")
    L.append("")

    L.append("## Step 2 -- Constrained selection")
    L.append("")
    L.append("Rule: maximise validation MEL Recall subject to validation macro-F1 >= "
             + format(anchor, ".6f") + ". Exhaustive sweep over every distinct observed "
             "validation p_mel value (" + str(len(sweep)) + " candidates), not a fixed-step grid.")
    L.append("")
    L.append("| | Threshold | Val Macro-F1 | Val MEL Recall | Test Macro-F1 | Test MEL Recall |")
    L.append("|---|---|---|---|---|---|")
    L.append("| Previous (superseded) | " + "{:.6f}".format(PREVIOUS_T_STAR) + " | "
             + "{:.4f}".format(PREVIOUS_VAL_F1) + " | " + "{:.4f}".format(PREVIOUS_VAL_MR)
             + " | " + "{:.4f}".format(PREVIOUS_TEST_F1) + " | " + "{:.4f}".format(PREVIOUS_TEST_MR) + " |")
    L.append("| **New (this report)** | **" + "{:.6f}".format(t_star) + "** | "
             + "{:.4f}".format(val_f1_star) + " | " + "{:.4f}".format(val_mr_star)
             + " | " + "{:.4f}".format(test_f1_star) + " | " + "{:.4f}".format(test_mr_star) + " |")
    L.append("")
    L.append("**t* differs from the previous 0.014023:** " + str(differs_from_previous) + ".")
    L.append("")
    L.append("**Is the constraint binding at t*?** val macro-F1 (" + "{:.4f}".format(val_f1_star)
             + ") exceeds the anchor (" + "{:.4f}".format(anchor) + ") by "
             + "{:+.4f}".format(binding_margin) + " -- "
             + ("binding, sitting essentially at the floor." if binding_margin < 0.005
                else "comfortably above the floor, not binding.") )
    L.append("")

    L.append("## Step 3 -- Feasible region")
    L.append("")
    L.append("Feasible thresholds (val macro-F1 >= " + "{:.4f}".format(anchor) + "): "
             + str(len(feasible)) + " of " + str(len(sweep)) + " candidates, spanning t in ["
             + "{:.6f}".format(min(feasible_ts)) + ", " + "{:.6f}".format(max(feasible_ts))
             + "]" + (" (one contiguous interval)." if contiguous else " (NOT contiguous -- fragmented feasible region)."))
    L.append("")
    L.append("Maximum validation MEL Recall achievable anywhere in the feasible region: **"
             + "{:.6f}".format(max_recall) + "**, achieved at " + str(len(tied))
             + " tied threshold(s) (selected t* is the highest-val-F1 tie, then largest threshold).")
    L.append("")
    if higher_recall_than_previous:
        L.append("**The relaxation ADMITS higher recall than the previous selection's "
                 + "{:.4f}".format(PREVIOUS_VAL_MR) + "** -- the looser constraint changes "
                 "what's achievable, not just which point gets picked.")
    else:
        L.append("**The relaxation does NOT admit higher recall than the previous "
                 "selection's " + "{:.4f}".format(PREVIOUS_VAL_MR) + "** -- the same "
                 "recall ceiling holds even with the wider feasible region; only the "
                 "selected operating point changes (if it changes at all).")
    L.append("")

    L.append("## Step 4 -- Rounded threshold check")
    L.append("")
    L.append("| Threshold | Val Macro-F1 | Val MEL Recall | Test Macro-F1 | Test MEL Recall |")
    L.append("|---|---|---|---|---|")
    L.append("| t* = " + "{:.6f}".format(t_star) + " | " + "{:.4f}".format(val_f1_star) + " | "
             + "{:.4f}".format(val_mr_star) + " | " + "{:.4f}".format(test_f1_star) + " | "
             + "{:.4f}".format(test_mr_star) + " |")
    L.append("| t = " + "{:.3f}".format(ROUNDED_T) + " (rounded, but this is the OLD "
             "threshold's rounding, not t*'s) | " + "{:.4f}".format(val_f1_rounded)
             + " | " + "{:.4f}".format(val_mr_rounded) + " | " + "{:.4f}".format(test_f1_rounded)
             + " | " + "{:.4f}".format(test_mr_rounded) + " |")
    L.append("| t = " + "{:.3f}".format(t_star_rounded) + " (natural 3-decimal rounding of t* itself) | "
             + "{:.4f}".format(val_f1_tsr) + " | " + "{:.4f}".format(val_mr_tsr) + " | "
             + "{:.4f}".format(test_f1_tsr) + " | " + "{:.4f}".format(test_mr_tsr) + " |")
    L.append("")
    # "Free" rounding means recall doesn't drop and F1 doesn't drop meaningfully
    # -- NOT a symmetric small-magnitude-difference test, which would wrongly
    # flag a favorable move (recall flat, F1 slightly UP) as "meaningful".
    rounding_is_free = (test_mr_tsr >= test_mr_star - 1e-9) and (test_f1_tsr >= test_f1_star - 0.002)
    L.append("**Note:** t=" + "{:.3f}".format(ROUNDED_T) + " is the natural rounding of the "
             "OLD, superseded 0.014023 -- included because it was explicitly requested, but "
             "it is not a rounding of this report's t*. Its test metrics being identical to "
             "the old 0.014023 row above just confirms no validation or test sample's "
             "probability falls between 0.014 and 0.014023. The more relevant rounding "
             "check for the NEW t* is the t=" + "{:.3f}".format(t_star_rounded) + " row: "
             + ("rounding costs nothing here -- test MEL Recall is unchanged ("
                + "{:.4f}".format(test_mr_tsr) + " vs " + "{:.4f}".format(test_mr_star)
                + ") and test Macro-F1 is not lower (" + "{:.4f}".format(test_f1_tsr) + " vs "
                + "{:.4f}".format(test_f1_star) + "). Report t=" + "{:.3f}".format(t_star_rounded)
                + ", not the six-decimal " + "{:.6f}".format(t_star) + " -- no accuracy cost, "
                "looks far less overfit to validation."
                if rounding_is_free else
                "rounding costs real accuracy here -- MEL Recall drops from "
                + "{:.4f}".format(test_mr_star) + " to " + "{:.4f}".format(test_mr_tsr)
                + " and/or Macro-F1 drops by more than 0.002. Keep the six-decimal t*."))
    L.append("")

    L.append("## Melanoma precision at the selected threshold")
    L.append("")
    L.append("At t* = " + "{:.6f}".format(t_star) + " (test set): **melanoma precision = "
             + "{:.4f}".format(test_prec_star) + "**, recall = " + "{:.4f}".format(test_mr_star) + ".")
    L.append("At the rounded t=" + "{:.3f}".format(ROUNDED_T) + ": melanoma precision = "
             + "{:.4f}".format(test_prec_rounded) + ", recall = " + "{:.4f}".format(test_mr_rounded) + ".")
    L.append("At t=" + "{:.3f}".format(t_star_rounded) + " (recommended -- natural rounding of "
             "t* itself, no cost): melanoma precision = " + "{:.4f}".format(test_prec_tsr)
             + ", recall = " + "{:.4f}".format(test_mr_tsr) + ".")
    L.append("")

    L.append("## Step 5 -- Updated Pareto front")
    L.append("")
    L.append("Across all " + str(len(results_list)) + " named results on record (every "
             "experiment in `Final Results.txt`, the loss-ablation and probe-ablation runs, "
             "and this report's new threshold pick; excludes raw diagnostic-curve points "
             "such as t=0 that were never proposed as an answer). " + str(len(front))
             + " are non-dominated.")
    L.append("")
    L.append("| Macro-F1 | MEL Recall | Model |")
    L.append("|---|---|---|")
    for name, f1, mr in front:
        L.append("| " + "{:.4f}".format(f1) + " | " + "{:.4f}".format(mr) + " | " + name + " |")
    L.append("")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(NL.join(L), encoding="utf-8")
    print(NL + "Report written: " + str(OUT_PATH))


if __name__ == "__main__":
    main()
