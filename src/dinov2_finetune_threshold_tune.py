"""
Post-hoc MEL-threshold tuning for the full-fine-tuned DINOv2 ViT-S/14 model.

The 30-epoch DINOv2 full fine-tune (train_dinov2_finetune.py) gave the best
overall Macro-F1 in the whole thesis (0.744) but regressed on melanoma recall
(0.569) — the better-calibrated model stops over-predicting mel. This is the
exact F1-vs-MEL-recall trade-off that ensemble_concat_v2.py's Variant 3 fixed
"for free" (post-hoc, no retraining), recovering MEL recall from 0.575 -> 0.754.

This script tries the same free fix on the fine-tuned model BEFORE deciding
whether an expensive (~4.5 h) re-run with a melanoma-aware checkpoint criterion
is even worth it. No training, no gradients — it just re-scores the already-
trained model's softmax outputs with different MEL decision thresholds:

  1. Load the existing epoch-22 checkpoint (outputs/dinov2_finetune/…).
  2. Run inference once on VAL and TEST -> per-class softmax probabilities.
  3. Grid-search the MEL threshold on VALIDATION only:
       if mel_prob > t  -> predict mel
       else             -> argmax over the remaining 6 classes
     scored by  composite = 0.5 * Macro-F1 + 0.5 * MEL_Recall
  4. Pick the best-composite threshold, apply it BLIND on the TEST set.

Threshold logic, composite score, and the grid-table format are identical to
ensemble_concat_v2.py so the two threshold experiments are directly comparable.

Saves results/dinov2_finetune_threshold_results.txt.

Run from project root (seconds, no training):
  python src/dinov2_finetune_threshold_tune.py \\
    --checkpoint outputs/dinov2_finetune/best_dinov2_finetune.pth \\
    --metadata-csv data/HAM10000_metadata.csv \\
    --images-dir data
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import Config
from helpers import get_device, set_seed
# Reuse the exact model + split used at training time so nothing diverges.
from train_dinov2_finetune import DINOv2Classifier, load_dinov2_backbone, make_loaders
from extract_features_v2 import load_multiclass_table
from dataset import lesion_level_split

# Hard-coded baselines for the comparison table (from earlier experiments).
_EFFNET_MACRO_F1, _EFFNET_MEL_REC = 0.692, 0.713   # EfficientNet-B0, full fine-tune
_DINOV2_PROBE_F1, _DINOV2_PROBE_REC = 0.545, 0.689  # DINOv2 frozen linear probe


# ---------------------------------------------------------------------------
# Threshold logic (identical to ensemble_concat_v2.py)
# ---------------------------------------------------------------------------

def apply_mel_threshold(probs: np.ndarray, mel_idx: int, threshold: float) -> np.ndarray:
    """
    Override argmax with MEL when mel_prob > threshold; otherwise predict the
    argmax over the remaining 6 classes.
    """
    preds = np.empty(len(probs), dtype=np.int64)
    for i, p in enumerate(probs):
        if p[mel_idx] > threshold:
            preds[i] = mel_idx
        else:
            masked = p.copy()
            masked[mel_idx] = -1.0
            preds[i] = int(np.argmax(masked))
    return preds


def composite_score(y_true: np.ndarray, y_pred: np.ndarray, mel_idx: int) -> float:
    """0.5 * Macro-F1 + 0.5 * MEL Recall."""
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mr = recall_score(y_true, y_pred, labels=[mel_idx], average="micro", zero_division=0)
    return 0.5 * mf1 + 0.5 * mr


def format_classification_report(y_true, y_pred, class_names) -> str:
    rep = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
    )
    lines = ["              precision    recall  f1-score   support", ""]
    for name in class_names:
        row = rep[name]
        lines.append(
            f"{name:>12} {row['precision']:>10.2f} {row['recall']:>9.2f}"
            f" {row['f1-score']:>9.2f} {int(row['support']):>9d}"
        )
    lines.append("")
    lines.append(
        f"{'accuracy':>12} {'':>10} {'':>9} {rep['accuracy']:>9.2f} {int(rep['macro avg']['support']):>9d}"
    )
    for agg in ("macro avg", "weighted avg"):
        row = rep[agg]
        lines.append(
            f"{agg:>12} {row['precision']:>10.2f} {row['recall']:>9.2f}"
            f" {row['f1-score']:>9.2f} {int(row['support']):>9d}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Inference — softmax probabilities over a split
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_probs(model, loader, device, use_amp: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, probs) where probs has shape (N, num_classes)."""
    model.eval()
    all_y: list[int] = []
    all_probs: list[np.ndarray] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
        all_probs.append(F.softmax(logits.float(), dim=1).cpu().numpy())
        all_y.extend(labels.tolist())
    return np.array(all_y), np.vstack(all_probs)


# ---------------------------------------------------------------------------
# Per-sample probability dump (--dump-probs)
# ---------------------------------------------------------------------------

def reconstruct_split_frames(cfg: Config):
    """
    Rebuild df_val / df_test exactly as make_loaders() does internally (same CSV,
    same lesion_level_split seed), purely to recover image_id/lesion_id for the CSV
    dump -- make_loaders() itself only returns DataLoaders, which drop those columns.

    Safe because both steps are deterministic given the same CSV + seed: reading the
    CSV never reorders rows, and lesion_level_split's train_test_split calls use an
    explicit random_state=seed independent of global RNG state, so calling this a
    second time reproduces byte-identical df_val/df_test. The val/test DataLoaders are
    built with shuffle=False, so predict_probs() visits rows in this same order --
    verified below via the write_probs_csv() label sanity-check, not just assumed.
    """
    df, label_map = load_multiclass_table(
        cfg.metadata_csv, cfg.col_image_id, cfg.col_lesion_id, cfg.col_diagnosis
    )
    _, df_val, df_test = lesion_level_split(
        df, cfg.col_lesion_id, cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, cfg.seed
    )
    return df_val, df_test, label_map


def write_probs_csv(
    path: Path,
    df,
    y_true: np.ndarray,
    probs: np.ndarray,
    cfg: Config,
    idx_to_class: dict,
    mel_idx: int,
) -> None:
    """Write image_id, lesion_id, y_true, p_mel, p_argmax_class, p_argmax_value."""
    if not (len(df) == len(y_true) == len(probs)):
        raise RuntimeError(
            f"Row-count mismatch: df={len(df)} y_true={len(y_true)} probs={len(probs)} "
            "-- split reconstruction is out of sync with the model's loader order."
        )
    # The only real proof the row order lines up: the label recovered from the
    # independently-rebuilt dataframe must match the label the loader actually
    # produced, position-for-position.
    df_labels = df["_binary_label"].to_numpy()
    n_mismatch = int((df_labels != y_true).sum())
    if n_mismatch:
        raise RuntimeError(
            f"{n_mismatch} label mismatches between reconstructed split and loader "
            "order -- refusing to write a CSV whose row alignment can't be trusted."
        )

    argmax_idx = probs.argmax(axis=1)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["image_id", "lesion_id", "y_true", "p_mel", "p_argmax_class", "p_argmax_value"]
        )
        for i in range(len(df)):
            row = df.iloc[i]
            writer.writerow(
                [
                    row[cfg.col_image_id],
                    row[cfg.col_lesion_id],
                    idx_to_class[int(y_true[i])],
                    f"{probs[i, mel_idx]:.6f}",
                    idx_to_class[int(argmax_idx[i])],
                    f"{probs[i, argmax_idx[i]]:.6f}",
                ]
            )
    print(f"Saved per-sample probabilities -> {path}  ({len(df)} rows)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Post-hoc MEL-threshold tuning for the fine-tuned DINOv2 model"
    )
    p.add_argument("--checkpoint", type=Path,
                   default=Path("outputs/dinov2_finetune/best_dinov2_finetune.pth"))
    p.add_argument("--metadata-csv", type=Path, default=None)
    p.add_argument("--images-dir", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    # MEL-threshold grid. Default extends below ensemble_concat_v2's 0.10 floor:
    # for the fine-tuned model the best composite lands at t=0.02, so the grid
    # starts at 0.02 (step 0.02) to reproduce that operating point by default.
    p.add_argument("--min-threshold", type=float, default=0.02)
    p.add_argument("--max-threshold", type=float, default=0.50)
    p.add_argument("--threshold-step", type=float, default=0.02)
    p.add_argument("--results-name", type=str, default="dinov2_finetune_threshold_results.txt",
                   help="Filename for the report inside --results-dir (override so a "
                        "re-run with a different grid doesn't clobber the default one)")
    p.add_argument("--dump-probs", action="store_true",
                   help="Also write per-sample probabilities (val + test) to CSV")
    p.add_argument("--probs-val-name", type=str, default="dinov2_finetune_probs_val.csv")
    p.add_argument("--probs-test-name", type=str, default="dinov2_finetune_probs_test.csv")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cfg = Config()
    if args.metadata_csv is not None:
        cfg.metadata_csv = args.metadata_csv
    if args.images_dir is not None:
        cfg.images_dir = args.images_dir
    cfg.seed = args.seed
    set_seed(cfg.seed)

    device = get_device()
    use_amp = (not args.no_amp) and device.type == "cuda"
    print(f"Device : {device}   AMP : {use_amp}")

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Run src/train_dinov2_finetune.py first (or pass --checkpoint)."
        )

    # --- Rebuild the model exactly as trained, load weights ------------------
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    label_map: dict[str, int] = ckpt["label_map"]
    num_classes = len(label_map)
    feat_dim = ckpt.get("feat_dim", 384)
    image_size = ckpt.get("image_size", 224)
    idx_to_class = {v: k for k, v in label_map.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]
    mel_idx = label_map["mel"]

    print(f"Loaded checkpoint : {args.checkpoint}")
    print(f"  trained epoch={ckpt.get('epoch','?')}  "
          f"val_macro_f1={ckpt.get('val_macro_f1', float('nan')):.4f}  "
          f"image_size={image_size}")

    print("Loading DINOv2 ViT-S/14 backbone ...")
    model = DINOv2Classifier(load_dinov2_backbone(), num_classes, feat_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # --- Same lesion-level split (seed 42) via the training helper -----------
    _, val_loader, test_loader, lm2, _ = make_loaders(
        cfg, image_size, args.batch_size, args.num_workers
    )
    assert lm2 == label_map, "Label map from split disagrees with checkpoint."

    # --- Inference once on val + test ---------------------------------------
    print("Scoring validation set ...")
    y_true_val, probs_val = predict_probs(model, val_loader, device, use_amp)
    print("Scoring test set ...")
    y_true_test, probs_test = predict_probs(model, test_loader, device, use_amp)

    if args.dump_probs:
        args.results_dir.mkdir(parents=True, exist_ok=True)
        df_val_ids, df_test_ids, label_map_check = reconstruct_split_frames(cfg)
        assert label_map_check == label_map, "Label map from split disagrees with checkpoint."
        write_probs_csv(
            args.results_dir / args.probs_val_name, df_val_ids, y_true_val, probs_val,
            cfg, idx_to_class, mel_idx,
        )
        write_probs_csv(
            args.results_dir / args.probs_test_name, df_test_ids, y_true_test, probs_test,
            cfg, idx_to_class, mel_idx,
        )

    # Sanity: plain-argmax test metrics (should reproduce 0.744 / 0.569).
    argmax_test = probs_test.argmax(axis=1)
    argmax_f1 = f1_score(y_true_test, argmax_test, average="macro", zero_division=0)
    argmax_mr = recall_score(y_true_test, argmax_test, labels=[mel_idx], average="micro", zero_division=0)

    # --- Grid-search MEL threshold on VALIDATION only ------------------------
    thresholds = [
        round(float(t), 3)
        for t in np.arange(args.min_threshold, args.max_threshold + args.threshold_step / 2, args.threshold_step)
    ]
    # Threshold column width adapts to the grid's step size: the old 0.02-step grid
    # still prints 2 decimals (byte-identical to before), but a finer step (e.g. 0.001)
    # needs 3+ decimals or distinct thresholds silently collapse to the same label.
    step_str = f"{args.threshold_step:.10f}".rstrip("0")
    step_decimals = len(step_str.split(".")[1]) if "." in step_str else 0
    t_decimals = max(2, step_decimals)

    print(f"\nGrid-searching MEL threshold over {thresholds}")
    print("Selection uses VALIDATION composite only; TEST columns are diagnostic "
          "(shown for transparency, never used to pick the threshold).")
    grid_header = (
        f"{'Thr':>6}  {'val_F1':>8} {'val_MEL':>8} {'val_Comp':>9}"
        f"  | {'test_F1':>8} {'test_MEL':>8}"
    )
    print(grid_header)

    best_t, best_score = thresholds[0], -1.0
    grid_rows: list[str] = []
    for t in thresholds:
        # Validation (used for selection)
        yv = apply_mel_threshold(probs_val, mel_idx, t)
        vf1 = f1_score(y_true_val, yv, average="macro", zero_division=0)
        vmr = recall_score(y_true_val, yv, labels=[mel_idx], average="micro", zero_division=0)
        vcomp = 0.5 * vf1 + 0.5 * vmr
        # Test (diagnostic only — NOT used for selection)
        yt = apply_mel_threshold(probs_test, mel_idx, t)
        tf1 = f1_score(y_true_test, yt, average="macro", zero_division=0)
        tmr = recall_score(y_true_test, yt, labels=[mel_idx], average="micro", zero_division=0)

        row = f"{t:>6.{t_decimals}f}  {vf1:>8.4f} {vmr:>8.4f} {vcomp:>9.4f}  | {tf1:>8.4f} {tmr:>8.4f}"
        print(row)
        grid_rows.append(row)
        if vcomp > best_score:
            best_score, best_t = vcomp, t

    print(f"\nBest threshold on val: t={best_t:.{t_decimals}f}  (val composite={best_score:.4f})")

    # --- Apply the best threshold BLIND on the test set ----------------------
    y_pred_test_thresh = apply_mel_threshold(probs_test, mel_idx, best_t)
    thr_f1 = f1_score(y_true_test, y_pred_test_thresh, average="macro", zero_division=0)
    thr_mr = recall_score(y_true_test, y_pred_test_thresh, labels=[mel_idx], average="micro", zero_division=0)

    # --- Build report --------------------------------------------------------
    cm = confusion_matrix(y_true_test, y_pred_test_thresh, labels=list(range(num_classes)))

    lines = [
        "=" * 64,
        "DINOv2 ViT-S/14 FULL FINE-TUNE  —  post-hoc MEL-threshold tuning",
        "=" * 64,
        f"Checkpoint          : {args.checkpoint}  (epoch {ckpt.get('epoch','?')})",
        f"Threshold rule      : predict MEL if mel_prob > t, else argmax of other 6",
        f"Selection criterion : composite = 0.5*Macro-F1 + 0.5*MEL_Recall (val set)",
        f"Best threshold (val): t={best_t:.{t_decimals}f}  (val composite={best_score:.4f})",
        "",
        "Threshold grid search (val = selection metric; test = diagnostic only):",
        grid_header,
        *grid_rows,
        "",
        f"TEST confusion matrix at t={best_t:.{t_decimals}f} (rows = true, cols = predicted):",
        "         " + "  ".join(f"{c:>6}" for c in class_names),
    ]
    for i, row in enumerate(cm):
        lines.append(f"{class_names[i]:>8} " + "  ".join(f"{v:>6}" for v in row))
    lines += ["", f"TEST per-class report at t={best_t:.{t_decimals}f}:",
              format_classification_report(y_true_test, y_pred_test_thresh, class_names), ""]

    # Comparison table
    sep = "-" * 60
    rows = [
        ("EfficientNet (full fine-tune)", _EFFNET_MACRO_F1, _EFFNET_MEL_REC),
        ("DINOv2 (linear probe, frozen)", _DINOV2_PROBE_F1, _DINOV2_PROBE_REC),
        ("DINOv2 (full fine-tune, argmax)", argmax_f1, argmax_mr),
        (f"DINOv2 (full fine-tune, threshold t={best_t:.{t_decimals}f})", thr_f1, thr_mr),
    ]
    lines += [
        "=" * 60,
        "  Model Comparison",
        "=" * 60,
        f"{'Model':<40} | {'Macro-F1':>8} | {'MEL Rec':>7}",
        sep,
    ]
    for name, f1v, rec in rows:
        lines.append(f"{name:<40} | {f1v:>8.3f} | {rec:>7.3f}")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print("\n" + report)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.results_dir / args.results_name
    out_file.write_text(report, encoding="utf-8")
    print(f"\nSaved results -> {out_file}")


if __name__ == "__main__":
    main()
