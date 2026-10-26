"""
Regenerate the missing result files for the "first-approach" experiments.

train_dino_head.py (all DINO/BiomedCLIP linear probes) and train_efficientnet.py
print their test metrics to stdout but never wrote a results/*.txt — so those
reports were lost after the terminal scrolled. Every trained CHECKPOINT and every
cached FEATURE set still exists, though, so this script reloads them and rebuilds
the reports deterministically (no retraining).

Two evaluation types, driven by a config table below:

  1. Linear probes (DINOv1 focal/combined/dice, DINOv2, DINOv2-reg, DINOv3,
     BiomedCLIP): rebuild nn.Linear(feat_dim -> 7), load the saved head, and run
     it over the cached TEST features. Exact, deterministic, seconds.

  2. EfficientNet-B0 7-class (baseline + improved): rebuild the CNN, load the
     saved weights, and run over the TEST images. --effnet-mode selects
     plain (single forward, default), tta (5-view), or tta_mc (5-view + MC
     Dropout) — the last reproduces train_efficientnet.py's headline number.

Writes one results/<name>_results.txt per model (confusion matrix + per-class
report + Macro-F1 + MEL recall, same format as the other experiments) and prints
a consolidated leaderboard at the end.

Run from project root:
  python src/eval_saved_models.py                 # probes + EfficientNet (plain)
  python src/eval_saved_models.py --only probes
  python src/eval_saved_models.py --only effnet --effnet-mode tta_mc --mc-passes 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import Config
from dataset import SkinLesionDataset, build_transforms, lesion_level_split
from helpers import get_device, set_seed
from model import build_efficientnet_b0
from extract_features_v2 import load_multiclass_table
from train_dino_head import format_classification_report
# Reuse EfficientNet's own TTA / TTA+MC evaluation so the numbers match exactly.
from train_efficientnet import (
    TTASkinLesionDataset,
    build_tta_transforms,
    evaluate_tta,
    evaluate_tta_mc,
)

# ---------------------------------------------------------------------------
# Config — every saved model to re-evaluate
# ---------------------------------------------------------------------------

# (report name, checkpoint, cached TEST features) — linear heads on frozen feats.
PROBE_CONFIGS = [
    ("dinov1_probe_focal",    "outputs/dino_linear/best_dino_head.pth",        "data/processed/dino_feats_test.pt"),
    ("dinov1_probe_combined", "outputs/dino_combined/best_dino_head.pth",      "data/processed/dino_feats_test.pt"),
    ("dinov1_probe_dice",     "outputs/dino_dice/best_dino_head.pth",          "data/processed/dino_feats_test.pt"),
    ("dinov2_probe",          "outputs/dinov2_linear/best_dino_head.pth",      "data/processed/dinov2/dinov2_feats_test.pt"),
    ("dinov2_reg_probe",      "outputs/dinov2_reg_linear/best_dino_head.pth",  "data/processed/dinov2_reg/dinov2_reg_feats_test.pt"),
    ("dinov3_probe",          "outputs/dinov3_linear/best_dino_head.pth",      "data/processed/dinov3/dinov3_feats_test.pt"),
    ("biomedclip_probe",      "outputs/biomedclip_linear/best_dino_head.pth",  "data/processed/biomedclip/biomedclip_feats_test.pt"),
]

# Full EfficientNet-B0 7-class model, run on TEST images. NOTE: outputs/
# efficientnet_baseline and outputs/efficientnet_mc are the SAME trained weights
# (identical stored val_macro_f1); "baseline" vs "improved" is the EVAL method,
# not different training. So we evaluate the one model two ways:
#   baseline = plain single forward pass
#   improved = 5-view TTA + MC Dropout  (the thesis headline number)
EFFNET_CKPT = "outputs/efficientnet_mc/best_efficientnet.pth"
EFFNET_EVALS = [
    ("efficientnet_baseline_7class", "plain"),
    ("efficientnet_improved_7class", "tta_mc"),
]


# ---------------------------------------------------------------------------
# Metrics + report
# ---------------------------------------------------------------------------

def _metrics(y_true, y_pred, mel_idx):
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mr = (recall_score(y_true, y_pred, labels=[mel_idx], average="micro", zero_division=0)
          if mel_idx >= 0 else float("nan"))
    return mf1, mr


def build_report(title, subtitle, label_map, y_true, y_pred):
    idx_to_class = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)
    class_names = [idx_to_class[i] for i in range(num_classes)]
    mel_idx = label_map.get("mel", -1)
    mf1, mr = _metrics(y_true, y_pred, mel_idx)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    lines = [
        "=" * 64, title, "=" * 64, subtitle,
        f"Regenerated from saved checkpoint (no retraining).",
        "",
        "Confusion Matrix  (rows = true class, cols = predicted class):",
        "         " + "  ".join(f"{c:>6}" for c in class_names),
    ]
    for i, row in enumerate(cm):
        lines.append(f"{class_names[i]:>8} " + "  ".join(f"{v:>6}" for v in row))
    lines += ["", "Per-class Classification Report:",
              format_classification_report(y_true, y_pred, class_names),
              "",
              "=" * 40,
              f"  Macro-F1        : {mf1:.4f}",
              f"  Melanoma Recall : {mr:.4f}",
              "=" * 40]

    # Structured mirror of the same numbers (full float precision, not the
    # report's 2-decimal text) -- used by callers that want to dump JSON
    # instead of / alongside the text report, e.g. the loss-ablation driver.
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True,
    )
    structured = {
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
        "per_class": {
            name: {
                "precision": report_dict[name]["precision"],
                "recall": report_dict[name]["recall"],
                "f1": report_dict[name]["f1-score"],
                "support": int(report_dict[name]["support"]),
            }
            for name in class_names
        },
        "accuracy": report_dict["accuracy"],
        "macro_f1": mf1,
        "mel_recall": mr,
    }
    return "\n".join(lines), mf1, mr, structured


# ---------------------------------------------------------------------------
# Linear-probe evaluation (cached features)
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_probe(name, ckpt_path, feats_path, device, results_dir, dump_json: Path | None = None):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    data = torch.load(feats_path, map_location="cpu", weights_only=False)
    label_map = ckpt["label_map"]
    feat_dim = ckpt.get("feat_dim", data["features"].shape[1])

    head = nn.Linear(feat_dim, len(label_map)).to(device)
    head.load_state_dict(ckpt["model_state_dict"])
    head.eval()

    X, y = data["features"].to(device), data["labels"].long().numpy()
    y_pred = head(X).argmax(dim=1).cpu().numpy()

    subtitle = (f"Model: frozen-feature linear probe  |  feat_dim={feat_dim}  "
                f"|  ckpt={ckpt_path}  (val_macro_f1={ckpt.get('val_macro_f1', float('nan')):.4f})")
    report, mf1, mr, structured = build_report(f"{name}  —  linear probe (7-class HAM10000)",
                                   subtitle, label_map, y, y_pred)
    out = results_dir / f"{name}_results.txt"
    out.write_text(report, encoding="utf-8")
    if dump_json is not None:
        payload = {"name": name, "mode": "probe", "ckpt": str(ckpt_path),
                   "feats": str(feats_path), **structured}
        dump_json.parent.mkdir(parents=True, exist_ok=True)
        dump_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"  {name:<26} Macro-F1={mf1:.4f}  MEL-Recall={mr:.4f}  -> {out}")
    return name, mf1, mr


# ---------------------------------------------------------------------------
# EfficientNet evaluation (full model on images)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_plain(model, loader, device):
    model.eval()
    all_y, all_pred = [], []
    for images, labels in loader:
        images = images.to(device)
        all_pred.extend(model(images).argmax(dim=1).cpu().tolist())
        all_y.extend(labels.tolist())
    return np.array(all_y), np.array(all_pred)


def eval_effnet(name, ckpt_path, cfg, mode, mc_passes, batch_size, num_workers,
                device, results_dir, df_test, dump_json: Path | None = None):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    label_map = ckpt["label_map"]
    model = build_efficientnet_b0(num_classes=len(label_map), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    pin = device.type == "cuda"
    if mode == "plain":
        ds = SkinLesionDataset(df_test, cfg.images_dir, cfg.col_image_id,
                               build_transforms(cfg.image_size, is_train=False))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
        y_true, y_pred = _eval_plain(model, loader, device)
        mdesc = "plain single forward pass (no TTA / no MC Dropout)"
    else:
        tta = build_tta_transforms(cfg.image_size)
        ds = TTASkinLesionDataset(df_test, cfg.images_dir, cfg.col_image_id, tta)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
        if mode == "tta":
            y_true, y_pred = evaluate_tta(model, loader, device)
            mdesc = f"{len(tta)}-view TTA (averaged softmax)"
        else:  # tta_mc
            y_true, y_pred, _ = evaluate_tta_mc(model, loader, device, mc_passes)
            mdesc = f"{len(tta)}-view TTA + MC Dropout ({mc_passes} passes/view)"

    subtitle = f"Model: EfficientNet-B0 (7-class)  |  eval={mdesc}  |  ckpt={ckpt_path}"
    report, mf1, mr, structured = build_report(f"{name}  —  EfficientNet-B0 (7-class HAM10000)",
                                   subtitle, label_map, y_true, y_pred)
    out = results_dir / f"{name}_results.txt"
    out.write_text(report, encoding="utf-8")
    if dump_json is not None:
        payload = {"name": name, "mode": mode, "ckpt": str(ckpt_path), **structured}
        dump_json.parent.mkdir(parents=True, exist_ok=True)
        dump_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"  {name:<30} [{mode}]  Macro-F1={mf1:.4f}  MEL-Recall={mr:.4f}  -> {out}")
    return f"{name} [{mode}]", mf1, mr


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Regenerate result files from saved checkpoints")
    p.add_argument("--metadata-csv", type=Path, default=None)
    p.add_argument("--images-dir", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--only", choices=["all", "probes", "effnet"], default="all")
    p.add_argument("--effnet-mode", choices=["plain", "tta", "tta_mc"], default="plain",
                   help="EfficientNet eval mode. tta_mc reproduces the headline number.")
    p.add_argument("--probe-ckpt", type=Path, default=None,
                   help="Override checkpoint path for a single linear-probe eval "
                        "(requires --probe-feats and --probe-report-name).")
    p.add_argument("--probe-feats", type=Path, default=None,
                   help="Override cached TEST features (.pt) for a single probe eval.")
    p.add_argument("--probe-report-name", type=str, default=None,
                   help="If set, evaluate only this single probe against --probe-ckpt / "
                        "--probe-feats instead of the built-in PROBE_CONFIGS list. Used by "
                        "the probe loss-ablation driver; leaves default behaviour untouched.")
    p.add_argument("--effnet-ckpt", type=Path, default=None,
                   help="Override checkpoint path for EfficientNet eval "
                        f"(default: {EFFNET_CKPT})")
    p.add_argument("--effnet-report-name", type=str, default=None,
                   help="If set, evaluate only this single (name, --effnet-mode) pair "
                        "against --effnet-ckpt instead of the default baseline+improved "
                        "pair against the default checkpoint. Used by scripts that "
                        "re-evaluate other EfficientNet checkpoints (e.g. the loss "
                        "ablation driver) without touching the default behaviour here.")
    p.add_argument("--dump-json", type=Path, default=None,
                   help="If set (together with --effnet-report-name), also write a "
                        "structured JSON sidecar with the confusion matrix and "
                        "per-class metrics next to the .txt report.")
    p.add_argument("--mc-passes", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
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
    print(f"Device : {device}")
    args.results_dir.mkdir(parents=True, exist_ok=True)

    leaderboard: list[tuple[str, float, float]] = []

    # --- Linear probes -------------------------------------------------------
    if args.only in ("all", "probes"):
        print("\nLinear probes (frozen features):")
        if args.probe_report_name is not None:
            # Single explicit probe against an arbitrary checkpoint/feature file --
            # used by the probe loss-ablation driver. Default path below untouched.
            if not (Path(args.probe_ckpt).exists() and Path(args.probe_feats).exists()):
                print(f"  {args.probe_report_name:<26} SKIPPED (missing ckpt or features)")
            else:
                leaderboard.append(eval_probe(
                    args.probe_report_name, str(args.probe_ckpt), str(args.probe_feats),
                    device, args.results_dir, dump_json=args.dump_json,
                ))
        else:
            for name, ckpt, feats in PROBE_CONFIGS:
                if not (Path(ckpt).exists() and Path(feats).exists()):
                    print(f"  {name:<26} SKIPPED (missing ckpt or features)")
                    continue
                leaderboard.append(eval_probe(name, ckpt, feats, device, args.results_dir))

    # --- EfficientNet --------------------------------------------------------
    if args.only in ("all", "effnet"):
        print(f"\nEfficientNet-B0 (eval mode = {args.effnet_mode}):")
        df, label_map = load_multiclass_table(
            cfg.metadata_csv, cfg.col_image_id, cfg.col_lesion_id, cfg.col_diagnosis
        )
        _, _, df_test = lesion_level_split(
            df, cfg.col_lesion_id, cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, cfg.seed
        )
        ckpt_path = str(args.effnet_ckpt) if args.effnet_ckpt is not None else EFFNET_CKPT
        if not Path(ckpt_path).exists():
            print(f"  {ckpt_path} SKIPPED (missing ckpt)")
        elif args.effnet_report_name is not None:
            # Single explicit (name, mode) pair against an arbitrary checkpoint --
            # used by scripts driving other EfficientNet runs (e.g. the loss
            # ablation driver), leaving the default all-checkpoint behaviour above
            # completely untouched.
            leaderboard.append(eval_effnet(
                args.effnet_report_name, ckpt_path, cfg, args.effnet_mode, args.mc_passes,
                args.batch_size, args.num_workers, device, args.results_dir, df_test,
                dump_json=args.dump_json,
            ))
        else:
            for name, mode in EFFNET_EVALS:
                leaderboard.append(eval_effnet(
                    name, EFFNET_CKPT, cfg, mode, args.mc_passes,
                    args.batch_size, args.num_workers, device, args.results_dir, df_test
                ))

    # --- Consolidated leaderboard -------------------------------------------
    leaderboard.sort(key=lambda r: r[1], reverse=True)
    print("\n" + "=" * 60)
    print("  Regenerated leaderboard (sorted by Macro-F1)")
    print("=" * 60)
    print(f"{'Model':<38} | {'Macro-F1':>8} | {'MEL Rec':>7}")
    print("-" * 60)
    for nm, mf1, mr in leaderboard:
        print(f"{nm:<38} | {mf1:>8.4f} | {mr:>7.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
