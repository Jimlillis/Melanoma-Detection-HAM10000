"""
Full end-to-end fine-tuning of DINOv2 ViT-S/14 on HAM10000 (7-class).

WHAT MAKES THIS DIFFERENT FROM EVERY OTHER DINO SCRIPT IN THIS REPO
-------------------------------------------------------------------
All previous DINO experiments (extract_features_v2.py + train_dino_head.py, and
the ensemble scripts) use a FROZEN backbone: images are pushed through DINOv2
once, the 384-dim CLS features are cached to .pt files, and only a linear/MLP
head is trained on those frozen features.

This script does REAL fine-tuning:
  - The DINOv2 ViT-S/14 backbone is NOT frozen — gradients flow through the whole
    transformer, so the pretrained features themselves adapt to dermoscopy.
  - There are NO pre-extracted features: every batch of raw images is passed
    through DINOv2 during training (much heavier, but the whole point).
  - Model = dinov2_vits14 backbone  →  Linear(384 → 7) head on the CLS token,
    ALL parameters trainable.

FINE-TUNING STRATEGY — full unfreeze from the start + layer-wise LR decay (LLRD)
-------------------------------------------------------------------------------
Two mainstream options exist for adapting a pretrained ViT:
  (a) gradual/progressive unfreezing (ULMFiT-style): train the head first, then
      unfreeze blocks top-down over several stages.
  (b) full unfreeze from epoch 0 with a *discriminative* (layer-wise) learning
      rate — every layer trains at once, but lower/earlier layers get an
      exponentially smaller LR than later layers and the head.

We use (b). It is the de-facto standard for modern ViT fine-tuning (BEiT, DeiT,
MAE, and the DINOv2 downstream recipes all use LLRD) and, for a small dataset,
is both simpler and more robust than staged unfreezing: LLRD already protects
the generic low-level features (they barely move) while letting the top blocks
and head specialise, which is exactly what gradual unfreezing tries to achieve
manually — without the extra stage-scheduling hyperparameters and the risk of
the head over-fitting during its frozen-backbone warmup phase.

Concretely (justified by ViT fine-tuning literature; LLRD factor 0.75 is the
common value for ViT-S, backbone base LR ~1e-5, head LR ~1e-3):
  - head:            head_lr            (default 1e-3)
  - last block:      backbone_lr        (default 1e-5)
  - block i:         backbone_lr * layer_decay**(n_blocks-1 - i)
  - patch/pos/cls:   backbone_lr * layer_decay**n_blocks   (smallest — preserved)
This "MUCH lower LR on the backbone than the head" is what avoids catastrophic
forgetting of the self-supervised pretrained features.

OVERFITTING CONTROLS (small dataset + 22M-param ViT = high overfitting risk):
  - Layer-wise LR decay (bottom layers effectively near-frozen).
  - AdamW weight decay 0.05 (ViT-standard), disabled on norms/biases/tokens.
  - Same augmentation pipeline as the EfficientNet baseline (flip/flip/jitter) so
    the comparison is fair, plus early stopping on val Macro-F1 (patience 10).
  - Cosine LR schedule with linear warmup.
  - Mixed-precision (AMP) + gradient clipping for stable, memory-light training.

LOSS: Combined (0.5 * Focal[gamma=2] + 0.5 * Dice), the exact CombinedLoss used
by our best DINO linear-probe experiments (this combo avoided mode collapse),
imported unchanged from train_dino_head.py. Inverse-frequency class weights.

IMAGE PREPROCESSING: 224x224, ImageNet mean/std (dataset.build_transforms — the
same transforms the EfficientNet baseline and the frozen probe used). 224 keeps
16x16 = 256 patches (224 / patch-size 14); 518 would quadruple the token count
and memory for a marginal gain and is left off by default for a fair, affordable
comparison — see --image-size if you want to try it.

Best checkpoint (by val Macro-F1) →  outputs/dinov2_finetune/best_dinov2_finetune.pth
Test report + comparison table    →  results/dinov2_finetune_results.txt

Run from project root:
  python src/train_dinov2_finetune.py \\
    --metadata-csv data/HAM10000_metadata.csv \\
    --images-dir data \\
    --output-dir outputs/dinov2_finetune

Just measure GPU memory + estimated wall-clock time and exit (no training):
  python src/train_dinov2_finetune.py --estimate-only
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import Config
from dataset import SkinLesionDataset, build_transforms, lesion_level_split
from helpers import get_device, set_seed

# Reuse the EXACT loss, class-weight, sampler, label-order and report helpers used
# by the frozen linear-probe experiments so nothing but the training regime differs.
from extract_features_v2 import load_multiclass_table
from train_dino_head import (
    CombinedLoss,
    DiceLoss,
    FocalLoss,
    compute_class_weights,
    format_classification_report,
    make_weighted_sampler,
)

# Hard-coded baselines for the final comparison table (provided by the thesis).
_EFFNET_MACRO_F1, _EFFNET_MEL_REC = 0.692, 0.713    # EfficientNet-B0, full fine-tune
_DINOV2_PROBE_F1, _DINOV2_PROBE_REC = 0.545, 0.689  # DINOv2 frozen linear probe
# Prior DINOv2 full-fine-tune results (best-val-F1 checkpoint), for the 5-row
# comparison printed by the melanoma-aware (composite-criterion) variant.
_DINOV2_ARGMAX_F1, _DINOV2_ARGMAX_REC = 0.744, 0.569   # best-F1 ckpt, plain argmax
_DINOV2_THRESH_F1, _DINOV2_THRESH_REC = 0.732, 0.641   # best-F1 ckpt, MEL threshold t=0.10


# ---------------------------------------------------------------------------
# Model — DINOv2 backbone + linear classification head on the CLS token
# ---------------------------------------------------------------------------

class DINOv2Classifier(nn.Module):
    """
    dinov2_vits14 backbone (all params trainable) + Linear(feat_dim -> num_classes).

    The hub backbone's forward returns the normalised CLS token of shape
    (B, 384) — the same tensor extract_features_v2.py caches — so the head is a
    single Linear on top. Because we never wrap the backbone in no_grad and keep
    it in train() mode, gradients flow through the entire ViT.
    """

    def __init__(self, backbone: nn.Module, num_classes: int, feat_dim: int = 384) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)          # (B, feat_dim) CLS token
        return self.head(feats)


def load_dinov2_backbone() -> nn.Module:
    """Load dinov2_vits14 from torch.hub (cached after first download)."""
    return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")


# ---------------------------------------------------------------------------
# Discriminative / layer-wise LR param groups
# ---------------------------------------------------------------------------

def _layer_id_of(param_name: str, n_blocks: int) -> int:
    """
    Map a parameter name to a depth id used for layer-wise LR decay:
      0                = embeddings (patch_embed, cls/pos/mask tokens)
      1 .. n_blocks    = transformer blocks (block index + 1)
      n_blocks + 1     = top of backbone (final norm) / head
    """
    if param_name.startswith("head"):
        return n_blocks + 1
    name = param_name[len("backbone."):] if param_name.startswith("backbone.") else param_name
    if name.startswith(("cls_token", "pos_embed", "patch_embed", "mask_token", "register_tokens")):
        return 0
    if name.startswith("blocks."):
        return int(name.split(".")[1]) + 1
    if name.startswith("norm"):          # final norm layer of the backbone
        return n_blocks + 1
    return n_blocks + 1                    # anything unexpected → top (safe default)


def build_param_groups(
    model: DINOv2Classifier,
    backbone_lr: float,
    head_lr: float,
    layer_decay: float,
    weight_decay: float,
) -> list[dict]:
    """
    One AdamW param group per (layer, weight-decay-flag), with layer-wise LR decay.

    - Backbone: lr = backbone_lr * layer_decay**(n_blocks-1 - block_idx); the last
      block sits at backbone_lr, earlier blocks and the embeddings train slower.
    - Head: lr = head_lr.
    - Weight decay is disabled for 1-D params (biases, LayerNorm weights) and the
      cls/pos/mask tokens — the standard "no-WD" set for ViTs.

    A single LambdaLR schedule (warmup+cosine) multiplies every group's base lr by
    the same factor, so these discriminative ratios are preserved throughout.
    """
    n_blocks = len(model.backbone.blocks)
    groups: dict[tuple[int, bool], dict] = {}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        layer_id = _layer_id_of(name, n_blocks)

        if name.startswith("head"):
            base_lr = head_lr
        elif layer_id == n_blocks + 1:            # final backbone norm → top of backbone
            base_lr = backbone_lr
        else:                                      # embeddings (0) .. last block (n_blocks)
            base_lr = backbone_lr * (layer_decay ** (n_blocks - layer_id))

        no_decay = p.ndim == 1 or name.endswith((".bias",)) or any(
            t in name for t in ("cls_token", "pos_embed", "mask_token", "register_tokens")
        )
        wd = 0.0 if no_decay else weight_decay

        key = (layer_id, no_decay)
        if key not in groups:
            groups[key] = {"params": [], "lr": base_lr, "weight_decay": wd}
        groups[key]["params"].append(p)

    return list(groups.values())


def make_warmup_cosine(optimizer, warmup_steps: int, total_steps: int):
    """LambdaLR: linear warmup to 1.0 over warmup_steps, then cosine decay to 0."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_loaders(
    cfg: Config, image_size: int, batch_size: int, num_workers: int,
    split_seed: int | None = None,
):
    df, label_map = load_multiclass_table(
        cfg.metadata_csv, cfg.col_image_id, cfg.col_lesion_id, cfg.col_diagnosis
    )
    # Decoupled from cfg.seed (training RNG) -- see --split-seed. Defaults to
    # cfg.seed when not given, i.e. identical to the previous behaviour.
    effective_split_seed = split_seed if split_seed is not None else cfg.seed
    df_train, df_val, df_test = lesion_level_split(
        df, cfg.col_lesion_id, cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, effective_split_seed
    )

    ds_train = SkinLesionDataset(
        df_train, cfg.images_dir, cfg.col_image_id, build_transforms(image_size, is_train=True)
    )
    ds_val = SkinLesionDataset(
        df_val, cfg.images_dir, cfg.col_image_id, build_transforms(image_size, is_train=False)
    )
    ds_test = SkinLesionDataset(
        df_test, cfg.images_dir, cfg.col_image_id, build_transforms(image_size, is_train=False)
    )

    num_classes = len(label_map)
    y_train = torch.tensor(df_train["_binary_label"].values, dtype=torch.long)
    sampler = make_weighted_sampler(y_train, num_classes)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        ds_train, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin
    )
    test_loader = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin
    )
    return train_loader, val_loader, test_loader, label_map, y_train


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, criterion, use_amp: bool):
    model.eval()
    losses, all_y, all_pred = [], [], []
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        losses.append(loss.item())
        all_y.extend(labels.cpu().tolist())
        all_pred.extend(logits.argmax(dim=1).cpu().tolist())
    return float(np.mean(losses)), np.array(all_y), np.array(all_pred)


# ---------------------------------------------------------------------------
# Resource estimate (GPU memory + wall-clock time) — printed before the full run
# ---------------------------------------------------------------------------

def estimate_resources(model, train_loader, criterion, optimizer, scaler, device,
                       epochs: int, use_amp: bool, n_probe: int = 8) -> None:
    """
    Run a few forward+backward passes (NO optimizer.step, so weights stay pristine),
    measure peak GPU memory and average step time, and extrapolate to the full run.
    """
    model.train()
    steps_per_epoch = len(train_loader)
    step_times: list[float] = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    it = iter(train_loader)
    for i in range(n_probe + 1):  # first iteration is warmup (cudnn autotune, alloc)
        try:
            images, labels = next(it)
        except StopIteration:
            break
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = criterion(model(images), labels)
        scaler.scale(loss).backward()   # backward measured; step deliberately skipped

        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i > 0:                        # drop warmup step
            step_times.append(dt)

    optimizer.zero_grad(set_to_none=True)  # clear probe gradients before real training

    avg_step = float(np.mean(step_times)) if step_times else float("nan")
    epoch_time = avg_step * steps_per_epoch
    total_time = epoch_time * epochs
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "-" * 64)
    print("RESOURCE ESTIMATE (before full run)")
    print("-" * 64)
    print(f"  Trainable params      : {n_train / 1e6:.1f} M")
    print(f"  Batch size            : {train_loader.batch_size}")
    print(f"  Steps / epoch         : {steps_per_epoch}")
    print(f"  Avg train step        : {avg_step * 1000:.0f} ms  (fwd+bwd, AMP={use_amp})")
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        total_mem = torch.cuda.get_device_properties(device).total_memory / 1024**3
        # + AdamW state (2 moments, fp32) ~ 8 bytes/param, allocated on first real step
        opt_state_gb = n_train * 8 / 1024**3
        print(f"  Peak GPU mem (fwd+bwd): {peak:.2f} GB  (+~{opt_state_gb:.2f} GB AdamW state at step 1)")
        print(f"  GPU total             : {total_mem:.2f} GB  ({torch.cuda.get_device_name(device)})")
    else:
        print("  (CPU run — no GPU memory to report; expect this to be very slow)")
    print(f"  Est. time / epoch     : {epoch_time:.1f} s")
    print(f"  Est. total ({epochs} ep)   : {total_time / 60:.1f} min  (upper bound; early stopping usually ends sooner)")
    print("-" * 64 + "\n")


# ---------------------------------------------------------------------------
# Results file
# ---------------------------------------------------------------------------

def build_report(args, label_map, y_true, y_pred, best_epoch, best_val_f1, best_val_mel) -> str:
    idx_to_class = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)
    class_names = [idx_to_class[i] for i in range(num_classes)]
    mel_idx = label_map.get("mel", -1)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mel_recall = (
        recall_score(y_true, y_pred, labels=[mel_idx], average="micro", zero_division=0)
        if mel_idx >= 0 else float("nan")
    )
    composite = args.best_criterion == "composite"

    lines = [
        "=" * 64,
        "DINOv2 ViT-S/14 — FULL FINE-TUNE (7-class HAM10000)"
        + ("  [melanoma-aware selection]" if composite else ""),
        "=" * 64,
        f"Backbone            : dinov2_vits14 (all params trainable)",
        f"Image size          : {args.image_size}",
        f"Loss                : combined (0.5*Focal[gamma={args.gamma}] + 0.5*Dice), inverse-freq weights",
        f"Backbone LR / Head LR / layer-decay : {args.backbone_lr:g} / {args.head_lr:g} / {args.layer_decay}",
        f"Weight decay        : {args.weight_decay}  |  warmup ep: {args.warmup_epochs}  |  patience: {args.patience}",
        f"Checkpoint criterion: {args.best_criterion}"
        + ("  (0.5*Macro-F1 + 0.5*MEL_Recall)" if composite else ""),
        f"Best epoch          : {best_epoch}  "
        f"(val Macro-F1 = {best_val_f1:.4f}, val MEL Recall = {best_val_mel:.4f})",
        "",
        "Confusion Matrix  (rows = true class, cols = predicted class):",
        "         " + "  ".join(f"{c:>6}" for c in class_names),
    ]
    for i, row in enumerate(cm):
        lines.append(f"{class_names[i]:>8} " + "  ".join(f"{v:>6}" for v in row))

    lines += ["", "Per-class Classification Report:",
              format_classification_report(y_true, y_pred, class_names), ""]

    # Final comparison table. The melanoma-aware (composite) variant prints the
    # full 5-row table against all prior DINOv2 results; the default best-F1 run
    # keeps its original 3-row table.
    if composite:
        new_label = "DINOv2 (full fine-tune, melanoma-aware) [NEW]"
        rows = [
            ("EfficientNet (full fine-tune)", _EFFNET_MACRO_F1, _EFFNET_MEL_REC),
            ("DINOv2 (linear probe, frozen)", _DINOV2_PROBE_F1, _DINOV2_PROBE_REC),
            ("DINOv2 (full fine-tune, best-F1 ckpt)", _DINOV2_ARGMAX_F1, _DINOV2_ARGMAX_REC),
            ("DINOv2 (full fine-tune, best-F1, threshold)", _DINOV2_THRESH_F1, _DINOV2_THRESH_REC),
            (new_label, macro_f1, mel_recall),
        ]
        width = max(len(r[0]) for r in rows)
    else:
        rows = [
            ("EfficientNet (full fine-tune)", _EFFNET_MACRO_F1, _EFFNET_MEL_REC),
            ("DINOv2 (linear probe, frozen)", _DINOV2_PROBE_F1, _DINOV2_PROBE_REC),
            ("DINOv2 (FULL fine-tune) [NEW]", macro_f1, mel_recall),
        ]
        width = 31

    table_w = width + 24
    lines += [
        "=" * table_w,
        "  Model Comparison",
        "=" * table_w,
        f"{'Model':<{width}} | {'Macro-F1':>8} | {'MEL Rec':>7}",
        "-" * table_w,
    ]
    for name, f1v, rec in rows:
        lines.append(f"{name:<{width}} | {f1v:>8.3f} | {rec:>7.3f}")
    lines.append("=" * table_w)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full fine-tune DINOv2 ViT-S/14 on HAM10000 (7-class)")
    p.add_argument("--metadata-csv", type=Path, default=None)
    p.add_argument("--images-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/dinov2_finetune"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--image-size", type=int, default=224, help="Must be divisible by 14 (default 224)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--backbone-lr", type=float, default=1e-5, help="LR of the top backbone block")
    p.add_argument("--head-lr", type=float, default=1e-3, help="LR of the classification head")
    p.add_argument("--layer-decay", type=float, default=0.75, help="Layer-wise LR decay factor (ViT-S)")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma")
    p.add_argument("--patience", type=int, default=10,
                   help="Early stopping patience (on the --best-criterion score)")
    p.add_argument("--best-criterion", choices=["macro_f1", "composite"], default="macro_f1",
                   help="Checkpoint/early-stop selection metric on val: 'macro_f1' (default) "
                        "or 'composite' = 0.5*Macro-F1 + 0.5*MEL_Recall (melanoma-aware)")
    p.add_argument("--checkpoint-name", type=str, default="best_dinov2_finetune.pth",
                   help="Filename for the best checkpoint inside --output-dir")
    p.add_argument("--results-name", type=str, default="dinov2_finetune_results.txt",
                   help="Filename for the report inside --results-dir")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=None,
                   help="Seed for the lesion-level train/val/test split, decoupled "
                        "from --seed (training RNG). Defaults to --seed's value if "
                        "omitted, i.e. identical to old behaviour.")
    p.add_argument("--estimate-only", action="store_true",
                   help="Print GPU memory + time estimate and exit without training")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    # Console may be a non-UTF-8 codepage (e.g. Greek cp1253); never crash on output.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.image_size % 14 != 0:
        raise ValueError(f"--image-size must be divisible by 14 (got {args.image_size})")

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

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Data ----------------------------------------------------------------
    train_loader, val_loader, test_loader, label_map, y_train = make_loaders(
        cfg, args.image_size, args.batch_size, args.num_workers,
        split_seed=args.split_seed,
    )
    num_classes = len(label_map)
    idx_to_class = {v: k for k, v in label_map.items()}
    mel_idx = label_map.get("mel", -1)
    print(f"Classes : {label_map}")
    print(f"Split   : train={len(train_loader.dataset)}  "
          f"val={len(val_loader.dataset)}  test={len(test_loader.dataset)}")

    # --- Model ---------------------------------------------------------------
    print("Loading DINOv2 ViT-S/14 backbone (torch.hub, cached after first run) ...")
    backbone = load_dinov2_backbone()
    feat_dim = backbone.embed_dim if hasattr(backbone, "embed_dim") else 384
    model = DINOv2Classifier(backbone, num_classes, feat_dim).to(device)

    # --- Loss (identical to the frozen linear-probe 'combined' setting) -------
    class_weights = compute_class_weights(y_train, num_classes, device)
    criterion = CombinedLoss(
        FocalLoss(gamma=args.gamma, weight=class_weights),
        DiceLoss(weight=class_weights),
    ).to(device)
    print("Class weights:", {idx_to_class[i]: round(w, 3) for i, w in enumerate(class_weights.cpu().tolist())})

    # --- Optimiser (discriminative LR) + warmup-cosine schedule --------------
    param_groups = build_param_groups(
        model, args.backbone_lr, args.head_lr, args.layer_decay, args.weight_decay
    )
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))
    # Report base LRs BEFORE the scheduler's warmup scales them down.
    lr_span = [g["lr"] for g in param_groups]
    print(f"LR groups : {len(param_groups)}  base-lr min {min(lr_span):.2e} .. max {max(lr_span):.2e}")

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    scheduler = make_warmup_cosine(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    # --- Resource estimate (always shown before the full run) ----------------
    estimate_resources(model, train_loader, criterion, optimizer, scaler, device,
                        args.epochs, use_amp)
    if args.estimate_only:
        print("--estimate-only set -> exiting before training.")
        return

    # --- Training loop with early stopping on val Macro-F1 -------------------
    best_score, best_epoch, epochs_no_improve = -1.0, -1, 0
    best_val_f1, best_val_mel = -1.0, -1.0     # metrics at the selected epoch
    best_path = out_dir / args.checkpoint_name
    global_step = 0
    print(f"Checkpoint criterion : {args.best_criterion}"
          + ("  (0.5*Macro-F1 + 0.5*MEL_Recall)" if args.best_criterion == "composite" else ""))

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n_batches = 0.0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            running += loss.item()
            n_batches += 1

        val_loss, y_true_val, y_pred_val = evaluate(model, val_loader, device, criterion, use_amp)
        val_macro_f1 = f1_score(y_true_val, y_pred_val, average="macro", zero_division=0)
        val_mel_recall = (
            recall_score(y_true_val, y_pred_val, labels=[mel_idx], average="micro", zero_division=0)
            if mel_idx >= 0 else float("nan")
        )
        # Selection score: only the checkpoint/early-stop criterion changes
        # between the two runs — everything else is identical.
        val_composite = 0.5 * val_macro_f1 + 0.5 * val_mel_recall
        sel_score = val_composite if args.best_criterion == "composite" else val_macro_f1

        print(f"Epoch {epoch:3d}  train_loss={running / max(1, n_batches):.4f}  "
              f"val_loss={val_loss:.4f}  val_macro_f1={val_macro_f1:.4f}  "
              f"val_mel_recall={val_mel_recall:.4f}  val_composite={val_composite:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        if sel_score > best_score:
            best_score, best_epoch, epochs_no_improve = sel_score, epoch, 0
            best_val_f1, best_val_mel = val_macro_f1, val_mel_recall
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_map": label_map,
                    "arch": "dinov2_vits14_finetune",
                    "image_size": args.image_size,
                    "feat_dim": feat_dim,
                    "epoch": epoch,
                    "best_criterion": args.best_criterion,
                    "val_macro_f1": val_macro_f1,
                    "val_mel_recall": val_mel_recall,
                    "val_composite": val_composite,
                },
                best_path,
            )
            print(f"  -> New best checkpoint ({args.best_criterion}={best_score:.4f}  "
                  f"val_macro_f1={val_macro_f1:.4f}  val_mel_recall={val_mel_recall:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no val {args.best_criterion} improvement for {args.patience} epochs).")
                break

    print(f"\nTraining complete. Best val {args.best_criterion} = {best_score:.4f} "
          f"(epoch {best_epoch};  val_macro_f1={best_val_f1:.4f}  val_mel_recall={best_val_mel:.4f}).")

    # --- Test evaluation on the best checkpoint ------------------------------
    print("\n" + "=" * 64 + "\nTEST SET EVALUATION\n" + "=" * 64)
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _, y_true_test, y_pred_test = evaluate(model, test_loader, device, criterion, use_amp)

    report = build_report(args, label_map, y_true_test, y_pred_test,
                          best_epoch, best_val_f1, best_val_mel)
    print("\n" + report)

    out_file = results_dir / args.results_name
    out_file.write_text(report, encoding="utf-8")
    print(f"\nSaved results -> {out_file}")
    print(f"Saved best checkpoint -> {best_path}")


if __name__ == "__main__":
    main()
