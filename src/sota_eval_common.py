"""
Shared inference + metrics for the state-of-the-art comparison (Chapter 8).

Configurations (all inference only, existing checkpoints, no re-tuning):
  (a) DINOv2 ViT-S/14 full fine-tune, rule t = 0.003 (final model)
  (b) DINOv2 ViT-S/14 full fine-tune, argmax
  (c) EfficientNet-B0, 5-view TTA + MC Dropout (10 passes/view, seed 42), argmax
  (d) EfficientNet-B0, same protocol, rule t = 0.185
  (e) EfficientNet-B0, plain forward (eval mode, no TTA/MC), argmax
Rule: predict mel if p(mel) > t, otherwise argmax over the other six classes
(src/dinov2_finetune_threshold_tune.py::apply_mel_threshold).

Preprocessing is the thesis test pipeline: Resize((224, 224)) -> ToTensor ->
ImageNet Normalize (src/dataset.py::build_transforms(224, is_train=False)).
  * DINOv2: GPU with autocast when CUDA is available, batch 64, exactly as
    src/dinov2_finetune_threshold_tune.py::predict_probs.
  * EfficientNet: CPU. Plain = view 0 of the TTA set (the same transform).
    TTA+MC = app/tta_mc.py::tta_mc_predict, seeded with 42 per image, i.e.
    the protocol whose test result (0.693 / 0.725) the thesis reports.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support, roc_auc_score)
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset import build_transforms                                   # noqa: E402
from dinov2_finetune_threshold_tune import apply_mel_threshold         # noqa: E402
from model import build_efficientnet_b0                                # noqa: E402
from train_dinov2_finetune import DINOv2Classifier, load_dinov2_backbone  # noqa: E402

# app/tta_mc.py is loaded by path: app/ and src/ both contain a module named "model".
_spec = importlib.util.spec_from_file_location("tta_mc", ROOT / "app" / "tta_mc.py")
tta_mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tta_mc)

DINOV2_CKPT = ROOT / "outputs" / "dinov2_finetune" / "best_dinov2_finetune.pth"
EFFNET_CKPT = ROOT / "outputs" / "efficientnet_mc" / "best_efficientnet.pth"
T_DINO = 0.003
T_EFF = 0.185
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MEL = CLASSES.index("mel")
NV = CLASSES.index("nv")
CONFIGS = {
    "a": "DINOv2 FT, t = 0.003 (final)",
    "b": "DINOv2 FT, argmax",
    "c": "EfficientNet-B0, TTA+MC, argmax",
    "d": "EfficientNet-B0, TTA+MC, t = 0.185",
    "e": "EfficientNet-B0, plain, argmax",
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def load_dinov2(device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(DINOV2_CKPT, map_location=device, weights_only=False)
    assert [k for k, _ in sorted(ckpt["label_map"].items(), key=lambda kv: kv[1])] == CLASSES
    model = DINOv2Classifier(load_dinov2_backbone(), len(CLASSES), ckpt.get("feat_dim", 384)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval()


def load_effnet() -> torch.nn.Module:
    ckpt = torch.load(EFFNET_CKPT, map_location="cpu", weights_only=False)
    assert [k for k, _ in sorted(ckpt["label_map"].items(), key=lambda kv: kv[1])] == CLASSES
    model = build_efficientnet_b0(num_classes=len(CLASSES), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class PathDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = list(paths)
        self.tf = build_transforms(224, is_train=False)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), 0


@torch.no_grad()
def dinov2_probs_from_loader(model, loader, device) -> np.ndarray:
    """Same computation as dinov2_finetune_threshold_tune.predict_probs."""
    use_amp = device.type == "cuda"
    out = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
        out.append(F.softmax(logits.float(), dim=1).cpu().numpy())
    return np.vstack(out)


def dinov2_probs_from_paths(model, paths, device) -> np.ndarray:
    loader = DataLoader(PathDataset(paths), batch_size=64, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    return dinov2_probs_from_loader(model, loader, device)


@torch.no_grad()
def effnet_probs_from_paths(model, paths, log_every: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Returns (plain, tta_mc) probabilities, both (N, 7), computed on CPU."""
    device = torch.device("cpu")
    plain_tf = tta_mc.build_tta_transforms(224)[0]
    plain, tm = [], []
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        model.eval()
        plain.append(torch.softmax(model(plain_tf(img).unsqueeze(0)), 1)[0].numpy())
        tm.append(tta_mc.tta_mc_predict(model, img, device)["avg_probs"].numpy())
        if log_every and (i + 1) % log_every == 0:
            print(f"    EfficientNet {i + 1}/{len(paths)}", flush=True)
    return np.stack(plain), np.stack(tm)


def predictions(P_dino: np.ndarray, P_tm: np.ndarray, P_plain: np.ndarray) -> dict:
    """Predicted labels and the probability matrix behind each configuration."""
    return {
        "a": (apply_mel_threshold(P_dino, MEL, T_DINO), P_dino),
        "b": (P_dino.argmax(1), P_dino),
        "c": (P_tm.argmax(1), P_tm),
        "d": (apply_mel_threshold(P_tm, MEL, T_EFF), P_tm),
        "e": (P_plain.argmax(1), P_plain),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(y: np.ndarray, pred: np.ndarray, P: np.ndarray | None) -> dict:
    labels = list(range(len(CLASSES)))
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    mel_true, mel_pred = y == MEL, pred == MEL
    tn = int((~mel_true & ~mel_pred).sum())
    fp = int((~mel_true & mel_pred).sum())
    m = {
        "n": int(len(y)),
        "acc": accuracy_score(y, pred),
        "bacc": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, labels=labels, average="macro", zero_division=0),
        "mel_recall": rc[MEL],
        "mel_precision": pr[MEL],
        "mel_specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "per_class_recall": dict(zip(CLASSES, rc)),
        "per_class_precision": dict(zip(CLASSES, pr)),
        "per_class_f1": dict(zip(CLASSES, f1)),
        "support": dict(zip(CLASSES, sup.tolist())),
        "cm": confusion_matrix(y, pred, labels=labels),
        "mel_auroc": float("nan"),
        "macro_auroc": float("nan"),
    }
    if P is not None:
        m["mel_auroc"] = roc_auc_score(mel_true.astype(int), P[:, MEL])
        present = sorted(set(y.tolist()))
        if len(present) == len(CLASSES):
            m["macro_auroc"] = roc_auc_score(y, P, multi_class="ovr", average="macro", labels=labels)
    return m


def always_nv(y: np.ndarray) -> dict:
    return metrics(y, np.full_like(y, NV), None)


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

def f3(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def summary_table(rows: list[tuple[str, dict]]) -> list[str]:
    L = ["| Configuration | n | Acc | BACC | macro-F1 | MEL Recall | MEL Precision | MEL Specificity | MEL AUROC | macro AUROC |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in rows:
        L.append(f"| {name} | {m['n']} | {f3(m['acc'])} | {f3(m['bacc'])} | {f3(m['macro_f1'])} | {f3(m['mel_recall'])} | "
                 f"{f3(m['mel_precision'])} | {f3(m['mel_specificity'])} | {f3(m['mel_auroc'])} | {f3(m['macro_auroc'])} |")
    return L


def per_class_table(rows: list[tuple[str, dict]], key: str) -> list[str]:
    L = ["| Configuration | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    for name, m in rows:
        L.append(f"| {name} | " + " | ".join(f3(m[key][c]) for c in CLASSES) + " |")
    return L


def cm_table(cm: np.ndarray) -> list[str]:
    L = ["| true \\ pred | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    for i, c in enumerate(CLASSES):
        L.append(f"| **{c}** | " + " | ".join(str(int(v)) for v in cm[i]) + " |")
    return L


def per_sample_frame(image_ids, y, P_dino, P_tm, P_plain, preds) -> "pd.DataFrame":
    import pandas as pd
    d = {"image_id": list(image_ids), "y_true": [CLASSES[i] for i in y]}
    for tag, P in (("dino", P_dino), ("eff_ttamc", P_tm), ("eff_plain", P_plain)):
        for j, c in enumerate(CLASSES):
            d[f"p_{tag}_{c}"] = np.round(P[:, j], 8)
    for k, (pred, _) in preds.items():
        d[f"pred_{k}"] = [CLASSES[i] for i in pred]
    return pd.DataFrame(d)
