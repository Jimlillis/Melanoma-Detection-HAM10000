"""
Part B: extra metrics for configurations (a)-(e) on OUR lesion-level test split
(1502 images, 167 melanomas). Inference only; see src/sota_eval_common.py.

The DINOv2 probabilities come from the same DataLoader the thesis used
(train_dinov2_finetune.make_loaders, test split, shuffle=False); the
EfficientNet probabilities from the same per-image CPU loop as the dashboard
evaluation. A sanity gate checks that macro-F1 and MEL Recall reproduce the
thesis values at the 3rd decimal; a row that fails is excluded from Part B
and Part C.

Outputs:
  outputs/sota_eval/ours_probs.npz          cached probabilities (git-ignored)
  outputs/sota_eval/ours_metrics.json       metrics, read by write_sota_report.py
  results/sota_per_sample_ours.csv          per-sample probabilities and predictions

Run from the project root:
  python src/eval_sota_metrics.py               # uses the cache if present
  python src/eval_sota_metrics.py --recompute
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from sota_eval_common import (CLASSES, CONFIGS, ROOT, always_nv, dinov2_probs_from_loader,
                              effnet_probs_from_paths, load_dinov2, load_effnet, metrics,
                              per_sample_frame, predictions)
from config import Config
from helpers import get_device, resolve_image_path, set_seed
import train_dinov2_finetune as tdf

OUT = ROOT / "outputs" / "sota_eval"
EXPECTED = {"a": (0.695, 0.814), "b": (0.744, 0.569), "c": (0.693, 0.725), "d": (0.664, 0.814), "e": (0.690, 0.719)}


def to_jsonable(m: dict) -> dict:
    out = {}
    for k, v in m.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = {kk: float(vv) for kk, vv in v.items()}
        else:
            out[k] = float(v) if isinstance(v, (np.floating, float)) else v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "ours_probs.npz"

    cfg = Config()
    cfg.metadata_csv = ROOT / "data" / "HAM10000_metadata.csv"
    cfg.images_dir = ROOT / "data"
    df, label_map = tdf.load_multiclass_table(cfg.metadata_csv, cfg.col_image_id, cfg.col_lesion_id, cfg.col_diagnosis)
    assert [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])] == CLASSES
    _, _, df_test = tdf.lesion_level_split(df, cfg.col_lesion_id, cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, cfg.seed)
    df_test = df_test.reset_index(drop=True)
    ids = df_test[cfg.col_image_id].tolist()
    y = df_test["_binary_label"].to_numpy()

    if cache.exists() and not args.recompute:
        z = np.load(cache, allow_pickle=True)
        assert list(z["image_id"]) == ids
        P_dino, P_tm, P_plain = z["P_dino"], z["P_tm"], z["P_plain"]
        print("loaded cached probabilities from", cache)
    else:
        set_seed(cfg.seed)
        device = get_device()
        t0 = time.time()
        _, _, test_loader, lm2, _ = tdf.make_loaders(cfg, 224, 64, 0)
        assert lm2 == label_map
        P_dino = dinov2_probs_from_loader(load_dinov2(device), test_loader, device)
        print(f"  DINOv2 done ({time.time() - t0:.0f}s, device={device})", flush=True)
        paths = [resolve_image_path(cfg.images_dir, i) for i in ids]
        P_plain, P_tm = effnet_probs_from_paths(load_effnet(), paths)
        print(f"  EfficientNet done ({time.time() - t0:.0f}s)", flush=True)
        np.savez(cache, image_id=np.array(ids), y=y, P_dino=P_dino, P_tm=P_tm, P_plain=P_plain)

    preds = predictions(P_dino, P_tm, P_plain)
    res = {"gate": {}, "metrics": {}, "always_nv": to_jsonable(always_nv(y)), "n": int(len(y)),
           "class_counts": {c: int((y == i).sum()) for i, c in enumerate(CLASSES)}}
    for k, (pred, P) in preds.items():
        m = metrics(y, pred, P)
        got = (round(m["macro_f1"], 3), round(m["mel_recall"], 3))
        ok = got == EXPECTED[k]
        res["gate"][k] = {"expected": EXPECTED[k], "got": got, "got_unrounded": (m["macro_f1"], m["mel_recall"]), "pass": ok}
        print(f"  gate {k} {CONFIGS[k]}: expected {EXPECTED[k]} got {got} -> {'PASS' if ok else 'FAIL'}")
        if ok:
            res["metrics"][k] = to_jsonable(m)
    (OUT / "ours_metrics.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    per_sample_frame(ids, y, P_dino, P_tm, P_plain, preds).to_csv(ROOT / "results" / "sota_per_sample_ours.csv", index=False)
    print("wrote", OUT / "ours_metrics.json", "and results/sota_per_sample_ours.csv")


if __name__ == "__main__":
    main()
