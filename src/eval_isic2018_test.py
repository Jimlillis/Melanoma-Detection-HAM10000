"""
Part C: inference of configurations (a)-(e) on the official ISIC 2018 Task 3
test set (1512 images). Inference only: thresholds t = 0.003 (DINOv2) and
t = 0.185 (EfficientNet) stay as chosen on our validation set; MC Dropout seed 42.
Preprocessing is identical to our test pipeline (src/sota_eval_common.py).

Inputs (data/ is git-ignored):
  data/isic2018_test/ISIC2018_Task3_Test_Input/*.jpg
  data/isic2018_test/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv
      official challenge ground truth (one-hot MEL, NV, BCC, AKIEC, BKL, DF, VASC)
  data/isic2018_test/ISIC2018_Task3_Test_GroundTruth_dataverse.orig
      HAM10000 Dataverse release (doi:10.7910/DVN/DBW86T, file id 6924466):
      lesion_id, dx and the `dataset` column (source; 'external' marks the 316
      external images)

Outputs:
  outputs/sota_eval/isic_probs.npz       cached probabilities (git-ignored)
  outputs/sota_eval/isic_metrics.json    metrics + integrity checks
  results/sota_per_sample_isic2018.csv   per-sample probabilities and predictions

Run from the project root:
  python src/eval_isic2018_test.py               # uses the cache if present
  python src/eval_isic2018_test.py --recompute
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from sota_eval_common import (CLASSES, ROOT, always_nv, dinov2_probs_from_paths,
                              effnet_probs_from_paths, load_dinov2, load_effnet, metrics,
                              per_sample_frame, predictions)
from eval_sota_metrics import to_jsonable
from config import Config
from dataset import lesion_level_split
from extract_features_v2 import load_multiclass_table
from helpers import get_device, set_seed

D = ROOT / "data" / "isic2018_test"
IMG_DIR = D / "ISIC2018_Task3_Test_Input"
GT_CSV = D / "ISIC2018_Task3_Test_GroundTruth" / "ISIC2018_Task3_Test_GroundTruth.csv"
DV_CSV = D / "ISIC2018_Task3_Test_GroundTruth_dataverse.orig"
OUT = ROOT / "outputs" / "sota_eval"
EASTER_EGG = "ISIC_0035068"


def integrity(gt: pd.DataFrame, dv: pd.DataFrame) -> dict:
    cfg = Config()
    ham, _ = load_multiclass_table(ROOT / "data" / "HAM10000_metadata.csv", cfg.col_image_id,
                                   cfg.col_lesion_id, cfg.col_diagnosis)
    tr, va, te = lesion_level_split(ham, cfg.col_lesion_id, cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, cfg.seed)
    jpgs = sorted(p.stem for p in IMG_DIR.glob("*.jpg"))
    ids = set(gt["image_id"])
    merged = gt.merge(dv, on="image_id", how="outer", suffixes=("", "_dv"), indicator=True)
    r = {
        "n_images": len(jpgs),
        "n_gt_rows": int(len(gt)),
        "gt_ids_unique": int(gt["image_id"].nunique()),
        "images_without_gt": len(set(jpgs) - ids),
        "gt_without_image": len(ids - set(jpgs)),
        "class_counts_gt": {c: int((gt["dx"] == c).sum()) for c in CLASSES},
        "overlap_image_ids_with_HAM10000_all": len(ids & set(ham[cfg.col_image_id])),
        "overlap_image_ids_with_train": len(ids & set(tr[cfg.col_image_id])),
        "overlap_image_ids_with_val": len(ids & set(va[cfg.col_image_id])),
        "overlap_image_ids_with_test": len(ids & set(te[cfg.col_image_id])),
        "dataverse_rows": int(len(dv)),
        "dataverse_vs_official_join": merged["_merge"].value_counts().to_dict(),
        "dataverse_dx_agrees_with_official": int((merged["dx"] == merged["dx_dv"]).sum()),
        "dataverse_lesion_ids": int(dv["lesion_id"].nunique()),
        "dataverse_lesion_id_prefixes": sorted({s.split("_")[0] for s in dv["lesion_id"]}),
        "lesion_id_overlap_with_HAM10000": len(set(dv["lesion_id"]) & set(ham[cfg.col_lesion_id])),
        "dataverse_dataset_counts": {str(k): int(v) for k, v in dv["dataset"].value_counts(dropna=False).items()},
        "easter_egg_present": EASTER_EGG in ids,
        "easter_egg_row": dv[dv["image_id"] == EASTER_EGG].astype(str).to_dict("records"),
    }
    r["merged_dataverse_ok"] = r["dataverse_vs_official_join"].get("both", 0) == len(gt)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "isic_probs.npz"

    raw = pd.read_csv(GT_CSV)
    cols = {"AKIEC": "akiec", "BCC": "bcc", "BKL": "bkl", "DF": "df", "MEL": "mel", "NV": "nv", "VASC": "vasc"}
    assert (raw[list(cols)].sum(axis=1) == 1).all(), "ground truth is not one-hot"
    gt = pd.DataFrame({"image_id": raw["image"], "dx": raw[list(cols)].idxmax(axis=1).map(cols)})
    dv = pd.read_csv(DV_CSV)
    checks = integrity(gt, dv)
    print(json.dumps({k: v for k, v in checks.items() if k != "easter_egg_row"}, indent=1, default=str))

    ids = gt["image_id"].tolist()
    y = gt["dx"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    paths = [IMG_DIR / f"{i}.jpg" for i in ids]

    if cache.exists() and not args.recompute:
        z = np.load(cache, allow_pickle=True)
        assert list(z["image_id"]) == ids
        P_dino, P_tm, P_plain = z["P_dino"], z["P_tm"], z["P_plain"]
        print("loaded cached probabilities from", cache)
    else:
        set_seed(42)
        device = get_device()
        t0 = time.time()
        P_dino = dinov2_probs_from_paths(load_dinov2(device), paths, device)
        print(f"  DINOv2 done ({time.time() - t0:.0f}s, device={device})", flush=True)
        P_plain, P_tm = effnet_probs_from_paths(load_effnet(), paths)
        print(f"  EfficientNet done ({time.time() - t0:.0f}s)", flush=True)
        np.savez(cache, image_id=np.array(ids), y=y, P_dino=P_dino, P_tm=P_tm, P_plain=P_plain)

    preds = predictions(P_dino, P_tm, P_plain)
    ds = gt.merge(dv[["image_id", "dataset"]], on="image_id", how="left")["dataset"]
    subsets = {
        "all_1512": np.ones(len(ids), bool),
        "without_easter_egg_1511": np.array([i != EASTER_EGG for i in ids]),
        "internal_1196": (ds != "external").to_numpy(),
        "external_316": (ds == "external").to_numpy(),
    }
    res = {"integrity": checks, "subsets": {}}
    for name, mask in subsets.items():
        ys = y[mask]
        block = {"n": int(mask.sum()), "class_counts": {c: int((ys == i).sum()) for i, c in enumerate(CLASSES)},
                 "always_nv": to_jsonable(always_nv(ys)), "metrics": {}}
        for k, (pred, P) in preds.items():
            block["metrics"][k] = to_jsonable(metrics(ys, pred[mask], P[mask]))
        res["subsets"][name] = block
        print(name, block["n"], {k: round(v["bacc"], 3) for k, v in block["metrics"].items()})
    # how the two thresholds behave on this set: share of images flagged as mel
    res["mel_flag_rate"] = {k: float((pred == CLASSES.index("mel")).mean()) for k, (pred, _) in preds.items()}
    (OUT / "isic_metrics.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    df = per_sample_frame(ids, y, P_dino, P_tm, P_plain, preds)
    df.insert(2, "dataset_dataverse", ds.fillna("").to_numpy())
    df.to_csv(ROOT / "results" / "sota_per_sample_isic2018.csv", index=False)
    print("wrote", OUT / "isic_metrics.json", "and results/sota_per_sample_isic2018.csv")


if __name__ == "__main__":
    main()
