"""
Evaluate the dashboard's own inference code (app/main.py::_predict) on the full
HAM10000 test split, and check it against the thesis's Section 6.3 numbers.

Run from the project root. It runs on CPU by default, to mirror the deployed
Space exactly (CUDA uses a different RNG stream for the dropout masks):

    python src/eval_dashboard_protocol.py

Writes:
    results/dashboard_protocol_eval.md          (the report)
    results/dashboard_protocol_per_sample.csv   (one row per test image)
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import inspect
import io
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
SRC = ROOT / "src"
# app/ first: app/model.py and src/model.py are duplicates; main.py imports "model".
sys.path[:0] = [str(APP), str(SRC)]

import main as appmain                     # noqa: E402  (the dashboard backend)
import tta_mc                              # noqa: E402
import train_efficientnet as thesis        # noqa: E402  (the Section 6.3 source)
from config import Config                  # noqa: E402
from dataset import lesion_level_split     # noqa: E402
from helpers import resolve_image_path     # noqa: E402

THESIS = {
    "macro_f1": 0.701, "mel_recall": 0.725, "macro_f1_range": (0.687, 0.701),
    "mean_var": 0.0067, "mean_var_correct": 0.0047, "mean_var_wrong": 0.0139,
    "auroc": 0.81, "ref10": 23.5, "ref20": 45.4,
    "mel_detected": 0.0075, "mel_missed": 0.0102,
    "agree_0032214": -0.153, "agree_correct_range": (0.546, 0.762),
}
CORRECT_GRADCAM = ["ISIC_0025387", "ISIC_0026981", "ISIC_0027169", "ISIC_0027488",
                   "ISIC_0028238", "ISIC_0031201", "ISIC_0032550"]
SAMPLES = ["mel", "bcc", "akiec", "nv", "bkl", "df", "vasc"]


def flag_fires(d: dict) -> bool:
    """Same rule as isThresholdTriggered() in app/index.html."""
    return d["predicted_class"] == "mel" and d["scores"][0]["class"] != "mel"


def referral(unc: np.ndarray, is_err: np.ndarray, pct: int) -> float:
    """Share of all errors among the round(N*pct/100) most uncertain images."""
    k = round(len(unc) * pct / 100.0)
    top = np.argsort(-unc, kind="stable")[:k]
    return 100.0 * is_err[top].sum() / is_err.sum()


def stat_block(unc: np.ndarray, is_err: np.ndarray) -> dict:
    return {
        "mean": unc.mean(),
        "mean_correct": unc[~is_err].mean(),
        "mean_wrong": unc[is_err].mean(),
        "auroc": roc_auc_score(is_err.astype(int), unc),
        "ref10": referral(unc, is_err, 10),
        "ref20": referral(unc, is_err, 20),
    }


def load_old_app():
    """The pre-change backend (git HEAD of app/main.py), for before/after agreement."""
    src = subprocess.run(["git", "show", "HEAD:main.py"], cwd=APP,
                         capture_output=True, check=True).stdout.decode("utf-8")
    path = Path(tempfile.gettempdir()) / "old_app_main_head.py"
    path.write_text(src, encoding="utf-8")
    try:
        spec = importlib.util.spec_from_file_location("old_app_main", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        path.unlink()
    return mod


def reuse_checks(test_paths: list[Path]) -> list[str]:
    lines = []
    same_src = inspect.getsource(tta_mc.build_tta_transforms) == \
        inspect.getsource(thesis.build_tta_transforms)
    lines.append(f"- `build_tta_transforms` source text identical to "
                 f"`src/train_efficientnet.py`: **{same_src}**")
    lines.append(f"- `_NORM` identical: **{tta_mc._NORM == thesis._NORM}**")
    equal = 0
    for p in test_paths:
        img = Image.open(p).convert("RGB")
        torch.manual_seed(42)
        a = torch.stack([t(img) for t in tta_mc.build_tta_transforms(224)])
        torch.manual_seed(42)
        b = torch.stack([t(img) for t in thesis.build_tta_transforms(224)])
        equal += int(torch.equal(a, b))
    lines.append(f"- Views produced by app vs thesis transforms (same seed), "
                 f"bitwise equal: **{equal}/{len(test_paths)}** images")
    return lines


def run(args) -> None:
    cfg = Config()
    images_dir = ROOT / "data"
    df, _ = thesis.load_multiclass_table(
        ROOT / "data" / "HAM10000_metadata.csv",
        cfg.col_image_id, cfg.col_lesion_id, cfg.col_diagnosis)
    _, _, df_test = lesion_level_split(
        df, cfg.col_lesion_id, cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, cfg.seed)
    df_test = df_test.reset_index(drop=True)
    if args.limit:
        df_test = df_test.head(args.limit)

    def predict_file(p: Path) -> tuple[dict, float]:
        data = p.read_bytes()
        t0 = time.perf_counter()
        d = appmain._predict(data)
        return d, time.perf_counter() - t0

    # ── full test set ─────────────────────────────────────────────────────────
    rows, lat = [], []
    for i, r in df_test.iterrows():
        image_id = r[cfg.col_image_id]
        d, dt = predict_file(resolve_image_path(images_dir, image_id))
        lat.append(dt)
        rows.append({
            "image_id": image_id,
            "y_true": r[cfg.col_diagnosis],
            "eff_pred": d["eff_predicted_class"],
            "eff_conf": d["eff_confidence"],
            "eff_std_pred_class": d["eff_confidence_std"],
            "eff_mean_variance": d["eff_mean_variance"],
            "primary_pred": d["predicted_class"],
            "flag": flag_fires(d),
            "agreement_r": d["agreement_score"],
        })
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(df_test)}  median {np.median(lat):.2f}s/img", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results" / "dashboard_protocol_per_sample.csv", index=False)

    y, yp = res["y_true"].to_numpy(), res["eff_pred"].to_numpy()
    is_err = y != yp
    macro_f1 = f1_score(y, yp, average="macro")
    mel_rec = recall_score(y == "mel", yp == "mel")
    mv = stat_block(res["eff_mean_variance"].to_numpy(), is_err)
    sd = stat_block(res["eff_std_pred_class"].to_numpy(), is_err)
    mel = y == "mel"
    det, miss = mel & (yp == "mel"), mel & (yp != "mel")

    # ── comparison with the thesis's per-sample CSV ─────────────────────────
    ref = pd.read_csv(ROOT / "results" / "mc_uncertainty_per_sample.csv")
    m = res.merge(ref, on="image_id", suffixes=("", "_thesis"))
    pred_agree = (m["eff_pred"] == m["y_pred"]).mean()
    rho = spearmanr(m["eff_mean_variance"], m["mc_uncertainty"]).correlation
    pr = pearsonr(m["eff_mean_variance"], m["mc_uncertainty"])[0]

    # ── specific images ───────────────────────────────────────────────────────
    old = load_old_app()
    old.MODEL, old.DINOV2_MODEL = appmain.MODEL, appmain.DINOV2_MODEL
    old.DEVICE, old.CLASS_ORDER, old.MEL_IDX = appmain.DEVICE, appmain.CLASS_ORDER, appmain.MEL_IDX
    old._DINOV2_ATTN = appmain._DINOV2_ATTN      # the attention hook writes to the new module's dict

    agree_rows = []
    for image_id in ["ISIC_0032214"] + CORRECT_GRADCAM:
        data = resolve_image_path(images_dir, image_id).read_bytes()
        new = appmain._predict(data)
        before = old._predict(data)
        agree_rows.append((image_id, before["eff_predicted_class"], before["agreement_score"],
                           new["eff_predicted_class"], new["agreement_score"]))

    samples_dir = APP / "static" / "samples"
    sample_rows = []
    for c in SAMPLES:
        d = appmain._predict((samples_dir / f"sample_{c}.jpg").read_bytes())
        sample_rows.append((c, d["predicted_class"], d["confidence"], d["eff_predicted_class"],
                            d["eff_confidence"], d["eff_confidence_std"], flag_fires(d)))

    challenge = {}
    for name in ["challenge_ISIC_0030443.jpg", "challenge_ISIC_0032214.jpg"]:
        data = (samples_dir / name).read_bytes()
        d1, d2 = appmain._predict(data), appmain._predict(data)
        challenge[name] = (d1, json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True))

    reuse = reuse_checks([resolve_image_path(images_dir, i) for i in df_test["image_id"].head(20)])

    # ── report ────────────────────────────────────────────────────────────────
    T = THESIS
    lo, hi = T["macro_f1_range"]
    in_range = lo <= round(macro_f1, 3) <= hi
    L = []
    L += ["# Dashboard uncertainty protocol: evaluation on the test set", ""]
    L += [f"Produced by `src/eval_dashboard_protocol.py`, calling the dashboard's own "
          f"`app/main.py::_predict()` on every test image (device: "
          f"`{appmain.DEVICE}`, as on the Space). Protocol: "
          f"{tta_mc.TTA_VIEWS} TTA views × {tta_mc.PASSES_PER_VIEW} MC-Dropout passes = "
          f"{tta_mc.MC_SAMPLES} samples, seed {tta_mc.MC_SEED} per request. "
          f"Test split: lesion-level, seed {cfg.seed}, **{len(res)} images** "
          f"({int(mel.sum())} melanomas, {int(is_err.sum())} EfficientNet errors).", ""]
    L += ["## Reuse of the thesis code", ""] + reuse + [
          f"- EfficientNet prediction agrees with the thesis's own run "
          f"(`results/mc_uncertainty_per_sample.csv`) on **{100 * pred_agree:.1f}%** of images; "
          f"per-image mean variance correlates with the thesis values at Spearman ρ = "
          f"**{rho:.3f}** (Pearson r = {pr:.3f}). Differences are MC sampling noise: the "
          f"thesis run was unseeded.", ""]
    L += ["## (a) Classification (EfficientNet, argmax of the mean of 50 samples)", "",
          "| Metric | App | Thesis |", "|---|---|---|",
          f"| macro-F1 | **{macro_f1:.3f}** | {T['macro_f1']:.3f} (re-runs {lo:.3f}–{hi:.3f}) |",
          f"| MEL Recall | **{mel_rec:.3f}** ({int(det.sum())}/{int(mel.sum())}) | {T['mel_recall']:.3f} |",
          "", f"Within the thesis re-run range: **{'yes' if in_range else 'NO'}**", ""]

    thesis_col = [f"{T['mean_var']:.4f}", f"{T['mean_var_correct']:.4f}",
                  f"{T['mean_var_wrong']:.4f}",
                  f"{T['mean_var_wrong'] / T['mean_var_correct']:.2f}×",
                  f"{T['auroc']:.2f}", f"{T['ref10']:.1f}%", f"{T['ref20']:.1f}%"]

    def stat_table(s: dict, ref: list[str]) -> list[str]:
        app_col = [f"{s['mean']:.4f}", f"{s['mean_correct']:.4f}", f"{s['mean_wrong']:.4f}",
                   f"{s['mean_wrong'] / s['mean_correct']:.2f}×", f"{s['auroc']:.3f}",
                   f"{s['ref10']:.1f}%", f"{s['ref20']:.1f}%"]
        names = ["Mean, all images", "Mean, correct predictions", "Mean, incorrect predictions",
                 "Ratio incorrect / correct", "AUROC for predicting an error",
                 "Errors captured, top 10% referred", "Errors captured, top 20% referred"]
        return (["| Quantity | App | Thesis |", "|---|---|---|"]
                + [f"| {n} | {a} | {r} |" for n, a, r in zip(names, app_col, ref)])

    L += ["## (b) Section 6.3 statistic: mean variance", ""] + stat_table(mv, thesis_col) + [""]
    L += ["## (c) Displayed statistic: std of the predicted class's probability", "",
          "This is the ± users see (0.052 is shown as ± 5.2%). The thesis did not analyse "
          "this statistic, so there is no thesis value.", ""] + stat_table(sd, ["–"] * 7) + [""]
    L += ["## (d) Melanomas: correctly detected vs missed", "",
          "| Statistic | Detected (n={}) | Missed (n={}) | Ratio | Thesis |".format(int(det.sum()), int(miss.sum())),
          "|---|---|---|---|---|",
          f"| Mean variance | {res.loc[det, 'eff_mean_variance'].mean():.4f} | "
          f"{res.loc[miss, 'eff_mean_variance'].mean():.4f} | "
          f"{res.loc[miss, 'eff_mean_variance'].mean() / res.loc[det, 'eff_mean_variance'].mean():.2f}× | "
          f"{T['mel_detected']:.4f} / {T['mel_missed']:.4f} ({T['mel_missed'] / T['mel_detected']:.2f}×) |",
          f"| Displayed std | {res.loc[det, 'eff_std_pred_class'].mean():.4f} | "
          f"{res.loc[miss, 'eff_std_pred_class'].mean():.4f} | "
          f"{res.loc[miss, 'eff_std_pred_class'].mean() / res.loc[det, 'eff_std_pred_class'].mean():.2f}× | – |", ""]
    L += ["## (e) Agreement score (Grad-CAM vs DINOv2 attention)", "",
          "Before = the committed backend (git HEAD: 20 passes, single view). Grad-CAM always "
          "targets EfficientNet's predicted class, so r can only change if that class changes.", "",
          "| Image | Before: EffNet class | Before: r | After: EffNet class | After: r |",
          "|---|---|---|---|---|"]
    L += [f"| {i} | {bc} | {br:.3f} | {ac} | {ar:.3f} |" for i, bc, br, ac, ar in agree_rows]
    L += ["", "Thesis Section 7.2 quotes −0.153 for ISIC_0032214 and 0.55–0.76 for the seven "
          "correct examples.", ""]
    L += ["## (f) The seven representative samples", "",
          "| Sample | Primary | Primary confidence | EfficientNet | EfficientNet ± | Flag fires |",
          "|---|---|---|---|---|---|"]
    L += [f"| sample_{c}.jpg | {p} | {100 * pc:.1f}% | {e} | {100 * ec:.1f}% ± {100 * es:.1f}% | "
          f"{'yes' if f else 'no'} |" for c, p, pc, e, ec, es, f in sample_rows]
    L += ["", "## (g) The two challenging samples", ""]
    for name, (d, same) in challenge.items():
        L += [f"- `{name}`: primary **{d['predicted_class']}**, flag fires: "
              f"**{'yes' if flag_fires(d) else 'no'}**, EfficientNet {d['eff_predicted_class']} "
              f"{100 * d['eff_confidence']:.1f}% ± {100 * d['eff_confidence_std']:.1f}%, "
              f"agreement r = {d['agreement_score']:.3f} (\"{d['agreement_label']}\"), "
              f"mean variance {d['eff_mean_variance']:.4f}. Two identical requests gave "
              f"byte-identical JSON: **{same}**"]
    L += ["", "## Local latency (not the Space)", "",
          f"`_predict()` on this machine's CPU ({torch.get_num_threads()} threads): median "
          f"{np.median(lat):.2f} s, 90th percentile {np.percentile(lat, 90):.2f} s per image. "
          "The Space has different hardware; its latency is measured separately after deployment.", ""]
    out = ROOT / "results" / "dashboard_protocol_eval.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out}")
    print(f"macro-F1={macro_f1:.4f} MEL recall={mel_rec:.4f} in_range={in_range}")


async def main_async(args) -> None:
    # Pin the device the app's own startup code will pick (the Space is CPU-only).
    appmain.get_device = lambda: torch.device(args.device)
    async with appmain.lifespan(appmain.app):
        run(args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="smoke test on the first N test images")
    ap.add_argument("--device", default="cpu",
                    help="cpu (default) mirrors the deployed Space, incl. its dropout RNG stream")
    asyncio.run(main_async(ap.parse_args()))
