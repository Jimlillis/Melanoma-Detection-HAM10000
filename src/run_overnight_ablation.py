"""
Overnight run: (A) loss ablation on the frozen linear probes, and
(B) seed-variance estimate for EfficientNet-B0.

PART A -- 4 cached feature sets x 3 losses = 12 runs of train_dino_head.py,
with --loss and --seed passed EXPLICITLY on every run (never relying on the
default; settling which loss each probe actually used is the point).
DINOv1 is deliberately not re-run: all three of its losses are already on
record in results/dinov1_probe_*_results.txt.

PART B -- 3 runs of train_efficientnet.py, weighted CE, seeds 42/1337/2024,
each evaluated under both protocols. Seed 42 doubles as a determinism check
against the earlier loss-ablation ce run.

Robustness (unattended):
  - every training and every evaluation is its own subprocess, so one crash
    (or a CUDA OOM) cannot take down the rest of the job;
  - the manifest is written to disk after EVERY individual step, never
    batched to the end;
  - any failure is caught, appended with its traceback to
    results/overnight_errors.log, and the driver continues to the next run;
  - already-completed runs are skipped on re-invocation (--force to redo),
    so an interrupted job resumes instead of restarting.

Run from project root:
  python src/run_overnight_ablation.py                 # part A then part B
  python src/run_overnight_ablation.py --part a
  python src/run_overnight_ablation.py --part b
  python src/run_overnight_ablation.py --summarize-only
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

NL = chr(10)

_ROOT = Path(__file__).resolve().parent.parent
_SRC = Path(__file__).resolve().parent

RESULTS_DIR = _ROOT / "results"
PROBE_OUT_ROOT = _ROOT / "outputs" / "probe_loss_ablation"
EFFNET_OUT_ROOT = _ROOT / "outputs" / "effnet_seed_variance"
MANIFEST_PATH = _ROOT / "outputs" / "overnight_manifest.json"
ERROR_LOG = RESULTS_DIR / "overnight_errors.log"
SUMMARY_PATH = RESULTS_DIR / "overnight_summary.md"

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# (name, features dir, filename prefix)
FEATURE_SETS = [
    ("dinov2", "data/processed/dinov2", "dinov2_feats"),
    ("dinov2_reg", "data/processed/dinov2_reg", "dinov2_reg_feats"),
    ("dinov3", "data/processed/dinov3", "dinov3_feats"),
    ("biomedclip", "data/processed/biomedclip", "biomedclip_feats"),
]
LOSSES = ["focal", "dice", "combined"]
SEEDS = [42, 1337, 2024]
# The shared lesion-level split used everywhere else in this repo. Part B
# varies --seed (training RNG) across SEEDS but must pin the train/val/test
# partition itself to this value -- see train_efficientnet.py's --split-seed.
SPLIT_SEED = 42

DEGENERATE_NV_RECALL = 0.10

DEFAULT_METADATA_CSV = _ROOT / "data" / "HAM10000_metadata.csv"
DEFAULT_IMAGES_DIR = _ROOT / "data"

# Already-recorded DINOv1 probe results -- reused, not re-run.
# loss -> (macro_f1, mel_recall, nv_recall)
DINOV1_RECORDED = {
    "focal": (0.526, 0.623, 0.66),
    "dice": (0.436, 0.814, 0.00),
    "combined": (0.554, 0.629, 0.68),
}
DINOV1_REPORTS = {
    "focal": RESULTS_DIR / "dinov1_probe_focal_results.txt",
    "dice": RESULTS_DIR / "dinov1_probe_dice_results.txt",
    "combined": RESULTS_DIR / "dinov1_probe_combined_results.txt",
}

# Recorded numbers whose originating loss is disputed (Progress-main.docx says
# Combined; Final Results.txt records "(default)" = focal).
RECORDED_PROBE = {
    "dinov2": (0.545, 0.689),
    "dinov2_reg": (0.463, 0.617),
    "dinov3": (0.4831, 0.6228),  # CLS-token features; was mean-pooled 0.3797 / 0.4970
    "biomedclip": (0.420, 0.509),
}

PRIOR_EFFNET_MANIFEST = _ROOT / "outputs" / "effnet_loss_ablation" / "manifest.json"


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

def log_error(label: str, step: str, exc: BaseException) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().isoformat(timespec="seconds")
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(NL + ("=" * 72) + NL)
        f.write("[" + ts + "] run=" + label + "  step=" + step + NL)
        f.write(("=" * 72) + NL)
        f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        f.write(NL)
    print("  !! FAILED (" + step + ") -- logged to " + str(ERROR_LOG), flush=True)


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        m = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    else:
        m = {}
    m.setdefault("part_a", {})
    m.setdefault("part_b", {})
    return m


def save_manifest(m: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")


def run_subprocess(cmd: list, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(str(c) for c in cmd)
    print("  $ " + printable, flush=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run([str(c) for c in cmd], cwd=str(_ROOT),
                              stdout=logf, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        tail = ""
        try:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        except OSError:
            pass
        raise RuntimeError("exit " + str(proc.returncode) + " from: " + printable + NL
                           + "--- tail of " + str(log_path) + " ---" + NL + tail)


# ---------------------------------------------------------------------------
# PART A -- probe loss ablation
# ---------------------------------------------------------------------------

def part_a_train(fs_name, feats_dir, prefix, loss):
    run_dir = PROBE_OUT_ROOT / (fs_name + "_" + loss)
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, _SRC / "train_dino_head.py",
           "--features-dir", _ROOT / feats_dir,
           "--features-prefix", prefix,
           "--output-dir", run_dir,
           "--loss", loss,
           "--seed", "42"]
    t0 = time.time()
    run_subprocess(cmd, run_dir / "train_stdout.log")
    wall = time.time() - t0
    ckpt = run_dir / "best_dino_head.pth"
    if not ckpt.exists():
        raise RuntimeError("training finished but no checkpoint at " + str(ckpt))
    import torch
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    return {"feature_set": fs_name, "loss": loss, "ckpt_path": str(ckpt),
            "epoch": c.get("epoch"), "seed": c.get("seed"),
            "loss_name": c.get("loss_name"),
            "val_macro_f1": c.get("val_macro_f1"),
            "val_mel_recall": c.get("val_mel_recall"),
            "wall_clock_seconds_to_best": c.get("wall_clock_seconds"),
            "wall_clock_seconds_total": wall,
            "training_log_csv": str(run_dir / "training_log.csv")}


def part_a_eval(fs_name, feats_dir, prefix, loss, ckpt_path):
    run_dir = PROBE_OUT_ROOT / (fs_name + "_" + loss)
    report_name = "probe_loss_" + fs_name + "_" + loss
    json_path = run_dir / (report_name + ".json")
    cmd = [sys.executable, _SRC / "eval_saved_models.py",
           "--only", "probes",
           "--results-dir", RESULTS_DIR,
           "--seed", "42",
           "--probe-ckpt", ckpt_path,
           "--probe-feats", _ROOT / feats_dir / (prefix + "_test.pt"),
           "--probe-report-name", report_name,
           "--dump-json", json_path]
    run_subprocess(cmd, run_dir / "eval_stdout.log")
    if not json_path.exists():
        raise RuntimeError("eval finished but no JSON at " + str(json_path))
    return json.loads(json_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# PART B -- EfficientNet seed variance
# ---------------------------------------------------------------------------

def part_b_train(seed, args):
    run_dir = EFFNET_OUT_ROOT / ("seed" + str(seed))
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, _SRC / "train_efficientnet.py",
           "--metadata-csv", args.metadata_csv,
           "--images-dir", args.images_dir,
           "--output-dir", run_dir,
           "--loss", "ce",
           "--seed", str(seed),
           "--split-seed", str(SPLIT_SEED),
           "--phase1-epochs", str(args.phase1_epochs),
           "--phase2-epochs", str(args.phase2_epochs),
           "--no-mc"]
    t0 = time.time()
    run_subprocess(cmd, run_dir / "train_stdout.log")
    wall = time.time() - t0
    ckpt = run_dir / "best_efficientnet.pth"
    if not ckpt.exists():
        raise RuntimeError("training finished but no checkpoint at " + str(ckpt))
    import torch
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    return {"seed": seed, "ckpt_path": str(ckpt), "epoch": c.get("epoch"),
            "phase": c.get("phase"), "stored_seed": c.get("seed"),
            "stored_split_seed": c.get("split_seed"),
            "val_macro_f1": c.get("val_macro_f1"),
            "wall_clock_seconds_total": wall,
            "training_log_csv": str(run_dir / "training_log.csv")}


def part_b_eval(seed, ckpt_path, mode, args):
    run_dir = EFFNET_OUT_ROOT / ("seed" + str(seed))
    report_name = "effnet_seed" + str(seed) + "_" + mode
    json_path = run_dir / (report_name + ".json")
    cmd = [sys.executable, _SRC / "eval_saved_models.py",
           "--metadata-csv", args.metadata_csv,
           "--images-dir", args.images_dir,
           "--results-dir", RESULTS_DIR,
           "--seed", "42",
           "--only", "effnet",
           "--effnet-mode", mode,
           "--effnet-ckpt", ckpt_path,
           "--effnet-report-name", report_name,
           "--dump-json", json_path]
    if mode == "tta_mc":
        cmd += ["--mc-passes", "10"]
    run_subprocess(cmd, run_dir / ("eval_" + mode + "_stdout.log"))
    if not json_path.exists():
        raise RuntimeError("eval finished but no JSON at " + str(json_path))
    return json.loads(json_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def pred_shares(cm):
    a = np.array(cm)
    total = a.sum()
    cols = a.sum(axis=0)
    return {CLASS_NAMES[j]: float(cols[j]) / float(total) for j in range(len(CLASS_NAMES))}


def parse_recorded_nv_recall(path: Path):
    """Pull nv recall out of an existing per-class report (read-only)."""
    if not path.exists():
        return None
    pattern = re.compile(r"^\s*nv\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9]+)\s*$")
    for line in path.read_text(encoding="utf-8").splitlines():
        m = pattern.match(line)
        if m:
            return float(m.group(2))
    return None


def stats(values):
    a = np.array([v for v in values if v is not None], dtype=float)
    if a.size == 0:
        return None
    return {"mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max()),
            "range": float(a.max() - a.min()), "n": int(a.size)}


def prior_effnet_numbers():
    """ce / dice plain macro-F1 from the earlier loss ablation, if present."""
    if not PRIOR_EFFNET_MANIFEST.exists():
        return {}
    try:
        m = json.loads(PRIOR_EFFNET_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for loss in ("ce", "dice"):
        r = (m.get("runs", {}).get(loss) or {}).get("eval_plain")
        if r:
            out[loss] = (r["macro_f1"], r["mel_recall"])
    return out


def generate_summary(manifest) -> str:
    a_runs = manifest.get("part_a", {})
    b_runs = manifest.get("part_b", {})
    L = []

    L.append("# Overnight Run -- Probe Loss Ablation + EfficientNet Seed Variance")
    L.append("")
    L.append("Generated: " + datetime.now().isoformat(timespec="seconds"))
    L.append("")
    L.append("Part A: 12 linear-probe runs (4 cached feature sets x 3 losses), "
             "`--loss` and `--seed 42` passed explicitly on every run, all other "
             "`train_dino_head.py` hyperparameters at their defaults. DINOv1 rows "
             "are reused from the existing saved reports, not re-run.")
    L.append("")
    L.append("Part B: 3 EfficientNet-B0 runs (weighted CE, seeds 42/1337/2024), "
             "two-phase 15+15, otherwise identical to the earlier loss ablation.")
    L.append("")

    # ---------------- 1. probe ablation table ----------------
    L.append("## 1. Probe ablation table")
    L.append("")
    L.append("| Feature set | Loss | Test Macro-F1 | Test MEL Recall | nv recall | Degenerate |")
    L.append("|---|---|---|---|---|---|")
    for loss in LOSSES:
        mf1, mrec, nv_fallback = DINOV1_RECORDED[loss]
        nv = parse_recorded_nv_recall(DINOV1_REPORTS[loss])
        if nv is None:
            nv = nv_fallback
        deg = "YES" if nv < DEGENERATE_NV_RECALL else "no"
        L.append("| dinov1 (recorded) | " + loss + " | " + "{:.3f}".format(mf1)
                 + " | " + "{:.3f}".format(mrec) + " | " + "{:.2f}".format(nv)
                 + " | " + deg + " |")
    for fs_name, _, _ in FEATURE_SETS:
        for loss in LOSSES:
            rec = a_runs.get(fs_name + "_" + loss) or {}
            ev = rec.get("eval")
            if not ev:
                state = "NOT RUN" if not rec else "FAILED"
                L.append("| " + fs_name + " | " + loss + " | " + state + " | - | - | - |")
                continue
            nv = ev["per_class"]["nv"]["recall"]
            deg = "YES" if nv < DEGENERATE_NV_RECALL else "no"
            L.append("| " + fs_name + " | " + loss + " | "
                     + "{:.4f}".format(ev["macro_f1"]) + " | "
                     + "{:.4f}".format(ev["mel_recall"]) + " | "
                     + "{:.4f}".format(nv) + " | " + deg + " |")
    L.append("")
    L.append("Degenerate = nv recall < " + "{:.2f}".format(DEGENERATE_NV_RECALL)
             + " (the DINOv1+Dice failure mode, where nv recall was exactly 0.00).")
    L.append("")

    L.append("### 1b. Share of all test predictions landing in each class (12 new runs)")
    L.append("")
    L.append("| Feature set | Loss | " + " | ".join(CLASS_NAMES) + " |")
    L.append("|---|---|" + ("---|" * len(CLASS_NAMES)))
    for fs_name, _, _ in FEATURE_SETS:
        for loss in LOSSES:
            ev = (a_runs.get(fs_name + "_" + loss) or {}).get("eval")
            if not ev:
                L.append("| " + fs_name + " | " + loss + " | "
                         + " | ".join(["-"] * len(CLASS_NAMES)) + " |")
                continue
            sh = pred_shares(ev["confusion_matrix"])
            L.append("| " + fs_name + " | " + loss + " | "
                     + " | ".join("{:.1f}%".format(sh[c] * 100.0) for c in CLASS_NAMES) + " |")
    L.append("")
    L.append("Reference: `nv` is 1004/1502 = 66.8% of the test set by true support.")
    L.append("")

    # ---------------- 2. best loss per feature set ----------------
    L.append("## 2. Best loss per feature set")
    L.append("")
    best = {}
    best["dinov1"] = max(LOSSES, key=lambda l: DINOV1_RECORDED[l][0])
    L.append("| Feature set | Best loss (by Macro-F1) | Macro-F1 | Runner-up | Gap |")
    L.append("|---|---|---|---|---|")
    bl = best["dinov1"]
    ordered = sorted(LOSSES, key=lambda l: DINOV1_RECORDED[l][0], reverse=True)
    L.append("| dinov1 (recorded) | " + bl + " | " + "{:.3f}".format(DINOV1_RECORDED[bl][0])
             + " | " + ordered[1] + " | "
             + "{:+.3f}".format(DINOV1_RECORDED[ordered[0]][0] - DINOV1_RECORDED[ordered[1]][0]) + " |")
    for fs_name, _, _ in FEATURE_SETS:
        scored = []
        for loss in LOSSES:
            ev = (a_runs.get(fs_name + "_" + loss) or {}).get("eval")
            if ev:
                scored.append((loss, ev["macro_f1"]))
        if not scored:
            L.append("| " + fs_name + " | NOT AVAILABLE | - | - | - |")
            continue
        scored.sort(key=lambda t: t[1], reverse=True)
        best[fs_name] = scored[0][0]
        runner = scored[1][0] if len(scored) > 1 else "-"
        gap = (scored[0][1] - scored[1][1]) if len(scored) > 1 else 0.0
        L.append("| " + fs_name + " | " + scored[0][0] + " | " + "{:.4f}".format(scored[0][1])
                 + " | " + runner + " | " + "{:+.4f}".format(gap) + " |")
    L.append("")
    distinct = sorted(set(best.values()))
    if len(distinct) == 1:
        L.append("**The best loss is the same for all " + str(len(best))
                 + " feature sets: `" + distinct[0] + "`.** Cross-version probe "
                 "comparisons in the thesis are therefore not confounded by loss "
                 "choice -- every version is being compared at its own best loss, "
                 "which is also the same loss.")
    else:
        L.append("**The best loss is NOT the same across feature sets** (" +
                 ", ".join(k + " -> " + v for k, v in best.items()) + ").")
        L.append("")
        L.append("Affected comparisons: any cross-version statement that reads a "
                 "single number per backbone off a table where the backbones did "
                 "not all use the same loss. Concretely, in `Final Results.txt` "
                 "Section 3 and in the thesis narrative, the DINOv1 vs DINOv2 vs "
                 "DINOv3 vs BiomedCLIP ordering is only meaningful if read at a "
                 "FIXED loss (same column of the Part 1 table above), not at each "
                 "backbone's own best loss. The matched-loss comparison is in "
                 "Section 5.")
    L.append("")

    # ---------------- 3. which loss did the original runs use ----------------
    L.append("## 3. Which loss did the recorded probe runs actually use?")
    L.append("")
    L.append("`Progress-main.docx` says these probes used **Combined**; "
             "`Final Results.txt` records `(default)`, i.e. **focal**. "
             "`EXPERIMENTAL_RECORD.md` tags it INFERRED. Matching each recorded "
             "pair against the three explicit re-runs settles it empirically. "
             "Distance = |d Macro-F1| + |d MEL Recall| against the recorded pair.")
    L.append("")
    L.append("| Feature set | Recorded (F1/MEL) | focal | dice | combined | Closest |")
    L.append("|---|---|---|---|---|---|")
    verdicts = {}
    for fs_name, _, _ in FEATURE_SETS:
        rec_f1, rec_mr = RECORDED_PROBE[fs_name]
        cells, dists = [], {}
        for loss in LOSSES:
            ev = (a_runs.get(fs_name + "_" + loss) or {}).get("eval")
            if not ev:
                cells.append("-")
                continue
            d = abs(ev["macro_f1"] - rec_f1) + abs(ev["mel_recall"] - rec_mr)
            dists[loss] = d
            cells.append("{:.4f}/{:.4f} (d={:.4f})".format(ev["macro_f1"], ev["mel_recall"], d))
        if dists:
            winner = min(dists, key=dists.get)
            verdicts[fs_name] = (winner, dists[winner])
        else:
            winner = "-"
        L.append("| " + fs_name + " | " + "{:.3f}/{:.3f}".format(rec_f1, rec_mr)
                 + " | " + " | ".join(cells) + " | **" + winner + "** |")
    L.append("")
    if verdicts:
        agree = sorted(set(v[0] for v in verdicts.values()))
        if len(agree) == 1:
            L.append("**All " + str(len(verdicts)) + " recorded probe results match `"
                     + agree[0] + "` most closely.** On this evidence the original "
                     "runs used " + agree[0] + ", i.e. "
                     + ("`Final Results.txt` (focal / the script default) is correct and "
                        "`Progress-main.docx` is wrong."
                        if agree[0] == "focal" else
                        ("`Progress-main.docx` (Combined) is correct and "
                         "`Final Results.txt`'s `(default)` annotation is wrong."
                         if agree[0] == "combined" else
                         "neither document's claim is correct.")))
        else:
            L.append("**The recorded results do not all match the same loss** ("
                     + ", ".join(k + " -> " + v[0] for k, v in verdicts.items())
                     + "), which means the original probes were not all trained "
                       "under one consistent loss -- neither document's blanket "
                       "claim holds for every backbone.")
        L.append("")
        L.append("Caveat: identification is by closest match, not exact "
                 "reproduction. Distances are reported so a near-tie is visible "
                 "rather than hidden behind a verdict.")
    L.append("")

    # ---------------- 4. seed variance ----------------
    L.append("## 4. EfficientNet-B0 seed variance (weighted CE)")
    L.append("")
    L.append("**Data-leakage bug found and fixed during this run, documented here rather "
             "than silently corrected.** The first attempt at seeds 1337/2024 passed the "
             "varying seed straight into `train_efficientnet.py --seed`, which controls "
             "BOTH the training RNG and the lesion-level train/val/test split -- so each "
             "seed silently produced a *different* split, not just different training "
             "randomness. Verified directly: 86% of the canonical seed=42 test set (1289 "
             "of 1502 images) was inside seed=1337's own train+val split, so that model's "
             "test-set score was largely measuring memorized training data. Symptom: "
             "macro-F1 of 0.86-0.87, versus 0.69-0.70 for every other EfficientNet-B0 run "
             "in this repo. Fix: `train_efficientnet.py` now takes a separate "
             "`--split-seed` (defaults to `--seed` for full backward compatibility), and "
             "this driver pins `--split-seed 42` for every Part B run while `--seed` still "
             "varies training-only randomness. Seeds 1337/2024 were re-run under the fix; "
             "seed 42 was untouched (its split-seed and training-seed already coincided at "
             "42, so it was never affected). The original leaked runs are preserved, not "
             "deleted, at `outputs/effnet_seed_variance/seed{1337,2024}_LEAKED_splitbug/` "
             "and `results/LEAKED_splitbug/` -- never cite those numbers.")
    L.append("")
    L.append("| Seed | Macro-F1 (plain) | MEL Recall (plain) | Macro-F1 (TTA+MC) | MEL Recall (TTA+MC) |")
    L.append("|---|---|---|---|---|")
    cols = {"pf1": [], "pmr": [], "tf1": [], "tmr": []}
    for seed in SEEDS:
        rec = b_runs.get(str(seed)) or {}
        p, t = rec.get("eval_plain"), rec.get("eval_tta_mc")
        if not rec.get("train"):
            L.append("| " + str(seed) + " | NOT RUN / FAILED | - | - | - |")
            continue
        row = [str(seed)]
        for ev, k1, k2 in ((p, "pf1", "pmr"), (t, "tf1", "tmr")):
            if ev:
                cols[k1].append(ev["macro_f1"])
                cols[k2].append(ev["mel_recall"])
                row += ["{:.4f}".format(ev["macro_f1"]), "{:.4f}".format(ev["mel_recall"])]
            else:
                row += ["FAILED", "FAILED"]
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    st = {k: stats(v) for k, v in cols.items()}
    L.append("| Statistic | Macro-F1 (plain) | MEL Recall (plain) | Macro-F1 (TTA+MC) | MEL Recall (TTA+MC) |")
    L.append("|---|---|---|---|---|")
    for label, key in (("mean", "mean"), ("std dev (ddof=1)", "std"),
                       ("min", "min"), ("max", "max"), ("range (max-min)", "range")):
        cells = []
        for k in ("pf1", "pmr", "tf1", "tmr"):
            cells.append("{:.4f}".format(st[k][key]) if st[k] else "-")
        L.append("| " + label + " | " + " | ".join(cells) + " |")
    L.append("")
    L.append("**plain** columns isolate training variance (the evaluation itself is "
             "deterministic). **TTA+MC** columns combine training variance with "
             "MC-Dropout sampling noise. They are deliberately not merged.")
    L.append("")

    # determinism check
    L.append("### Determinism check (seed 42)")
    L.append("")
    prior = prior_effnet_numbers()
    s42 = (b_runs.get("42") or {}).get("eval_plain")
    if prior.get("ce") and s42:
        p_f1, p_mr = prior["ce"]
        d_f1 = s42["macro_f1"] - p_f1
        d_mr = s42["mel_recall"] - p_mr
        exact = abs(d_f1) < 1e-9 and abs(d_mr) < 1e-9
        L.append("| | Macro-F1 (plain) | MEL Recall (plain) |")
        L.append("|---|---|---|")
        L.append("| Earlier loss-ablation `ce` run (seed 42) | " + "{:.6f}".format(p_f1)
                 + " | " + "{:.6f}".format(p_mr) + " |")
        L.append("| This run, seed 42 | " + "{:.6f}".format(s42["macro_f1"])
                 + " | " + "{:.6f}".format(s42["mel_recall"]) + " |")
        L.append("| Difference | " + "{:+.6f}".format(d_f1) + " | " + "{:+.6f}".format(d_mr) + " |")
        L.append("")
        if exact:
            L.append("**Seed 42 reproduced the earlier run exactly on the plain "
                     "evaluation.** The training pipeline is seed-deterministic on "
                     "this machine, so the seed-to-seed spread below is genuine "
                     "training variance and not run-to-run jitter.")
        else:
            L.append("**Seed 42 did NOT reproduce the earlier run exactly** "
                     "(difference above). The pipeline is therefore not "
                     "seed-deterministic on this machine: identical seed and "
                     "identical hyperparameters still give different weights, "
                     "most likely via non-deterministic cuDNN kernels "
                     "(`torch.use_deterministic_algorithms` is never set). "
                     "This matters more than the variance estimate itself: it "
                     "means no result in this thesis is exactly reproducible "
                     "from its seed alone, and the spread below therefore mixes "
                     "seed effects with pure run-to-run nondeterminism.")
    else:
        L.append("Could not compare: earlier loss-ablation `ce` result or this "
                 "run's seed-42 plain evaluation is missing.")
    L.append("")

    # ---------------- 5. verdict ----------------
    L.append("## 5. Verdict: which differences exceed the noise floor?")
    L.append("")
    sd = st["pf1"]["std"] if st["pf1"] and st["pf1"]["n"] > 1 else None
    if sd is None:
        L.append("Not enough completed seed runs to estimate a noise floor.")
        return NL.join(L)

    L.append("Noise floor: measured std dev of test Macro-F1 across seeds on the "
             "**plain** evaluation, sigma = " + "{:.4f}".format(sd)
             + " (n=" + str(st["pf1"]["n"]) + ", EfficientNet-B0, weighted CE). "
             "Criterion used below: a gap is called real only if it exceeds "
             "2*sigma = " + "{:.4f}".format(2 * sd) + "; between 1 and 2 sigma it "
             "is 'suggestive'; below 1 sigma it is indistinguishable from noise.")
    L.append("")

    def verdict(delta):
        ad = abs(delta)
        if ad > 2 * sd:
            return "**exceeds noise floor** (> 2 sigma)"
        if ad > sd:
            return "suggestive only (1-2 sigma)"
        return "**within noise** (< 1 sigma)"

    pe = prior_effnet_numbers()
    rows = []
    if pe.get("dice") and pe.get("ce"):
        d = pe["dice"][0] - pe["ce"][0]
        rows.append(("Dice (" + "{:.4f}".format(pe["dice"][0]) + ") vs CE ("
                     + "{:.4f}".format(pe["ce"][0]) + "), EfficientNet, plain",
                     d, verdict(d), "same architecture + same eval as the sigma above"))
    d2 = DINOV1_RECORDED["combined"][0] - RECORDED_PROBE["dinov2"][0]
    rows.append(("DINOv1 (0.554) vs DINOv2 (0.545), frozen probes", d2, verdict(d2),
                 "sigma is an EfficientNet proxy -- see caveat"))
    d3 = RECORDED_PROBE["dinov2"][0] - RECORDED_PROBE["dinov3"][0]
    # DINOv3 label value follows RECORDED_PROBE above (CLS-token features; was 0.380 mean-pooled)
    rows.append(("DINOv2 (0.545) vs DINOv3 (0.483), frozen probes", d3, verdict(d3),
                 "sigma is an EfficientNet proxy -- see caveat"))

    L.append("| Comparison | Difference | Verdict | Note |")
    L.append("|---|---|---|---|")
    for name, d, v, note in rows:
        L.append("| " + name + " | " + "{:+.4f}".format(d) + " | " + v + " | " + note + " |")
    L.append("")
    L.append("**Caveat that limits rows 2 and 3:** sigma was measured on "
             "EfficientNet-B0, not on the linear probes. No seed-variance estimate "
             "exists for the probes -- every probe result in this repo, including "
             "the 12 new runs above, is a single seed. Using an EfficientNet-derived "
             "sigma for probe comparisons is a proxy, not a measurement. A probe "
             "seed sweep is cheap (each run is under a minute) and would remove "
             "this caveat entirely.")
    L.append("")
    L.append("**Matched-loss check for row 2.** The recorded 0.554 (DINOv1) and "
             "0.545 (DINOv2) pair may not even be a like-for-like comparison, "
             "since it depends on both having used the same loss (Section 3). "
             "The Part A grid supplies the properly matched numbers:")
    L.append("")
    L.append("| Loss | DINOv1 (recorded) | DINOv2 (this run) | Difference | Verdict |")
    L.append("|---|---|---|---|---|")
    for loss in LOSSES:
        ev = (a_runs.get("dinov2_" + loss) or {}).get("eval")
        if not ev:
            L.append("| " + loss + " | " + "{:.3f}".format(DINOV1_RECORDED[loss][0])
                     + " | - | - | - |")
            continue
        d = DINOV1_RECORDED[loss][0] - ev["macro_f1"]
        L.append("| " + loss + " | " + "{:.3f}".format(DINOV1_RECORDED[loss][0]) + " | "
                 + "{:.4f}".format(ev["macro_f1"]) + " | " + "{:+.4f}".format(d)
                 + " | " + verdict(d) + " |")
    L.append("")

    failed_a = [k for k, v in a_runs.items() if not (v or {}).get("eval")]
    failed_b = [s for s in SEEDS if not (b_runs.get(str(s)) or {}).get("eval_tta_mc")]
    if failed_a or failed_b or ERROR_LOG.exists():
        L.append("## Errors")
        L.append("")
        if failed_a:
            L.append("Part A incomplete: " + ", ".join(sorted(failed_a)))
        if failed_b:
            L.append("Part B incomplete: seeds " + ", ".join(str(s) for s in failed_b))
        if ERROR_LOG.exists():
            L.append("Tracebacks: `" + ERROR_LOG.name + "`")
        L.append("")

    return NL.join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Overnight probe loss ablation + EfficientNet seed variance")
    p.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    p.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    p.add_argument("--part", choices=["a", "b", "all"], default="all")
    p.add_argument("--phase1-epochs", type=int, default=15)
    p.add_argument("--phase2-epochs", type=int, default=15)
    p.add_argument("--force", action="store_true",
                   help="Redo runs the manifest already shows complete.")
    p.add_argument("--summarize-only", action="store_true",
                   help="Rebuild the summary from the existing manifest; run nothing.")
    return p.parse_args()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    manifest = load_manifest()

    if not args.summarize_only and args.part in ("a", "all"):
        print("")
        print("#" * 64)
        print("PART A -- probe loss ablation (12 runs)")
        print("#" * 64)
        for fs_name, feats_dir, prefix in FEATURE_SETS:
            for loss in LOSSES:
                key = fs_name + "_" + loss
                rec = manifest["part_a"].get(key) or {}
                if rec.get("train") and rec.get("eval") and not args.force:
                    print("")
                    print("== " + key + " -- already complete, skipping ==")
                    continue
                print("")
                print("== " + key + " ==")
                rec = {}
                manifest["part_a"][key] = rec
                try:
                    rec["train"] = part_a_train(fs_name, feats_dir, prefix, loss)
                    save_manifest(manifest)
                    print("  OK  train  epoch=" + str(rec["train"]["epoch"])
                          + "  val_macro_f1=" + "{:.4f}".format(rec["train"]["val_macro_f1"]))
                except Exception as exc:
                    log_error(key, "train", exc)
                    rec["train"] = None
                    save_manifest(manifest)
                    continue
                try:
                    rec["eval"] = part_a_eval(fs_name, feats_dir, prefix, loss,
                                              rec["train"]["ckpt_path"])
                    save_manifest(manifest)
                    ev = rec["eval"]
                    print("  OK  eval   macro_f1=" + "{:.4f}".format(ev["macro_f1"])
                          + "  mel_recall=" + "{:.4f}".format(ev["mel_recall"])
                          + "  nv_recall=" + "{:.4f}".format(ev["per_class"]["nv"]["recall"]))
                except Exception as exc:
                    log_error(key, "eval", exc)
                    rec["eval"] = None
                    save_manifest(manifest)

        # Part A results matter regardless of Part B: write the summary now.
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(generate_summary(manifest), encoding="utf-8")
        print("")
        print("Interim summary written (Part A complete): " + str(SUMMARY_PATH))

    if not args.summarize_only and args.part in ("b", "all"):
        print("")
        print("#" * 64)
        print("PART B -- EfficientNet seed variance (3 runs)")
        print("#" * 64)
        for seed in SEEDS:
            key = str(seed)
            rec = manifest["part_b"].get(key) or {}
            if rec.get("train") and rec.get("eval_plain") and rec.get("eval_tta_mc") and not args.force:
                print("")
                print("== seed " + key + " -- already complete, skipping ==")
                continue
            print("")
            print("== seed " + key + " ==")
            rec = {}
            manifest["part_b"][key] = rec
            try:
                rec["train"] = part_b_train(seed, args)
                save_manifest(manifest)
                t = rec["train"]
                print("  OK  train  epoch=" + str(t["epoch"]) + " phase=" + str(t["phase"])
                      + "  val_macro_f1=" + "{:.4f}".format(t["val_macro_f1"])
                      + "  wall=" + "{:.1f}".format(t["wall_clock_seconds_total"] / 60.0) + " min")
            except Exception as exc:
                log_error("seed" + key, "train", exc)
                rec["train"] = None
                save_manifest(manifest)
                continue
            for mode, mkey in (("plain", "eval_plain"), ("tta_mc", "eval_tta_mc")):
                try:
                    rec[mkey] = part_b_eval(seed, rec["train"]["ckpt_path"], mode, args)
                    save_manifest(manifest)
                    print("  OK  eval[" + mode + "]  macro_f1="
                          + "{:.4f}".format(rec[mkey]["macro_f1"])
                          + "  mel_recall=" + "{:.4f}".format(rec[mkey]["mel_recall"]))
                except Exception as exc:
                    log_error("seed" + key, "eval[" + mode + "]", exc)
                    rec[mkey] = None
                    save_manifest(manifest)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(generate_summary(manifest), encoding="utf-8")
    print("")
    print("Summary written: " + str(SUMMARY_PATH))


if __name__ == "__main__":
    main()
