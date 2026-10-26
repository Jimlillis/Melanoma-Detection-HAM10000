"""
Corrected EfficientNet-B0 seed-variance run: seeds 1337 and 2024 only (seed
42 is already valid and is never re-run here -- its numbers are read from
outputs/overnight_manifest.json).

This exists because the first attempt leaked: passing the varying seed into
train_efficientnet.py's single --seed coupled training randomness to the
train/val/test split itself, so seed=1337 trained on 86% of what became its
own "test" set. Fixed via --split-seed (added to train_efficientnet.py
separately). This script pins --split-seed 42 explicitly on every run here
and never relies on a default to get that right.

Two sanity gates, applied per run, BEFORE its numbers are allowed into the
report:
  1. Split-integrity gate -- the checkpoint's own stored seed/split_seed
     fields, AND an independent recomputation of the seed=42 split, must
     both match the canonical test-set class counts. A mismatch aborts that
     run's evaluation and marks it INVALID rather than reporting numbers.
  2. Leakage-threshold gate -- test macro-F1 > 0.80 on either protocol is
     flagged SUSPECTED LEAKAGE and excluded from the summary statistics
     (every legitimate EfficientNet-B0 run in this repo sits at 0.69-0.70).

Writes to:
  outputs/effnet_seed_variance/seed{1337,2024}/
  results/effnet_seed1337_*.txt, results/effnet_seed2024_*.txt (+ .json)
  results/effnet_seed_variance.md

Run from project root:
  python src/run_effnet_seed_variance.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_SRC = Path(__file__).resolve().parent

RESULTS_DIR = _ROOT / "results"
OUT_ROOT = _ROOT / "outputs" / "effnet_seed_variance"
OVERNIGHT_MANIFEST = _ROOT / "outputs" / "overnight_manifest.json"
LOCAL_MANIFEST = OUT_ROOT / "manifest_corrected.json"
SUMMARY_PATH = RESULTS_DIR / "effnet_seed_variance.md"

DEFAULT_METADATA_CSV = _ROOT / "data" / "HAM10000_metadata.csv"
DEFAULT_IMAGES_DIR = _ROOT / "data"

SEEDS_TO_RUN = [1337, 2024]
SPLIT_SEED = 42
EXPECTED_TEST_COUNTS = {"akiec": 52, "bcc": 71, "bkl": 167, "df": 20,
                        "mel": 167, "nv": 1004, "vasc": 21}
LEAKAGE_MACRO_F1_THRESHOLD = 0.80

# For the final verdict section -- Part A's actual (combined-loss) numbers,
# not the old ambiguous-loss recorded figures.
VERDICT_COMPARISONS = [
    ("Dice (0.7029) vs CE (0.6903), EfficientNet, plain", 0.7029, 0.6903),
    ("DINOv1 (0.554) vs DINOv2 (0.5446), frozen probes", 0.554, 0.5446),
    # DINOv3 = CLS-token features (0.4831); was mean-pooled 0.3797 / 0.4970
    ("DINOv2 (0.5446) vs DINOv3 (0.4831), frozen probes", 0.5446, 0.4831),
]


def verify_canonical_split() -> dict:
    """Independently recompute the seed=42 lesion-level split and return its
    test-set per-class counts. Pure function of split_seed=42 -- does not
    depend on which training seed is being evaluated."""
    sys.path.insert(0, str(_SRC))
    from config import Config
    from extract_features_v2 import load_multiclass_table
    from dataset import lesion_level_split

    cfg = Config()
    df, _ = load_multiclass_table(cfg.metadata_csv, cfg.col_image_id, cfg.col_lesion_id, cfg.col_diagnosis)
    _, _, df_test = lesion_level_split(
        df, cfg.col_lesion_id, cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, SPLIT_SEED,
    )
    return df_test["dx"].value_counts().sort_index().to_dict()


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
        raise RuntimeError("exit " + str(proc.returncode) + " from: " + printable
                           + chr(10) + "--- tail of " + str(log_path) + " ---"
                           + chr(10) + tail)


def train_one(seed: int, args) -> dict:
    run_dir = OUT_ROOT / ("seed" + str(seed))
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, _SRC / "train_efficientnet.py",
           "--metadata-csv", args.metadata_csv,
           "--images-dir", args.images_dir,
           "--output-dir", run_dir,
           "--loss", "ce",
           "--seed", str(seed),
           "--split-seed", str(SPLIT_SEED),
           "--phase1-epochs", "15",
           "--phase2-epochs", "15",
           "--no-mc"]

    # Verify the two flags are genuinely independent entries in this exact
    # command before running it -- not assumed, checked.
    seed_idx = cmd.index("--seed")
    split_seed_idx = cmd.index("--split-seed")
    assert seed_idx != split_seed_idx, "internal error: same flag matched twice"
    assert str(cmd[seed_idx + 1]) == str(seed), "internal error: --seed value wrong"
    assert str(cmd[split_seed_idx + 1]) == str(SPLIT_SEED), "internal error: --split-seed value wrong"
    assert cmd[seed_idx + 1] != cmd[split_seed_idx + 1] or seed == SPLIT_SEED, \
        "internal error: --seed and --split-seed collapsed to the same value unexpectedly"
    print("  [pre-flight] --seed=" + str(cmd[seed_idx + 1])
          + "  --split-seed=" + str(cmd[split_seed_idx + 1])
          + "  (independent entries, verified)", flush=True)

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
            "wall_clock_seconds_total": wall}


def eval_one(seed: int, ckpt_path: str, mode: str, args) -> dict:
    run_dir = OUT_ROOT / ("seed" + str(seed))
    report_name = "effnet_seed" + str(seed) + "_" + mode
    json_path = run_dir / (report_name + ".json")
    cmd = [sys.executable, _SRC / "eval_saved_models.py",
           "--metadata-csv", args.metadata_csv,
           "--images-dir", args.images_dir,
           "--results-dir", RESULTS_DIR,
           "--seed", str(SPLIT_SEED),  # canonical split for evaluation, always
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


def gate_split_integrity(seed: int, train_rec: dict, canonical_counts: dict) -> list:
    """Returns a list of failure reasons; empty list = gate passed."""
    failures = []
    if train_rec.get("stored_seed") != seed:
        failures.append("checkpoint stored seed=" + str(train_rec.get("stored_seed"))
                        + ", expected " + str(seed))
    if train_rec.get("stored_split_seed") != SPLIT_SEED:
        failures.append("checkpoint stored split_seed=" + str(train_rec.get("stored_split_seed"))
                        + ", expected " + str(SPLIT_SEED))
    if canonical_counts != EXPECTED_TEST_COUNTS:
        failures.append("recomputed split_seed=" + str(SPLIT_SEED) + " test counts "
                        + str(canonical_counts) + " != expected " + str(EXPECTED_TEST_COUNTS))
    return failures


def stats(values: list) -> dict | None:
    a = np.array([v for v in values if v is not None], dtype=float)
    if a.size == 0:
        return None
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max()), "range": float(a.max() - a.min()),
            "n": int(a.size)}


def load_seed42_reference() -> dict | None:
    if not OVERNIGHT_MANIFEST.exists():
        return None
    m = json.loads(OVERNIGHT_MANIFEST.read_text(encoding="utf-8"))
    r = (m.get("part_b") or {}).get("42")
    if not r or not r.get("train") or not r.get("eval_plain") or not r.get("eval_tta_mc"):
        return None
    return {"seed": 42, "status": "OK", "plain": r["eval_plain"], "tta_mc": r["eval_tta_mc"]}


def verdict_line(delta: float, sigma: float) -> str:
    ad = abs(delta)
    if ad > 2 * sigma:
        return "**exceeds noise floor** (> 2 sigma)"
    if ad > sigma:
        return "suggestive only (1-2 sigma)"
    return "**within noise** (< 1 sigma)"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    class Args:
        metadata_csv = DEFAULT_METADATA_CSV
        images_dir = DEFAULT_IMAGES_DIR
    args = Args()

    print("=" * 64)
    print("PRE-FLIGHT: verifying split_seed=42 resolves to the canonical split")
    print("=" * 64, flush=True)
    preflight_counts = verify_canonical_split()
    print("  recomputed test counts:", preflight_counts, flush=True)
    if preflight_counts != EXPECTED_TEST_COUNTS:
        raise SystemExit(
            "ABORTING before any training: split_seed=" + str(SPLIT_SEED)
            + " does NOT resolve to the canonical split. Got " + str(preflight_counts)
            + ", expected " + str(EXPECTED_TEST_COUNTS) + ". This is an environment/"
            "data problem, not something fixable by rerunning."
        )
    print("  MATCHES canonical split. Proceeding.", flush=True)

    results = []  # list of dicts: seed, status, plain, tta_mc, notes

    for seed in SEEDS_TO_RUN:
        print(chr(10) + "=" * 64)
        print("SEED " + str(seed))
        print("=" * 64, flush=True)

        entry = {"seed": seed, "status": None, "plain": None, "tta_mc": None, "notes": []}
        try:
            train_rec = train_one(seed, args)
        except Exception as exc:
            entry["status"] = "TRAIN_FAILED"
            entry["notes"].append(str(exc)[:500])
            print("  !! TRAIN FAILED: " + str(exc)[:500], flush=True)
            results.append(entry)
            continue

        print("  OK  train  epoch=" + str(train_rec["epoch"]) + " phase=" + str(train_rec["phase"])
              + "  val_macro_f1=" + "{:.4f}".format(train_rec["val_macro_f1"])
              + "  wall=" + "{:.1f}".format(train_rec["wall_clock_seconds_total"] / 60.0) + " min",
              flush=True)

        canonical_counts = verify_canonical_split()  # re-verified per run, as specified
        failures = gate_split_integrity(seed, train_rec, canonical_counts)
        if failures:
            entry["status"] = "INVALID_SPLIT"
            entry["notes"].extend(failures)
            print(flush=True)
            print("  *** SPLIT-INTEGRITY GATE FAILED -- ABORTING THIS RUN, NOT EVALUATING ***", flush=True)
            for f in failures:
                print("      - " + f, flush=True)
            print(flush=True)
            results.append(entry)
            continue
        print("  OK  split-integrity gate passed (seed=" + str(seed) + ", split_seed=" + str(SPLIT_SEED) + ")",
              flush=True)

        try:
            plain = eval_one(seed, train_rec["ckpt_path"], "plain", args)
            tta_mc = eval_one(seed, train_rec["ckpt_path"], "tta_mc", args)
        except Exception as exc:
            entry["status"] = "EVAL_FAILED"
            entry["notes"].append(str(exc)[:500])
            print("  !! EVAL FAILED: " + str(exc)[:500], flush=True)
            results.append(entry)
            continue

        print("  OK  eval[plain]   macro_f1=" + "{:.4f}".format(plain["macro_f1"])
              + "  mel_recall=" + "{:.4f}".format(plain["mel_recall"]), flush=True)
        print("  OK  eval[tta_mc]  macro_f1=" + "{:.4f}".format(tta_mc["macro_f1"])
              + "  mel_recall=" + "{:.4f}".format(tta_mc["mel_recall"]), flush=True)

        if plain["macro_f1"] > LEAKAGE_MACRO_F1_THRESHOLD or tta_mc["macro_f1"] > LEAKAGE_MACRO_F1_THRESHOLD:
            entry["status"] = "SUSPECTED_LEAKAGE"
            entry["plain"] = plain
            entry["tta_mc"] = tta_mc
            entry["notes"].append(
                "plain macro_f1=" + "{:.4f}".format(plain["macro_f1"])
                + ", tta_mc macro_f1=" + "{:.4f}".format(tta_mc["macro_f1"])
                + " -- one or both exceed the " + "{:.2f}".format(LEAKAGE_MACRO_F1_THRESHOLD)
                + " leakage-suspicion threshold. Every legitimate EfficientNet-B0 run in "
                  "this repo sits at 0.69-0.70. Numbers are recorded below but EXCLUDED "
                  "from summary statistics."
            )
            print(flush=True)
            print("  *** SUSPECTED LEAKAGE -- macro-F1 exceeds " + "{:.2f}".format(LEAKAGE_MACRO_F1_THRESHOLD)
                  + " threshold ***", flush=True)
            print(flush=True)
            results.append(entry)
            continue

        entry["status"] = "OK"
        entry["plain"] = plain
        entry["tta_mc"] = tta_mc
        results.append(entry)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOCAL_MANIFEST.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    seed42 = load_seed42_reference()
    all_rows = ([seed42] if seed42 else []) + results

    # --- Build the report ----------------------------------------------------
    NL = chr(10)
    L = []
    L.append("# EfficientNet-B0 Seed Variance (corrected)")
    L.append("")
    L.append("Generated: " + datetime.now().isoformat(timespec="seconds"))
    L.append("")
    L.append("Seeds 1337 and 2024 re-run here with `--split-seed 42` pinned explicitly "
             "(decoupled from the varying `--seed`); seed 42 is reused unchanged from the "
             "earlier valid run (its split-seed and training-seed already coincided at 42, "
             "so it was never affected by the leakage bug). Two gates applied per new run "
             "before its numbers are trusted: (1) split-integrity -- checkpoint's own "
             "stored seed/split_seed fields plus an independent recomputation of the "
             "seed=42 split must match the canonical test-set counts exactly; (2) "
             "leakage-threshold -- test macro-F1 > " + "{:.2f}".format(LEAKAGE_MACRO_F1_THRESHOLD)
             + " on either protocol is flagged and excluded from statistics.")
    L.append("")

    L.append("## Pre-flight check")
    L.append("")
    L.append("Recomputed seed=42 split independently before any training: "
             + str(preflight_counts) + " -- matches the canonical counts exactly.")
    L.append("")

    L.append("## Results")
    L.append("")
    L.append("| Seed | Status | Macro-F1 (plain) | MEL Recall (plain) | Macro-F1 (TTA+MC) | MEL Recall (TTA+MC) |")
    L.append("|---|---|---|---|---|---|")
    for r in all_rows:
        seed = r["seed"]
        status = r["status"]
        p, t = r.get("plain"), r.get("tta_mc")
        p_f1 = "{:.4f}".format(p["macro_f1"]) if p else "-"
        p_mr = "{:.4f}".format(p["mel_recall"]) if p else "-"
        t_f1 = "{:.4f}".format(t["macro_f1"]) if t else "-"
        t_mr = "{:.4f}".format(t["mel_recall"]) if t else "-"
        flag = "" if status == "OK" else "  <-- " + status
        L.append("| " + str(seed) + " | " + status + flag + " | " + p_f1 + " | " + p_mr
                 + " | " + t_f1 + " | " + t_mr + " |")
    L.append("")

    for r in results:
        if r["status"] != "OK" and r["notes"]:
            L.append("**" + str(r["seed"]) + " (" + r["status"] + "):** " + " | ".join(r["notes"]))
            L.append("")

    ok_rows = [r for r in all_rows if r["status"] == "OK"]
    cols = {"pf1": [], "pmr": [], "tf1": [], "tmr": []}
    for r in ok_rows:
        cols["pf1"].append(r["plain"]["macro_f1"])
        cols["pmr"].append(r["plain"]["mel_recall"])
        cols["tf1"].append(r["tta_mc"]["macro_f1"])
        cols["tmr"].append(r["tta_mc"]["mel_recall"])
    st = {k: stats(v) for k, v in cols.items()}

    L.append("## Summary statistics (OK runs only, n=" + str(len(ok_rows)) + ")")
    L.append("")
    L.append("| Statistic | Macro-F1 (plain) | MEL Recall (plain) | Macro-F1 (TTA+MC) | MEL Recall (TTA+MC) |")
    L.append("|---|---|---|---|---|")
    for label, key in (("mean", "mean"), ("std dev (ddof=1)", "std"), ("min", "min"),
                       ("max", "max"), ("range (max-min)", "range")):
        cells = ["{:.4f}".format(st[k][key]) if st[k] else "-" for k in ("pf1", "pmr", "tf1", "tmr")]
        L.append("| " + label + " | " + " | ".join(cells) + " |")
    L.append("")
    L.append("**plain** isolates training variance (deterministic evaluation). **TTA+MC** "
             "mixes in MC-Dropout sampling noise. Kept separate, not merged, by design.")
    L.append("")

    L.append("## Verdict: do these differences exceed the noise floor?")
    L.append("")
    if st["pf1"] and st["pf1"]["n"] > 1:
        sigma = st["pf1"]["std"]
        L.append("Noise floor: std dev of test Macro-F1 across seeds on the **plain** "
                 "evaluation, sigma = " + "{:.4f}".format(sigma) + " (n=" + str(st["pf1"]["n"])
                 + "). A gap counts as real only if it exceeds 2*sigma = "
                 + "{:.4f}".format(2 * sigma) + "; 1-2 sigma is suggestive; below 1 sigma is "
                 "indistinguishable from noise.")
        L.append("")
        L.append("| Comparison | Difference | Verdict |")
        L.append("|---|---|---|")
        for name, a, b in VERDICT_COMPARISONS:
            d = a - b
            L.append("| " + name + " | " + "{:+.4f}".format(d) + " | " + verdict_line(d, sigma) + " |")
        L.append("")
        L.append("Caveat: sigma is measured on EfficientNet-B0 (3 seeds, weighted CE). The "
                 "DINOv1/DINOv2/DINOv3 comparisons use it as a proxy noise floor -- no "
                 "seed-variance estimate exists for the frozen linear probes themselves "
                 "(every probe result in this repo, Part A included, is a single seed).")
    else:
        L.append("Not enough OK runs (need >= 2) to compute a noise floor. "
                 "n=" + str(st["pf1"]["n"] if st["pf1"] else 0) + ".")
    L.append("")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(NL.join(L), encoding="utf-8")
    print(NL + "Summary written: " + str(SUMMARY_PATH), flush=True)


if __name__ == "__main__":
    main()
