"""
Seed-variance estimate for the DINOv2 frozen linear probe, combined loss --
three runs, seeds 42/1337/2024, everything else identical to the Part A
probe ablation (default train_dino_head.py settings: WeightedRandomSampler
on, no train-split rebalancing).

Purpose: results/effnet_seed_variance.md used EfficientNet's seed-to-seed
sigma (0.0105 macro-F1) as a proxy noise floor for probe comparisons,
flagging that a linear head on frozen features is likely far less variable
than 30 epochs of full CNN training. This measures the probe's own sigma
directly and re-states the two verdicts that depended on the proxy.

Seed 42 is rerun (not reused from Part A's saved JSON) specifically as a
determinism check, mirroring the same check already done for Run 1 of the
dice-degeneracy 2x2 -- same seed, same code, same features, should
reproduce exactly.

Writes to:
  outputs/probe_seed_variance/dinov2_combined_seed{42,1337,2024}/
  results/probe_seed_variance_dinov2_combined_seed{42,1337,2024}_results.txt (+ .json)

Appends a new "Probe-specific seed variance" section to
results/effnet_seed_variance.md -- existing content in that file is never
touched, only appended after.

Run from project root:
  python src/run_probe_seed_variance.py
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
OUT_ROOT = _ROOT / "outputs" / "probe_seed_variance"
EFFNET_SEED_VARIANCE_MD = RESULTS_DIR / "effnet_seed_variance.md"

FEATS_DIR = _ROOT / "data" / "processed" / "dinov2"
FEATS_PREFIX = "dinov2_feats"
SEEDS = [42, 1337, 2024]

# Part A's original dinov2+combined control (outputs/probe_loss_ablation/
# dinov2_combined/) -- seed 42 here should reproduce it exactly.
PART_A_CONTROL = {"macro_f1": 0.5446405854092092, "mel_recall": 0.688622754491018}
DETERMINISM_TOLERANCE = 1e-6

# From results/effnet_seed_variance.md, using EfficientNet's sigma as a proxy.
VERDICT_COMPARISONS = [
    ("DINOv1 (0.554) vs DINOv2 (0.5446), frozen probes", 0.554, 0.5446),
    # DINOv3 = CLS-token features (0.4831); was mean-pooled 0.3797 / 0.4970
    ("DINOv2 (0.5446) vs DINOv3 (0.4831), frozen probes", 0.5446, 0.4831),
]


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


def train_one(seed: int) -> dict:
    run_dir = OUT_ROOT / ("dinov2_combined_seed" + str(seed))
    run_dir.mkdir(parents=True, exist_ok=True)
    # No --rebalance-train, no --no-sampler: identical to the Part A default.
    cmd = [sys.executable, _SRC / "train_dino_head.py",
           "--features-dir", FEATS_DIR,
           "--features-prefix", FEATS_PREFIX,
           "--output-dir", run_dir,
           "--loss", "combined",
           "--seed", str(seed)]
    t0 = time.time()
    run_subprocess(cmd, run_dir / "train_stdout.log")
    wall = time.time() - t0
    ckpt = run_dir / "best_dino_head.pth"
    if not ckpt.exists():
        raise RuntimeError("training finished but no checkpoint at " + str(ckpt))
    import torch
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    return {"seed": seed, "ckpt_path": str(ckpt), "epoch": c.get("epoch"),
            "stored_seed": c.get("seed"), "val_macro_f1": c.get("val_macro_f1"),
            "wall_clock_seconds_total": wall}


def eval_one(seed: int, ckpt_path: str) -> dict:
    run_dir = OUT_ROOT / ("dinov2_combined_seed" + str(seed))
    report_name = "probe_seed_variance_dinov2_combined_seed" + str(seed)
    json_path = run_dir / (report_name + ".json")
    cmd = [sys.executable, _SRC / "eval_saved_models.py",
           "--only", "probes",
           "--results-dir", RESULTS_DIR,
           "--seed", "42",
           "--probe-ckpt", ckpt_path,
           "--probe-feats", FEATS_DIR / (FEATS_PREFIX + "_test.pt"),
           "--probe-report-name", report_name,
           "--dump-json", json_path]
    run_subprocess(cmd, run_dir / "eval_stdout.log")
    if not json_path.exists():
        raise RuntimeError("eval finished but no JSON at " + str(json_path))
    return json.loads(json_path.read_text(encoding="utf-8"))


def stats(values: list) -> dict:
    a = np.array(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)), "n": int(a.size)}


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

    runs = []
    determinism_text = ""
    for seed in SEEDS:
        print(chr(10) + "=" * 64)
        print("SEED " + str(seed) + "  (DINOv2, combined loss)")
        print("=" * 64, flush=True)
        train_rec = train_one(seed)
        print("  OK  train  epoch=" + str(train_rec["epoch"])
              + "  val_macro_f1=" + "{:.4f}".format(train_rec["val_macro_f1"]), flush=True)
        eval_rec = eval_one(seed, train_rec["ckpt_path"])
        print("  OK  eval   macro_f1=" + "{:.4f}".format(eval_rec["macro_f1"])
              + "  mel_recall=" + "{:.4f}".format(eval_rec["mel_recall"]), flush=True)
        runs.append({"seed": seed, "train": train_rec, "eval": eval_rec})

        if seed == 42:
            d_f1 = eval_rec["macro_f1"] - PART_A_CONTROL["macro_f1"]
            d_mr = eval_rec["mel_recall"] - PART_A_CONTROL["mel_recall"]
            matches = abs(d_f1) < DETERMINISM_TOLERANCE and abs(d_mr) < DETERMINISM_TOLERANCE
            if matches:
                determinism_text = (
                    "Seed 42 reproduced Part A's dinov2+combined control EXACTLY "
                    "(macro_f1={:.6f}, mel_recall={:.6f}). Pipeline is deterministic; "
                    "the three seeds below are comparable."
                ).format(eval_rec["macro_f1"], eval_rec["mel_recall"])
            else:
                determinism_text = (
                    "*** DETERMINISM PROBLEM *** Seed 42 did NOT reproduce Part A's "
                    "dinov2+combined control despite identical seed, code, and features. "
                    "Part A: macro_f1={:.6f}, mel_recall={:.6f}. This run: "
                    "macro_f1={:.6f}, mel_recall={:.6f}. Differences: d_macro_f1={:+.6f}, "
                    "d_mel_recall={:+.6f}."
                ).format(PART_A_CONTROL["macro_f1"], PART_A_CONTROL["mel_recall"],
                         eval_rec["macro_f1"], eval_rec["mel_recall"], d_f1, d_mr)
            print(chr(10) + determinism_text + chr(10), flush=True)

    macro_f1_vals = [r["eval"]["macro_f1"] for r in runs]
    mel_recall_vals = [r["eval"]["mel_recall"] for r in runs]
    f1_stats = stats(macro_f1_vals)
    mr_stats = stats(mel_recall_vals)

    NL = chr(10)
    L = []
    L.append("## Probe-specific seed variance")
    L.append("")
    L.append("Generated: " + datetime.now().isoformat(timespec="seconds"))
    L.append("")
    L.append("DINOv2 frozen linear probe, `--loss combined`, seeds 42/1337/2024, "
             "everything else identical to the Part A probe ablation (default "
             "`train_dino_head.py` settings -- WeightedRandomSampler on, no "
             "train-split rebalancing). Purpose: the verdicts above used "
             "EfficientNet's seed-to-seed sigma (0.0105 macro-F1) as a proxy noise "
             "floor for probe comparisons; this measures the probe's own sigma "
             "directly, since a linear head on frozen features is expected to be "
             "far less variable than 30 epochs of full CNN training.")
    L.append("")
    L.append("Determinism check (seed 42 vs. Part A control): " + determinism_text)
    L.append("")
    L.append("| Seed | Macro-F1 | MEL Recall |")
    L.append("|---|---|---|")
    for r in runs:
        L.append("| " + str(r["seed"]) + " | " + "{:.4f}".format(r["eval"]["macro_f1"])
                 + " | " + "{:.4f}".format(r["eval"]["mel_recall"]) + " |")
    L.append("")
    L.append("| Statistic | Macro-F1 | MEL Recall |")
    L.append("|---|---|---|")
    L.append("| mean | " + "{:.4f}".format(f1_stats["mean"]) + " | " + "{:.4f}".format(mr_stats["mean"]) + " |")
    L.append("| std dev (ddof=1) | " + "{:.4f}".format(f1_stats["std"]) + " | " + "{:.4f}".format(mr_stats["std"]) + " |")
    L.append("")
    L.append("**Probe-specific Macro-F1 sigma = " + "{:.4f}".format(f1_stats["std"])
             + "** (n=" + str(f1_stats["n"]) + "), vs. the EfficientNet proxy of 0.0105 used "
             "above. **Probe-specific MEL Recall sigma = " + "{:.4f}".format(mr_stats["std"])
             + "** (n=" + str(mr_stats["n"])
             + ") -- reported explicitly here since it was omitted for the plain "
             "evaluation in the EfficientNet section above.")
    L.append("")
    L.append("### Re-stated verdicts, using the probe-specific Macro-F1 sigma")
    L.append("")
    sigma = f1_stats["std"]
    L.append("sigma = " + "{:.4f}".format(sigma) + ", 2*sigma = " + "{:.4f}".format(2 * sigma)
             + ". Same criterion as above: a gap counts as real only if it exceeds 2*sigma; "
             "1-2 sigma is suggestive; below 1 sigma is indistinguishable from noise.")
    L.append("")
    L.append("| Comparison | Difference | Verdict (proxy sigma=0.0105) | Verdict (probe sigma=" + "{:.4f}".format(sigma) + ") |")
    L.append("|---|---|---|---|")
    for name, a, b in VERDICT_COMPARISONS:
        d = a - b
        proxy_verdict = verdict_line(d, 0.0105)
        probe_verdict = verdict_line(d, sigma)
        L.append("| " + name + " | " + "{:+.4f}".format(d) + " | " + proxy_verdict
                 + " | " + probe_verdict + " |")
    L.append("")

    section = NL.join(L)
    existing = EFFNET_SEED_VARIANCE_MD.read_text(encoding="utf-8")
    EFFNET_SEED_VARIANCE_MD.write_text(existing.rstrip(NL) + NL + NL + section, encoding="utf-8")
    print(NL + "Appended 'Probe-specific seed variance' to " + str(EFFNET_SEED_VARIANCE_MD), flush=True)


if __name__ == "__main__":
    main()
