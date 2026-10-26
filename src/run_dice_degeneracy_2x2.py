"""
Isolate why Dice loss degenerates on the frozen DINOv2 linear probe (nv
recall = 0.00) but was the best loss on EfficientNet-B0 full fine-tuning
(nv recall = 0.8486).

Two competing explanations:
  H1 (capacity)    -- a linear layer on frozen features cannot escape the
                       degenerate solution; a full network can reshape its
                       own features to satisfy the Dice objective.
  H2 (rebalancing) -- EfficientNet does train-split resampling BEFORE the
                       loss sees the data; the probes rely on
                       WeightedRandomSampler alone.

2x2 on DINOv2 features, Dice loss only, using train_dino_head.py's
--rebalance-train / --no-sampler flags added for this experiment. All four
runs are trained fresh here -- Run 1 is NOT reused from Part A's saved JSON,
specifically so its result can be compared against Part A's number as a
determinism check (same seed, same code, same features -- it should
reproduce exactly; if it doesn't, that's reported before anything else):

  Run 1 (rebalance=no,  sampler=yes) -- Part A's exact control config, rerun
  Run 2 (rebalance=yes, sampler=no)
  Run 3 (rebalance=yes, sampler=yes)
  Run 4 (rebalance=no,  sampler=no)

Writes to:
  outputs/dice_degeneracy_2x2/run{1,2,3,4}/
  results/dice_degeneracy_run{1,2,3,4}_results.txt (+ .json)
  results/dice_degeneracy_2x2.md

Run from project root:
  python src/run_dice_degeneracy_2x2.py
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
OUT_ROOT = _ROOT / "outputs" / "dice_degeneracy_2x2"
SUMMARY_PATH = RESULTS_DIR / "dice_degeneracy_2x2.md"

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
DEGENERATE_NV_RECALL = 0.10

FEATS_DIR = _ROOT / "data" / "processed" / "dinov2"
FEATS_PREFIX = "dinov2_feats"

# Part A's original dinov2+dice control result (outputs/probe_loss_ablation/
# dinov2_dice/), for the Run 1 determinism check -- same seed, same code,
# same features as Run 1 here; it should reproduce exactly.
PART_A_CONTROL = {"macro_f1": 0.4067328642589158, "mel_recall": 0.9221556886227545,
                  "nv_recall": 0.0}
DETERMINISM_TOLERANCE = 1e-6

# (run_id, rebalance, sampler, extra train flags)
RUNS = [
    (1, False, True, []),
    (2, True, False, ["--rebalance-train", "--no-sampler"]),
    (3, True, True, ["--rebalance-train"]),
    (4, False, False, ["--no-sampler"]),
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


def train_one(run_id: int, extra_args: list) -> dict:
    run_dir = OUT_ROOT / ("run" + str(run_id))
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, _SRC / "train_dino_head.py",
           "--features-dir", FEATS_DIR,
           "--features-prefix", FEATS_PREFIX,
           "--output-dir", run_dir,
           "--loss", "dice",
           "--seed", "42",
           *extra_args]
    t0 = time.time()
    run_subprocess(cmd, run_dir / "train_stdout.log")
    wall = time.time() - t0
    ckpt = run_dir / "best_dino_head.pth"
    if not ckpt.exists():
        raise RuntimeError("training finished but no checkpoint at " + str(ckpt))
    import torch
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    return {"run_id": run_id, "ckpt_path": str(ckpt), "epoch": c.get("epoch"),
            "rebalance_train": c.get("rebalance_train"), "used_sampler": c.get("used_sampler"),
            "val_macro_f1": c.get("val_macro_f1"), "wall_clock_seconds_total": wall}


def eval_one(run_id: int, ckpt_path: str) -> dict:
    run_dir = OUT_ROOT / ("run" + str(run_id))
    report_name = "dice_degeneracy_run" + str(run_id)
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


def pred_shares(cm) -> dict:
    a = np.array(cm)
    total = a.sum()
    cols = a.sum(axis=0)
    return {CLASS_NAMES[j]: float(cols[j]) / float(total) for j in range(len(CLASS_NAMES))}


def build_row_table(runs_data: dict) -> tuple[list, dict]:
    lines = ["| Run | Rebalance | Sampler | Macro-F1 | MEL Recall | nv recall | Degenerate |",
             "|---|---|---|---|---|---|---|"]
    degenerate_flags = {}
    for rid in (1, 2, 3, 4):
        r = runs_data[rid]
        ev = r["eval"]
        nv_recall = ev["per_class"]["nv"]["recall"]
        deg = nv_recall < DEGENERATE_NV_RECALL
        degenerate_flags[rid] = deg
        lines.append(
            "| " + str(rid) + " | " + ("yes" if r["rebalance"] else "no")
            + " | " + ("yes" if r["sampler"] else "no")
            + " | " + "{:.4f}".format(ev["macro_f1"])
            + " | " + "{:.4f}".format(ev["mel_recall"])
            + " | " + "{:.4f}".format(nv_recall)
            + " | " + ("YES" if deg else "no") + " |"
        )
    return lines, degenerate_flags


def build_share_table(runs_data: dict) -> list:
    lines = ["| Run | " + " | ".join(CLASS_NAMES) + " |",
             "|---|" + ("---|" * len(CLASS_NAMES))]
    for rid in (1, 2, 3, 4):
        ev = runs_data[rid]["eval"]
        sh = pred_shares(ev["confusion_matrix"])
        lines.append("| " + str(rid) + " | "
                     + " | ".join("{:.1f}%".format(sh[c] * 100.0) for c in CLASS_NAMES) + " |")
    return lines


def verdict_text(degenerate_flags: dict) -> str:
    d1, d2, d3, d4 = (degenerate_flags[1], degenerate_flags[2],
                      degenerate_flags[3], degenerate_flags[4])
    lines = []
    if (not d2) and (not d3):
        lines.append(
            "**H2 supported: rebalancing is the cause, model capacity is irrelevant.** "
            "Both runs with train-split rebalancing (2: rebalance-only, 3: rebalance+sampler) "
            "do NOT degenerate, regardless of whether the sampler is also present. The linear "
            "probe is perfectly capable of avoiding the collapse once the class distribution it "
            "actually trains on is rebalanced -- the earlier failure on frozen features was a "
            "property of the imbalance-handling scheme, not a capacity limit of a linear layer."
        )
    elif d1 and d2 and d3 and d4:
        lines.append(
            "**H1 supported: all four configurations degenerate, including both rebalanced "
            "runs.** Model capacity is the more likely explanation -- a linear layer on frozen "
            "features collapses under Dice regardless of the imbalance-handling scheme, and "
            "escaping the collapse requires reshaping the features themselves (full "
            "fine-tuning), not just rebalancing which examples it sees."
        )
    else:
        lines.append(
            "**Mixed result -- neither H1 nor H2 cleanly explains all four runs.** "
            "Rebalancing changes the outcome for at least one of runs 2/3 but not both, "
            "or a rebalanced run still degenerates while the other does not. Read the "
            "per-run table directly rather than collapsing this to a single verdict."
        )
    if d1 != d4:
        lines.append("")
        lines.append(
            "**Additional finding: the sampler is not inert.** Run 4 (neither rebalancing "
            "nor sampler) " + ("also degenerates like Run 1" if d4 else "does NOT degenerate, "
            "unlike Run 1") + " (sampler-only). Run 1 and Run 4 differ only in the "
            "WeightedRandomSampler, so " + (
                "the sampler's presence/absence made no difference here -- rebalancing (or its "
                "absence) is doing all the work, further supporting H2."
                if d1 == d4 else
                "the sampler itself is actively contributing to (or protecting against) the "
                "collapse, independent of rebalancing -- a third mechanism beyond the H1/H2 "
                "framing as originally posed."
            )
        )
    else:
        lines.append("")
        lines.append(
            "Run 1 and Run 4 (identical except for the sampler) both "
            + ("degenerate" if d1 else "avoid degenerating")
            + " -- the sampler alone does not flip the outcome one way or the other here."
        )
    return "\n".join(lines)


def determinism_check(eval_rec: dict) -> tuple[bool, str]:
    """Run 1 uses the identical seed/code/features as Part A's dinov2+dice
    control -- it should reproduce exactly. Returns (matches, report_text)."""
    d_f1 = eval_rec["macro_f1"] - PART_A_CONTROL["macro_f1"]
    d_mr = eval_rec["mel_recall"] - PART_A_CONTROL["mel_recall"]
    d_nv = eval_rec["per_class"]["nv"]["recall"] - PART_A_CONTROL["nv_recall"]
    matches = (abs(d_f1) < DETERMINISM_TOLERANCE and abs(d_mr) < DETERMINISM_TOLERANCE
              and abs(d_nv) < DETERMINISM_TOLERANCE)
    if matches:
        text = ("Run 1 reproduced Part A's dinov2+dice control EXACTLY "
                "(macro_f1={:.6f}, mel_recall={:.6f}, nv_recall={:.6f}). "
                "Pipeline is deterministic; the runs below are comparable.").format(
                    eval_rec["macro_f1"], eval_rec["mel_recall"], eval_rec["per_class"]["nv"]["recall"])
    else:
        text = ("*** DETERMINISM PROBLEM *** Run 1 did NOT reproduce Part A's "
                "dinov2+dice control despite identical seed=42, identical code, "
                "identical features. Part A: macro_f1={:.6f}, mel_recall={:.6f}, "
                "nv_recall={:.6f}. This run: macro_f1={:.6f}, mel_recall={:.6f}, "
                "nv_recall={:.6f}. Differences: d_macro_f1={:+.6f}, d_mel_recall={:+.6f}, "
                "d_nv_recall={:+.6f}. The pipeline is NOT bit-reproducible on this "
                "machine -- treat every cross-run comparison in this report (and "
                "possibly elsewhere in this repo) with that in mind.").format(
                    PART_A_CONTROL["macro_f1"], PART_A_CONTROL["mel_recall"], PART_A_CONTROL["nv_recall"],
                    eval_rec["macro_f1"], eval_rec["mel_recall"], eval_rec["per_class"]["nv"]["recall"],
                    d_f1, d_mr, d_nv)
    return matches, text


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    runs_data: dict = {}
    determinism_text = ""

    for run_id, rebalance, sampler, extra_args in RUNS:
        print(chr(10) + "=" * 64)
        print("RUN " + str(run_id) + "  (rebalance=" + str(rebalance) + ", sampler=" + str(sampler) + ")")
        print("=" * 64, flush=True)
        train_rec = train_one(run_id, extra_args)
        print("  OK  train  epoch=" + str(train_rec["epoch"])
              + "  val_macro_f1=" + "{:.4f}".format(train_rec["val_macro_f1"])
              + "  rebalance_train(stored)=" + str(train_rec["rebalance_train"])
              + "  used_sampler(stored)=" + str(train_rec["used_sampler"]), flush=True)
        eval_rec = eval_one(run_id, train_rec["ckpt_path"])
        print("  OK  eval   macro_f1=" + "{:.4f}".format(eval_rec["macro_f1"])
              + "  mel_recall=" + "{:.4f}".format(eval_rec["mel_recall"])
              + "  nv_recall=" + "{:.4f}".format(eval_rec["per_class"]["nv"]["recall"]), flush=True)
        runs_data[run_id] = {"rebalance": rebalance, "sampler": sampler,
                             "train": train_rec, "eval": eval_rec}

        if run_id == 1:
            matches, determinism_text = determinism_check(eval_rec)
            print(flush=True)
            print(determinism_text, flush=True)
            print(flush=True)

    table_lines, degenerate_flags = build_row_table(runs_data)
    share_lines = build_share_table(runs_data)
    verdict = verdict_text(degenerate_flags)

    md = []
    md.append("# Dice Degeneracy 2x2 -- DINOv2 Frozen Probe vs. EfficientNet")
    md.append("")
    md.append("Generated: " + datetime.now().isoformat(timespec="seconds"))
    md.append("")
    md.append("Question: Dice loss gives nv recall = 0.00 on all 5 frozen linear probes "
              "(Part A), but was the BEST loss on EfficientNet-B0 full fine-tuning "
              "(0.7029 Macro-F1, nv recall 0.8486). H1 (capacity) says a linear head on "
              "frozen features cannot escape the collapse; H2 (rebalancing scheme) says "
              "EfficientNet's train-split resampling (done before the loss ever sees the "
              "data) is what actually prevents it, not model capacity. All 4 runs below "
              "are DINOv2 features, Dice loss, seed=42 -- only rebalancing and sampler vary.")
    md.append("")
    md.append("## Determinism check (Run 1 vs. Part A control)")
    md.append("")
    md.append(determinism_text)
    md.append("")
    md.append("## Results")
    md.append("")
    md.extend(table_lines)
    md.append("")
    md.append("Degenerate = nv recall < " + "{:.2f}".format(DEGENERATE_NV_RECALL) + ".")
    md.append("")
    md.append("## Share of all test predictions landing in each class")
    md.append("")
    md.extend(share_lines)
    md.append("")
    md.append("Reference: `nv` is 1004/1502 = 66.8% of the test set by true support.")
    md.append("")
    md.append("## Verdict")
    md.append("")
    md.append(verdict)
    md.append("")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(chr(10).join(md), encoding="utf-8")
    print(chr(10) + "Summary written: " + str(SUMMARY_PATH))


if __name__ == "__main__":
    main()
