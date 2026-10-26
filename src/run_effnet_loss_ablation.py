"""
Overnight loss-function ablation for EfficientNet-B0 (7-class HAM10000).

Runs four otherwise-identical two-phase EfficientNet-B0 trainings, one per
loss function (ce / focal / dice / combined), then evaluates each of the
four resulting checkpoints under two protocols (plain argmax, and 5-view
TTA + 10-pass-per-view MC Dropout), and writes a consolidated summary.

Reuses, does not reimplement:
  - train_efficientnet.py's own --loss flag (FocalLoss/DiceLoss/CombinedLoss
    are imported there from train_dino_head.py).
  - eval_saved_models.py's eval_effnet() (via its --effnet-ckpt /
    --effnet-report-name / --dump-json flags) for both eval protocols.

Robustness, for an unattended overnight run:
  - Each of the 4 trainings and each of the 8 evaluations is its own
    subprocess, so a crash (or CUDA OOM) in one can never corrupt or block
    the others -- CUDA context is fully released when the subprocess exits.
  - Every step's result is written to `outputs/effnet_loss_ablation/manifest.json`
    immediately after that step finishes, never buffered to the end.
  - A failure in any one step is caught, logged (with the child's full
    stdout/stderr tail, which contains its traceback) to
    `results/effnet_loss_ablation_errors.log`, and the driver moves on --
    it never aborts the whole job over one failed run.
  - Already-completed loss configs are skipped on a re-invocation (see
    --force to override) -- so if this process itself is killed partway
    through, re-running it resumes rather than restarting from scratch.

Run from project root:
  python src/run_effnet_loss_ablation.py                  # full overnight run
  python src/run_effnet_loss_ablation.py --smoke-test      # 1+1 epoch sanity pass
  python src/run_effnet_loss_ablation.py --only dice        # (re-)run one config
  python src/run_effnet_loss_ablation.py --summarize-only   # rebuild summary.md
                                                             # from whatever is
                                                             # already on disk
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
from typing import NamedTuple, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent  # project root
_SRC = Path(__file__).resolve().parent

RESULTS_DIR = _ROOT / "results"
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# (loss_name, extra CLI args for train_efficientnet.py) -- run in this exact
# order; Run A (ce) is not optional, it is the controlled baseline.
LOSS_CONFIGS = [
    ("ce", []),
    ("focal", ["--gamma", "2.0"]),
    ("dice", []),
    ("combined", ["--gamma", "2.0"]),
]
LOSS_NAMES = [name for name, _ in LOSS_CONFIGS]

DEFAULT_METADATA_CSV = _ROOT / "data" / "HAM10000_metadata.csv"
DEFAULT_IMAGES_DIR = _ROOT / "data"

LEGACY_BASELINE_TXT = RESULTS_DIR / "efficientnet_baseline_7class_results.txt"
LEGACY_IMPROVED_TXT = RESULTS_DIR / "efficientnet_improved_7class_results.txt"

SEED = 42


class Paths(NamedTuple):
    output_root: Path
    manifest_path: Path
    error_log: Path
    summary_path: Path
    label_suffix: str  # "" normally, "_smoketest" under --smoke-test


def make_paths(smoke_test: bool) -> Paths:
    if smoke_test:
        output_root = _ROOT / "outputs" / "effnet_loss_ablation_smoketest"
        return Paths(
            output_root=output_root,
            manifest_path=output_root / "manifest.json",
            error_log=RESULTS_DIR / "effnet_loss_ablation_errors_smoketest.log",
            summary_path=RESULTS_DIR / "effnet_loss_ablation_summary_smoketest.md",
            label_suffix="_smoketest",
        )
    output_root = _ROOT / "outputs" / "effnet_loss_ablation"
    return Paths(
        output_root=output_root,
        manifest_path=output_root / "manifest.json",
        error_log=RESULTS_DIR / "effnet_loss_ablation_errors.log",
        summary_path=RESULTS_DIR / "effnet_loss_ablation_summary.md",
        label_suffix="",
    )


# ---------------------------------------------------------------------------
# Manifest + error log
# ---------------------------------------------------------------------------

def load_manifest(paths: Paths) -> dict:
    if paths.manifest_path.exists():
        return json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    return {"runs": {}}


def save_manifest(paths: Paths, manifest: dict) -> None:
    paths.output_root.mkdir(parents=True, exist_ok=True)
    paths.manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def log_error(paths: Paths, loss_name: str, step: str, exc: BaseException) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().isoformat(timespec="seconds")
    with open(paths.error_log, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 72}\n[{ts}] loss={loss_name}  step={step}\n{'=' * 72}\n")
        f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        f.write("\n")
    print(f"  !! FAILED ({step}) -- see {paths.error_log}", flush=True)


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def run_subprocess(cmd: list[str], log_path: Path) -> None:
    """Run a subprocess, tee combined stdout+stderr to log_path. Raises
    RuntimeError (with the log's tail attached) on a nonzero exit code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  $ {' '.join(cmd)}", flush=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd, cwd=str(_ROOT), stdout=logf, stderr=subprocess.STDOUT, text=True,
        )
    if proc.returncode != 0:
        tail = ""
        try:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        except OSError:
            pass
        raise RuntimeError(
            f"Command failed (exit {proc.returncode}): {' '.join(cmd)}\n"
            f"--- last 4000 chars of {log_path} ---\n{tail}"
        )


# ---------------------------------------------------------------------------
# One training run / one evaluation
# ---------------------------------------------------------------------------

def train_one(loss_name: str, extra_args: list[str], args, paths: Paths) -> dict:
    run_dir = paths.output_root / loss_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best_efficientnet.pth"

    cmd = [
        sys.executable, str(_SRC / "train_efficientnet.py"),
        "--metadata-csv", str(args.metadata_csv),
        "--images-dir", str(args.images_dir),
        "--output-dir", str(run_dir),
        "--loss", loss_name,
        "--seed", str(SEED),
        "--phase1-epochs", str(args.phase1_epochs),
        "--phase2-epochs", str(args.phase2_epochs),
        # The script's own final in-process eval is throwaway (never saved to
        # a file) -- the real plain/tta_mc numbers come from eval_one() below,
        # which reloads the checkpoint independently. --no-mc keeps that
        # throwaway eval to 5 forward passes instead of 50, saving real time
        # across 4 runs for no loss of information.
        "--no-mc",
        *extra_args,
    ]
    t0 = time.time()
    run_subprocess(cmd, run_dir / "train_stdout.log")
    wall_clock_total = time.time() - t0

    if not ckpt_path.exists():
        raise RuntimeError(f"Training reported success but checkpoint not found: {ckpt_path}")

    import torch  # local import: --summarize-only never needs torch at all
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return {
        "loss_name": loss_name,
        "ckpt_path": str(ckpt_path),
        "epoch": ckpt.get("epoch"),
        "phase": ckpt.get("phase"),
        "seed": ckpt.get("seed"),
        "val_macro_f1": ckpt.get("val_macro_f1"),
        "val_mel_recall": ckpt.get("val_mel_recall"),
        "wall_clock_seconds_to_best": ckpt.get("wall_clock_seconds"),
        "wall_clock_seconds_total": wall_clock_total,
        "training_log_csv": str(run_dir / "training_log.csv"),
    }


def eval_one(loss_name: str, ckpt_path: str, mode: str, args, paths: Paths) -> dict:
    run_dir = paths.output_root / loss_name
    report_name = f"effnet_loss_{loss_name}_{mode}{paths.label_suffix}"
    json_path = run_dir / f"{report_name}.json"
    cmd = [
        sys.executable, str(_SRC / "eval_saved_models.py"),
        "--metadata-csv", str(args.metadata_csv),
        "--images-dir", str(args.images_dir),
        "--results-dir", str(RESULTS_DIR),
        "--seed", str(SEED),
        "--only", "effnet",
        "--effnet-mode", mode,
        "--effnet-ckpt", ckpt_path,
        "--effnet-report-name", report_name,
        "--dump-json", str(json_path),
    ]
    if mode == "tta_mc":
        cmd += ["--mc-passes", "10"]
    run_subprocess(cmd, run_dir / f"eval_{mode}_stdout.log")
    if not json_path.exists():
        raise RuntimeError(f"Eval reported success but JSON not found: {json_path}")
    return json.loads(json_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def parse_legacy_report(path: Path) -> Optional[dict]:
    """Parse an existing (pre-JSON-dump) results/*.txt for its headline
    Macro-F1 / Melanoma Recall pair. Returns None rather than raising if the
    file is missing or doesn't match the expected format, so the summary can
    say "could not compare" instead of crashing."""
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    mf1_m = re.search(r"Macro-F1\s*:\s*([\d.]+)", text)
    mr_m = re.search(r"Melanoma Recall\s*:\s*([\d.]+)", text)
    if not (mf1_m and mr_m):
        return None
    return {"macro_f1": float(mf1_m.group(1)), "mel_recall": float(mr_m.group(1))}


def predicted_class_shares(confusion_matrix: list[list[int]]) -> dict[str, float]:
    """Share of ALL test-set predictions landing in each class (column sums
    / grand total) -- this is what reveals a degenerate "always predict X"
    solution, which per-class recall alone does not."""
    cm = np.array(confusion_matrix)
    total = cm.sum()
    col_sums = cm.sum(axis=0)
    return {CLASS_NAMES[j]: float(col_sums[j]) / float(total) for j in range(len(CLASS_NAMES))}


def generate_summary(manifest: dict, paths: Paths) -> str:
    runs = manifest.get("runs", {})
    lines: list[str] = []

    lines.append("# EfficientNet-B0 Loss-Function Ablation -- Summary")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(
        "Four otherwise-identical two-phase EfficientNet-B0 runs (15 epochs "
        "frozen-backbone + 15 epochs full fine-tune, seed=42, AdamW, "
        "weight_decay=1e-4, batch size 32, same lesion-level split, same "
        "train-split rebalancing), differing only in the training loss. "
        "FocalLoss/DiceLoss/CombinedLoss are the exact same implementations "
        "already used by train_dino_head.py for the DINO/BiomedCLIP linear "
        "probes -- not reimplemented here."
    )
    lines.append("")

    # --- 1. Headline table --------------------------------------------------
    lines.append("## 1. Headline results")
    lines.append("")
    lines.append(
        "| Loss | Test Macro-F1 (plain) | Test MEL Recall (plain) | "
        "Test Macro-F1 (TTA+MC) | Test MEL Recall (TTA+MC) | Best epoch | Phase | Wall-clock (train) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for loss_name in LOSS_NAMES:
        r = runs.get(loss_name)
        if r is None:
            lines.append(f"| {loss_name} | NOT RUN YET | - | - | - | - | - | - |")
            continue
        train = r.get("train")
        if not train:
            lines.append(f"| {loss_name} | FAILED (training) | - | - | - | - | - | - |")
            continue
        plain = r.get("eval_plain")
        tta_mc = r.get("eval_tta_mc")
        epoch = train.get("epoch", "?")
        phase = train.get("phase", "?")
        wc = train.get("wall_clock_seconds_total")
        wc_str = f"{wc / 60:.1f} min" if isinstance(wc, (int, float)) else "?"
        plain_f1 = f"{plain['macro_f1']:.4f}" if plain else "FAILED"
        plain_mr = f"{plain['mel_recall']:.4f}" if plain else "FAILED"
        mc_f1 = f"{tta_mc['macro_f1']:.4f}" if tta_mc else "FAILED"
        mc_mr = f"{tta_mc['mel_recall']:.4f}" if tta_mc else "FAILED"
        lines.append(
            f"| {loss_name} | {plain_f1} | {plain_mr} | {mc_f1} | {mc_mr} | "
            f"{epoch} | {phase} | {wc_str} |"
        )
    lines.append("")

    # --- 2. Per-class F1 (TTA+MC) -------------------------------------------
    lines.append("## 2. Per-class F1 (TTA+MC evaluation)")
    lines.append("")
    lines.append("| Loss | " + " | ".join(CLASS_NAMES) + " |")
    lines.append("|---|" + "---|" * len(CLASS_NAMES))
    for loss_name in LOSS_NAMES:
        tta_mc = (runs.get(loss_name) or {}).get("eval_tta_mc")
        if not tta_mc:
            lines.append(f"| {loss_name} | " + " | ".join(["-"] * len(CLASS_NAMES)) + " |")
            continue
        row = [f"{tta_mc['per_class'][c]['f1']:.2f}" for c in CLASS_NAMES]
        lines.append(f"| {loss_name} | " + " | ".join(row) + " |")
    lines.append("")

    # --- 3. Degenerate-solution check ---------------------------------------
    lines.append("## 3. Degenerate-solution check (TTA+MC evaluation)")
    lines.append("")
    lines.append(
        "Share of ALL test-set predictions landing in each class (column "
        "sums of the confusion matrix / total). `nv` is 1004/1502 = 66.8% of "
        "the test set by true support, so some nv-heaviness in the share "
        "column is expected -- the verdict flags actual collapse: nv recall "
        "dropping toward 0 (the failure mode Dice produced on the DINO "
        "linear probes, where nv recall hit exactly 0.00), or any single "
        "class absorbing an implausible share of all predictions."
    )
    lines.append("")
    lines.append(
        "| Loss | " + " | ".join(f"{c} share" for c in CLASS_NAMES) + " | nv recall | Verdict |"
    )
    lines.append("|---|" + "---|" * (len(CLASS_NAMES) + 2))
    for loss_name in LOSS_NAMES:
        tta_mc = (runs.get(loss_name) or {}).get("eval_tta_mc")
        if not tta_mc:
            lines.append(f"| {loss_name} | " + " | ".join(["-"] * (len(CLASS_NAMES) + 2)) + " |")
            continue
        shares = predicted_class_shares(tta_mc["confusion_matrix"])
        nv_recall = tta_mc["per_class"]["nv"]["recall"]
        max_class, max_share = max(shares.items(), key=lambda kv: kv[1])
        notes = []
        if nv_recall < 0.05:
            notes.append("DEGENERATE: nv recall near zero")
        elif nv_recall < 0.30:
            notes.append("nv recall collapsing")
        if max_share > 0.80:
            notes.append(f"possible degenerate: {max_class} absorbs {max_share * 100:.0f}% of all predictions")
        verdict = "; ".join(notes) if notes else "OK"
        row = [f"{shares[c] * 100:.1f}%" for c in CLASS_NAMES]
        lines.append(f"| {loss_name} | " + " | ".join(row) + f" | {nv_recall:.4f} | {verdict} |")
    lines.append("")

    # --- 4. Run A vs existing saved reports ---------------------------------
    lines.append("## 4. Run A (weighted CE) vs. the existing saved EfficientNet reports")
    lines.append("")
    legacy_plain = parse_legacy_report(LEGACY_BASELINE_TXT)
    legacy_mc = parse_legacy_report(LEGACY_IMPROVED_TXT)
    ce_run = runs.get("ce") or {}
    ce_plain = ce_run.get("eval_plain")
    ce_mc = ce_run.get("eval_tta_mc")

    lines.append("| | Macro-F1 (plain) | MEL Recall (plain) | Macro-F1 (TTA+MC) | MEL Recall (TTA+MC) |")
    lines.append("|---|---|---|---|---|")
    if legacy_plain and legacy_mc:
        lines.append(
            f"| Existing saved (`efficientnet_baseline_7class` / `_improved_7class`) "
            f"| {legacy_plain['macro_f1']:.4f} | {legacy_plain['mel_recall']:.4f} "
            f"| {legacy_mc['macro_f1']:.4f} | {legacy_mc['mel_recall']:.4f} |"
        )
    else:
        lines.append("| Existing saved | N/A -- could not parse legacy report(s) | | | |")
    if ce_plain and ce_mc:
        lines.append(
            f"| This run (Run A, weighted CE, seed=42, fixed checkpoint format) "
            f"| {ce_plain['macro_f1']:.4f} | {ce_plain['mel_recall']:.4f} "
            f"| {ce_mc['macro_f1']:.4f} | {ce_mc['mel_recall']:.4f} |"
        )
        if legacy_plain and legacy_mc:
            d_f1_plain = ce_plain["macro_f1"] - legacy_plain["macro_f1"]
            d_mr_plain = ce_plain["mel_recall"] - legacy_plain["mel_recall"]
            d_f1_mc = ce_mc["macro_f1"] - legacy_mc["macro_f1"]
            d_mr_mc = ce_mc["mel_recall"] - legacy_mc["mel_recall"]
            lines.append(
                f"| Difference (this run - existing) "
                f"| {d_f1_plain:+.4f} | {d_mr_plain:+.4f} | {d_f1_mc:+.4f} | {d_mr_mc:+.4f} |"
            )
    else:
        lines.append("| This run (Run A) | FAILED or not yet run -- see errors log | | | |")
    lines.append("")
    lines.append(
        "The existing saved report's checkpoint has an unrecoverable seed "
        "(never stored -- see `EXPERIMENTAL_RECORD.md`, action-list item 5), "
        "so any difference above reflects the combined effect of a (possibly "
        "different) historical seed plus ordinary run-to-run stochasticity "
        "(DataLoader shuffling, augmentation sampling, cuDNN nondeterminism) "
        "-- it is not a controlled ablation of the seed alone, since the "
        "historical run cannot be reproduced or inspected further. Run A "
        "here uses the now-fixed checkpoint format (seed=42 explicit; "
        "epoch, phase and wall-clock recorded), so this gap cannot recur "
        "for future runs of this script."
    )
    lines.append("")

    # --- Errors --------------------------------------------------------------
    failed_training = [ln for ln in LOSS_NAMES if not (runs.get(ln) or {}).get("train")]
    if failed_training or paths.error_log.exists():
        lines.append("## Errors")
        lines.append("")
        if failed_training:
            lines.append(f"Training did not complete for: {', '.join(failed_training)}.")
        if paths.error_log.exists():
            lines.append(f"Full tracebacks (if any) are in `{paths.error_log.name}`.")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Overnight EfficientNet-B0 loss-function ablation")
    p.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    p.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    p.add_argument("--phase1-epochs", type=int, default=15)
    p.add_argument("--phase2-epochs", type=int, default=15)
    p.add_argument("--smoke-test", action="store_true",
                   help="Override to 1+1 epochs and use a separate *_smoketest "
                        "output root / result names, for a fast end-to-end "
                        "pipeline check before committing to the real run.")
    p.add_argument("--only", choices=LOSS_NAMES, default=None,
                   help="Run/re-run a single loss config instead of all four.")
    p.add_argument("--force", action="store_true",
                   help="Redo a loss config even if the manifest already shows "
                        "it fully complete (default: skip, so an interrupted "
                        "overnight run can be resumed by simply re-invoking).")
    p.add_argument("--summarize-only", action="store_true",
                   help="Skip all training/eval; just rebuild the summary .md "
                        "from whatever manifest + JSON already exists on disk.")
    return p.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    if args.smoke_test:
        args.phase1_epochs = 1
        args.phase2_epochs = 1
    paths = make_paths(args.smoke_test)
    manifest = load_manifest(paths)
    manifest.setdefault("runs", {})

    configs = LOSS_CONFIGS if args.only is None else [c for c in LOSS_CONFIGS if c[0] == args.only]

    if not args.summarize_only:
        for loss_name, extra_args in configs:
            record = manifest["runs"].get(loss_name) or {}
            already_done = bool(record.get("train") and record.get("eval_plain") and record.get("eval_tta_mc"))
            if already_done and not args.force:
                print(f"\n{'=' * 64}\nLOSS = {loss_name}  (already complete -- skipping; use --force to redo)\n{'=' * 64}", flush=True)
                continue

            print(f"\n{'=' * 64}\nLOSS = {loss_name}\n{'=' * 64}", flush=True)
            record = {}
            manifest["runs"][loss_name] = record

            try:
                record["train"] = train_one(loss_name, extra_args, args, paths)
                save_manifest(paths, manifest)
                t = record["train"]
                print(f"  OK  train  (epoch={t['epoch']}, phase={t['phase']}, "
                      f"val_macro_f1={t['val_macro_f1']:.4f}, "
                      f"wall_clock={t['wall_clock_seconds_total'] / 60:.1f} min)", flush=True)
            except Exception as exc:
                log_error(paths, loss_name, "train", exc)
                record["train"] = None
                save_manifest(paths, manifest)
                continue  # nothing to evaluate without a checkpoint

            for mode, key in (("plain", "eval_plain"), ("tta_mc", "eval_tta_mc")):
                try:
                    record[key] = eval_one(loss_name, record["train"]["ckpt_path"], mode, args, paths)
                    save_manifest(paths, manifest)
                    print(f"  OK  eval[{mode}]  macro_f1={record[key]['macro_f1']:.4f}  "
                          f"mel_recall={record[key]['mel_recall']:.4f}", flush=True)
                except Exception as exc:
                    log_error(paths, loss_name, f"eval[{mode}]", exc)
                    record[key] = None
                    save_manifest(paths, manifest)

    summary_md = generate_summary(manifest, paths)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    paths.summary_path.write_text(summary_md, encoding="utf-8")
    print(f"\nSummary written: {paths.summary_path}", flush=True)


if __name__ == "__main__":
    main()
