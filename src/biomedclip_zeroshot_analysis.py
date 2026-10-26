"""
BiomedCLIP Zero-Shot Analysis — Part A + Part B.

Part A — Semantic Crowding Visualization:
    Encodes averaged class embeddings for Phase 1 (generic) and Phase 2
    (MedGPT-derived dermoscopic) prompts, then plots side-by-side 7×7 cosine
    similarity heatmaps showing how well the text space separates classes.
    Output: results/semantic_crowding_heatmap.png

Part B — Phase 3 Zero-Shot (Gemini-derived prompts):
    Runs Averaged + Max-Sim zero-shot evaluation on the HAM10000 test split
    and prints a 6-row comparison table covering all three prompt phases.
    Output: results/biomedclip_zeroshot_phase3_results.txt

Run from project root:
    python src/biomedclip_zeroshot_analysis.py \\
        --metadata-csv data/HAM10000_metadata.csv \\
        --images-dir data \\
        --output-dir results
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import Config
from dataset import SkinLesionDataset, lesion_level_split
from helpers import get_device, set_seed

# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------

LABEL_MAP: dict[str, int] = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,
}
IDX_TO_CLASS: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}
CLASS_ORDER: list[str] = [IDX_TO_CLASS[i] for i in range(len(LABEL_MAP))]
MEL_IDX: int = LABEL_MAP["mel"]

# ---------------------------------------------------------------------------
# Phase 1 prompts — generic labels
# ---------------------------------------------------------------------------

PROMPTS_P1: dict[str, list[str]] = {
    "mel": [
        "This is a melanoma",
        "This is a malignant melanoma",
        "This is a skin cancer called melanoma",
        "A dermoscopy image of melanoma",
        "This is a cancerous pigmented skin lesion",
    ],
    "nv": [
        "This is a nevus",
        "This is a benign mole",
        "This is a melanocytic nevus",
        "A dermoscopy image of a common mole",
        "This is a benign pigmented skin lesion",
    ],
    "bkl": [
        "This is a benign keratosis",
        "This is a seborrheic keratosis",
        "This is a solar lentigo",
        "A dermoscopy image of benign keratosis",
        "This is a benign keratosis-like lesion",
    ],
    "bcc": [
        "This is a basal cell carcinoma",
        "This is a basal cell cancer",
        "This is a skin cancer called basal cell carcinoma",
        "A dermoscopy image of basal cell carcinoma",
        "This is a non-melanoma skin cancer",
    ],
    "akiec": [
        "This is an actinic keratosis",
        "This is a solar keratosis",
        "This is a precancerous skin lesion",
        "A dermoscopy image of actinic keratosis",
        "This is an intraepithelial carcinoma",
    ],
    "vasc": [
        "This is a vascular lesion",
        "This is an angioma",
        "This is a vascular skin growth",
        "A dermoscopy image of a vascular lesion",
        "This is a benign vascular skin lesion",
    ],
    "df": [
        "This is a dermatofibroma",
        "This is a benign skin fibroma",
        "This is a dermal fibroma",
        "A dermoscopy image of dermatofibroma",
        "This is a fibrous skin lesion",
    ],
}

# ---------------------------------------------------------------------------
# Phase 2 prompts — MedGPT dermoscopic descriptions
# ---------------------------------------------------------------------------

PROMPTS_P2: dict[str, list[str]] = {
    "mel": [
        "A dermoscopy image of an asymmetric melanoma with irregular borders, multicomponent pattern, and marked color variegation including brown, black, red, and white",
        "A dermoscopy image of melanoma showing chaotic architecture, atypical pigment network, eccentric black blotches, and regression structures",
        "A dermoscopy image of a pigmented melanoma with irregular structureless areas, dark clods, whitish scar-like depigmentation, and gray peppering",
        "A dermoscopy image of melanoma with asymmetrically distributed colors, ill-defined margins, hemorrhagic red area, and black structureless pigmentation",
        "A dermoscopy image of cutaneous melanoma demonstrating multicolor asymmetry, irregular blotches, focal regression, and a disorganized dermoscopic pattern",
    ],
    "nv": [
        "A dermoscopy image of a small oval brown melanocytic nevus with a darker central structureless blotch and softer peripheral pigmentation",
        "A dermoscopy image of a benign-appearing melanocytic nevus showing a light-to-medium brown reticuloglobular pattern with mild central hyperpigmentation",
        "A dermoscopy image of an oval tan-brown nevus with delicate pigment network, subtle globular areas, and overall limited color variation",
        "A dermoscopy image of a melanocytic nevus with symmetric brown pigmentation, a faint network, and a central darker blotch on pink skin",
        "A dermoscopy image of a common melanocytic nevus demonstrating uniform brown shades, soft borders, and a reticular to structureless central pattern",
    ],
    "bkl": [
        "A dermoscopy image of a well-circumscribed light-to-dark brown seborrheic keratosis with milia-like cysts and comedo-like openings",
        "A dermoscopy image of a keratotic tan-brown plaque showing cerebriform fissures and ridges on a pink background",
        "A dermoscopy image of a sharply demarcated benign keratosis with moth-eaten borders, surface scale, and multiple keratin plugs",
        "A dermoscopy image of a lobulated brown seborrheic keratosis with structureless pigmentation and whitish-yellow clods",
        "A dermoscopy image of a papillomatous benign keratosis displaying fissures, milia-like cysts, and absent pigment network",
    ],
    "bcc": [
        "A dermoscopy image of basal cell carcinoma with pink translucent stroma, arborizing telangiectasia, and blue-gray ovoid nests",
        "A dermoscopy image of pigmented basal cell carcinoma showing a structureless pink background, brown-gray globules, and focal ulceration",
        "A dermoscopy image of basal cell carcinoma with fine branching vessels, shiny white blotches, and blue-gray pigmented clods",
        "A dermoscopy image of basal cell carcinoma lacking a pigment network and containing arborizing vessels with gray-brown ovoid foci",
        "A dermoscopy image of pigmented basal cell carcinoma with asymmetric pink-white areas, microarborizing telangiectasia, and focal erosive crusts",
    ],
    "akiec": [
        "A dermoscopy image of a pink scaly actinic keratosis with a follicular pseudonetwork and perifollicular white halos",
        "A dermoscopy image of intraepithelial carcinoma showing erythematous structureless areas, adherent keratin, and enlarged follicular openings",
        "A dermoscopy image of pigmented actinic keratosis with focal brown-gray blotches on a pink keratinizing background",
        "A dermoscopy image of Bowen disease with surface scale, white structureless zones, and subtle dotted to linear irregular vessels",
        "A dermoscopy image of a non-melanocytic keratinizing plaque with follicular accentuation and a strawberry-pattern appearance",
    ],
    "vasc": [
        "A dermoscopy image of a well-circumscribed symmetric red-purple angioma with multiple lacunae on a pink background",
        "A dermoscopy image of a dome-shaped cherry angioma showing violaceous to red clods separated by pale septa",
        "A dermoscopy image of a round vascular papule with homogeneous reddish-purple coloration and a classic lacunar pattern",
        "A dermoscopy image of an angioma with sharply defined borders, lobulated red-purple lagoons, and no pigment network",
        "A dermoscopy image of a benign-appearing vascular lesion composed of clustered dark red to violaceous lacunae",
    ],
    "df": [
        "A dermoscopy image of a symmetric dermatofibroma with a central white scar-like patch and faint peripheral pigment network",
        "A dermoscopy image of a pink-brown dermatofibroma showing a structureless fibrotic white center and delicate light brown rim",
        "A dermoscopy image of a dermatofibroma with central shiny white scar-like areas and subtle symmetric peripheral reticular pigmentation",
        "A dermoscopy image of a round pink dermatofibroma with a pale central stellate white area and a thin brown peripheral network",
        "A dermoscopy image of a benign-appearing dermatofibroma with central white fibrosis, mild erythema, and symmetric peripheral pigmentation",
    ],
}

# ---------------------------------------------------------------------------
# Phase 3 prompts — Gemini-derived dermoscopic descriptions
# ---------------------------------------------------------------------------

PROMPTS_P3: dict[str, list[str]] = {
    "mel": [
        "Malignant melanoma characterized by an atypical pigment network and peripheral pseudopods",
        "Invasive melanoma exhibiting chaotic structural disorganization and a blue-white veil",
        "Malignant melanocytic lesion showing distinct chrysalis structures and abrupt edge cutoff",
        "Advanced melanoma displaying prominent regression structures and blue-peppered milky areas",
        "Malignant skin tumor featuring a multicomponent pattern with atypical streaks and streaming",
    ],
    "nv": [
        "Benign melanocytic nevus showing a perfectly symmetric typical reticular grid",
        "Common nevus with a uniform regular cobblestone pattern and even melanin distribution",
        "Benign dermal nevus displaying homogeneous tan coloration and regular dot-globular patterns",
        "Symmetric junctional nevus characterized by orderly gradient fading toward the periphery",
        "Benign mole presenting a typical targetoid architecture without structural atypia",
    ],
    "bkl": [
        "Seborrheic keratosis presenting distinct milia-like horn cysts and comedo-like openings",
        "Benign keratosis displaying classic cerebriform ridges, fissures, and a stuck-on warty appearance",
        "Solar lentigo characterized by sharply demarcated moth-eaten borders and fingerprint structures",
        "Lichen planus-like keratosis showing prominent opaque scale crust and dense hairpin vascular loops",
        "Pigmented seborrheic keratosis featuring sharp marginations, fat fingers, and crypts",
    ],
    "bcc": [
        "Basal cell carcinoma distinctly featuring sharp arborizing telangiectatic vessels",
        "Nodular basal cell carcinoma showing prominent blue-gray ovoid nests and maple-leaf-like areas",
        "Superficial basal cell carcinoma displaying spoke-wheel structures and translucent opalescent areas",
        "Malignant basaloid tumor presenting in-focus branching capillaries and multiple gray-blue globules",
        "Pigmented basal cell carcinoma characterized by concentric structures and superficial rodent ulcerations",
    ],
    "akiec": [
        "Actinic keratosis showing a prominent strawberry pattern with a red pseudonetwork",
        "Squamous cell carcinoma in situ characterized by clustered glomerular vessels and yellow surface keratin",
        "Bowen disease presenting opaque keratotic scales and targetoid hair follicles",
        "Pre-malignant epidermal lesion displaying four-dot rosettes and white-haloed follicular ostia",
        "Keratinizing actinic dysplasia featuring a hyperkeratotic surface and pinpoint dotted vessels",
    ],
    "vasc": [
        "Cherry angioma composed entirely of sharply demarcated red-purple lagoons",
        "Benign hemangioma displaying multiple distinct violaceous vascular lacunae",
        "Angiokeratoma characterized by dark blue-to-black blood-filled spaces divided by pale septa",
        "Vascular malformation presenting a pure lacunar pattern without any melanocytic structures",
        "Thrombosed angioma showing clotted erythrocytic pools and distinct red saccular dilations",
    ],
    "df": [
        "Dermatofibroma distinctly featuring a hard central white scar-like patch",
        "Benign fibrous histiocytoma showing a bright stellate central fibrotic area",
        "Dermatofibroma displaying a dense collagenous center with a fine peripheral hyperpigmented ring",
        "Fibroblastic skin nodule characterized by a classic dimple-sign center",
        "Benign dermal fibrosis presenting a central shiny patch surrounded by a delicate faint rim",
    ],
}

# ---------------------------------------------------------------------------
# Hard-coded prior-phase baselines (from previous runs)
# ---------------------------------------------------------------------------

BASELINES: dict[str, dict[str, float]] = {
    "Phase1 Averaged": {"macro_f1": 0.201, "mel_recall": 0.401},
    "Phase1 Max-Sim":  {"macro_f1": 0.157, "mel_recall": 0.443},
    "Phase2 Averaged": {"macro_f1": 0.151, "mel_recall": 0.108},
    "Phase2 Max-Sim":  {"macro_f1": 0.142, "mel_recall": 0.012},
}

# ---------------------------------------------------------------------------
# BiomedCLIP model loading
# ---------------------------------------------------------------------------

BIOMEDCLIP_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def load_biomedclip(device: torch.device):
    """Return (model, preprocess, tokenizer) for BiomedCLIP."""
    try:
        from open_clip import create_model_from_pretrained, get_tokenizer
    except ImportError:
        raise SystemExit(
            "open_clip_torch is not installed.\nRun:  pip install open_clip_torch"
        )
    model, preprocess = create_model_from_pretrained(BIOMEDCLIP_ID)
    tokenizer = get_tokenizer(BIOMEDCLIP_ID)
    model = model.to(device)
    model.eval()
    return model, preprocess, tokenizer


# ---------------------------------------------------------------------------
# Text embedding helpers
# ---------------------------------------------------------------------------

def build_averaged_embeddings(
    model: torch.nn.Module,
    tokenizer,
    prompts_dict: dict[str, list[str]],
    device: torch.device,
) -> torch.Tensor:
    """
    For each class (in CLASS_ORDER): encode all prompts → L2-norm → mean → L2-norm.
    Returns (C, D).
    """
    embeddings: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for cls in CLASS_ORDER:
            tokens = tokenizer(prompts_dict[cls]).to(device)
            feats = F.normalize(model.encode_text(tokens), dim=-1)  # (P, D)
            avg = F.normalize(feats.mean(dim=0), dim=-1)            # (D,)
            embeddings.append(avg)
    return torch.stack(embeddings, dim=0)  # (C, D)


def build_per_prompt_embeddings(
    model: torch.nn.Module,
    tokenizer,
    prompts_dict: dict[str, list[str]],
    device: torch.device,
) -> torch.Tensor:
    """
    For each class (in CLASS_ORDER): encode all prompts → L2-norm each.
    Returns (C, P, D).
    """
    embeddings: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for cls in CLASS_ORDER:
            tokens = tokenizer(prompts_dict[cls]).to(device)
            feats = F.normalize(model.encode_text(tokens), dim=-1)  # (P, D)
            embeddings.append(feats)
    return torch.stack(embeddings, dim=0)  # (C, P, D)


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def encode_test_images(
    model: torch.nn.Module,
    df_test: pd.DataFrame,
    images_dir: Path,
    col_image_id: str,
    preprocess,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (image_features, true_labels) with L2-normalised image features."""
    ds = SkinLesionDataset(df_test, images_dir, col_image_id, transform=preprocess)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    all_feats: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Encoding test images"):
            feats = F.normalize(model.encode_image(images.to(device)), dim=-1)
            all_feats.append(feats.cpu())
            all_labels.append(labels)
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)


# ---------------------------------------------------------------------------
# Prediction strategies
# ---------------------------------------------------------------------------

def predict_averaged(
    image_feats: torch.Tensor,      # (N, D)
    text_emb: torch.Tensor,         # (C, D)
) -> np.ndarray:
    return (image_feats @ text_emb.T).argmax(dim=-1).numpy()


def predict_max_sim(
    image_feats: torch.Tensor,      # (N, D)
    per_prompt_emb: torch.Tensor,   # (C, P, D)
) -> np.ndarray:
    # sims: (N, C, P) → max over P → (N, C) → argmax over C
    sims = torch.einsum("nd,cpd->ncp", image_feats, per_prompt_emb)
    return sims.max(dim=-1).values.argmax(dim=-1).numpy()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mel_recall = recall_score(
        y_true, y_pred, labels=[MEL_IDX], average="macro", zero_division=0
    )
    return macro_f1, mel_recall


def build_detail_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    strategy_name: str,
) -> str:
    macro_f1, mel_recall = compute_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_ORDER))))
    cls_report = classification_report(
        y_true, y_pred,
        labels=list(range(len(CLASS_ORDER))),
        target_names=CLASS_ORDER,
        zero_division=0,
    )
    lines: list[str] = [
        "=" * 70,
        f"  Strategy: {strategy_name}",
        "=" * 70,
        "",
        "Confusion Matrix (rows = true, cols = predicted):",
        "  " + " ".join(f"{c:>6}" for c in CLASS_ORDER),
    ]
    for i, row in enumerate(cm):
        lines.append(f"{CLASS_ORDER[i]:>6}  " + " ".join(f"{v:>6d}" for v in row))
    lines += ["", "Per-class metrics:", cls_report,
              f"  Macro-F1   : {macro_f1:.4f}",
              f"  MEL Recall : {mel_recall:.4f}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Part A — Semantic Crowding Visualization
# ---------------------------------------------------------------------------

def compute_sim_matrix(emb: torch.Tensor) -> np.ndarray:
    """(C, D) averaged L2-normed embeddings → (C, C) cosine similarity matrix."""
    sim = (emb @ emb.T).cpu().numpy()
    return sim


def plot_crowding_heatmaps(
    emb_p1: torch.Tensor,
    emb_p2: torch.Tensor,
    out_path: Path,
) -> None:
    sim_p1 = compute_sim_matrix(emb_p1)
    sim_p2 = compute_sim_matrix(emb_p2)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    common_kwargs = dict(
        xticklabels=CLASS_ORDER,
        yticklabels=CLASS_ORDER,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "Cosine similarity"},
    )

    sns.heatmap(sim_p1, ax=axes[0], **common_kwargs)
    axes[0].set_title("Phase 1 — Generic prompts", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Predicted class", fontsize=10)
    axes[0].set_ylabel("True class", fontsize=10)

    sns.heatmap(sim_p2, ax=axes[1], **common_kwargs)
    axes[1].set_title("Phase 2 — MedGPT dermoscopic prompts", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Predicted class", fontsize=10)
    axes[1].set_ylabel("")

    fig.suptitle(
        "BiomedCLIP text-space semantic crowding\n"
        "(cosine similarity between averaged class embeddings)",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Semantic crowding heatmap saved -> {out_path}")


# ---------------------------------------------------------------------------
# Part B — Comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(p3_avg_f1: float, p3_avg_rec: float,
                            p3_max_f1: float, p3_max_rec: float) -> str:
    sep = "-" * 54
    rows = list(BASELINES.items()) + [
        ("Phase3 Averaged", {"macro_f1": p3_avg_f1, "mel_recall": p3_avg_rec}),
        ("Phase3 Max-Sim",  {"macro_f1": p3_max_f1, "mel_recall": p3_max_rec}),
    ]
    lines = [
        "",
        "=" * 54,
        "  Zero-Shot Results — All Phases",
        "=" * 54,
        f"{'Strategy':<22} | {'Macro-F1':>9} | {'MEL Recall':>10}",
        sep,
    ]
    for name, vals in rows:
        lines.append(
            f"{name:<22} | {vals['macro_f1']:>9.3f} | {vals['mel_recall']:>10.3f}"
        )
    lines.append("=" * 54)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_multiclass_table(
    csv_path: Path,
    col_image_id: str,
    col_lesion_id: str,
    col_diagnosis: str,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {col_image_id, col_lesion_id, col_diagnosis}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df = df.copy()
    df["_binary_label"] = df[col_diagnosis].astype(str).map(LABEL_MAP).astype(int)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BiomedCLIP zero-shot analysis: semantic crowding (Part A) "
                    "and Phase 3 evaluation (Part B)"
    )
    p.add_argument("--metadata-csv", type=Path, default=None)
    p.add_argument("--images-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--skip-part-b", action="store_true",
        help="Only run Part A (heatmap); skip Phase 3 zero-shot evaluation"
    )
    return p.parse_args()


def main() -> None:
    # The Greek-locale (cp1253) Windows console cannot encode arrows; all
    # print() calls here are plain ASCII, this is the second line of defense.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()

    cfg = Config()
    if args.metadata_csv is not None:
        cfg.metadata_csv = args.metadata_csv
    if args.images_dir is not None:
        cfg.images_dir = args.images_dir
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.seed is not None:
        cfg.seed = args.seed

    set_seed(cfg.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")
    print(f"Loading BiomedCLIP ({BIOMEDCLIP_ID}) …")
    model, preprocess, tokenizer = load_biomedclip(device)

    # =========================================================================
    # PART A — Semantic Crowding Heatmap
    # =========================================================================
    print("\n[Part A] Encoding Phase 1 text embeddings …")
    emb_p1 = build_averaged_embeddings(model, tokenizer, PROMPTS_P1, device)
    print(f"  Phase 1 averaged embeddings: {tuple(emb_p1.shape)}")

    print("[Part A] Encoding Phase 2 text embeddings …")
    emb_p2 = build_averaged_embeddings(model, tokenizer, PROMPTS_P2, device)
    print(f"  Phase 2 averaged embeddings: {tuple(emb_p2.shape)}")

    heatmap_path = out_dir / "semantic_crowding_heatmap.png"
    print("[Part A] Plotting semantic crowding heatmaps …")
    plot_crowding_heatmaps(emb_p1.cpu(), emb_p2.cpu(), heatmap_path)

    if args.skip_part_b:
        print("\nPart B skipped (--skip-part-b).")
        return

    # =========================================================================
    # PART B — Phase 3 Zero-Shot Evaluation
    # =========================================================================
    print("\n[Part B] Loading metadata and reproducing test split …")
    df = load_multiclass_table(
        cfg.metadata_csv, cfg.col_image_id, cfg.col_lesion_id, cfg.col_diagnosis
    )
    print(f"  Label map: {LABEL_MAP}")

    _, _, df_test = lesion_level_split(
        df, cfg.col_lesion_id,
        cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, cfg.seed,
    )
    print(f"  Test set: {len(df_test)} images")

    print("\n[Part B] Encoding test images …")
    image_feats, y_true_t = encode_test_images(
        model, df_test, cfg.images_dir, cfg.col_image_id,
        preprocess, args.batch_size, cfg.num_workers, device,
    )
    y_true = y_true_t.numpy()
    print(f"  Image features: {tuple(image_feats.shape)}")

    print("\n[Part B] Encoding Phase 3 text embeddings …")
    avg_p3 = build_averaged_embeddings(model, tokenizer, PROMPTS_P3, device).cpu()
    pp_p3  = build_per_prompt_embeddings(model, tokenizer, PROMPTS_P3, device).cpu()
    print(f"  Averaged embeddings  : {tuple(avg_p3.shape)}")
    print(f"  Per-prompt embeddings: {tuple(pp_p3.shape)}")

    print("\n[Part B] Running Phase 3 zero-shot predictions …")
    y_pred_avg = predict_averaged(image_feats, avg_p3)
    y_pred_max = predict_max_sim(image_feats, pp_p3)

    p3_avg_f1, p3_avg_rec = compute_metrics(y_true, y_pred_avg)
    p3_max_f1, p3_max_rec = compute_metrics(y_true, y_pred_max)

    report_avg = build_detail_report(
        y_true, y_pred_avg,
        "Phase 3 Averaged Prompts (per-class mean embedding) — Gemini prompts"
    )
    report_max = build_detail_report(
        y_true, y_pred_max,
        "Phase 3 Max-Sim Ensemble (per-prompt max similarity) — Gemini prompts"
    )
    comparison = build_comparison_table(p3_avg_f1, p3_avg_rec, p3_max_f1, p3_max_rec)

    header = textwrap.dedent(f"""\
        BiomedCLIP Zero-Shot Classification — Phase 3 (Gemini-derived prompts)
        =======================================================================
        Model  : {BIOMEDCLIP_ID}
        Split  : test set, lesion-level stratified split (seed={cfg.seed})
        Images : {len(df_test)}
        Classes: {CLASS_ORDER}
        Prompts: {len(next(iter(PROMPTS_P3.values())))} per class (Gemini-derived dermoscopic descriptions)

    """)

    full_report = header + report_avg + "\n" + report_max + comparison

    print("\n" + full_report)

    out_file = out_dir / "biomedclip_zeroshot_phase3_results.txt"
    out_file.write_text(full_report, encoding="utf-8")
    print(f"\nPhase 3 results saved -> {out_file}")


if __name__ == "__main__":
    main()
