"""
GHG (Greenhouse Gas) Predictor
=========================================
Predicts ghg from material composition, product category, circularity-origin,
and end-of-life pathways.

Materials are embedded via Word2Vec-style / fastText-style vectors (300-d)
and weighted by mass fractions. Category is one-hot encoded and concatenated
with the material embedding and circularity features.

What this script does:
- Keeps only products with reference_unit == "kg"
- Filters out categories with fewer than MIN_CATEGORY_COUNT products
- Drops any product with invalid material percentages
- Drops any product with invalid/missing target values
- Keeps only products with GHG_MIN <= ghg <= GHG_MAX
- Uses circularity-origin and end-of-life percentages as extra numeric features
- Applies British->American English normalisation on material tokens
- Filters stopwords from material token sequences
- Uses zero-vector fallback for OOV tokens (instead of silently skipping)
- Assigns equal mass fractions when all percentages are missing or zero
- L2-normalises the final product embedding before concatenation
- Stratifies train/val/test splits by category
- Trains a regression model with log1p target transform
- Prints epoch-by-epoch training/validation metrics
- Evaluates on a held-out test set with per-category breakdown
- Saves plots and diagnostics
- Includes a prediction helper for inference

JSON structure expected:
  product["c_pcr"]          -> standardised category string
  product["product_integrity"]["materials"] -> list of {name, percentage}
  product["ghg_footprint"]["total_ghg"]     -> float target
  product["cyclability"]                    -> dict with lowercase future_use_* keys

Dependencies:
    pip install numpy scikit-learn torch matplotlib
"""

import json
import math
import re
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────────────────────
# PATHS / CONFIG
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset.json"

# ── Embedding backend ─────────────────────────────────────────────────────────
# Options:
#   "google_news" : local Word2Vec binary (.bin)
#   "fasttext"    : local/downloaded .vec
#   "custom_vec"  : user-provided .vec
EMBEDDING_BACKEND = "fasttext"

GOOGLE_NEWS_PATH  = BASE_DIR / "GoogleNews-vectors-negative300" / "GoogleNews-vectors-negative300.bin"

FASTTEXT_URL      = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
FASTTEXT_ZIP_PATH = BASE_DIR / "wiki-news-300d-1M.vec.zip"
FASTTEXT_DIR      = BASE_DIR / "wiki-news-300d-1M"
FASTTEXT_VEC_PATH = FASTTEXT_DIR / "wiki-news-300d-1M.vec"

CUSTOM_VEC_PATH   = BASE_DIR / "custom_vectors.vec"

# ── Output paths ──────────────────────────────────────────────────────────────
MODEL_PATH         = BASE_DIR / "ghg_model.pt"
DIAGNOSTICS_PATH   = BASE_DIR / "diagnostics_ghg.json"
TRAINING_PLOT_PATH = BASE_DIR / "training_curves_ghg.png"
PRED_SCATTER_PATH  = BASE_DIR / "pred_vs_actual_ghg.png"
RESIDUALS_PATH     = BASE_DIR / "residuals_ghg.png"

# ── Model hyperparameters ─────────────────────────────────────────────────────
EMBED_DIM          = 300
HIDDEN_DIMS        = [256, 128]
DROPOUT            = 0.2
LR                 = 1e-3
EPOCHS             = 200
BATCH_SIZE         = 64
PATIENCE           = 30
RANDOM_SEED        = 42

TARGET_FIELD       = "total_ghg"

GHG_MIN            = 0.0
GHG_MAX            = 10.0

MIN_CATEGORY_COUNT = 20

# 1 circularity-origin + 4 end-of-life percentages
N_CIRC_FEATURES    = 5

# ── Material token normalisation ──────────────────────────────────────────────
MATERIAL_VARIATIONS: Dict[str, str] = {
    "aluminium": "aluminum",
    "braces":    "suspenders",
    "colour":    "color",
    "fibre":     "fiber",
    "flavour":   "flavor",
    "gaol":      "jail",
    "kerb":      "curb",
    "litre":     "liter",
    "mould":     "mold",
    "petrol":    "gasoline",
    "sulphur":   "sulfur",
    "tyre":      "tire",
    "woollen":   "woolen",
}

STOP_WORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "for", "from", "has", "have", "in", "is", "it", "its", "of",
    "on", "or", "that", "the", "their", "there", "they", "this",
    "to", "was", "were", "which", "with",
}

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str):
            s = x.strip().lower()
            if s in {"", "undefined", "null", "none", "nan"}:
                return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def tokenise(text: str) -> List[str]:
    """Plain alphabetic tokeniser — no variation mapping, no stopword filtering."""
    return re.findall(r"[a-zA-Z]+", str(text).lower())


def tokenise_material(text: str) -> List[str]:
    """
    Tokenise a material name with British->American normalisation and stopword removal.
    Uses \b word-boundary anchors (single backslash) so re.findall splits
    hyphenated tokens like "glass-fibre" correctly.
    """
    tokens = re.findall(r"\b[a-zA-Z]+\b", str(text).lower())   # ← single \b
    tokens = [MATERIAL_VARIATIONS.get(t, t) for t in tokens]
    return [t for t in tokens if t not in STOP_WORDS]


def r2_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def normalise_shares_to_100(values: List[float]) -> List[float]:
    total = sum(values)
    if total <= 0:
        return [0.0 for _ in values]
    return [100.0 * v / total for v in values]


def _get_materials(product: dict) -> list:
    """
    Return the materials list from:
      product["product_integrity"]["materials"]
    """
    return (product.get("product_integrity") or {}).get("materials") or []


def extract_circularity_features(product: dict) -> Optional[Dict[str, float]]:
    cyclability = product.get("cyclability") or {}

    circularity_origin_pct = safe_float(cyclability.get("circularity_origin_percentage")) or 0.0

    fu_recycling  = safe_float(cyclability.get("future_use_recycling"))                          or 0.0
    fu_composting = safe_float(cyclability.get("future_use_composting"))                         or 0.0
    fu_val_fill   = safe_float(cyclability.get("future_use_valorisation / filling"))             or 0.0
    fu_recond     = safe_float(cyclability.get("future_use_reconditioning"))                     or 0.0
    fu_reuse      = safe_float(cyclability.get("future_use_reuse"))                              or 0.0
    fu_hazardous  = safe_float(cyclability.get("future_use_hazardous waste"))                    or 0.0
    fu_inert      = safe_float(cyclability.get("future_use_inert and non-hazardous landfills"))  or 0.0
    fu_inciner    = safe_float(cyclability.get("future_use_incineration"))                       or 0.0

    recycling_pct = fu_recycling + fu_composting + fu_val_fill + fu_recond + fu_reuse
    eol = normalise_shares_to_100([recycling_pct, fu_hazardous, fu_inert, fu_inciner])

    return {
        "circularity_origin_pct": circularity_origin_pct,
        "recycling_pct":          eol[0],
        "hazardous_pct":          eol[1],
        "inert_pct":              eol[2],
        "incineration_pct":       eol[3],
    }


# ─────────────────────────────────────────────
# DATA LOADING / VALIDATION
# ─────────────────────────────────────────────

def load_dataset(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def filter_reference_unit_kg(products: list) -> list:
    filtered, skipped = [], 0
    for p in products:
        unit = p.get("reference_unit")
        if unit is None or str(unit).strip().lower() != "kg":
            skipped += 1
            continue
        filtered.append(p)
    print(f"Reference unit filter (kg only): {len(filtered)} kept | {skipped} removed")
    return filtered


def build_category_index(products: list, min_count: int = MIN_CATEGORY_COUNT) -> Dict[str, int]:
    counts = Counter(
        str(p.get("c_pcr", "")).strip()
        for p in products
        if str(p.get("c_pcr", "")).strip()
    )

    print(f"\n{chr(9472) * 54}")
    print(f"  Category distribution ({len(counts)} unique categories found)")
    print(f"{chr(9472) * 54}")
    print(f"  {'Category':<32}  {'Count':>6}  Status")
    print(f"  {chr(9472) * 32}  {chr(9472) * 6}  {chr(9472) * 20}")

    kept: Dict[str, int] = {}
    dropped: Dict[str, int] = {}
    for cat, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        if cnt >= min_count:
            kept[cat] = cnt
            status = "KEPT"
        else:
            dropped[cat] = cnt
            status = f"DROPPED (< {min_count})"
        print(f"  {cat:<32}  {cnt:>6}  {status}")

    print(f"{chr(9472) * 54}")
    print(f"  Categories kept    : {len(kept)}")
    print(f"  Categories dropped : {len(dropped)}  (threshold: < {min_count} products)")
    print(f"  Products in kept categories   : {sum(kept.values())}")
    print(f"  Products in dropped categories: {sum(dropped.values())}")
    print(f"{chr(9472) * 54}\n")

    if not kept:
        raise ValueError(
            f"No category has >= {min_count} products. "
            "Lower MIN_CATEGORY_COUNT or check your dataset."
        )

    return {cat: i for i, cat in enumerate(sorted(kept.keys()))}


def normalize_product(
    product: dict,
    cat_index: Dict[str, int],
    require_target: bool = True,
    ghg_min: float = GHG_MIN,
    ghg_max: float = GHG_MAX,
) -> Optional[dict]:
    if str(product.get("reference_unit", "")).strip().lower() != "kg":
        return None

    category = str(product.get("c_pcr", "")).strip()
    if category not in cat_index:
        return None

    ghg = None
    if require_target:
        ghg_raw = (product.get("ghg_footprint") or {}).get(TARGET_FIELD)
        ghg = safe_float(ghg_raw)
        if ghg is None:
            return None
        if ghg < ghg_min or ghg > ghg_max:
            return None

    materials = _get_materials(product)
    if not materials:
        return None

    all_pcts_missing = all(
        safe_float(m.get("percentage")) in (None, 0.0)
        for m in materials
    )

    cleaned_materials = []
    for m in materials:
        name = str(m.get("name", "")).strip()
        if not name:
            return None
        pct = safe_float(m.get("percentage")) or 0.0
        cleaned_materials.append({"name": name, "percentage": pct})

    if all_pcts_missing:
        equal_weight = 100.0 / len(cleaned_materials)
        for m in cleaned_materials:
            m["percentage"] = equal_weight
    elif sum(m["percentage"] for m in cleaned_materials) <= 0:
        return None

    circ_feats = extract_circularity_features(product)
    if circ_feats is None:
        return None

    return {
        "ghg":       ghg,
        "category":  category,
        "materials": cleaned_materials,
        "raw":       product,
        **circ_feats,
    }


def filter_valid_products(products: list, cat_index: Dict[str, int]) -> list:
    out = []
    skipped_category = skipped_target = skipped_materials = skipped_other = 0

    for p in products:
        category = str(p.get("c_pcr", "")).strip()
        if category not in cat_index:
            skipped_category += 1
            continue

        c = normalize_product(p, cat_index, require_target=True, ghg_min=GHG_MIN, ghg_max=GHG_MAX)
        if c is None:
            ghg_raw   = (p.get("ghg_footprint") or {}).get(TARGET_FIELD)
            ghg       = safe_float(ghg_raw)
            materials = _get_materials(p)
            if ghg is None or not (GHG_MIN <= ghg <= GHG_MAX):
                skipped_target += 1
            elif not materials:
                skipped_materials += 1
            else:
                skipped_other += 1
            continue
        out.append(c)

    total_skipped = skipped_category + skipped_target + skipped_materials + skipped_other
    print("Product validation summary:")
    print(f"  Valid products                  : {len(out)}")
    print(f"  Skipped (low-count category)    : {skipped_category}")
    print(f"  Skipped (target GHG)            : {skipped_target}"
          f"  (missing, invalid, or outside [{GHG_MIN}, {GHG_MAX}])")
    print(f"  Skipped (materials)             : {skipped_materials}")
    print(f"  Skipped (other)                 : {skipped_other}")
    print(f"  Total skipped                   : {total_skipped}")
    return out


# ─────────────────────────────────────────────
# EMBEDDING LOADING
# ─────────────────────────────────────────────

def _load_word2vec_binary(path: Path, needed_tokens: Set[str]) -> Dict[str, np.ndarray]:
    print(f"Loading Word2Vec binary from {path} ...")
    vecs: Dict[str, np.ndarray] = {}
    bytes_per_vec = EMBED_DIM * 4

    with open(path, "rb") as f:
        header = f.readline().decode("utf-8").strip()
        vocab_size, vector_size = map(int, header.split())

        if vector_size != EMBED_DIM:
            raise ValueError(
                f"Binary file has vector_size={vector_size} but EMBED_DIM={EMBED_DIM}. "
                "Update EMBED_DIM to match."
            )

        for _ in range(vocab_size):
            if len(vecs) == len(needed_tokens):
                break

            word_bytes = bytearray()
            while True:
                ch = f.read(1)
                if ch in (b" ", b""):
                    break
                if ch == b"\n":     # ← single backslash = actual newline byte
                    word_bytes = bytearray()
                    break
                word_bytes.extend(ch)

            word = word_bytes.decode("utf-8", errors="ignore").strip()

            raw = f.read(bytes_per_vec)
            if len(raw) < bytes_per_vec:
                break

            if word in needed_tokens:
                vecs[word] = np.frombuffer(raw, dtype=np.float32).copy()

    return vecs


def _ensure_fasttext_downloaded() -> Path:
    FASTTEXT_DIR.mkdir(exist_ok=True)
    if FASTTEXT_VEC_PATH.exists():
        return FASTTEXT_VEC_PATH
    if not FASTTEXT_ZIP_PATH.exists():
        print(f"Downloading fastText vectors ...\n  -> {FASTTEXT_URL}")
        urllib.request.urlretrieve(FASTTEXT_URL, str(FASTTEXT_ZIP_PATH))
    print("Extracting fastText vectors ...")
    with zipfile.ZipFile(str(FASTTEXT_ZIP_PATH), "r") as zf:
        zf.extractall(str(FASTTEXT_DIR))
    if not FASTTEXT_VEC_PATH.exists():
        raise FileNotFoundError(f"Expected extracted file not found: {FASTTEXT_VEC_PATH}")
    return FASTTEXT_VEC_PATH


def _load_vec_subset(vec_path: Path, needed_tokens: Set[str]) -> Dict[str, np.ndarray]:
    print(f"Loading vec subset from {vec_path} ...")
    vecs: Dict[str, np.ndarray] = {}

    with open(vec_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
        if not first_line:
            return vecs

        first_parts = first_line.rstrip().split()
        has_header = (
            len(first_parts) == 2
            and first_parts[0].isdigit()
            and first_parts[1].isdigit()
        )

        def maybe_store(parts: List[str]) -> None:
            if len(parts) != EMBED_DIM + 1:
                return
            word = parts[0]
            if word not in needed_tokens:
                return
            try:
                vecs[word] = np.asarray(parts[1:], dtype=np.float32)
            except Exception:
                return

        if not has_header:
            maybe_store(first_parts)

        for line in f:
            if len(vecs) == len(needed_tokens):
                break
            maybe_store(line.rstrip().split())

    return vecs


def collect_needed_tokens(products: list) -> Set[str]:
    tokens: Set[str] = set()
    for item in products:
        for m in item["materials"]:
            tokens.update(tokenise_material(m["name"]))
    return tokens


def _report_missing_tokens(needed_tokens: Set[str], vocab: Dict[str, np.ndarray]) -> None:
    missing = needed_tokens - set(vocab.keys())
    found   = len(needed_tokens) - len(missing)
    print(f"  Tokens found : {found} / {len(needed_tokens)}")
    if missing:
        print(f"  Missing tokens ({len(missing)}) -- zero-vector fallback: {sorted(missing)}")


def get_vocab(products: list) -> Dict[str, np.ndarray]:
    needed_tokens = collect_needed_tokens(products)
    print(f"Unique tokens needed from embeddings: {len(needed_tokens):,}")
    print(f"Embedding backend: {EMBEDDING_BACKEND}")

    if EMBEDDING_BACKEND == "google_news":
        if not GOOGLE_NEWS_PATH.exists():
            raise FileNotFoundError(
                f"Google News binary not found at {GOOGLE_NEWS_PATH}.\n"
                "Switch EMBEDDING_BACKEND to 'fasttext' or fix the path."
            )
        vocab = _load_word2vec_binary(GOOGLE_NEWS_PATH, needed_tokens)
        _report_missing_tokens(needed_tokens, vocab)

    elif EMBEDDING_BACKEND == "fasttext":
        vec_path = _ensure_fasttext_downloaded()
        vocab    = _load_vec_subset(vec_path, needed_tokens)
        _report_missing_tokens(needed_tokens, vocab)

    elif EMBEDDING_BACKEND == "custom_vec":
        if not CUSTOM_VEC_PATH.exists():
            raise FileNotFoundError(
                f"Custom vec file not found at {CUSTOM_VEC_PATH}.\n"
                "Set CUSTOM_VEC_PATH to your .vec file and ensure EMBED_DIM matches."
            )
        vocab = _load_vec_subset(CUSTOM_VEC_PATH, needed_tokens)
        _report_missing_tokens(needed_tokens, vocab)

    else:
        raise ValueError(
            f"Unknown EMBEDDING_BACKEND '{EMBEDDING_BACKEND}'. "
            "Valid options: 'google_news', 'fasttext', 'custom_vec'."
        )

    return vocab


# ─────────────────────────────────────────────
# EMBEDDINGS
# ─────────────────────────────────────────────

def embed_material(name: str, vocab: Dict[str, np.ndarray]) -> np.ndarray:
    tokens = tokenise_material(name)
    if not tokens:
        return np.zeros(EMBED_DIM, dtype=np.float32)

    vecs = [
        np.asarray(vocab[t], dtype=np.float32) if t in vocab
        else np.zeros(EMBED_DIM, dtype=np.float32)
        for t in tokens
    ]
    return np.mean(vecs, axis=0).astype(np.float32)


def product_embedding(materials: list, vocab: Dict[str, np.ndarray]) -> np.ndarray:
    """L2-normalised, mass-fraction-weighted average embedding for a product."""
    total = sum(m["percentage"] for m in materials)
    if total <= 0:
        return np.zeros(EMBED_DIM, dtype=np.float32)

    vec = np.zeros(EMBED_DIM, dtype=np.float32)
    for m in materials:
        vec += (m["percentage"] / total) * embed_material(m["name"], vocab)

    norm = float(np.linalg.norm(vec))
    return (vec / norm).astype(np.float32) if norm > 0.0 else vec


def category_onehot(category: str, cat_index: Dict[str, int]) -> np.ndarray:
    vec = np.zeros(len(cat_index), dtype=np.float32)
    idx = cat_index.get(category)
    if idx is not None:
        vec[idx] = 1.0
    return vec


# ─────────────────────────────────────────────
# FEATURE BUILDING
# ─────────────────────────────────────────────

def build_features(
    valid_products: list,
    vocab: Dict[str, np.ndarray],
    cat_index: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X, y, categories = [], [], []
    for item in valid_products:
        mat_emb    = product_embedding(item["materials"], vocab)
        cat_emb    = category_onehot(item["category"], cat_index)
        circ_feats = np.array([
            item["circularity_origin_pct"],
            item["recycling_pct"],
            item["hazardous_pct"],
            item["inert_pct"],
            item["incineration_pct"],
        ], dtype=np.float32)

        X.append(np.concatenate([mat_emb, cat_emb, circ_feats]))
        y.append(item["ghg"])
        categories.append(item["category"])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), categories


# ─────────────────────────────────────────────
# PYTORCH DATASET
# ─────────────────────────────────────────────

class GHGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

class GHGNet(nn.Module):
    def __init__(self, input_dim: int, hidden=None, drop=DROPOUT):
        super().__init__()
        if hidden is None:
            hidden = HIDDEN_DIMS

        layers = []
        in_dim = input_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(drop)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train_model(model, train_loader, val_loader, device, scaler_y_mean, scaler_y_scale):
    opt     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)
    loss_fn = nn.HuberLoss(delta=1.0)

    best_val     = float("inf")
    patience_ctr = 0
    best_state   = None

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_mae": [], "val_r2": []}

    print(f"\n{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}  {'Val MAE':>10}  {'Val R2':>8}")
    print("-" * 58)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            t_loss += loss.item() * len(yb)
        t_loss /= len(train_loader.dataset)

        model.eval()
        v_loss, preds_s, actuals_s = 0.0, [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                p = model(Xb)
                v_loss += loss_fn(p, yb).item() * len(yb)
                preds_s.extend(p.cpu().numpy())
                actuals_s.extend(yb.cpu().numpy())
        v_loss /= len(val_loader.dataset)

        preds   = np.expm1(np.asarray(preds_s)  * scaler_y_scale + scaler_y_mean)
        actuals = np.expm1(np.asarray(actuals_s) * scaler_y_scale + scaler_y_mean)

        v_mae = mean_absolute_error(actuals, preds)
        v_r2  = r2_safe(actuals, preds)

        sched.step(v_loss)

        history["epoch"].append(epoch)
        history["train_loss"].append(float(t_loss))
        history["val_loss"].append(float(v_loss))
        history["val_mae"].append(float(v_mae))
        history["val_r2"].append(float(v_r2) if np.isfinite(v_r2) else float("nan"))

        print(f"{epoch:>6}  {t_loss:>12.4f}  {v_loss:>10.4f}  {v_mae:>10.4f}  {v_r2:>8.4f}")

        if v_loss < best_val - 1e-5:
            best_val     = v_loss
            patience_ctr = 0
            best_state   = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(model, loader, device, scaler_y_mean, scaler_y_scale):
    model.eval()
    preds_s, actuals_s = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            preds_s.extend(model(Xb).cpu().numpy())
            actuals_s.extend(yb.numpy())

    preds   = np.expm1(np.asarray(preds_s,  dtype=np.float32) * scaler_y_scale + scaler_y_mean)
    actuals = np.expm1(np.asarray(actuals_s, dtype=np.float32) * scaler_y_scale + scaler_y_mean)

    ghg_range = GHG_MAX - GHG_MIN
    mae        = float(mean_absolute_error(actuals, preds))
    rmse       = float(math.sqrt(np.mean((actuals - preds) ** 2)))
    nrmse      = rmse / ghg_range
    ss_res     = float(np.sum((actuals - preds) ** 2))
    ss_range   = float(len(actuals) * ghg_range ** 2)
    r2_range   = float(1.0 - ss_res / ss_range)
    r2_sample  = r2_safe(actuals, preds)

    abs_err    = np.abs(actuals - preds)
    thresholds = [0.5, 1.0, 2.0, 5.0]
    within     = {f"within_{t}kg": float(np.mean(abs_err <= t) * 100) for t in thresholds}

    return {
        "mae": mae, "rmse": rmse, "nrmse": nrmse,
        "r2_range": r2_range, "r2_sample": r2_sample,
        **within,
        "preds": preds, "actuals": actuals,
    }


def print_category_metrics(
    actuals: np.ndarray,
    preds: np.ndarray,
    categories: List[str],
) -> Dict[str, dict]:
    ghg_range = GHG_MAX - GHG_MIN
    cat_data: Dict[str, list] = defaultdict(list)
    for a, p, c in zip(actuals, preds, categories):
        cat_data[c].append((a, p))

    print(f"\n{chr(9472) * 80}")
    print("  Per-category test metrics")
    print(f"{chr(9472) * 80}")
    print(
        f"  {'Category':<28}  {'N':>5}  {'MAE':>8}  {'RMSE':>8}  "
        f"{'NRMSE':>7}  {'+-0.5':>6}  {'+-1.0':>6}  {'+-2.0':>6}"
    )
    print(
        f"  {chr(9472) * 28}  {chr(9472) * 5}  {chr(9472) * 8}  {chr(9472) * 8}  "
        f"{chr(9472) * 7}  {chr(9472) * 6}  {chr(9472) * 6}  {chr(9472) * 6}"
    )

    per_cat = {}
    for cat in sorted(cat_data.keys()):
        pairs = cat_data[cat]
        n     = len(pairs)
        a_arr = np.array([x[0] for x in pairs])
        p_arr = np.array([x[1] for x in pairs])
        mae   = float(mean_absolute_error(a_arr, p_arr))
        rmse  = float(math.sqrt(np.mean((a_arr - p_arr) ** 2)))
        nrmse = rmse / ghg_range
        ae    = np.abs(a_arr - p_arr)
        w05   = float(np.mean(ae <= 0.5) * 100)
        w10   = float(np.mean(ae <= 1.0) * 100)
        w20   = float(np.mean(ae <= 2.0) * 100)

        print(
            f"  {cat:<28}  {n:>5}  {mae:>8.4f}  {rmse:>8.4f}  "
            f"{nrmse:>7.4f}  {w05:>5.1f}%  {w10:>5.1f}%  {w20:>5.1f}%"
        )
        per_cat[cat] = {
            "n": n, "mae": mae, "rmse": rmse, "nrmse": nrmse,
            "within_0.5kg": w05, "within_1.0kg": w10, "within_2.0kg": w20,
        }

    print(f"{chr(9472) * 80}")
    return per_cat


def print_worst_predictions(
    actuals: np.ndarray,
    preds: np.ndarray,
    categories: List[str],
    n: int = 10,
) -> None:
    abs_err = np.abs(actuals - preds)
    worst   = np.argsort(abs_err)[::-1][:n]

    print(f"\n-- Worst {n} predictions (test set) --")
    print(f"  {'Actual':>10}  {'Predicted':>10}  {'Abs Err':>10}  Category")
    for i in worst:
        a, p, c = actuals[i], preds[i], categories[i]
        print(f"  {a:>10.4f}  {p:>10.4f}  {abs(a - p):>10.4f}  {c}")


# ─────────────────────────────────────────────
# PLOTS / DIAGNOSTICS
# ─────────────────────────────────────────────

def save_plots(history: dict, actuals: np.ndarray, preds: np.ndarray):
    import matplotlib.pyplot as plt

    epochs = np.array(history["epoch"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_loss"], label="Train loss")
    ax.plot(epochs, history["val_loss"],   label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber loss (log-scaled target)")
    ax.set_title("Training and validation loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(TRAINING_PLOT_PATH, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actuals, preds, alpha=0.55)
    min_v = float(min(actuals.min(), preds.min()))
    max_v = float(max(actuals.max(), preds.max()))
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--")
    ax.set_xlabel("Actual GHG (kg CO2-eq)")
    ax.set_ylabel("Predicted GHG (kg CO2-eq)")
    ax.set_title("Predicted vs actual GHG on test set")
    fig.tight_layout()
    fig.savefig(PRED_SCATTER_PATH, dpi=160)
    plt.close(fig)

    residuals = preds - actuals
    fig, ax   = plt.subplots(figsize=(10, 6))
    ax.hist(residuals, bins=40)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Residual (predicted - actual)")
    ax.set_ylabel("Count")
    ax.set_title("Residual distribution on test set")
    fig.tight_layout()
    fig.savefig(RESIDUALS_PATH, dpi=160)
    plt.close(fig)


def save_diagnostics(summary: dict):
    with open(DIAGNOSTICS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


# ─────────────────────────────────────────────
# INFERENCE HELPER
# ─────────────────────────────────────────────

def predict_ghg(
    product: dict,
    vocab: Dict[str, np.ndarray],
    checkpoint: Union[str, Path] = MODEL_PATH,
) -> float:
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)

    y_mean:    float          = float(ckpt["y_mean"])
    y_scale:   float          = float(ckpt["y_scale"])
    cat_index: Dict[str, int] = ckpt["cat_index"]
    input_dim: int            = ckpt["input_dim"]

    model = GHGNet(input_dim=input_dim, hidden=ckpt["hidden_dims"], drop=ckpt["dropout"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    normalized = normalize_product(
        product, cat_index, require_target=False, ghg_min=GHG_MIN, ghg_max=GHG_MAX
    )
    if normalized is None:
        raise ValueError(
            "Invalid product for inference: missing kg unit, unknown/dropped category, "
            "invalid materials, or invalid circularity/material values."
        )

    mat_emb    = product_embedding(normalized["materials"], vocab)
    cat_emb    = category_onehot(normalized["category"], cat_index)
    circ_feats = np.array([
        normalized["circularity_origin_pct"],
        normalized["recycling_pct"],
        normalized["hazardous_pct"],
        normalized["inert_pct"],
        normalized["incineration_pct"],
    ], dtype=np.float32)

    x = torch.tensor(
        np.concatenate([mat_emb, cat_emb, circ_feats]), dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        pred_scaled = model(x).item()

    return float(np.expm1(pred_scaled * y_scale + y_mean))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Script directory: {BASE_DIR}")
    print(f"Current working directory: {Path.cwd()}")

    print(f"\nLoading '{DATASET_PATH.name}' ...")
    products = load_dataset(DATASET_PATH)
    print(f"Total products: {len(products)}")

    products  = filter_reference_unit_kg(products)
    cat_index = build_category_index(products, min_count=MIN_CATEGORY_COUNT)
    print(f"Active categories ({len(cat_index)}): {sorted(cat_index.keys())}")

    valid_products = filter_valid_products(products, cat_index)

    if len(valid_products) < 10:
        raise ValueError("Need at least 10 valid labelled samples after filtering.")

    ghg_values = np.array([p["ghg"] for p in valid_products], dtype=np.float32)
    print(
        f"\nGHG summary after filtering -> "
        f"min: {ghg_values.min():.4f}, "
        f"median: {np.median(ghg_values):.4f}, "
        f"max: {ghg_values.max():.4f}, "
        f"mean: {ghg_values.mean():.4f}"
    )

    cat_counts_valid = Counter(p["category"] for p in valid_products)
    print("\nCategory distribution in valid set:")
    for cat, cnt in sorted(cat_counts_valid.items(), key=lambda x: -x[1]):
        print(f"  {cat:<32}  {cnt:>5} products")

    vocab = get_vocab(valid_products)
    print(f"\nVocabulary loaded: {len(vocab):,}")

    n_categories = len(cat_index)
    input_dim    = EMBED_DIM + n_categories + N_CIRC_FEATURES
    print(
        f"Input dimension: {EMBED_DIM} (material embedding) "
        f"+ {n_categories} (category one-hot) "
        f"+ {N_CIRC_FEATURES} (circularity features) = {input_dim}"
    )

    X, y, categories_all = build_features(valid_products, vocab, cat_index)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Products used for training: {len(y)}")

    y_log    = np.log1p(y)
    scaler_y = StandardScaler()
    sy       = scaler_y.fit_transform(y_log.reshape(-1, 1)).ravel().astype(np.float32)

    if not np.isfinite(sy).all():
        raise ValueError("Target scaling produced non-finite values. Check the dataset.")

    idx = np.arange(len(y))
    idx_tr, idx_tmp = train_test_split(
        idx, test_size=0.30, random_state=RANDOM_SEED,
        stratify=[categories_all[i] for i in idx],
    )
    idx_val, idx_te = train_test_split(
        idx_tmp, test_size=0.50, random_state=RANDOM_SEED,
        stratify=[categories_all[i] for i in idx_tmp],
    )

    print(f"\nStratified split  ->  train: {len(idx_tr)}  val: {len(idx_val)}  test: {len(idx_te)}")

    for split_name, split_idx in [("train", idx_tr), ("val", idx_val), ("test", idx_te)]:
        cats_in_split = set(categories_all[i] for i in split_idx)
        missing       = set(cat_index.keys()) - cats_in_split
        status        = "all categories present" if not missing else f"MISSING: {missing}"
        print(f"  {split_name:<6}: {status}")

    def make_loader(indices, shuffle=False):
        ds = GHGDataset(X[indices], sy[indices])
        return DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            drop_last=(shuffle and len(indices) > BATCH_SIZE),
        )

    train_loader = make_loader(idx_tr, shuffle=True)
    val_loader   = make_loader(idx_val)
    test_loader  = make_loader(idx_te)

    model = GHGNet(input_dim=input_dim).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    scaler_y_mean  = float(scaler_y.mean_[0])
    scaler_y_scale = float(scaler_y.scale_[0])

    model, history = train_model(
        model, train_loader, val_loader, device, scaler_y_mean, scaler_y_scale
    )

    test_results = evaluate_model(model, test_loader, device, scaler_y_mean, scaler_y_scale)

    print("\n" + "=" * 50)
    print("  TEST SET RESULTS")
    print("=" * 50)
    print(f"  MAE           {test_results['mae']:.4f} kg CO2-eq")
    print(f"  RMSE          {test_results['rmse']:.4f} kg CO2-eq")
    print(f"  NRMSE         {test_results['nrmse']:.4f}  (fraction of [{GHG_MIN}, {GHG_MAX}] range)")
    print(f"  R2 (range)    {test_results['r2_range']:.6f}  (vs fixed scale)")
    print(f"  R2 (sample)   {test_results['r2_sample']:.4f}  (vs sample variance, for reference)")
    print(f"  Within +-0.5  {test_results['within_0.5kg']:.1f}%")
    print(f"  Within +-1.0  {test_results['within_1.0kg']:.1f}%")
    print(f"  Within +-2.0  {test_results['within_2.0kg']:.1f}%")
    print(f"  Within +-5.0  {test_results['within_5.0kg']:.1f}%")
    print("=" * 50)

    print("\n-- Sample predictions (first 5 of test set) --")
    print(f"  {'Actual':>10}  {'Predicted':>10}  {'Abs Err':>10}  Category")
    test_categories = [categories_all[i] for i in idx_te]
    for a, p, c in zip(
        test_results["actuals"][:5], test_results["preds"][:5], test_categories[:5]
    ):
        print(f"  {a:>10.4f}  {p:>10.4f}  {abs(a - p):>10.4f}  {c}")

    per_cat_metrics = print_category_metrics(
        test_results["actuals"], test_results["preds"], test_categories
    )

    print_worst_predictions(
        test_results["actuals"], test_results["preds"], test_categories, n=10
    )

    save_plots(history, test_results["actuals"], test_results["preds"])
    print(f"\nSaved plot -> {TRAINING_PLOT_PATH}")
    print(f"Saved plot -> {PRED_SCATTER_PATH}")
    print(f"Saved plot -> {RESIDUALS_PATH}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "y_mean":      scaler_y_mean,
            "y_scale":     scaler_y_scale,
            "hidden_dims": HIDDEN_DIMS,
            "dropout":     DROPOUT,
            "cat_index":   cat_index,
            "input_dim":   input_dim,
        },
        str(MODEL_PATH),
    )
    print(f"Model saved -> {MODEL_PATH}")

    diagnostics = {
        "total_products":         len(products),
        "valid_products":         len(valid_products),
        "train_size":             len(idx_tr),
        "val_size":               len(idx_val),
        "test_size":              len(idx_te),
        "vocab_size":             len(vocab),
        "n_categories":           n_categories,
        "n_circularity_features": N_CIRC_FEATURES,
        "input_dim":              input_dim,
        "categories":             sorted(cat_index.keys()),
        "ghg_min":                GHG_MIN,
        "ghg_max":                GHG_MAX,
        "min_category_count":     MIN_CATEGORY_COUNT,
        "test_metrics": {
            "mae":          test_results["mae"],
            "rmse":         test_results["rmse"],
            "nrmse":        test_results["nrmse"],
            "r2_range":     test_results["r2_range"],
            "r2_sample":    test_results["r2_sample"],
            **{k: v for k, v in test_results.items() if k.startswith("within_")},
            "per_category": per_cat_metrics,
        },
        "history": history,
    }
    save_diagnostics(diagnostics)
    print(f"Diagnostics saved -> {DIAGNOSTICS_PATH}")

    example_raw  = valid_products[idx_te[0]]["raw"]
    example_pred = predict_ghg(example_raw, vocab, checkpoint=MODEL_PATH)
    example_true = valid_products[idx_te[0]]["ghg"]
    print("\n-- Inference demo --")
    print(f"  True GHG : {example_true:.4f} kg CO2-eq")
    print(f"  Pred GHG : {example_pred:.4f} kg CO2-eq")


if __name__ == "__main__":
    main()