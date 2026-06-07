"""
Configuración central del curador de dataset.

Un único lugar para ajustar rutas y umbrales; el resto del paquete importa
desde aquí para evitar valores mágicos dispersos.
"""

from __future__ import annotations

from pathlib import Path

# ── raíz del proyecto ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── dataset existente (organizado por dieta, no por especie) ────────────────────
DATASET_DIR  = ROOT / "dataset"
DIET_CLASSES = ["carnivore", "herbivore", "omnivore"]
SPLITS       = ["train", "val"]

# ── datos persistentes del curador ──────────────────────────────────────────────
DATA_DIR         = ROOT / "curator" / "data"
INDEX_PATH       = DATA_DIR / "embeddings.npz"        # índice de similitud
REGISTRY_PATH    = DATA_DIR / "species_registry.json" # especies + nº de fotos
DIET_LABELS_PATH = DATA_DIR / "diet_labels.json"      # especie → dieta (editable a mano)

# ── áreas de trabajo (las imágenes aceptadas se copian aquí, no al dataset) ──────
STAGING_DIR        = ROOT / "curator" / "staging"
PENDING_REVIEW_DIR = STAGING_DIR / "_pending_review"  # especie conocida, dieta desconocida
UNIDENTIFIED_DIR   = STAGING_DIR / "_unidentified"    # visualmente nueva, sin especie

# ── extensiones de imagen admitidas ─────────────────────────────────────────────
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── modelo de embeddings ─────────────────────────────────────────────────────────
EMBED_ARCH    = "resnet50"   # backbone torchvision usado como extractor de features
EMBED_DIM     = 2048         # 2048 para resnet50, 512 para resnet18
INPUT_SIZE    = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── umbrales de la cascada (opción C) ────────────────────────────────────────────
# Etapa 1 — similitud visual (coseno sobre embeddings normalizados L2):
DUPLICATE_SIM   = 0.95   # >= → casi idéntica a una que ya tenemos → descartar
NEW_SPECIES_SIM = 0.80   # <  → visualmente distinta a todo → casi seguro especie nueva

# Etapa 2 — por especie:
TARGET_PER_SPECIES = 60  # nº de fotos deseado por especie; alcanzado → descartar nuevas

# ── división al hacer commit del staging al dataset ──────────────────────────────
VAL_RATIO = 0.2          # ~20% de lo aceptado va a val/, el resto a train/
