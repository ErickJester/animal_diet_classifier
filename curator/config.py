"""
Configuración central del curador de dataset.

Un único lugar para ajustar rutas y umbrales; el resto del paquete importa
desde aquí para evitar valores mágicos dispersos.
"""

from __future__ import annotations

__version__ = "1.4.0"  # 1.0 rutas/umbrales base · 1.1 config descarga · 1.2 DOWNLOAD_WORKERS · 1.3 NEW_SPECIES_SIM+TARGET_PER_SPECIES · 1.4 fase C extintos (Commons + Wikipedia)

from pathlib import Path

# ── raíz del proyecto ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── dataset existente (organizado por dieta, no por especie) ────────────────────
DATASET_DIR  = ROOT / "dataset"
DIET_CLASSES = ["carnivore", "herbivore", "omnivore", "other"]
SPLITS       = ["train", "val"]

# ── datos persistentes del curador ──────────────────────────────────────────────
DATA_DIR         = ROOT / "curator" / "data"
INDEX_PATH       = DATA_DIR / "embeddings.npz"        # índice de similitud
REGISTRY_PATH    = DATA_DIR / "species_registry.json" # especies + nº de fotos
DIET_LABELS_PATH = DATA_DIR / "diet_labels.json"      # especie → dieta (editable a mano)

# Fase C (animales extintos): entrada y salida del flujo de Wikimedia Commons.
EXTINCT_SPECIES_PATH  = DATA_DIR / "extinct_species.json"   # lista de extintos a recopilar (editable)
DIET_SUGGESTIONS_PATH = DATA_DIR / "diet_suggestions.json"  # dietas auto-sugeridas (a revisar a mano)

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
NEW_SPECIES_SIM = 0.95   # <  → visualmente distinta a todo → casi seguro especie nueva

# Etapa 2 — por especie:
TARGET_PER_SPECIES = 175  # nº de fotos deseado por especie; alcanzado → descartar nuevas

# ── división al hacer commit del staging al dataset ──────────────────────────────
VAL_RATIO = 0.2          # ~20% de lo aceptado va a val/, el resto a train/

# ── descarga de imágenes (downloader) ────────────────────────────────────────────
# Las imágenes descargadas NO van directo al dataset: aterrizan aquí, ya
# organizadas por dieta cuando se conoce la especie. La curación es un paso aparte.
DOWNLOADS_DIR = ROOT / "downloads"             # destino de lo descargado (por dieta)
RAW_DIR       = DOWNLOADS_DIR / "_raw"         # datasets masivos crudos antes de reorganizar
UNSORTED_DIR  = DOWNLOADS_DIR / "_unsorted"    # especies sin dieta conocida (revisar a mano)
MANIFEST_PATH = DATA_DIR / "download_manifest.json"  # registro ligero anti-repetición

# Fase 2 (suplemento desde fuentes con especie verificada):
SUPP_PER_SPECIES = 40                           # tope de fotos por especie al suplementar
DOWNLOAD_WORKERS = 16                            # descargas HTTP en paralelo (el cuello es la red)
INAT_PAGE_SIZE   = 50                           # observaciones por página (máx. API: 200)
INAT_MAX_PAGES   = 20                           # cota de seguridad por especie
INAT_LICENSES    = "cc0,cc-by,cc-by-nc"         # solo fotos con licencia reutilizable
INAT_OBS_URL     = "https://api.inaturalist.org/v1/observations"  # endpoint público (sin token)

# Fase C (extintos desde Wikimedia Commons — iNaturalist no sirve: son observaciones de vivos):
COMMONS_API_URL       = "https://commons.wikimedia.org/w/api.php"  # API pública (sin token)
COMMONS_THUMB_PX      = 500    # ancho del thumbnail a descargar (≈ tamaño "medium" de iNat)
COMMONS_PER_SPECIES   = 60     # tope de fotos por especie extinta
COMMONS_SRLIMIT       = 50     # resultados por página de búsqueda (máx. API: 500)
COMMONS_MAX_PAGES     = 10     # cota de seguridad de paginación por especie
COMMONS_MIMES         = {"image/jpeg", "image/png", "image/webp"}  # se ignoran svg/tiff/gif
COMMONS_DIET_MIN_CONF = 0.60   # confianza mínima de la dieta auto-investigada para clasificarla;
                               # por debajo, la especie va a _unsorted a la espera de revisión

# Investigación automática de dieta (heurística sobre el texto de Wikipedia):
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"  # API pública (sin token)

# HTTP común a las descargas:
HTTP_TIMEOUT    = 30.0
HTTP_USER_AGENT = "animal-diet-classifier/1.0 (dataset downloader)"
