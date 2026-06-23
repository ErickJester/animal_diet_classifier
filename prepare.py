"""
Preparación del dataset a partir de las imágenes ya descargadas/curadas.

Extrae la especie del NOMBRE de archivo (formato del descargador:
'Genus_species_<photoid>.jpg'), busca su dieta en diet_labels.json y reparte
las imágenes en dataset/train/ y dataset/val/ con split 80/20.

A diferencia de curate.py (que usa embeddings + la API de iNaturalist para
identificar especie en vivo), este script es liviano, offline y totalmente
reproducible: solo necesita los archivos y diet_labels.json.

Fuente por defecto (en este orden):
  1. curator/staging/   → salida ya deduplicada por curate.py (recomendado)
  2. downloads/         → si no hay staging; incluye duplicados visuales

Lugar en el pipeline:
    download.py  →  curate.py (dedup, opcional)  →  prepare.py  →  train

Uso:
    python prepare.py                      # autodetecta la fuente
    python prepare.py --dry-run            # muestra qué haría, sin copiar nada
    python prepare.py --val-ratio 0.2      # ajusta el split (default 20% a val)
    python prepare.py downloads/carnivore  # una carpeta/archivo concreto

Idempotente: re-ejecutarlo no duplica (salta los archivos que ya están en
dataset/). Copia (no mueve), así la fuente queda intacta para re-correr.
"""

from __future__ import annotations

__version__ = "1.0.1"  # 1.0.1 excluir _pending_review/_unidentified; strip hex hashes

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# La consola de Windows suele ser cp1252 y no traga caracteres de caja/acentos.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── rutas del proyecto ─────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent
DOWNLOADS_DIR = ROOT / "downloads"
STAGING_DIR   = ROOT / "curator" / "staging"
DATASET_DIR   = ROOT / "dataset"
DIET_LABELS   = ROOT / "curator" / "data" / "diet_labels.json"
DIET_CLASSES  = ["carnivore", "herbivore", "omnivore", "other"]
IMG_EXTS      = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXCLUDE_PARTS = {"_raw", "_pending_review", "_unidentified"}  # carpetas internas del curator
_HEX_RE       = re.compile(r'^[0-9a-f]{8,}$')  # hash hex generado por curator (ej. 23e24f836e)


# ── utilidades ─────────────────────────────────────────────────────────────────
def load_diet_labels() -> dict:
    if not DIET_LABELS.is_file():
        print(f"ERROR: no se encontró {DIET_LABELS}")
        sys.exit(1)
    with open(DIET_LABELS, encoding="utf-8") as fh:
        return json.load(fh)


def species_from_filename(path: Path) -> str:
    """
    Especie a partir del nombre de archivo del descargador.

        Panthera_leo_123456.jpg    → 'Panthera_leo'
        Bos_taurus_987654.jpg      → 'Bos_taurus'
        lion.jpg                   → 'lion'  (nombre suelto, sin photoid)

    Quita el último token si es numérico (el photo_id).
    """
    parts = path.stem.split("_")
    if len(parts) > 1 and (parts[-1].isdigit() or _HEX_RE.match(parts[-1])):
        parts = parts[:-1]
    return "_".join(parts)


def lookup_diet(species_us: str, labels: dict) -> Optional[str]:
    """Busca la dieta probando guión bajo y espacios, case-insensitive."""
    candidates = {
        species_us,
        species_us.replace("_", " "),
        species_us.lower(),
        species_us.replace("_", " ").lower(),
    }
    for variant in candidates:
        diet = labels.get(variant)
        if diet in DIET_CLASSES:
            return diet
    return None


def existing_names() -> set:
    """Nombres de archivo ya presentes en dataset/{train,val}/ (idempotencia)."""
    names = set()
    for split in ("train", "val"):
        for diet in DIET_CLASSES:
            folder = DATASET_DIR / split / diet
            if folder.is_dir():
                names.update(
                    f.name for f in folder.iterdir() if f.suffix.lower() in IMG_EXTS
                )
    return names


def unique_dest(dest_dir: Path, name: str) -> Path:
    candidate = dest_dir / name
    stem, ext = Path(name).stem, Path(name).suffix
    i = 1
    while candidate.exists():
        candidate = dest_dir / f"{stem}_{i}{ext}"
        i += 1
    return candidate


def collect_images(sources: List[Path]) -> List[Path]:
    """Recoge imágenes de las rutas dadas (recursivo), excluyendo _raw/."""
    images = []
    for src in sources:
        if src.is_file() and src.suffix.lower() in IMG_EXTS:
            images.append(src)
        elif src.is_dir():
            for f in sorted(src.rglob("*")):
                if not f.is_file() or f.suffix.lower() not in IMG_EXTS:
                    continue
                if EXCLUDE_PARTS.intersection(f.parts):
                    continue
                images.append(f)
    return images


def default_source() -> Path:
    """staging/ si sus carpetas de clase tienen imágenes; si no, downloads/."""
    for diet in DIET_CLASSES:
        diet_dir = STAGING_DIR / diet
        if diet_dir.is_dir() and any(
            f.suffix.lower() in IMG_EXTS for f in diet_dir.rglob("*") if f.is_file()
        ):
            return STAGING_DIR
    return DOWNLOADS_DIR


# ── lógica principal ────────────────────────────────────────────────────────────
def prepare(sources: List[Path], val_ratio: float, dry_run: bool) -> None:
    labels = load_diet_labels()
    every  = max(2, round(1 / val_ratio))   # cada cuántas, una va a val

    for diet in DIET_CLASSES:
        for split in ("train", "val"):
            (DATASET_DIR / split / diet).mkdir(parents=True, exist_ok=True)

    already = existing_names()
    images  = collect_images(sources)
    if not images:
        print("No se encontraron imágenes en las rutas indicadas.")
        sys.exit(0)

    stats: Dict[str, Dict[str, int]] = {d: {"train": 0, "val": 0} for d in DIET_CLASSES}
    unknown: List[str]        = []
    counters: Dict[str, int]  = defaultdict(int)
    skipped_existing          = 0

    print(f"{'archivo':<42} {'especie':<26} {'dieta':<10} split")
    print("─" * 90)

    for img in images:
        if img.name in already:
            skipped_existing += 1
            continue

        species = species_from_filename(img)
        # Si la imagen ya vive en una carpeta de clase (downloads/<diet>/ o
        # staging/<diet>/), esa carpeta manda — imprescindible para clases
        # pre-clasificadas como "other" (no-animal), que no están en
        # diet_labels.json. Si no, se infiere la dieta desde la especie.
        diet = img.parent.name if img.parent.name in DIET_CLASSES else lookup_diet(species, labels)
        if diet is None:
            unknown.append(f"{img.name}  [{species}]")
            continue

        split = "val" if (counters[diet] % every == 0) else "train"
        counters[diet] += 1
        stats[diet][split] += 1
        already.add(img.name)   # evita duplicar si la misma fuente repite nombre

        print(f"{img.name[:42]:<42} {species.replace('_', ' ')[:25]:<26} "
              f"{diet:<10} {split}" + ("   [DRY RUN]" if dry_run else ""))

        if not dry_run:
            dest = unique_dest(DATASET_DIR / split / diet, img.name)
            shutil.copy2(img, dest)

    _print_summary(stats, skipped_existing, unknown, dry_run)


def _print_summary(stats, skipped_existing, unknown, dry_run) -> None:
    print("\n── Resumen ───────────────────────────────────────")
    total = 0
    for diet in DIET_CLASSES:
        tr, vl = stats[diet]["train"], stats[diet]["val"]
        total += tr + vl
        print(f"  {diet:<12}  train: {tr:>6}   val: {vl:>5}   total: {tr + vl:>6}")
    print(f"  {'TOTAL':<12}  {'':>13}{'':>12}  total: {total:>6}")

    if skipped_existing:
        print(f"\n  Ya en dataset (saltadas): {skipped_existing}")

    if unknown:
        print(f"\n  Sin dieta en diet_labels.json ({len(unknown)}):")
        for u in unknown[:15]:
            print(f"    {u}")
        if len(unknown) > 15:
            print(f"    ... y {len(unknown) - 15} más")

    print("\n[DRY RUN] Nada fue modificado." if dry_run
          else "\nDataset actualizado. Listo para entrenar.")


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Organiza imágenes descargadas/curadas en dataset/train/ y dataset/val/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("sources", nargs="*",
                        help="carpetas/archivos a procesar (default: autodetecta staging/ o downloads/)")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="fracción para val, entre 0 y 1 (default: 0.2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="muestra qué haría sin copiar nada")
    return parser.parse_args()


def main():
    print(f"prepare.py v{__version__}")
    args = parse_args()

    if not 0 < args.val_ratio < 1:
        print(f"ERROR: --val-ratio debe estar entre 0 y 1 (recibí {args.val_ratio})")
        sys.exit(1)

    if args.sources:
        sources = [Path(s) for s in args.sources]
        missing = [s for s in sources if not s.exists()]
        if missing:
            for m in missing:
                print(f"ERROR: no existe: {m}")
            sys.exit(1)
    else:
        src = default_source()
        if not src.is_dir():
            print("ERROR: no hay nada que preparar.")
            print("  Corre primero:  python download.py more")
            sys.exit(1)
        print(f"Fuente: {src}\n")
        sources = [src]

    prepare(sources, val_ratio=args.val_ratio, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
