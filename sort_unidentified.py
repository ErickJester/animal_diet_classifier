"""
Clasifica por NOMBRE DE ARCHIVO las imágenes acumuladas en
curator/staging/_unidentified/, sin llamar a la API de iNaturalist.

Muchas imágenes llegan a _unidentified/ con la especie ya en el nombre
(el curador no la identificó por visión, pero el archivo sí la delata):

    Panthera_leo_123456.jpg     → especie 'Panthera leo'   (formato descargador)
    african_elephant-0001.jpg   → nombre común 'african elephant'
    zebra-0262.jpg              → nombre común 'zebra'
    4a813e64c6.jpg             → hash sin información → se deja para la API

Para cada imagen extrae el nombre, busca su dieta en diet_labels.json y la
mueve a:
    staging/<dieta>/        si la dieta es conocida   → lista para 'curate.py commit'
    _pending_review/        si hay nombre pero no dieta → asígnala y re-corre
    (se queda en _unidentified/)  si es un hash sin nombre → necesita la API

Es el complemento offline de 'curate.py reidentify': resuelve el 90 %+ de los
casos al instante y deja para la API solo lo que de verdad no tiene nombre.

Uso:
    python sort_unidentified.py            # mover todo
    python sort_unidentified.py --dry-run  # solo mostrar el plan, sin mover nada

Lugar en el pipeline:
    download.py → curate.py curate → sort_unidentified.py → curate.py commit → train
"""

from __future__ import annotations

__version__ = "1.0.0"  # 1.0 clasifica _unidentified/ por nombre de archivo (offline)

import argparse
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

# La consola de Windows suele ser cp1252 y no traga caracteres de caja/acentos.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from curator import config
from curator.registry import DietLabels

# Sufijo numérico de secuencia: 'african_elephant-0001' → 'african_elephant'
_SEQ_SUFFIX = re.compile(r"-\d+$")
# Hash sin información: solo hexadecimal y razonablemente largo (los del dataset
# son de 10 caracteres). Estas no tienen nombre → se dejan para la API.
_HASH_RE = re.compile(r"^[0-9a-f]{8,}$", re.IGNORECASE)


def name_from_filename(stem: str) -> str:
    """
    Extrae el nombre (especie científica o común) del stem del archivo.

        Panthera_leo_123456    → 'Panthera_leo'   (quita el photoID final)
        african_elephant-0001  → 'african_elephant'
        zebra-0262             → 'zebra'
        4a813e64c6             → '4a813e64c6'      (hash; no es un nombre real)
    """
    name = _SEQ_SUFFIX.sub("", stem)          # quita '-NNNN'
    parts = name.split("_")
    if len(parts) > 1 and parts[-1].isdigit():  # quita '_<photoID>'
        name = "_".join(parts[:-1])
    return name


def lookup_diet(labels: DietLabels, name: str) -> Optional[str]:
    """Busca la dieta probando guión bajo y espacios (DietLabels ya ignora may/min)."""
    for variant in (name, name.replace("_", " ")):
        diet = labels.get(variant)
        if diet in config.DIET_CLASSES:
            return diet
    return None


def _unique_dest(dest_dir: Path, src: Path) -> Path:
    candidate = dest_dir / src.name
    i = 1
    while candidate.exists():
        candidate = dest_dir / f"{src.stem}_{i}{src.suffix}"
        i += 1
    return candidate


def main() -> None:
    print(f"sort_unidentified.py v{__version__}")
    parser = argparse.ArgumentParser(
        description="Clasifica staging/_unidentified/ por nombre de archivo (offline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="muestra el plan sin mover archivos")
    args = parser.parse_args()

    src_dir = config.UNIDENTIFIED_DIR
    if not src_dir.is_dir():
        print(f"No existe {src_dir}")
        return

    images = sorted(f for f in src_dir.iterdir()
                    if f.suffix.lower() in config.IMG_EXTS)
    if not images:
        print("_unidentified/ está vacía.")
        return

    t_start    = time.time()
    by_diet    = Counter()        # dieta → nº movidas a staging/<dieta>/
    to_pending = 0                # con nombre pero sin dieta en diet_labels.json
    left_hash  = 0                # hash sin nombre → se queda para la API
    pending_names = Counter()     # nombres sin dieta (para saber qué añadir a labels)

    for diet in config.DIET_CLASSES:
        (config.STAGING_DIR / diet).mkdir(parents=True, exist_ok=True)

    print(f"\nProcesando {len(images)} imágenes de {src_dir}\n")

    labels = DietLabels.load()   # diet_labels.json una sola vez, no por imagen

    for img in images:
        name = name_from_filename(img.stem)
        diet = lookup_diet(labels, name)

        if diet:
            dest_dir = config.STAGING_DIR / diet
            by_diet[diet] += 1
        elif _HASH_RE.match(img.stem):
            left_hash += 1
            continue                          # se queda en _unidentified/
        else:
            dest_dir = config.PENDING_REVIEW_DIR
            to_pending += 1
            pending_names[name] += 1

        if not args.dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(img), str(_unique_dest(dest_dir, img)))

    _print_summary(by_diet, to_pending, left_hash, pending_names,
                   time.time() - t_start, args.dry_run, src_dir)


def _print_summary(by_diet, to_pending, left_hash, pending_names,
                   elapsed, dry_run, src_dir) -> None:
    accion = "se moverían" if dry_run else "movidas"
    print("── Resumen ───────────────────────────────────")
    total_diet = 0
    for diet in config.DIET_CLASSES:
        n = by_diet.get(diet, 0)
        total_diet += n
        print(f"  staging/{diet:<12} {n:>7}  {accion}")
    print(f"  _pending_review     {to_pending:>7}  (con nombre, sin dieta en diet_labels.json)")
    print(f"  _unidentified       {left_hash:>7}  (hash sin nombre → quedan para la API)")
    print("─" * 47)
    print(f"  A dieta: {total_diet}   A revisión: {to_pending}   "
          f"Sin tocar: {left_hash}   Tiempo: {elapsed:.1f}s")

    if pending_names:
        print(f"\n  Nombres sin dieta ({len(pending_names)} distintos) — añádelos a "
              f"{config.DIET_LABELS_PATH.name} y re-corre:")
        for nm, cnt in pending_names.most_common(20):
            print(f"    {nm:<32} {cnt:>5} fotos")
        if len(pending_names) > 20:
            print(f"    ... y {len(pending_names) - 20} nombres más")

    if dry_run:
        print("\n[DRY RUN] Nada fue modificado.")
    else:
        print(f"\nQuedan en _unidentified: {_count(src_dir)}")
        print("Siguiente paso:  python curate.py commit")


def _count(folder: Path) -> int:
    return sum(1 for f in folder.iterdir()
               if f.is_file() and f.suffix.lower() in config.IMG_EXTS)


if __name__ == "__main__":
    main()
