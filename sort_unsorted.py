"""
Mueve las imágenes de downloads/_unsorted/ a downloads/<dieta>/
usando el mapeo de curator/data/diet_labels.json.

El nombre de cada subcarpeta en _unsorted/ es la especie (nombre común).
Si la especie está en diet_labels.json, sus imágenes se mueven a la carpeta
de dieta correspondiente. Las que no tengan mapeo se dejan donde están.

Uso:
    python sort_unsorted.py           # mover todo
    python sort_unsorted.py --dry-run # solo mostrar qué haría, sin mover nada
"""

__version__ = "1.0.0"  # 1.0 reorganiza _unsorted/ → downloads/<dieta>/

import argparse
import shutil
import time
from pathlib import Path

from curator import config
from curator.registry import DietLabels


def main() -> None:
    print(f"sort_unsorted.py v{__version__}")
    parser = argparse.ArgumentParser(description="Reorganiza _unsorted/ → carpetas de dieta")
    parser.add_argument("--dry-run", action="store_true",
                        help="muestra el plan sin mover archivos")
    args = parser.parse_args()

    diet_labels = DietLabels.load()
    src_root    = config.UNSORTED_DIR

    if not src_root.is_dir():
        print(f"No existe {src_root}")
        return

    species_dirs = sorted(d for d in src_root.iterdir() if d.is_dir())
    if not species_dirs:
        print("_unsorted/ está vacía.")
        return

    t_start  = time.time()
    moved    = 0
    skipped  = 0
    no_diet  = []

    print(f"{'especie':<24} {'dieta':<12} {'imágenes':>8}  {'acción'}")
    print("─" * 60)

    for sp_dir in species_dirs:
        species = sp_dir.name
        diet    = diet_labels.get(species)

        images = sorted(f for f in sp_dir.iterdir()
                        if f.suffix.lower() in config.IMG_EXTS)
        if not images:
            continue

        if diet not in config.DIET_CLASSES:
            no_diet.append(species)
            print(f"{species:<24} {'?':<12} {len(images):>8}  sin dieta — se deja")
            skipped += len(images)
            continue

        dest_dir = config.DOWNLOADS_DIR / diet
        accion   = "simulado" if args.dry_run else "movido"

        if not args.dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)

        for f in images:
            dest = dest_dir / f.name
            i = 1
            while dest.exists():
                dest = dest_dir / f"{f.stem}_{i}{f.suffix}"
                i += 1
            if not args.dry_run:
                shutil.move(str(f), str(dest))
            moved += 1

        print(f"{species:<24} {diet:<12} {len(images):>8}  {accion}")

        if not args.dry_run and not any(sp_dir.iterdir()):
            sp_dir.rmdir()

    elapsed = time.time() - t_start
    print("─" * 60)
    print(f"Movidas: {moved}   Sin dieta (dejadas): {skipped}   Tiempo: {elapsed:.1f}s")
    if no_diet:
        print(f"Especies sin dieta: {', '.join(no_diet)}")
    if not args.dry_run and moved:
        print(f"Imágenes en: {config.DOWNLOADS_DIR}")
        print("Siguiente paso:  python curate.py curate downloads/")


if __name__ == "__main__":
    main()
