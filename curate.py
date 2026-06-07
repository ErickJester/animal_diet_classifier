"""
Curador de dataset — CLI.

Mantiene el dataset limpio y variado: indexa lo que ya tenemos y filtra
imágenes nuevas (p. ej. descargadas de iNaturalist) para quedarse solo con
especies que faltan o que aún no llegan al cupo, evitando duplicados.

Subcomandos:
    index                 Indexa el dataset/ existente (etapa previa, hazlo una vez).
    curate <ruta...>      Cura imágenes/carpetas nuevas → copia las útiles a staging/.
    status                Muestra el estado: nº indexado, especies, pendientes.
    commit                Mueve lo aceptado en staging/ al dataset (split train/val).

Ejemplos:
    python curate.py index
    python curate.py curate descargas_inaturalist/
    python curate.py status
    python curate.py commit

Requiere torch/torchvision (extractor de embeddings). La identificación de
especie es opcional: define INATURALIST_API_TOKEN o pasa --inat-token para
activarla; sin token, el curador decide solo por similitud visual.
"""

import argparse
import shutil
import sys
from pathlib import Path

from curator import DatasetCurator
from curator import config


# ── utilidades ──────────────────────────────────────────────────────────────────

def collect_images(paths) -> list:
    """Expande rutas: archivos sueltos o carpetas (recursivo)."""
    images = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            images.extend(sorted(
                f for f in path.rglob("*") if f.suffix.lower() in config.IMG_EXTS
            ))
        elif path.is_file():
            images.append(path)
        else:
            print(f"ADVERTENCIA: no existe: {path}")
    return images


def _count_images(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for f in folder.rglob("*") if f.suffix.lower() in config.IMG_EXTS)


# ── subcomando: index ───────────────────────────────────────────────────────────

def cmd_index(args):
    curator = _build_curator(args)
    print("Indexando dataset existente...")
    added = curator.index_existing()
    curator.save()
    print(f"  + {added} imágenes nuevas indexadas")
    print(f"  Índice total: {len(curator.index)} imágenes")


# ── subcomando: curate ──────────────────────────────────────────────────────────

def cmd_curate(args):
    curator = _build_curator(args)

    if not curator.identifier.is_available:
        print("NOTA: iNaturalist no disponible (sin token o sin 'requests').")
        print("      Se decidirá solo por similitud visual.\n")

    images = collect_images(args.inputs)
    if not images:
        print("No se encontraron imágenes para curar.")
        sys.exit(1)

    kept = skipped = 0
    print(f"{'archivo':<34} {'decisión':<8} {'sim':>5}  motivo / especie")
    print("─" * 78)
    for img_path in images:
        res = curator.curate_image(str(img_path))
        if res.decision == "keep":
            kept += 1
            detail = res.reason + (f"  [{res.species}]" if res.species else "")
            detail += f"  → {res.diet}" if res.diet else "  → (revisión)"
        else:
            skipped += 1
            detail = res.reason + (f"  [{res.species}]" if res.species else "")
        print(f"{img_path.name[:34]:<34} {res.decision:<8} {res.similarity:>5.2f}  {detail}")

    curator.save()
    print("─" * 78)
    print(f"Aceptadas: {kept}   Descartadas: {skipped}")
    print(f"Las aceptadas están en: {config.STAGING_DIR}")
    print("Revisa y luego ejecuta:  python curate.py commit")


# ── subcomando: status ──────────────────────────────────────────────────────────

def cmd_status(args):
    curator = _build_curator(args, require_model=False)

    print("── Índice de similitud ──")
    print(f"  Imágenes indexadas: {len(curator.index)}")

    print("\n── Registro de especies ──")
    print(f"  Especies: {curator.registry.total_species()}   "
          f"Fotos contabilizadas: {curator.registry.total_images()}")
    for species, info in sorted(curator.registry.items()):
        diet = info.get("diet") or "?"
        common = info.get("common_name") or ""
        print(f"    {species:<28} {info.get('count', 0):>4}  {diet:<10} {common}")

    print("\n── Staging (pendiente de commit) ──")
    for diet in config.DIET_CLASSES:
        print(f"    {diet:<14} {_count_images(config.STAGING_DIR / diet):>4}")
    print(f"    _pending_review {_count_images(config.PENDING_REVIEW_DIR):>4}  (especie conocida, dieta sin definir)")
    print(f"    _unidentified   {_count_images(config.UNIDENTIFIED_DIR):>4}  (visualmente nuevas, sin especie)")


# ── subcomando: commit ──────────────────────────────────────────────────────────

def cmd_commit(args):
    """Mueve las imágenes aceptadas con dieta conocida al dataset (split train/val)."""
    every = max(2, round(1 / config.VAL_RATIO))   # cada cuántas, una va a val
    moved_train = moved_val = 0

    for diet in config.DIET_CLASSES:
        staged = config.STAGING_DIR / diet
        if not staged.is_dir():
            continue
        files = sorted(f for f in staged.iterdir() if f.suffix.lower() in config.IMG_EXTS)
        for i, f in enumerate(files):
            split = "val" if (i % every == 0) else "train"
            dest_dir = config.DATASET_DIR / split / diet
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = _unique(dest_dir, f)
            shutil.move(str(f), str(dest))
            if split == "val":
                moved_val += 1
            else:
                moved_train += 1

    print(f"Movidas a train/: {moved_train}   a val/: {moved_val}")
    print("Quedan en staging (sin tocar) las de _pending_review y _unidentified:")
    print(f"  asigna su dieta en {config.DIET_LABELS_PATH} y vuelve a curar, o muévelas a mano.")


def _unique(dest_dir: Path, src: Path) -> Path:
    candidate = dest_dir / src.name
    i = 1
    while candidate.exists():
        candidate = dest_dir / f"{src.stem}_{i}{src.suffix}"
        i += 1
    return candidate


# ── construcción del curador ─────────────────────────────────────────────────────

def _build_curator(args, require_model: bool = True) -> DatasetCurator:
    curator = DatasetCurator(
        device=getattr(args, "device", ""),
        inat_token=getattr(args, "inat_token", ""),
    )
    if require_model and not curator.is_available:
        print("ERROR: extractor de embeddings no disponible.")
        print("  Instala torch/torchvision:  pip install torch torchvision")
        sys.exit(1)
    return curator


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Curador de dataset (opción C: similitud visual → especie)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device",     default="", help="cuda | cpu | mps | '' (autodetectar)")
    parser.add_argument("--inat-token", default="", help="token de la API de iNaturalist")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("index",  help="indexa el dataset existente")

    p_curate = sub.add_parser("curate", help="cura imágenes/carpetas nuevas")
    p_curate.add_argument("inputs", nargs="+", help="foto(s) o carpeta(s) a curar")

    sub.add_parser("status", help="muestra el estado del curador")
    sub.add_parser("commit", help="mueve lo aceptado al dataset (train/val)")

    return parser.parse_args()


def main():
    args = parse_args()
    dispatch = {
        "index":  cmd_index,
        "curate": cmd_curate,
        "status": cmd_status,
        "commit": cmd_commit,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
