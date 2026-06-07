"""
Descargador de imágenes para armar el dataset — CLI.

Flujo en dos fases:

  Fase A · datasets masivos (Kaggle / Hugging Face / Roboflow)
      python download.py datasets
      python download.py datasets --only animals90 mammals45

  Fase B · suplemento con especie verificada (iNaturalist), tras la fase A
      python download.py more
      python download.py more --per-species 60

  Todo de una:
      python download.py all

  Estado / catálogo:
      python download.py status
      python download.py sources

Lo descargado aterriza en downloads/<dieta>/ (y downloads/_unsorted/<especie>/
para especies sin dieta en diet_labels.json). En cada ejecución se salta lo ya
descargado: datasets completos y, a nivel imagen, duplicados exactos por hash.

La identificación de especie de iNaturalist NO requiere token (endpoint público
de observaciones). Las fuentes masivas sí requieren sus paquetes/credenciales;
si faltan, esa fuente se omite con un aviso y el resto continúa.
"""

import argparse
import os
import sys

# La consola de Windows suele ser cp1252 y no traga los caracteres de caja ni
# acentos. Forzamos UTF-8 en la salida para no reventar al imprimir.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from curator import config
from curator.downloader import DatasetDownloader
from curator.manifest import DownloadManifest
from curator.sources import BULK_SOURCES


def _count_images(folder) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for f in folder.rglob("*") if f.suffix.lower() in config.IMG_EXTS)


# ── subcomando: datasets (fase A) ─────────────────────────────────────────────
def cmd_datasets(args):
    DatasetDownloader().fetch_datasets(only=args.only)


# ── subcomando: more (fase B) ─────────────────────────────────────────────────
def cmd_more(args):
    _apply_insecure(args)
    DatasetDownloader().supplement(per_species=args.per_species)


# ── subcomando: all ───────────────────────────────────────────────────────────
def cmd_all(args):
    _apply_insecure(args)
    dl = DatasetDownloader()
    dl.fetch_datasets(only=None)
    dl.supplement(per_species=args.per_species)


def _apply_insecure(args):
    """--insecure → omite la verificación TLS (redes corporativas con proxy)."""
    if getattr(args, "insecure", False):
        os.environ["ADC_INSECURE_SSL"] = "1"
        from curator.sources import silence_insecure_warnings
        silence_insecure_warnings()
        print("AVISO: verificación TLS desactivada (--insecure). Úsalo solo en redes de confianza.\n")


# ── subcomando: status ────────────────────────────────────────────────────────
def cmd_status(args):
    m = DownloadManifest.load()

    print("── Fase A: datasets masivos ──")
    if not m.datasets:
        print("  (ninguno intentado todavía)")
    for ref, info in sorted(m.datasets.items()):
        print(f"    {info.get('status','?'):<9} {info.get('images',0):>6} imgs  {ref}")

    print("\n── Imágenes en disco (downloads/) ──")
    for diet in config.DIET_CLASSES:
        print(f"    {diet:<14} {_count_images(config.DOWNLOADS_DIR / diet):>6}")
    print(f"    _unsorted      {_count_images(config.UNSORTED_DIR):>6}  (especies sin dieta en diet_labels.json)")

    print("\n── Anti-repetición ──")
    print(f"    hashes únicos guardados: {len(m.hashes)}")
    print(f"    fotos iNaturalist vistas: {len(m.photo_ids.get('inaturalist', []))}")
    print(f"    fase A intentada: {'sí' if m.phase_a_attempted else 'no'}")
    if not m.phase_a_attempted:
        print("    (la fase B 'more' se habilita tras la primera corrida de 'datasets')")


# ── subcomando: sources ───────────────────────────────────────────────────────
def cmd_sources(args):
    print(f"{'nombre':<16} {'tipo':<9} {'activa':<7} descripción")
    print("─" * 78)
    for s in BULK_SOURCES:
        print(f"{s.name:<16} {s.kind:<9} {'sí' if s.enabled else 'no':<7} {s.note}")
    print("\nLas inactivas requieren verificar el slug o completar credenciales.")
    print("Puedes forzar una con:  python download.py datasets --only <nombre>")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Descargador de imágenes (fase A datasets → fase B iNaturalist verificado)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ds = sub.add_parser("datasets", help="fase A: baja los datasets masivos")
    p_ds.add_argument("--only", nargs="+", metavar="NOMBRE",
                      help="limita a estas fuentes (ver 'sources')")

    p_more = sub.add_parser("more", help="fase B: suplementa desde iNaturalist (tras fase A)")
    p_more.add_argument("--per-species", type=int, default=None,
                        help=f"tope de fotos por especie (def. {config.SUPP_PER_SPECIES})")
    p_more.add_argument("--insecure", action="store_true",
                        help="omite la verificación TLS (red corporativa con proxy)")

    p_all = sub.add_parser("all", help="fase A y luego fase B")
    p_all.add_argument("--per-species", type=int, default=None,
                       help=f"tope de fotos por especie (def. {config.SUPP_PER_SPECIES})")
    p_all.add_argument("--insecure", action="store_true",
                       help="omite la verificación TLS (red corporativa con proxy)")

    sub.add_parser("status",  help="muestra el estado de las descargas")
    sub.add_parser("sources", help="lista los datasets masivos disponibles")

    return parser.parse_args()


def main():
    args = parse_args()
    dispatch = {
        "datasets": cmd_datasets,
        "more":     cmd_more,
        "all":      cmd_all,
        "status":   cmd_status,
        "sources":  cmd_sources,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
