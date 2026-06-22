"""
Descargador de imágenes de dinosaurios OMNÍVOROS extintos desde Wikimedia Commons.

Lee la lista de especies de omnivore_extinct_candidates.csv y guarda las imágenes
bajo una sola carpeta `omnivoros/`, con una subcarpeta por especie:

    omnivoros/
    ├── Deinocheirus_mirificus/
    │   ├── Deinocheirus_mirificus_001.jpg
    │   └── ...
    ├── Gallimimus_bullatus/
    └── ...

Por qué se reescribió: la versión anterior raspaba miniaturas de Google Images con
Selenium y NO descargaba nada — las miniaturas de Google son pequeñas y no pasaban
el filtro de 300x300, y la imagen full-res no viene en el `src`. Esta versión usa
la API pública de Wikimedia Commons (la misma que el resto del proyecto): sin
navegador, sin captcha, con licencias libres y resolución usable.

Uso:
    python imagenes.py
    python imagenes.py --per-species 60 --insecure
    python imagenes.py --csv otra_lista.csv --out otra_carpeta --no-ocr

--insecure omite la verificación TLS (redes corporativas con proxy, igual que
download.py). Si Commons falla por TLS, el script te lo sugiere.
"""

from __future__ import annotations

__version__ = "2.0.0"  # 1.x scraping de Google (Selenium) · 2.0 Wikimedia Commons + CSV

import argparse
import csv
import io
import os
import sys

from PIL import Image

# La consola de Windows suele ser cp1252 y no traga acentos/caja; forzamos UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Fuente de imágenes: reutiliza la API de Commons ya probada del proyecto.
from curator.sources import (
    download_url,
    iter_commons_photos,
    probe_commons,
    silence_insecure_warnings,
)

# OCR opcional: descarta imágenes con texto incrustado (diagramas, infografías).
try:
    import pytesseract
    OCR_DISPONIBLE = True
except ImportError:
    OCR_DISPONIBLE = False

ROOT        = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(ROOT, "omnivore_extinct_candidates.csv")
DEFAULT_OUT = os.path.join(ROOT, "omnivoros")


# ── filtros de calidad ────────────────────────────────────────────────────────
def es_imagen_de_calidad(data, ancho_min=300, alto_min=300):
    """Devuelve (ok, ancho, alto, imagen_pil). ok=True si cumple la resolución mínima."""
    try:
        imagen = Image.open(io.BytesIO(data))
        ancho, alto = imagen.size
        if ancho >= ancho_min and alto >= alto_min:
            return True, ancho, alto, imagen
        return False, ancho, alto, None
    except Exception:
        return False, 0, 0, None


def tiene_texto_incrustado(imagen_pil):
    """True si la imagen tiene texto reconocible (probable diagrama/infografía)."""
    if not OCR_DISPONIBLE:
        return False
    try:
        texto = pytesseract.image_to_string(imagen_pil).strip()
        return len(texto) > 5
    except Exception:
        return False


# ── descarga de una especie ───────────────────────────────────────────────────
def descargar_especie(folder_name, display_name, out_dir, cantidad, usar_ocr):
    """Baja hasta `cantidad` imágenes de `display_name` en out_dir/folder_name/."""
    carpeta = os.path.join(out_dir, folder_name)
    os.makedirs(carpeta, exist_ok=True)

    # Reanudable: si ya hay suficientes, se salta.
    existentes = [f for f in os.listdir(carpeta) if f.lower().endswith(".jpg")]
    if len(existentes) >= cantidad:
        print(f"  saltando: ya hay {len(existentes)} imágenes en {folder_name}/")
        return 0

    descargadas = len(existentes)
    vistos      = set()
    # Pedimos de más (x4) porque el filtro de resolución/OCR descarta varias.
    for pageid, url in iter_commons_photos(display_name, max_photos=cantidad * 4):
        if descargadas >= cantidad:
            break
        if pageid in vistos:
            continue
        vistos.add(pageid)

        data = download_url(url)
        if not data:
            continue

        ok, _w, _h, img = es_imagen_de_calidad(data)
        if not (ok and img):
            continue
        if usar_ocr and tiene_texto_incrustado(img):
            continue

        nombre = f"{folder_name}_{descargadas + 1:03d}.jpg"
        ruta   = os.path.join(carpeta, nombre)
        try:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.save(ruta, "JPEG")
            descargadas += 1
        except Exception:
            continue

    nuevas = descargadas - len(existentes)
    print(f"  {display_name}: +{nuevas}  (total {descargadas} en {folder_name}/)")
    return nuevas


# ── lectura del CSV ───────────────────────────────────────────────────────────
def leer_csv(path):
    """Lee omnivore_extinct_candidates.csv → lista de filas (dict)."""
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    print(f"imagenes.py v{__version__}")
    ap = argparse.ArgumentParser(
        description="Baja imágenes de omnívoros extintos (CSV) desde Wikimedia Commons.",
    )
    ap.add_argument("--csv", default=DEFAULT_CSV, help="ruta al CSV de especies")
    ap.add_argument("--out", default=DEFAULT_OUT, help="carpeta raíz de salida (def. omnivoros/)")
    ap.add_argument("--per-species", type=int, default=100, help="imágenes por especie (def. 100)")
    ap.add_argument("--insecure", action="store_true", help="omite la verificación TLS (proxy corporativo)")
    ap.add_argument("--no-ocr", action="store_true", help="desactiva el filtro de texto OCR")
    args = ap.parse_args()

    if args.insecure:
        os.environ["ADC_INSECURE_SSL"] = "1"
        silence_insecure_warnings()
        print("AVISO: verificación TLS desactivada (--insecure). Úsalo solo en redes de confianza.")

    if not os.path.isfile(args.csv):
        print(f"ERROR: no encuentro el CSV: {args.csv}")
        sys.exit(1)

    problema = probe_commons()
    if problema:
        print(f"No puedo usar Wikimedia Commons: {problema}")
        if "TLS" in problema or "SSL" in problema:
            print("Sugerencia: vuelve a correr con  --insecure")
        sys.exit(1)

    filas    = leer_csv(args.csv)
    usar_ocr = OCR_DISPONIBLE and not args.no_ocr
    os.makedirs(args.out, exist_ok=True)

    print(f"\n{len(filas)} especies → '{args.out}/'  (hasta {args.per_species} c/u)")
    if not usar_ocr:
        print(" (filtro de texto OCR inactivo)")

    total = 0
    for row in filas:
        display = (row.get("display_name") or "").strip()
        folder  = (row.get("species_folder") or display.replace(" ", "_")).strip()
        if not display:
            continue
        conf = row.get("confidence_for_omnivore", "?")
        print(f"\n--- {display}  [{conf}] ---")
        total += descargar_especie(folder, display, args.out, args.per_species, usar_ocr)

    print(f"\n¡Listo! {total} imágenes nuevas en '{args.out}/'")


if __name__ == "__main__":
    main()
