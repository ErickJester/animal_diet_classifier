"""
Descargador de imágenes de dinosaurios OMNÍVOROS extintos desde Google Imágenes.

Lee la lista de especies de omnivore_extinct_candidates.csv y guarda las imágenes
bajo una sola carpeta `omnivoros/`, con una subcarpeta por especie:

    omnivoros/
    ├── Deinocheirus_mirificus/
    │   ├── Deinocheirus_mirificus_001.jpg
    │   └── ...
    ├── Gallimimus_bullatus/
    └── ...

POR QUÉ ESTA VERSIÓN SÍ DESCARGA:
La versión vieja tomaba el `src` crudo de la grilla: muchas eran base64 diminutas o
iconos que no pasaban 300x300, así que no bajaba nada. Aquí se cosechan las copias
que Google sirve en encrypted-tbn0.gstatic.com (token 'tbn:ANd9Gc'), que al
descargarlas dan ~400-700px — suficiente para entrenar (el modelo usa 224px) — y se
obtienen decenas por especie. El filtro de resolución descarta los iconos pequeños.

Google Imágenes da mucha más variedad (paleoarte) que Wikimedia Commons para
dinosaurios oscuros, a costa de ser frágil: el HTML cambia y puede aparecer
captcha/consentimiento. Por eso el script hace una pausa para que resuelvas el
captcha a mano antes de empezar (desactívala con --no-pause).

Uso:
    python imagenes.py
    python imagenes.py --per-species 100
    python imagenes.py --headless --no-pause --insecure
    python imagenes.py --min-res 250 --no-ocr

Requisitos: selenium + Google Chrome (chromedriver lo resuelve Selenium solo).
"""

from __future__ import annotations

__version__ = "3.0.0"  # 1.x google miniaturas (no servía) · 2.0 Commons · 3.0 google full-res por clic

import argparse
import base64
import csv
import io
import os
import sys
import time

import requests
from PIL import Image

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

# La consola de Windows suele ser cp1252 y no traga acentos/caja; forzamos UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# OCR opcional: descarta imágenes con texto incrustado (diagramas, infografías).
try:
    import pytesseract
    OCR_DISPONIBLE = True
except ImportError:
    OCR_DISPONIBLE = False

ROOT        = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(ROOT, "omnivore_extinct_candidates.csv")
DEFAULT_OUT = os.path.join(ROOT, "omnivoros")

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
             "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")


# ── filtros de calidad ────────────────────────────────────────────────────────
def es_imagen_de_calidad(data, ancho_min, alto_min):
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
        return len(pytesseract.image_to_string(imagen_pil).strip()) > 5
    except Exception:
        return False


# ── descarga de una URL (http o data:image), con reintentos ───────────────────
def descargar_bytes(src, timeout=8, reintentos=2):
    if src.startswith("data:image"):
        try:
            _, b64 = src.split(",", 1)
            return base64.b64decode(b64)
        except Exception:
            return None
    if not src.startswith("http"):
        return None
    for intento in range(reintentos):
        if intento:
            time.sleep(0.3 * intento)
        try:
            r = requests.get(src, headers={"User-Agent": USER_AGENT},
                             timeout=timeout, verify=not _INSECURE)
            if r.status_code == 200 and r.content:
                return r.content
            if r.status_code in (403, 404, 410):
                return None
        except Exception:
            pass
    return None


# ── Selenium / Google Imágenes ────────────────────────────────────────────────
def crear_driver(headless):
    opts = Options()
    opts.add_argument("--log-level=3")
    opts.add_argument(f"user-agent={USER_AGENT}")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    if headless:
        opts.add_argument("--headless=new")
        opts.add_argument("--window-size=1920,1080")
    return webdriver.Chrome(options=opts)


def aceptar_consentimiento(driver):
    """Intenta cerrar el aviso de cookies/consentimiento de Google (UE)."""
    textos = ["Aceptar todo", "Accept all", "Acepto", "I agree", "Rechazar todo", "Reject all"]
    for t in textos:
        try:
            botones = driver.find_elements(By.XPATH, f"//button[.//text()[contains(., '{t}')]] | //div[@role='button'][contains(., '{t}')]")
            if botones:
                botones[0].click()
                time.sleep(1)
                return
        except Exception:
            continue


def scroll_para_cargar(driver, veces=4):
    for _ in range(veces):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.5)


def cosechar_urls(driver):
    """
    Cosecha las URLs de imagen de la grilla de Google (sin clicar miniatura a miniatura).

    Google sirve sus copias en encrypted-tbn0.gstatic.com con el token 'tbn:ANd9Gc';
    al descargarlas dan ~400-700px (suficiente para entrenar; el modelo usa 224px).
    Es mucho más robusto y rápido que abrir la vista previa de cada imagen, y evita
    depender de los nombres de clase de Google, que cambian seguido.

    Nota: son copias re-comprimidas de Google, no el original del sitio fuente; para
    un clasificador de dieta es irrelevante, y a cambio se obtienen decenas por especie.
    """
    urls   = []
    vistas = set()
    for im in driver.find_elements(By.TAG_NAME, "img"):
        src = im.get_attribute("src") or im.get_attribute("data-src") or ""
        if src.startswith("http") and "tbn:ANd9Gc" in src and src not in vistas:
            vistas.add(src)
            urls.append(src)
    return urls


# ── descarga de una especie ───────────────────────────────────────────────────
def descargar_especie(driver, folder_name, display_name, query, out_dir, cantidad,
                      min_res, usar_ocr):
    carpeta = os.path.join(out_dir, folder_name)
    os.makedirs(carpeta, exist_ok=True)

    existentes = [f for f in os.listdir(carpeta) if f.lower().endswith(".jpg")]
    if len(existentes) >= cantidad:
        print(f"  saltando: ya hay {len(existentes)} imágenes en {folder_name}/")
        return 0

    descargadas = len(existentes)
    # La query del CSV ya está pensada para Google; añadimos variaciones de respaldo.
    variaciones = [
        query or display_name,
        f"{display_name} reconstruction",
        f"{display_name} paleoart",
        f"{display_name} dinosaur",
    ]

    for variacion in variaciones:
        if descargadas >= cantidad:
            break
        print(f"  buscando: {variacion!r}")
        driver.get("https://www.google.com/search?tbm=isch&q=" + requests.utils.quote(variacion))
        time.sleep(1.5)
        aceptar_consentimiento(driver)
        scroll_para_cargar(driver, veces=6)

        for url in cosechar_urls(driver):
            if descargadas >= cantidad:
                break
            data = descargar_bytes(url)
            if not data:
                continue
            ok, _w, _h, img = es_imagen_de_calidad(data, min_res, min_res)
            if not (ok and img):
                continue
            if usar_ocr and tiene_texto_incrustado(img):
                continue
            nombre = f"{folder_name}_{descargadas + 1:03d}.jpg"
            try:
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img.save(os.path.join(carpeta, nombre), "JPEG")
                descargadas += 1
            except Exception:
                continue

    nuevas = descargadas - len(existentes)
    print(f"  {display_name}: +{nuevas}  (total {descargadas} en {folder_name}/)")
    return nuevas


def leer_csv(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ── CLI ───────────────────────────────────────────────────────────────────────
_INSECURE = False   # se fija desde --insecure; lo usa descargar_bytes


def main():
    global _INSECURE
    print(f"imagenes.py v{__version__}")
    ap = argparse.ArgumentParser(
        description="Baja imágenes de omnívoros extintos (CSV) desde Google Imágenes con Selenium.",
    )
    ap.add_argument("--csv", default=DEFAULT_CSV, help="ruta al CSV de especies")
    ap.add_argument("--out", default=DEFAULT_OUT, help="carpeta raíz de salida (def. omnivoros/)")
    ap.add_argument("--per-species", type=int, default=100, help="imágenes por especie (def. 100)")
    ap.add_argument("--min-res", type=int, default=300, help="resolución mínima en px (def. 300)")
    ap.add_argument("--insecure", action="store_true", help="omite verificación TLS al bajar (proxy)")
    ap.add_argument("--no-ocr", action="store_true", help="desactiva el filtro de texto OCR")
    ap.add_argument("--headless", action="store_true", help="Chrome sin ventana (puede disparar captcha)")
    ap.add_argument("--no-pause", action="store_true", help="no esperar a resolver captcha manualmente")
    args = ap.parse_args()

    _INSECURE = args.insecure
    if not os.path.isfile(args.csv):
        print(f"ERROR: no encuentro el CSV: {args.csv}")
        sys.exit(1)

    filas    = leer_csv(args.csv)
    usar_ocr = OCR_DISPONIBLE and not args.no_ocr
    os.makedirs(args.out, exist_ok=True)

    driver = crear_driver(args.headless)
    try:
        # Chequeo de seguridad: deja resolver captcha/cookies antes de empezar.
        driver.get("https://www.google.com/search?tbm=isch&q=dinosaurios+omnivoros")
        aceptar_consentimiento(driver)
        if not args.no_pause:
            print("\n" + "=" * 70)
            print("Si aparece captcha o aviso de cookies, resuélvelo en la ventana de Chrome.")
            input("Cuando la página esté limpia, presiona ENTER para iniciar la descarga...")
            print("=" * 70)

        print(f"\n{len(filas)} especies → '{args.out}/'  (hasta {args.per_species} c/u, min {args.min_res}px)")
        if not usar_ocr:
            print(" (filtro de texto OCR inactivo)")

        total = 0
        for row in filas:
            display = (row.get("display_name") or "").strip()
            folder  = (row.get("species_folder") or display.replace(" ", "_")).strip()
            query   = (row.get("recommended_search_query") or "").strip()
            if not display:
                continue
            conf = row.get("confidence_for_omnivore", "?")
            print(f"\n--- {display}  [{conf}] ---")
            total += descargar_especie(driver, folder, display, query, args.out,
                                       args.per_species, args.min_res, usar_ocr)
            time.sleep(1)

        print(f"\n¡Listo! {total} imágenes nuevas en '{args.out}/'")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
