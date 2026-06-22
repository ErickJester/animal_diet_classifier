"""
Fuentes de imágenes para el descargador.

Dos familias:

  1. Datasets masivos (fase A) — Kaggle / Hugging Face / Roboflow.
     Cada fuente se declara de forma declarativa en BULK_SOURCES y se baja con
     su herramienta oficial. Si falta el paquete o las credenciales, el fetcher
     NO revienta: devuelve (None, motivo) y el descargador la salta con un aviso
     accionable (degradación elegante, igual que el resto del proyecto).

  2. Fuentes con especie verificada (fase B) — iNaturalist.
     El endpoint público de observaciones devuelve fotos cuya especie fue
     confirmada por la comunidad ("research grade"). Al consultar POR especie,
     toda foto descargada es con certeza ese animal: es la versión fiable de
     "buscar en Google pero solo en páginas que de verdad son del animal".

Requisitos:
  · iNaturalist y descarga por URL: solo 'requests' (ya en requirements.txt).
  · Kaggle:        pip install kaggle          + ~/.kaggle/kaggle.json
  · Hugging Face:  pip install huggingface_hub
  · Roboflow:      pip install roboflow        + variable ROBOFLOW_API_KEY
"""

from __future__ import annotations

__version__ = "1.1.0"  # 1.0 fetchers Kaggle/HF/Roboflow + iNat · 1.1 soporte paralelización

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

from . import config

# ── importación opcional de requests ──────────────────────────────────────────
_REQUESTS_AVAILABLE = False
try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    pass


# ── verificación TLS ──────────────────────────────────────────────────────────
# En redes corporativas con proxy (Zscaler, Netskope, etc.) el certificado lo
# firma un CA interno que Python no conoce → "CERTIFICATE_VERIFY_FAILED". Dos
# salidas, ambas estándar de 'requests':
#   · limpia : exporta REQUESTS_CA_BUNDLE apuntando al CA de tu empresa.
#   · rápida : ADC_INSECURE_SSL=1 omite la verificación (inseguro, solo si confías
#              en la red). Equivale al flag --insecure de la CLI.
def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _verify() -> bool:
    return not _env_truthy("ADC_INSECURE_SSL")


def silence_insecure_warnings() -> None:
    """Calla el aviso 'Unverified HTTPS request' al ir en modo --insecure."""
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        warnings.filterwarnings("ignore", message="Unverified HTTPS request")


if not _verify():  # caso: ADC_INSECURE_SSL ya venía en el entorno al importar
    silence_insecure_warnings()


def _explain(exc: Exception) -> str:
    text = str(exc)
    if "CERTIFICATE_VERIFY" in text or "SSL" in type(exc).__name__.upper():
        return ("fallo de verificación TLS/SSL (típico de red corporativa con proxy). "
                "Define REQUESTS_CA_BUNDLE con el CA de tu empresa, o usa --insecure "
                "(env ADC_INSECURE_SSL=1) para omitir la verificación.")
    return f"no se pudo conectar ({type(exc).__name__}: {text})"


def probe_inaturalist() -> Optional[str]:
    """Comprueba que iNaturalist responde. Devuelve None si OK, o un motivo legible."""
    if not _REQUESTS_AVAILABLE:
        return "el paquete 'requests' no está instalado"
    try:
        resp = requests.get(
            config.INAT_OBS_URL,
            params={"per_page": 1},
            headers={"User-Agent": config.HTTP_USER_AGENT},
            timeout=config.HTTP_TIMEOUT,
            verify=_verify(),
        )
        if resp.status_code == 200:
            return None
        return f"respuesta inesperada de iNaturalist (HTTP {resp.status_code})"
    except Exception as exc:
        return _explain(exc)


# ── catálogo de datasets masivos ──────────────────────────────────────────────
@dataclass
class BulkSource:
    name:    str           # identificador corto para la CLI (--only)
    kind:    str           # "kaggle" | "hf" | "roboflow"
    ref:     str           # slug/repo/coordenadas según la fuente
    note:    str = ""      # descripción legible
    enabled: bool = True    # las que requieren datos a completar van en False
    fixed_diet: Optional[str] = None  # si se fija, TODA la fuente se etiqueta con esta
                                       # dieta (p.ej. "other") sin pasar por diet_labels.json
    max_images: int = 0                # tope de imágenes a guardar (0 = sin tope); útil
                                       # para mantener balanceada una clase como "other"


# Sólo las habilitadas se intentan por defecto. Las deshabilitadas necesitan que
# verifiques el slug o completes credenciales antes de activarlas (--only o
# editando enabled=True aquí).
BULK_SOURCES = [
    BulkSource(
        name="animals90",
        kind="kaggle",
        ref="iamsouravbanerjee/animal-image-dataset-90-different-animals",
        note="Animal Image Dataset — 90 especies (~5.4k imgs, mucha variedad)",
    ),
    BulkSource(
        name="mammals45",
        kind="kaggle",
        ref="asaniczka/mammals-image-classification-dataset-45-animals",
        note="Mammals Image Classification — 45 mamíferos (~6k imgs, limpio)",
    ),
    BulkSource(
        name="animals10",
        kind="kaggle",
        ref="alessiocorrado99/animals10",
        note="Animals-10 — 10 especies (~28k imgs; carpetas en italiano)",
        enabled=False,   # volumen alto y nombres en italiano → habilítalo si lo quieres
    ),
    BulkSource(
        name="roboflow-ch",
        kind="roboflow",
        ref="workspace/project/version",   # ← completa con las coordenadas reales
        note="Roboflow Carnívoro/Herbívoro (~700 imgs ya etiquetadas por dieta)",
        enabled=False,   # requiere ROBOFLOW_API_KEY y coordenadas workspace/project/version
    ),
    BulkSource(
        name="hf-big-animals",
        kind="hf",
        ref="Isamu136/big-animal-dataset",
        note="Hugging Face — dataset grande y variado de animales",
        enabled=False,   # verifica el repo_id y que venga como carpetas de imágenes
    ),

    # ── clase "other" (NO animal) ──────────────────────────────────────────────
    # Fuentes PURAMENTE no-animales: enseñan al modelo a rechazar lo que no es un
    # animal (objetos, comida, paisajes, personas). fixed_diet="other" salta el
    # registro de especies (toda la fuente es "other"); max_images mantiene la
    # clase balanceada (~10k en total, el promedio de las otras tres clases).
    # IMPORTANTE: no metas aquí datasets con carpetas de animales (gato/perro/…),
    # los etiquetaría como "other", justo lo contrario de lo que queremos.
    BulkSource(
        name="other-scenes",
        kind="kaggle",
        ref="puneet6060/intel-image-classification",
        note="Intel Image Classification — escenas/paisajes/edificios/calle (no-animal)",
        fixed_diet="other",
        max_images=5500,
    ),
    BulkSource(
        name="other-food",
        kind="kaggle",
        ref="kmader/food41",
        note="Food-101 — platos de comida (no-animal). Verifica el slug si la descarga falla.",
        fixed_diet="other",
        max_images=4500,
    ),
    BulkSource(
        name="other-objects",
        kind="kaggle",
        ref="usuario/dataset-de-objetos-no-animales",   # ← completa con un slug real
        note="Objetos/personas no-animales — da variedad a 'other'. Completa el slug "
             "o usa una carpeta local con prepare.py (ver README).",
        enabled=False,   # requiere que verifiques/completes el slug antes de activarla
        fixed_diet="other",
        max_images=3000,
    ),
]


# ── fetchers de datasets masivos ──────────────────────────────────────────────
def fetch_bulk(source: BulkSource, dest: Path) -> Tuple[Optional[Path], str]:
    """Descarga y extrae una fuente masiva en `dest`. Devuelve (ruta, "") u (None, motivo)."""
    dispatch = {"kaggle": _fetch_kaggle, "hf": _fetch_hf, "roboflow": _fetch_roboflow}
    fn = dispatch.get(source.kind)
    if fn is None:
        return None, f"tipo de fuente desconocido: {source.kind}"
    dest.mkdir(parents=True, exist_ok=True)
    return fn(source.ref, dest)


def _fetch_kaggle(ref: str, dest: Path) -> Tuple[Optional[Path], str]:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        return None, "paquete 'kaggle' no instalado  →  pip install kaggle"
    try:
        api = KaggleApi()
        api.authenticate()  # usa ~/.kaggle/kaggle.json o KAGGLE_USERNAME/KAGGLE_KEY
        api.dataset_download_files(ref, path=str(dest), unzip=True, quiet=False)
        return dest, ""
    except Exception as e:
        return None, f"kaggle no autenticado o slug inválido ({e}). "\
                     "Coloca tu kaggle.json en ~/.kaggle/"


def _fetch_hf(ref: str, dest: Path) -> Tuple[Optional[Path], str]:
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return None, "paquete 'huggingface_hub' no instalado  →  pip install huggingface_hub"
    try:
        snapshot_download(
            repo_id=ref,
            repo_type="dataset",
            local_dir=str(dest),
            local_dir_use_symlinks=False,
        )
        return dest, ""
    except Exception as e:
        return None, f"huggingface: {e}"


def _fetch_roboflow(ref: str, dest: Path) -> Tuple[Optional[Path], str]:
    key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not key:
        return None, "falta la variable ROBOFLOW_API_KEY"
    try:
        from roboflow import Roboflow
    except Exception:
        return None, "paquete 'roboflow' no instalado  →  pip install roboflow"
    try:
        ws, proj, ver = ref.split("/")
    except ValueError:
        return None, f"ref roboflow debe ser 'workspace/project/version', no '{ref}'"
    try:
        rf = Roboflow(api_key=key)
        rf.workspace(ws).project(proj).version(int(ver)).download("folder", location=str(dest))
        return dest, ""
    except Exception as e:
        return None, f"roboflow: {e}"


# ── descarga por URL (genérica) ───────────────────────────────────────────────
def download_url(url: str) -> Optional[bytes]:
    """Descarga el contenido de una URL. Devuelve los bytes o None ante cualquier error."""
    if not _REQUESTS_AVAILABLE:
        return None
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": config.HTTP_USER_AGENT},
            timeout=config.HTTP_TIMEOUT,
            verify=_verify(),
        )
        if resp.status_code == 200 and resp.content:
            return resp.content
    except Exception:
        pass
    return None


# ── iNaturalist: fotos con especie verificada (fase B) ────────────────────────
def iter_inaturalist_photos(species: str, max_photos: int) -> Iterator[Tuple[str, str]]:
    """
    Itera (photo_id, url_media) de observaciones research-grade de una especie.

    `species` debe ser el nombre científico (taxon_name). Sólo se piden fotos con
    licencia reutilizable. Paginación acotada por config.INAT_MAX_PAGES.
    """
    if not _REQUESTS_AVAILABLE or max_photos <= 0:
        return

    yielded = 0
    for page in range(1, config.INAT_MAX_PAGES + 1):
        params = {
            "taxon_name":    species,
            "quality_grade": "research",
            "photos":        "true",
            "photo_license": config.INAT_LICENSES,
            "per_page":      config.INAT_PAGE_SIZE,
            "page":          page,
            "order_by":      "votes",
            "order":         "desc",
        }
        try:
            resp = requests.get(
                config.INAT_OBS_URL,
                params=params,
                headers={"User-Agent": config.HTTP_USER_AGENT},
                timeout=config.HTTP_TIMEOUT,
                verify=_verify(),
            )
        except Exception:
            return
        if resp.status_code != 200:
            return

        results = resp.json().get("results", [])
        if not results:
            return

        for obs in results:
            for photo in obs.get("photos", []) or []:
                pid = photo.get("id")
                url = photo.get("url")
                if not pid or not url:
                    continue
                yield str(pid), _to_medium(url)
                yielded += 1
                if yielded >= max_photos:
                    return


def _to_medium(url: str) -> str:
    """Las URLs de iNat vienen en tamaño 'square'; pedimos 'medium' (≈500px)."""
    return url.replace("square", "medium")
