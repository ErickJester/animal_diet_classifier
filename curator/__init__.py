"""
Curador de dataset para el clasificador de dieta animal.

Mantiene el dataset limpio y variado filtrando imágenes nuevas mediante una
cascada (opción C):

    Etapa 1 — similitud visual (rápida, offline) descarta duplicados.
    Etapa 2 — identificación de especie (iNaturalist) decide por cupo de especie.

También provee la descarga de imágenes para armar el dataset desde cero
(DatasetDownloader: datasets masivos → suplemento con especie verificada).

API pública:
    from curator import DatasetCurator, CurationResult   # requiere torch/opencv
    from curator import DatasetDownloader                 # solo requiere requests

Los símbolos se importan de forma perezosa: pedir DatasetDownloader no obliga a
tener torch/opencv instalados (solo se cargan si usas el curador).
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # solo para type-checkers; no se ejecuta en runtime
    from .curator import CurationResult, DatasetCurator
    from .downloader import DatasetDownloader

__all__ = ["DatasetCurator", "CurationResult", "DatasetDownloader"]

_LAZY = {
    "DatasetCurator": ".curator",
    "CurationResult": ".curator",
    "DatasetDownloader": ".downloader",
}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module, __name__), name)
