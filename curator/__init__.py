"""
Curador de dataset para el clasificador de dieta animal.

Mantiene el dataset limpio y variado filtrando imágenes nuevas mediante una
cascada (opción C):

    Etapa 1 — similitud visual (rápida, offline) descarta duplicados.
    Etapa 2 — identificación de especie (iNaturalist) decide por cupo de especie.

API pública:
    from curator import DatasetCurator, CurationResult
"""

from .curator import CurationResult, DatasetCurator

__all__ = ["DatasetCurator", "CurationResult"]
