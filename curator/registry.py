"""
Registro de especies y etiquetas de dieta, ambos respaldados por JSON.

SpeciesRegistry — cuántas fotos tenemos de cada especie y qué dieta tiene
                  asignada (si se conoce). Se actualiza a medida que el curador
                  acepta imágenes.

DietLabels     — mapa "nombre → dieta". Es la fuente de verdad de la dieta: se
                 edita a mano (curator/data/diet_labels.json) y el curador solo
                 lee de aquí. Si una especie no está, su dieta es desconocida y
                 la imagen se deja en revisión.
"""

from __future__ import annotations

__version__ = "1.0.1"  # 1.0 SpeciesRegistry + DietLabels · 1.0.1 sync con downloader

import json
from pathlib import Path
from typing import Dict, ItemsView, Optional

from . import config


class DietLabels:
    """Mapa nombre→dieta. Acepta tanto nombre científico como común (case-insensitive)."""

    def __init__(self, mapping: Dict[str, str]) -> None:
        self.raw = dict(mapping)                                # conserva la grafía original
        self._by_name = {k.lower(): v for k, v in mapping.items()}

    def get(self, scientific_name: str, common_name: str = "") -> Optional[str]:
        if scientific_name and scientific_name.lower() in self._by_name:
            return self._by_name[scientific_name.lower()]
        if common_name and common_name.lower() in self._by_name:
            return self._by_name[common_name.lower()]
        return None

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "DietLabels":
        path = Path(path or config.DIET_LABELS_PATH)
        if path.is_file():
            with open(path, encoding="utf-8") as fh:
                return cls(json.load(fh))
        return cls({})


class SpeciesRegistry:
    """Cuántas fotos hay por especie y su dieta asignada."""

    def __init__(self, data: Optional[Dict[str, dict]] = None) -> None:
        self._data: Dict[str, dict] = data or {}

    def count(self, species: str) -> int:
        return self._data.get(species, {}).get("count", 0)

    def has(self, species: str) -> bool:
        return species in self._data

    def add(self, species: str, common_name: str = "",
            diet: Optional[str] = None, n: int = 1) -> None:
        entry = self._data.setdefault(
            species, {"common_name": common_name, "count": 0, "diet": diet}
        )
        entry["count"] += n
        if common_name and not entry.get("common_name"):
            entry["common_name"] = common_name
        if diet:
            entry["diet"] = diet

    def items(self) -> ItemsView[str, dict]:
        return self._data.items()

    def total_species(self) -> int:
        return len(self._data)

    def total_images(self) -> int:
        return sum(e.get("count", 0) for e in self._data.values())

    def save(self, path: Optional[Path] = None) -> None:
        path = Path(path or config.REGISTRY_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "SpeciesRegistry":
        path = Path(path or config.REGISTRY_PATH)
        if path.is_file():
            with open(path, encoding="utf-8") as fh:
                return cls(json.load(fh))
        return cls({})
