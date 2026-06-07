"""
Manifiesto de descargas — registro ligero para no repetir trabajo.

A diferencia del curador (que compara embeddings visuales, costoso), aquí la
deduplicación es barata y de una sola pasada:

  · a nivel dataset  → marca cada fuente masiva como hecha; en la siguiente
                       ejecución la fuente se salta entera sin re-escanear.
  · a nivel imagen   → guarda el hash MD5 del contenido; una imagen ya vista
                       (aunque venga de otra fuente) no se vuelve a guardar.
  · a nivel foto-API → guarda los ids de foto ya traídos por fuente (iNaturalist)
                       para no volver a descargarlos.
  · por especie      → cuántas fotos llevamos por especie en el suplemento
                       (para respetar el tope por especie).

Todo se persiste en un único JSON (config.MANIFEST_PATH). Los conjuntos se
serializan como listas ordenadas.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional, Set

from . import config


class DownloadManifest:
    def __init__(self, data: Optional[dict] = None) -> None:
        data = data or {}
        self.datasets: Dict[str, dict] = data.get("datasets", {})          # ref → {status, images, at}
        self.hashes: Set[str]          = set(data.get("hashes", []))        # MD5 de contenido
        self.photo_ids: Dict[str, Set[str]] = {                            # fuente → ids de foto vistos
            src: set(str(i) for i in ids)
            for src, ids in data.get("photo_ids", {}).items()
        }
        self.species: Dict[str, int]   = data.get("species_counts", {})    # especie → nº suplementado
        self.phase_a_attempted: bool   = bool(data.get("phase_a_attempted", False))

    # ── datasets masivos (fase A) ──────────────────────────────────────────────
    def dataset_status(self, ref: str) -> Optional[str]:
        return self.datasets.get(ref, {}).get("status")

    def is_dataset_done(self, ref: str) -> bool:
        return self.dataset_status(ref) == "done"

    def mark_dataset(self, ref: str, status: str, images: int = 0) -> None:
        self.datasets[ref] = {
            "status": status,
            "images": images,
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    # ── deduplicación por contenido ────────────────────────────────────────────
    def seen_hash(self, digest: str) -> bool:
        return digest in self.hashes

    def add_hash(self, digest: str) -> None:
        self.hashes.add(digest)

    # ── fotos ya traídas de una fuente-API ─────────────────────────────────────
    def seen_photo(self, source: str, photo_id) -> bool:
        return str(photo_id) in self.photo_ids.get(source, set())

    def add_photo(self, source: str, photo_id) -> None:
        self.photo_ids.setdefault(source, set()).add(str(photo_id))

    # ── conteo por especie (suplemento) ────────────────────────────────────────
    def species_count(self, species: str) -> int:
        return self.species.get(species, 0)

    def add_species(self, species: str, n: int = 1) -> None:
        self.species[species] = self.species.get(species, 0) + n

    # ── persistencia ───────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "phase_a_attempted": self.phase_a_attempted,
            "datasets": self.datasets,
            "species_counts": self.species,
            "photo_ids": {src: sorted(ids) for src, ids in self.photo_ids.items()},
            "hashes": sorted(self.hashes),
        }

    def save(self, path: Optional[Path] = None) -> None:
        path = Path(path or config.MANIFEST_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "DownloadManifest":
        path = Path(path or config.MANIFEST_PATH)
        if path.is_file():
            with open(path, encoding="utf-8") as fh:
                return cls(json.load(fh))
        return cls({})
