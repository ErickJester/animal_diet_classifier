"""
Curador de dataset — opción C: cascada similitud-visual → especie.

Para cada imagen candidata:

  Etapa 1 — Similitud visual (rápida, offline)
    Calcula el embedding y lo compara con el índice de lo que ya tenemos.
      · similitud >= DUPLICATE_SIM   → DESCARTAR (casi idéntica a una existente)
      · similitud <  NEW_SPECIES_SIM → muy probablemente especie nueva → Etapa 2
      · en medio                     → ambiguo → Etapa 2 decide por especie

  Etapa 2 — Identificación de especie (lenta, online; solo para sobrevivientes)
    Pregunta a iNaturalist qué especie es.
      · especie con TARGET_PER_SPECIES fotos ya → DESCARTAR (suficientes)
      · especie con cupo libre                  → ACEPTAR
      · sin identificar pero visualmente nueva  → ACEPTAR (a revisión, sin especie)

  Al ACEPTAR, la imagen se copia a staging/ (por dieta si se conoce, o a
  _pending_review / _unidentified si no) y se actualizan índice y registro.

Degradación: si iNaturalist no está disponible (sin token / sin requests), la
Etapa 2 se salta y la decisión se toma solo con la similitud visual.
"""

from __future__ import annotations

__version__ = "1.0.0"  # 1.0 cascada similitud visual → especie (opción C)

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import config
from .embeddings import EmbeddingModel
from .index import SimilarityIndex
from .inaturalist import SpeciesIdentifier
from .registry import DietLabels, SpeciesRegistry


@dataclass
class CurationResult:
    path:       str
    decision:   str            # "keep" | "skip"
    reason:     str            # motivo legible
    species:    str   = ""     # especie identificada (si la hubo)
    diet:       str   = ""     # dieta asignada (si se conoce)
    similarity: float = 0.0    # similitud con la imagen más parecida del índice
    staged_to:  str   = ""     # ruta donde se copió (solo si keep)


class DatasetCurator:
    def __init__(self, device: str = "", inat_token: str = "") -> None:
        self.embedder    = EmbeddingModel(device=device)
        self.index       = SimilarityIndex.load()
        self.registry    = SpeciesRegistry.load()
        self.diet_labels = DietLabels.load()
        self.identifier  = SpeciesIdentifier(token=inat_token)
        self.is_available = self.embedder.is_available

    # ── etapa 0: indexar el dataset existente ──────────────────────────────────
    def index_existing(self) -> int:
        """Recorre dataset/ y añade al índice toda imagen aún no indexada."""
        known = set(self.index.paths)
        added = 0
        for diet in config.DIET_CLASSES:
            for split in config.SPLITS:
                folder = config.DATASET_DIR / split / diet
                if not folder.is_dir():
                    continue
                for img_path in sorted(folder.iterdir()):
                    if img_path.suffix.lower() not in config.IMG_EXTS:
                        continue
                    if str(img_path) in known:
                        continue
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    self.index.add(str(img_path), self.embedder.embed(img))
                    added += 1
        return added

    # ── curado de una imagen candidata (opción C) ──────────────────────────────
    def curate_image(self, image_path: str) -> CurationResult:
        img = cv2.imread(str(image_path))
        if img is None:
            return CurationResult(str(image_path), "skip", "no_image")

        emb    = self.embedder.embed(img)
        sim, _ = self.index.nearest(emb)

        # Etapa 1 — duplicado visual
        if sim >= config.DUPLICATE_SIM:
            return CurationResult(str(image_path), "skip", "duplicado_visual", similarity=sim)

        # Etapa 2 — identificación de especie
        guess = self.identifier.identify(str(image_path))

        if guess is None:
            # sin especie: aceptar solo si es visualmente muy distinta a todo
            if sim < config.NEW_SPECIES_SIM:
                return self._accept(image_path, emb, species="", common="",
                                    diet=None, sim=sim, reason="visualmente_nueva_sin_id")
            return CurationResult(str(image_path), "skip", "similar_sin_id", similarity=sim)

        species, common = guess.scientific_name, guess.common_name

        # cupo de especie ya alcanzado
        if self.registry.count(species) >= config.TARGET_PER_SPECIES:
            return CurationResult(str(image_path), "skip", "especie_completa",
                                  species=species, similarity=sim)

        diet   = self.diet_labels.get(species, common)
        reason = "especie_nueva" if not self.registry.has(species) else "amplia_especie"
        return self._accept(image_path, emb, species=species, common=common,
                            diet=diet, sim=sim, reason=reason)

    # ── aceptar: copiar a staging + actualizar estado ──────────────────────────
    def _accept(self, image_path: str, emb: np.ndarray, species: str, common: str,
                diet: Optional[str], sim: float, reason: str) -> CurationResult:
        src = Path(image_path)

        if diet in config.DIET_CLASSES:
            dest_dir = config.STAGING_DIR / diet
        elif species:
            dest_dir = config.PENDING_REVIEW_DIR / self._safe(species)
        else:
            dest_dir = config.UNIDENTIFIED_DIR
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest = self._unique_dest(dest_dir, src, species)
        shutil.copy2(src, dest)

        # el staged pasa a formar parte de lo "ya conocido"
        self.index.add(str(dest), emb)
        if species:
            self.registry.add(species, common,
                              diet if diet in config.DIET_CLASSES else None)

        return CurationResult(str(image_path), "keep", reason, species=species,
                              diet=diet or "", similarity=sim, staged_to=str(dest))

    # ── utilidades de nombres de archivo ───────────────────────────────────────
    @staticmethod
    def _safe(name: str) -> str:
        cleaned = "".join(c if c.isalnum() else "_" for c in name).strip("_")
        return cleaned or "desconocida"

    def _unique_dest(self, dest_dir: Path, src: Path, species: str) -> Path:
        stem = f"{self._safe(species)}_{src.stem}" if species else src.stem
        ext  = src.suffix.lower()
        candidate = dest_dir / f"{stem}{ext}"
        i = 1
        while candidate.exists():
            candidate = dest_dir / f"{stem}_{i}{ext}"
            i += 1
        return candidate

    # ── persistir estado ───────────────────────────────────────────────────────
    def save(self) -> None:
        self.index.save()
        self.registry.save()
