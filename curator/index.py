"""
Índice de similitud visual.

Guarda los embeddings (normalizados L2) de todas las imágenes ya conocidas,
junto con su ruta de origen. Dada una imagen nueva, responde cuál es la más
parecida y con qué similitud coseno.

Persistencia en un único .npz para no recalcular embeddings en cada ejecución.

Nota de rendimiento: la búsqueda del vecino más cercano es un producto
matriz-vector O(N·D), suficiente para algunos miles de imágenes. Para datasets
mucho mayores convendría un índice ANN (faiss/annoy).
"""

from __future__ import annotations

__version__ = "1.0.0"  # 1.0 índice coseno persistente en npz

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from . import config


class SimilarityIndex:
    def __init__(self) -> None:
        self._vecs:   List[np.ndarray] = []   # embeddings (cada uno EMBED_DIM,)
        self._paths:  List[str]        = []   # ruta paralela a cada embedding
        self._matrix: Optional[np.ndarray] = None  # caché [N, D], se invalida al añadir

    def __len__(self) -> int:
        return len(self._paths)

    @property
    def paths(self) -> List[str]:
        return list(self._paths)

    # ── construcción ───────────────────────────────────────────────────────────
    def add(self, path: str, embedding: np.ndarray) -> None:
        self._vecs.append(embedding.astype(np.float32).reshape(-1))
        self._paths.append(str(path))
        self._matrix = None   # invalidar caché

    def _as_matrix(self) -> np.ndarray:
        if self._matrix is None:
            self._matrix = (np.vstack(self._vecs) if self._vecs
                            else np.zeros((0, config.EMBED_DIM), np.float32))
        return self._matrix

    # ── consulta ───────────────────────────────────────────────────────────────
    def nearest(self, embedding: np.ndarray) -> Tuple[float, Optional[str]]:
        """Retorna (similitud_coseno_máxima, ruta) o (0.0, None) si está vacío."""
        if not self._vecs:
            return 0.0, None
        sims = self._as_matrix() @ embedding.reshape(-1)   # coseno (todo normalizado L2)
        idx  = int(np.argmax(sims))
        return float(sims[idx]), self._paths[idx]

    # ── persistencia ───────────────────────────────────────────────────────────
    def save(self, path: Optional[Path] = None) -> None:
        path = Path(path or config.INDEX_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            embeddings=self._as_matrix(),
            paths=np.array(self._paths, dtype=object),
        )

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "SimilarityIndex":
        path  = Path(path or config.INDEX_PATH)
        index = cls()
        if path.is_file():
            data = np.load(path, allow_pickle=True)
            matrix = data["embeddings"].astype(np.float32)
            index._vecs  = [row for row in matrix]
            index._paths = [str(p) for p in data["paths"]]
        return index
