"""
Identificación de especie vía la API de visión de iNaturalist.

iNaturalist expone un endpoint de visión por computadora que, dada una imagen,
sugiere las especies más probables. Requiere un token de API (gratuito, ligado
a una cuenta de iNaturalist).

  - El token se toma del constructor o de la variable de entorno
    INATURALIST_API_TOKEN.
  - Sin token, o si `requests` no está instalado, `is_available` es False y
    `identify()` devuelve None. El curador degrada con elegancia a decidir solo
    por similitud visual.

Cómo obtener el token (requiere haber iniciado sesión en iNaturalist):
    https://www.inaturalist.org/users/api_token

El parseo de la respuesta es defensivo: ante cualquier formato inesperado o
error de red, devuelve None en lugar de lanzar excepción.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

# ── importación opcional de requests ──────────────────────────────────────────
_REQUESTS_AVAILABLE = False
try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    pass

API_URL = "https://api.inaturalist.org/v1/computervision/score_image"


@dataclass
class SpeciesGuess:
    scientific_name: str
    common_name:     str
    score:           float


class SpeciesIdentifier:
    def __init__(self, token: str = "", timeout: float = 20.0) -> None:
        self.token        = token or os.environ.get("INATURALIST_API_TOKEN", "")
        self.timeout      = timeout
        self.is_available = bool(_REQUESTS_AVAILABLE and self.token)

    def identify(self, image_path: str) -> Optional[SpeciesGuess]:
        """Sube la imagen y devuelve la mejor sugerencia de especie, o None."""
        if not self.is_available:
            return None
        try:
            with open(image_path, "rb") as fh:
                resp = requests.post(
                    API_URL,
                    headers={"Authorization": self.token},
                    files={"image": fh},
                    timeout=self.timeout,
                )
            if resp.status_code != 200:
                return None

            results = resp.json().get("results", [])
            if not results:
                return None

            top   = results[0]
            taxon = top.get("taxon", {}) or {}
            name  = taxon.get("name")
            if not name:
                return None

            return SpeciesGuess(
                scientific_name=name,
                common_name=taxon.get("preferred_common_name", "") or "",
                score=float(top.get("combined_score", top.get("score", 0.0)) or 0.0),
            )
        except Exception:
            return None
