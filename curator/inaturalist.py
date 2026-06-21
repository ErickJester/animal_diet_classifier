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

__version__ = "1.1.0"  # 1.0 cliente visión iNaturalist · 1.1 throttle + retry 429 + last_status

import os
import time
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

# iNaturalist pide mantenerse en ≤60 peticiones/min (y ~10k/día); por encima
# responde 429 y la imagen quedaría sin identificar sin haber sido analizada.
MIN_INTERVAL    = 1.0   # segundos mínimos entre peticiones
MAX_RETRIES_429 = 3     # reintentos ante 429 antes de rendirse
BACKOFF_429     = 30.0  # espera (s) si el 429 no trae cabecera Retry-After


@dataclass
class SpeciesGuess:
    scientific_name: str
    common_name:     str
    score:           float


class SpeciesIdentifier:
    def __init__(self, token: str = "", timeout: float = 20.0,
                 min_interval: float = MIN_INTERVAL) -> None:
        self.token         = token or os.environ.get("INATURALIST_API_TOKEN", "")
        self.timeout       = timeout
        self.min_interval  = min_interval
        self.is_available  = bool(_REQUESTS_AVAILABLE and self.token)
        self.last_status   = ""    # causa del último identify(): "ok", "http_429", "error_red", ...
        self._last_request = 0.0

    def _throttle(self) -> None:
        """Espacia las peticiones para respetar el rate limit de iNaturalist."""
        wait = self.min_interval - (time.time() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.time()

    def identify(self, image_path: str) -> Optional[SpeciesGuess]:
        """Sube la imagen y devuelve la mejor sugerencia de especie, o None.

        Cuando devuelve None, `last_status` indica la causa:
          "sin_resultados"  la API respondió pero no reconoció ninguna especie
          "http_429"        rate limit agotado tras los reintentos
          "http_<código>"   otro error HTTP
          "error_red"       fallo de conexión o timeout
          "json_invalido" / "sin_taxon"   respuesta con formato inesperado
        """
        if not self.is_available:
            self.last_status = "no_disponible"
            return None

        resp = None
        for intento in range(MAX_RETRIES_429 + 1):
            self._throttle()
            try:
                with open(image_path, "rb") as fh:
                    resp = requests.post(
                        API_URL,
                        headers={"Authorization": self.token},
                        files={"image": fh},
                        timeout=self.timeout,
                    )
            except Exception:
                self.last_status = "error_red"
                return None

            if resp.status_code == 429:
                self.last_status = "http_429"
                if intento < MAX_RETRIES_429:
                    try:
                        espera = float(resp.headers.get("Retry-After", BACKOFF_429))
                    except (TypeError, ValueError):
                        espera = BACKOFF_429
                    time.sleep(espera)
                    continue
                return None
            if resp.status_code != 200:
                self.last_status = f"http_{resp.status_code}"
                return None
            break

        try:
            results = resp.json().get("results", [])
        except Exception:
            self.last_status = "json_invalido"
            return None
        if not results:
            self.last_status = "sin_resultados"
            return None

        top   = results[0]
        taxon = top.get("taxon", {}) or {}
        name  = taxon.get("name")
        if not name:
            self.last_status = "sin_taxon"
            return None

        self.last_status = "ok"
        return SpeciesGuess(
            scientific_name=name,
            common_name=taxon.get("preferred_common_name", "") or "",
            score=float(top.get("combined_score", top.get("score", 0.0)) or 0.0),
        )
