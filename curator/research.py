"""
Investigación automática de dieta por especie (fase C, animales extintos).

Para una especie sin dieta conocida en diet_labels.json, consulta el texto del
artículo de Wikipedia y cuenta menciones de familias de términos dietéticos
(carnívoro / herbívoro / omnívoro). Devuelve la clase dominante con una confianza
0..1 y la evidencia, o dieta=None si no hay señales claras.

IMPORTANTE — esto SUGIERE, no decide. Respeta la regla del proyecto ("la dieta
NUNCA se adivina, sale de diet_labels.json"): la sugerencia se vuelca a
diet_suggestions.json para que una persona la confirme. El downloader solo
clasifica automáticamente cuando la confianza supera COMMONS_DIET_MIN_CONF; por
debajo, la especie espera en _unsorted a la revisión.

    g = research_diet("Triceratops")
    print(g.diet, g.confidence)   # herbivore 0.67

No detecta morfología ni lee la imagen: es puro análisis de texto.
"""

from __future__ import annotations

__version__ = "1.0.0"  # 1.0 heurística de dieta sobre texto de Wikipedia

import re
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from . import config

# Reutiliza la lógica TLS centralizada de sources (proxy corporativo / --insecure).
from .sources import _REQUESTS_AVAILABLE, _verify

try:
    import requests
except Exception:  # pragma: no cover - mismo manejo opcional que sources
    requests = None  # type: ignore


# Familias de términos por dieta. Insectívoro/piscívoro cuentan como carnívoro;
# folívoro/granívoro/frugívoro como herbívoro. Se evita "predator/prey", demasiado
# contextual (un herbívoro suele mencionar a sus depredadores y daría falsos +).
_DIET_PATTERNS: Dict[str, re.Pattern] = {
    "carnivore": re.compile(r"carnivor|hypercarnivor|insectivor|piscivor"),
    "herbivore": re.compile(r"herbivor|folivor|granivor|frugivor"),
    "omnivore":  re.compile(r"omnivor"),
}


@dataclass
class DietGuess:
    diet:       Optional[str]            # "carnivore" | "herbivore" | "omnivore" | None
    confidence: float                    # 0..1 (proporción del término dominante)
    counts:     Dict[str, int] = field(default_factory=dict)  # menciones por dieta
    evidence:   str = ""                 # resumen legible de la decisión
    source_url: str = ""                 # artículo consultado


_WIKI_RETRIES = 3   # el proxy corporativo falla de forma intermitente bajo carga


def _wikipedia_fulltext(title: str) -> str:
    """
    Texto plano del artículo de Wikipedia (siguiendo redirecciones). '' si falla.

    Reintenta unas veces: en redes con proxy (modo --insecure) las peticiones
    fallan de forma intermitente y un único intento daría falsos "sin artículo".
    """
    if not _REQUESTS_AVAILABLE:
        return ""
    for attempt in range(_WIKI_RETRIES):
        if attempt:
            time.sleep(0.5 * attempt)   # backoff suave: ayuda con throttling/proxy
        try:
            resp = requests.get(
                config.WIKIPEDIA_API_URL,
                params={
                    "action":     "query",
                    "format":     "json",
                    "prop":       "extracts",
                    "explaintext": 1,
                    "redirects":  1,
                    "titles":     title,
                },
                headers={"User-Agent": config.HTTP_USER_AGENT},
                timeout=config.HTTP_TIMEOUT,
                verify=_verify(),
            )
            if resp.status_code != 200:
                continue
            pages = resp.json().get("query", {}).get("pages", {})
            for _, page in pages.items():
                return page.get("extract", "") or ""
        except Exception:
            continue
    return ""


def _count_diet(text: str) -> Dict[str, int]:
    return {diet: len(pat.findall(text)) for diet, pat in _DIET_PATTERNS.items()}


def research_diet(species: str) -> DietGuess:
    """
    Sugiere la dieta de `species` contando términos en su artículo de Wikipedia.

    Si el binomio no da señales, reintenta con el GÉNERO (primer token): en
    extintos la dieta suele describirse en el artículo del género, no de la especie.
    """
    url  = "https://en.wikipedia.org/wiki/" + species.replace(" ", "_")
    text = _wikipedia_fulltext(species).lower()

    counts = _count_diet(text) if text else {}
    total  = sum(counts.values())

    # Fallback al género si el binomio no aportó señales de dieta.
    if total == 0 and " " in species:
        genus = species.split()[0]
        gtext = _wikipedia_fulltext(genus).lower()
        if gtext:
            gcounts = _count_diet(gtext)
            if sum(gcounts.values()) > 0:
                counts, total = gcounts, sum(gcounts.values())
                diet = max(counts, key=counts.get)
                evid = "  ".join(f"{d}:{n}" for d, n in counts.items())
                return DietGuess(diet, counts[diet] / total, counts,
                                 f"menciones (vía género '{genus}') → {evid}", url)

    if not text:
        return DietGuess(None, 0.0, {}, "sin artículo de Wikipedia", url)
    if total == 0:
        return DietGuess(None, 0.0, counts, "el artículo no menciona la dieta", url)

    diet = max(counts, key=counts.get)
    evid = "  ".join(f"{d}:{n}" for d, n in counts.items())
    return DietGuess(diet, counts[diet] / total, counts, f"menciones → {evid}", url)
