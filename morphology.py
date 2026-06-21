"""
Inferencia de dieta por RASGOS MORFOLÓGICOS — el "plan B" del clasificador.

Idea: el clasificador principal (classifier.py) intenta reconocer la especie/dieta
con la CNN. Cuando su confianza es baja, este módulo entra en juego: dado un
conjunto de rasgos observados en la foto (ojos al frente, garras, pico ganchudo…),
suma sus pesos hacia carnívoro / herbívoro / omnívoro y devuelve la clase ganadora
con su porcentaje de confianza, o "uncertain" si no se alcanza el umbral.

La base de conocimiento (qué rasgo pesa cuánto hacia cada dieta) vive en
diet_morphology.json — editable sin tocar código.

    scorer = MorphologyDietScorer()
    res = scorer.score({
        "eye_placement": "forward_facing",
        "feet_type": "sharp_claws_or_talons",
        "canine_teeth": "long_sharp_prominent",
    })
    print(res.label, res.confidence)      # carnivore 0.74

Cada observación puede llevar una confianza propia (0..1) si el detector de rasgos
no está seguro:

    scorer.score({"eye_placement": ("forward_facing", 0.6), "feet_type": "hooves"})

NOTA: este módulo NO detecta rasgos desde la imagen; consume rasgos YA observados.
Convertir píxeles → rasgos (detección de atributos/keypoints) es un paso aparte
todavía por construir. Mientras tanto los rasgos pueden venir de anotación manual
o de un futuro detector. Ver README / classifier.py para la integración en cascada.
"""

from __future__ import annotations

__version__ = "1.0.0"  # 1.0 scoring ponderado de rasgos morfológicos (plan B)

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

# La consola de Windows suele ser cp1252 y no traga caracteres de barra/acentos.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

CLASSES = ["carnivore", "herbivore", "omnivore"]
DEFAULT_KB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diet_morphology.json")

# Una observación es el estado del rasgo, opcionalmente con su confianza (0..1).
Observation = Union[str, Tuple[str, float]]


@dataclass
class MorphologyResult:
    label:      str                       # "carnivore" | "herbivore" | "omnivore" | "uncertain"
    confidence: float                     # confianza de la clase ganadora (0..1)
    scores:     Dict[str, float]          # confianza por clase (suma 1.0)
    n_features: int                       # nº de rasgos válidos usados
    used:       Dict[str, str] = field(default_factory=dict)   # rasgo → estado aplicado
    ignored:    Dict[str, str] = field(default_factory=dict)   # rasgo → motivo de descarte
    source:     str = "morphology"


class MorphologyDietScorer:
    """Puntúa dieta a partir de rasgos morfológicos usando diet_morphology.json."""

    def __init__(self, kb_path: str = DEFAULT_KB) -> None:
        with open(kb_path, encoding="utf-8") as fh:
            self.kb = json.load(fh)

        self.features = {f["id"]: f for f in self.kb["features"]}
        scoring = self.kb.get("scoring", {})
        self.decision_threshold   = float(scoring.get("decision_threshold", 0.50))
        self.min_features_required = int(scoring.get("min_features_required", 2))
        self.classes = self.kb.get("classes", CLASSES)

    # ── API principal ─────────────────────────────────────────────────────────

    def score(self, observations: Dict[str, Observation]) -> MorphologyResult:
        """
        observations: {feature_id: estado}  o  {feature_id: (estado, confianza)}.
        Devuelve MorphologyResult con la clase ganadora y la confianza por clase.
        """
        contribution = {c: 0.0 for c in self.classes}
        used:    Dict[str, str] = {}
        ignored: Dict[str, str] = {}

        for feat_id, obs in observations.items():
            state, obs_conf = self._unpack(obs)

            feature = self.features.get(feat_id)
            if feature is None:
                ignored[feat_id] = "rasgo desconocido"
                continue
            state_def = feature.get("states", {}).get(state)
            if state_def is None:
                ignored[feat_id] = f"estado '{state}' no válido"
                continue
            if obs_conf <= 0.0:
                ignored[feat_id] = "confianza <= 0"
                continue

            reliability = float(feature.get("reliability", 1.0))
            affinity    = state_def["affinity"]
            weight      = reliability * obs_conf
            for c in self.classes:
                contribution[c] += weight * float(affinity.get(c, 0.0))
            used[feat_id] = state

        n = len(used)
        total = sum(contribution.values())
        if total <= 0.0 or n == 0:
            return MorphologyResult("uncertain", 0.0,
                                    {c: 0.0 for c in self.classes}, 0,
                                    used, ignored)

        scores = {c: contribution[c] / total for c in self.classes}
        best   = max(scores, key=scores.get)
        best_p = scores[best]

        label = best if (best_p >= self.decision_threshold
                         and n >= self.min_features_required) else "uncertain"
        return MorphologyResult(label, best_p, scores, n, used, ignored)

    # ── utilidades ────────────────────────────────────────────────────────────

    @staticmethod
    def _unpack(obs: Observation) -> Tuple[str, float]:
        """Acepta 'estado' o ('estado', confianza); por defecto confianza = 1.0."""
        if isinstance(obs, (tuple, list)):
            state = obs[0]
            conf  = float(obs[1]) if len(obs) > 1 else 1.0
            return state, conf
        return obs, 1.0

    def feature_ids(self) -> list:
        return list(self.features.keys())

    def states_of(self, feature_id: str) -> list:
        return list(self.features.get(feature_id, {}).get("states", {}).keys())


# ── CLI de prueba ────────────────────────────────────────────────────────────────
# Permite probar el scoring a mano:  python morphology.py eye_placement=forward_facing feet_type=hooves

def _parse_kv(pairs) -> Dict[str, Observation]:
    obs: Dict[str, Observation] = {}
    for p in pairs:
        if "=" not in p:
            print(f"ADVERTENCIA: ignorando '{p}' (usa rasgo=estado o rasgo=estado:conf)")
            continue
        key, val = p.split("=", 1)
        if ":" in val:
            state, conf = val.split(":", 1)
            obs[key] = (state, float(conf))
        else:
            obs[key] = val
    return obs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan B: infiere dieta a partir de rasgos morfológicos observados.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Ejemplo:\n"
               "  python morphology.py eye_placement=forward_facing "
               "canine_teeth=long_sharp_prominent feet_type=sharp_claws_or_talons\n\n"
               "Confianza por rasgo con ':':  feet_type=hooves:0.8\n"
               "Listar rasgos y estados:       python morphology.py --list",
    )
    parser.add_argument("observations", nargs="*", metavar="rasgo=estado",
                        help="rasgos observados (ver --list)")
    parser.add_argument("--list", action="store_true",
                        help="muestra todos los rasgos y sus estados posibles")
    parser.add_argument("--kb", default=DEFAULT_KB, help="ruta a diet_morphology.json")
    args = parser.parse_args()

    scorer = MorphologyDietScorer(args.kb)

    if args.list or not args.observations:
        print(f"morphology.py v{__version__}  —  base: {os.path.basename(args.kb)}\n")
        print("Rasgos disponibles (rasgo → estados):")
        for fid, feat in scorer.features.items():
            vis = "" if feat.get("visible_in_photo", True) else "  [no visible en foto]"
            print(f"\n  {fid}  (reliability {feat.get('reliability', 1.0)}){vis}")
            for st in feat.get("states", {}):
                print(f"      - {st}")
        if not args.observations:
            print("\nPasa observaciones para puntuar, p.ej.:")
            print("  python morphology.py eye_placement=forward_facing feet_type=sharp_claws_or_talons")
        return

    obs = _parse_kv(args.observations)
    res = scorer.score(obs)

    print(f"morphology.py v{__version__}\n")
    print("Rasgos usados:")
    for fid, st in res.used.items():
        print(f"  {fid:<22} = {st}")
    if res.ignored:
        print("\nRasgos ignorados:")
        for fid, why in res.ignored.items():
            print(f"  {fid:<22} → {why}")

    print("\nConfianza por dieta:")
    for c in scorer.classes:
        bar = "█" * int(round(res.scores[c] * 30))
        print(f"  {c:<11} {res.scores[c]*100:5.1f}%  {bar}")

    print(f"\n→ Decisión: {res.label.upper()}  "
          f"({res.confidence*100:.1f}% · {res.n_features} rasgos · "
          f"umbral {scorer.decision_threshold*100:.0f}%)")
    if res.label == "uncertain":
        print("  (no se alcanzó el umbral o faltan rasgos → quedaría en revisión)")


if __name__ == "__main__":
    main()
