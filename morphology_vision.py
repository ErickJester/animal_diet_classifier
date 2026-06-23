"""
Puente PÍXELES → RASGOS para el Plan C (morfología).

`morphology.py` sabe puntuar dieta a partir de rasgos YA observados
({"body_plan_extinct_mythic": "large_theropod_predator", ...}), pero alguien
tiene que MIRAR la foto y producir esas observaciones. Este módulo es ese
"alguien", usando únicamente visión por computadora clásica (OpenCV + numpy):
sin redes neuronales ni dependencias nuevas.

LIMITACIÓN IMPORTANTE (asumida a propósito):
    cv2 solo puede medir geometría global de la silueta (proporción, orientación,
    concavidades). NO puede ver dientes serrados, garras vs pezuñas ni posición
    de los ojos. Por eso este detector solo emite los rasgos de PLAN CORPORAL y
    EXTREMIDADES, y siempre con CONFIANZA BAJA. El voto ponderado de morphology.py
    se encarga de que, cuando las señales son ambiguas, el resultado quede
    "uncertain" y el clasificador caiga de vuelta al mejor resultado de la CNN.

    Es un Plan C aproximado y honesto, no un detector de atributos preciso. Para
    precisión real haría falta CLIP zero-shot o un detector de atributos entrenado.

Salida: dict {feature_id: (state, confianza)} listo para MorphologyDietScorer.score().

Uso directo (debug):
    python morphology_vision.py foto.jpg
"""

from __future__ import annotations

__version__ = "1.0.0"  # 1.0 detector geométrico (silueta) → plan corporal + extremidades

import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# La consola de Windows suele ser cp1252 y revienta con acentos/cajas.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

Observation = Tuple[str, float]


@dataclass
class VisionFeatures:
    """Métricas geométricas crudas de la silueta del sujeto (para debug/inspección)."""
    ok:          bool                       # True si la segmentación fue razonable
    aspect:      float = 0.0                # ancho/alto del bounding box del sujeto
    solidity:    float = 0.0                # area / area_convex_hull (concavidades)
    extent:      float = 0.0                # area / area_bbox (qué tan lleno está)
    orientation: str   = "unknown"          # "horizontal" | "vertical" | "square"
    coverage:    float = 0.0                # fracción del frame ocupada por el sujeto
    note:        str   = ""


class MorphologyVisionDetector:
    """
    Extrae rasgos morfológicos APROXIMADOS de una imagen con OpenCV.

    Parameters
    ----------
    min_coverage : fracción mínima del frame que debe ocupar el sujeto para
                   confiar en la segmentación (si no, baja la confianza).
    max_coverage : por encima de esto se asume que el "sujeto" es casi todo el
                   frame (fondo mal separado) y se penaliza la confianza.
    base_conf    : confianza base de las observaciones cuando la silueta es clara.
    """

    def __init__(
        self,
        min_coverage: float = 0.05,
        max_coverage: float = 0.95,
        base_conf:    float = 0.40,
    ) -> None:
        self.min_coverage = min_coverage
        self.max_coverage = max_coverage
        self.base_conf    = base_conf

    # ── API principal ───────────────────────────────────────────────────────────

    def detect(self, image_bgr: np.ndarray) -> Dict[str, Observation]:
        """
        Imagen BGR → observaciones {feature_id: (state, conf)} para morphology.py.
        Devuelve dict vacío si no se pudo segmentar nada útil.
        """
        feats = self.analyze(image_bgr)
        if not feats.ok:
            return {}

        # La confianza de detección escala con qué tan limpia fue la segmentación.
        det_conf = self._detection_confidence(feats)
        obs: Dict[str, Observation] = {}

        # ── 1) PLAN CORPORAL desde proporción + orientación de la silueta ─────────
        # Cuadrúpedo masivo y alargado horizontal  → saurópodo/herbívoro
        # Bípedo erguido (más alto que ancho)        → terópodo/depredador
        # Proporción intermedia                      → ornitomimosaurio/generalista
        if feats.aspect >= 1.7 and feats.orientation != "vertical":
            state = "sauropod_long_neck_quadruped"
            conf  = min(0.50, 0.30 + (feats.aspect - 1.7) * 0.10)
        elif feats.aspect <= 0.95 or feats.orientation == "vertical":
            state = "large_theropod_predator"
            conf  = min(0.50, 0.30 + (0.95 - min(feats.aspect, 0.95)) * 0.20)
        else:
            state = "ornithomimosaur_generalist"
            conf  = 0.28
        obs["body_plan_extinct_mythic"] = (state, round(conf * det_conf, 3))

        # ── 2) EXTREMIDADES desde solidity (concavidades = patas largas/cuello) ───
        # Silueta muy cóncava (patas separadas, cuello)  → extremidades columnares
        #                                                   de cuadrúpedo herbívoro.
        # Silueta compacta y no alargada                 → pies corredores
        #                                                   tipo generalista.
        if feats.solidity < 0.65:
            state = "columnar_quadruped_herbivore_limbs"
            conf  = 0.32
        elif feats.solidity > 0.85 and feats.aspect <= 1.2:
            state = "three_toed_cursorial_generalist"
            conf  = 0.25
        else:
            state = "three_toed_cursorial_generalist"
            conf  = 0.20
        obs["limb_and_feet_type"] = (state, round(conf * det_conf, 3))

        return obs

    # ── segmentación + métricas ───────────────────────────────────────────────────

    def analyze(self, image_bgr: np.ndarray) -> VisionFeatures:
        """Segmenta el sujeto principal y calcula métricas geométricas crudas."""
        if image_bgr is None or image_bgr.size == 0:
            return VisionFeatures(ok=False, note="imagen vacía")

        h, w = image_bgr.shape[:2]
        frame_area = float(h * w)

        contour = self._main_contour(image_bgr)
        if contour is None:
            return VisionFeatures(ok=False, note="sin contorno utilizable")

        area = cv2.contourArea(contour)
        if area <= 0:
            return VisionFeatures(ok=False, note="contorno de área nula")

        x, y, bw, bh = cv2.boundingRect(contour)
        aspect = bw / float(bh) if bh > 0 else 0.0
        extent = area / float(bw * bh) if bw * bh > 0 else 0.0

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        coverage = area / frame_area

        # Orientación por PCA de los puntos del contorno (más robusto que el bbox
        # para siluetas inclinadas).
        orientation = self._orientation(contour)

        ok = (self.min_coverage <= coverage <= self.max_coverage)
        note = "" if ok else f"cobertura fuera de rango ({coverage:.2f})"

        return VisionFeatures(
            ok=ok, aspect=round(aspect, 3), solidity=round(solidity, 3),
            extent=round(extent, 3), orientation=orientation,
            coverage=round(coverage, 3), note=note,
        )

    def _main_contour(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Devuelve el contorno externo más grande del sujeto.

        Estrategia en dos pasos:
          1. Otsu en ambos sentidos (rápido). Sirve cuando hay buen contraste
             sujeto/fondo y el sujeto NO llena el frame.
          2. Si Otsu solo encuentra siluetas que ocupan casi todo el frame
             (típico en renders/recortes), recurre a GrabCut con un rectángulo
             central, que modela el color de primer plano vs fondo y separa
             mucho mejor esos casos.
        """
        otsu = self._otsu_contour(image_bgr)
        if otsu is not None:
            h, w = image_bgr.shape[:2]
            cov = cv2.contourArea(otsu) / float(h * w)
            if cov <= 0.85:        # silueta de tamaño "sano": Otsu basta
                return otsu

        grab = self._grabcut_contour(image_bgr)
        if grab is not None:
            return grab
        return otsu   # último recurso: lo que diera Otsu (aunque sea grande)

    def _otsu_contour(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Contorno grande mejor centrado vía umbral de Otsu (sujeto oscuro/claro)."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        h, w = gray.shape[:2]
        frame_area = float(h * w)
        cx0, cy0 = w / 2.0, h / 2.0

        best, best_score = None, -1.0
        for inv in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
            _, mask = cv2.threshold(gray, 0, 255, inv + cv2.THRESH_OTSU)
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2
            )
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                cov = cv2.contourArea(c) / frame_area
                if cov < self.min_coverage or cov > 0.99:
                    continue
                M = cv2.moments(c)
                if M["m00"] <= 0:
                    continue
                ccx, ccy = M["m10"] / M["m00"], M["m01"] / M["m00"]
                dist = np.hypot(ccx - cx0, ccy - cy0) / np.hypot(cx0, cy0)
                score = cov * (1.0 - 0.5 * dist)
                if score > best_score:
                    best_score, best = score, c
        return best

    def _grabcut_contour(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Segmentación foreground/background con GrabCut, inicializada con un
        rectángulo que deja un margen del frame como fondo probable. Más lento
        que Otsu pero separa sujetos que llenan casi todo el cuadro.
        """
        h, w = image_bgr.shape[:2]
        mx, my = int(w * 0.06), int(h * 0.06)
        rect = (mx, my, max(1, w - 2 * mx), max(1, h - 2 * my))
        mask = np.zeros((h, w), np.uint8)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(image_bgr, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        except Exception:
            return None
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        biggest = max(cnts, key=cv2.contourArea)
        cov = cv2.contourArea(biggest) / float(h * w)
        if cov < self.min_coverage or cov > 0.985:
            return None
        return biggest

    @staticmethod
    def _orientation(contour: np.ndarray) -> str:
        """Orientación del eje principal del contorno vía PCA."""
        pts = contour.reshape(-1, 2).astype(np.float32)
        if len(pts) < 5:
            return "unknown"
        mean = pts.mean(axis=0)
        cov = np.cov((pts - mean).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal = eigvecs[:, int(np.argmax(eigvals))]
        angle = abs(np.degrees(np.arctan2(principal[1], principal[0])))
        angle = min(angle, 180 - angle)   # plegar a [0, 90]
        if angle < 35:
            return "horizontal"
        if angle > 55:
            return "vertical"
        return "square"

    def _detection_confidence(self, feats: VisionFeatures) -> float:
        """
        Factor [0..1] que refleja qué tan confiable fue la segmentación. Siluetas
        que ocupan una fracción "sana" del frame valen más; coberturas extremas
        (fondo mal separado o sujeto diminuto) penalizan la confianza.
        """
        cov = feats.coverage
        if cov < self.min_coverage or cov > self.max_coverage:
            return 0.4
        # óptimo alrededor de 0.25–0.70 de cobertura
        if 0.25 <= cov <= 0.70:
            return 1.0
        if cov < 0.25:
            return 0.6 + (cov - self.min_coverage) / (0.25 - self.min_coverage) * 0.4
        return 0.6 + (self.max_coverage - cov) / (self.max_coverage - 0.70) * 0.4


# ── CLI de debug ───────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print("Uso: python morphology_vision.py <imagen>")
        sys.exit(1)

    path = sys.argv[1]
    img = cv2.imread(path)
    if img is None:
        print(f"No se pudo leer la imagen: {path}")
        sys.exit(1)

    det = MorphologyVisionDetector()
    feats = det.analyze(img)
    obs = det.detect(img)

    print(f"morphology_vision.py v{__version__}\n")
    print("Métricas de la silueta:")
    print(f"  segmentación ok : {feats.ok}  {('· ' + feats.note) if feats.note else ''}")
    print(f"  aspect (w/h)    : {feats.aspect}")
    print(f"  solidity        : {feats.solidity}")
    print(f"  extent          : {feats.extent}")
    print(f"  orientación     : {feats.orientation}")
    print(f"  cobertura       : {feats.coverage}")

    print("\nObservaciones para morphology.py:")
    if not obs:
        print("  (ninguna — segmentación insuficiente)")
    for fid, (state, conf) in obs.items():
        print(f"  {fid:<26} = {state}  (conf {conf})")


if __name__ == "__main__":
    main()
