"""
Clasificador CNN de dieta animal a partir de una foto.

Arquitectura en cascada de confianza (3 planes):
  Plan A · Primario  — ResNet-18  (rápido, liviano)
  Plan B · Respaldo  — ResNet-50  (más preciso, activa cuando ResNet-18 duda)
  Plan C · Morfología — visión por computadora (OpenCV) que detecta rasgos de la
           silueta y los puntúa con la base de conocimiento diet_morphology.json.
           Se activa cuando AMBAS CNN quedan por debajo del umbral y la clase no
           es "other". Es un respaldo aproximado: si los rasgos son ambiguos
           devuelve "uncertain" y se conserva el mejor resultado de la CNN.

Clases: carnivore (0) | herbivore (1) | omnivore (2) | other (3 = no es un animal)
(El Plan C solo distingue carnivore/herbivore/omnivore; nunca produce "other".)

Uso:
    clf = DietClassifier()
    if clf.is_available:
        result = clf.classify_image("foto_animal.jpg")
        print(result.label, result.confidence, result.source)

Si torch/torchvision no están instalados, o si los pesos no existen,
`is_available` sigue siendo True mientras el Plan C (morfología) esté disponible;
solo es False si no hay NINGÚN plan utilizable. El módulo no lanza excepción al
importarse.
"""

from __future__ import annotations

__version__ = "1.1.0"  # 1.0 cascada ResNet-18/50 · 1.1 Plan C morfología (cv2 + diet_morphology.json)

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np

# ── importación opcional de torch ─────────────────────────────────────────────
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torchvision.models as tv_models
    _TORCH_AVAILABLE = True
except ImportError:
    pass

# ── importación opcional del Plan C (morfología) ──────────────────────────────
_MORPH_AVAILABLE = False
try:
    from morphology import MorphologyDietScorer
    from morphology_vision import MorphologyVisionDetector
    _MORPH_AVAILABLE = True
except ImportError:
    pass

# ── constantes ────────────────────────────────────────────────────────────────
CLASSES       = ["carnivore", "herbivore", "omnivore", "other"]
NUM_CLASSES   = len(CLASSES)
INPUT_SIZE    = 224          # estándar ResNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

PRIMARY_WEIGHTS  = os.path.join("weights", "diet_resnet18.pt")
FALLBACK_WEIGHTS = os.path.join("weights", "diet_resnet50.pt")
MORPHOLOGY_KB    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "diet_morphology.json")

PRIMARY_CONF_THR    = 0.65   # confianza mínima de ResNet-18; si < → activa ResNet-50
FALLBACK_CONF_THR   = 0.40   # confianza mínima de ResNet-50; si < → usa igual el mejor
MORPHOLOGY_CONF_THR = 0.45   # si el mejor de la CNN < esto (y no es "other") → Plan C
MORPHOLOGY_DECIDE_THR = 0.55  # el Plan C solo "gana" si decide con al menos esta confianza


@dataclass
class ClassifierResult:
    label:      str
    confidence: float
    source:     str   # "resnet18" | "resnet50" | "morphology" | "no_image"
    # detalle del Plan C cuando source == "morphology" (o cuando se intentó):
    morphology: Optional[Dict] = None


class DietClassifier:
    """
    Clasificador de dieta animal en cascada: ResNet-18 → ResNet-50 → morfología (cv2).

    Parameters
    ----------
    primary_weights     : ruta al state_dict de ResNet-18 (diet_resnet18.pt)
    fallback_weights    : ruta al state_dict de ResNet-50 (diet_resnet50.pt)
    primary_conf_thr    : umbral de confianza para activar el respaldo ResNet-50
    fallback_conf_thr   : umbral mínimo de confianza del respaldo (informativo)
    morphology_conf_thr : si el mejor resultado de la CNN cae por debajo, se intenta
                          el Plan C (morfología). Poner 0 lo desactiva.
    enable_morphology   : activa/desactiva globalmente el Plan C
    device              : "" → autodetectar; "cpu" | "cuda" | "mps"
    input_size          : tamaño de la imagen redimensionada (cuadrada)
    """

    def __init__(
        self,
        primary_weights:     str   = PRIMARY_WEIGHTS,
        fallback_weights:    str   = FALLBACK_WEIGHTS,
        primary_conf_thr:    float = PRIMARY_CONF_THR,
        fallback_conf_thr:   float = FALLBACK_CONF_THR,
        morphology_conf_thr: float = MORPHOLOGY_CONF_THR,
        enable_morphology:   bool  = True,
        device:              str   = "",
        input_size:          int   = INPUT_SIZE,
    ) -> None:
        self.primary_weights     = primary_weights
        self.fallback_weights    = fallback_weights
        self.primary_conf_thr    = primary_conf_thr
        self.fallback_conf_thr   = fallback_conf_thr
        self.morphology_conf_thr = morphology_conf_thr
        self.input_size          = input_size
        self.is_available        = False
        self.device              = None   # se fija abajo si torch está disponible

        self._primary  = None
        self._fallback = None
        self._scorer   = None    # MorphologyDietScorer (Plan C)
        self._detector = None    # MorphologyVisionDetector (Plan C)

        # ── CNN (Planes A y B) ────────────────────────────────────────────────
        if _TORCH_AVAILABLE:
            if device:
                self.device = torch.device(device)
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

            self._primary  = self._load_model(primary_weights,  "resnet18")
            self._fallback = self._load_model(fallback_weights, "resnet50")

        # ── Morfología (Plan C) ───────────────────────────────────────────────
        if enable_morphology and _MORPH_AVAILABLE and os.path.isfile(MORPHOLOGY_KB):
            try:
                self._scorer   = MorphologyDietScorer(MORPHOLOGY_KB)
                self._detector = MorphologyVisionDetector()
            except Exception as exc:
                print(f"[DietClassifier] Plan C deshabilitado: {exc}")
                self._scorer = self._detector = None

        # disponible si hay al menos un plan utilizable
        self.is_available = bool(
            self._primary or self._fallback or (self._scorer and self._detector)
        )

    # ── propiedades de conveniencia ───────────────────────────────────────────

    @property
    def has_cnn(self) -> bool:
        return self._primary is not None or self._fallback is not None

    @property
    def has_morphology(self) -> bool:
        return self._scorer is not None and self._detector is not None

    # ── construcción y carga ──────────────────────────────────────────────────

    def _build_resnet(self, arch: str) -> "nn.Module":
        """Construye un ResNet con cabeza de NUM_CLASSES (4) clases."""
        if arch == "resnet18":
            model = tv_models.resnet18(weights=None)
            in_features = 512
        elif arch == "resnet50":
            model = tv_models.resnet50(weights=None)
            in_features = 2048
        else:
            raise ValueError(f"Arquitectura no soportada: {arch}")
        model.fc = nn.Linear(in_features, NUM_CLASSES)
        return model

    def _load_model(self, weights_path: str, arch: str) -> "Optional[nn.Module]":
        """Carga pesos desde disco. Retorna None si el archivo no existe."""
        if not os.path.isfile(weights_path):
            return None
        try:
            model = self._build_resnet(arch)
            # weights_only=True es seguro para state_dicts (lo que guarda el train);
            # fallback para torch antiguos o checkpoints con objetos no permitidos.
            try:
                state = torch.load(weights_path, map_location=self.device, weights_only=True)
            except (TypeError, Exception):
                state = torch.load(weights_path, map_location=self.device)
            # acepta tanto state_dict directo como checkpoint con clave "model"
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()
            return model
        except Exception as exc:
            print(f"[DietClassifier] Error cargando {weights_path}: {exc}")
            return None

    # ── preprocesamiento ──────────────────────────────────────────────────────

    def preprocess(self, image_bgr: np.ndarray) -> "torch.Tensor":
        """
        BGR numpy → tensor normalizado listo para inferencia.

        Pasos:
          1. BGR → RGB
          2. padding a cuadrado (zero-pad centrado)
          3. resize a (input_size, input_size)
          4. float32, escala [0,1]
          5. normalización ImageNet
          6. batch dim + move to device
        """
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        h, w = rgb.shape[:2]
        side = max(h, w)
        pad_top    = (side - h) // 2
        pad_bottom = side - h - pad_top
        pad_left   = (side - w) // 2
        pad_right  = side - w - pad_left
        sq = cv2.copyMakeBorder(
            rgb, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        resized = cv2.resize(sq, (self.input_size, self.input_size),
                             interpolation=cv2.INTER_LINEAR)

        arr = resized.astype(np.float32) / 255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std  = np.array(IMAGENET_STD,  dtype=np.float32)
        arr  = (arr - mean) / std

        # HWC → CHW → batch
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device)

    # ── inferencia ────────────────────────────────────────────────────────────

    def _infer(self, model: "nn.Module", tensor: "torch.Tensor") -> Tuple[str, float]:
        """Ejecuta un modelo y retorna (label, confidence)."""
        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]
        idx        = int(probs.argmax().item())
        confidence = float(probs[idx].item())
        label      = CLASSES[idx]
        return label, confidence

    def _run_plan_c(self, image_bgr: np.ndarray) -> Optional[ClassifierResult]:
        """
        Plan C — detección morfológica con OpenCV + diet_morphology.json.

        Mira la silueta del animal, deduce rasgos de plan corporal / extremidades
        (aproximados, baja confianza) y los puntúa con el scorer. Devuelve un
        ClassifierResult solo si alcanza una decisión por encima del umbral; si
        los rasgos son ambiguos ("uncertain") o no se segmenta nada, devuelve None
        para que el llamador conserve el mejor resultado de la CNN.
        """
        if not self.has_morphology:
            return None

        obs = self._detector.detect(image_bgr)
        if not obs:
            return None

        res = self._scorer.score(obs)
        detail = {
            "label":       res.label,
            "confidence":  round(res.confidence, 3),
            "scores":      {k: round(v, 3) for k, v in res.scores.items()},
            "n_features":  res.n_features,
            "used":        res.used,
            "observations": {k: list(v) for k, v in obs.items()},
        }

        if res.label != "uncertain" and res.confidence >= MORPHOLOGY_DECIDE_THR:
            return ClassifierResult(res.label, res.confidence, "morphology", detail)

        # decisión débil o incierta: no impone resultado, pero deja el detalle
        return ClassifierResult(res.label, res.confidence, "morphology_uncertain", detail)

    def classify_array(self, image_bgr: np.ndarray) -> ClassifierResult:
        """
        Clasifica una imagen BGR (numpy) en cascada de 3 planes.

        1. ResNet-18 → si confianza ≥ primary_conf_thr → retorna (Plan A).
        2. ResNet-50 (si disponible) → se toma el MEJOR de las dos CNN (Plan B).
        3. Si el mejor de la CNN < morphology_conf_thr y la clase no es "other",
           se intenta el Plan C (morfología). Si el Plan C decide con confianza
           suficiente, su resultado gana; si no, se conserva el de la CNN.
        4. Sin CNN disponibles, se usa directamente el Plan C.
        """
        if not self.is_available or image_bgr is None or image_bgr.size == 0:
            return ClassifierResult("unknown", 0.0, "no_image")

        # ── Sin CNN: el Plan C es el único disponible ─────────────────────────
        if not self.has_cnn:
            plan_c = self._run_plan_c(image_bgr)
            if plan_c is not None and plan_c.source == "morphology":
                return plan_c
            # incierto o nulo
            if plan_c is not None:
                return plan_c
            return ClassifierResult("uncertain", 0.0, "morphology")

        tensor = self.preprocess(image_bgr)

        # ── Plan A: ResNet-18 ─────────────────────────────────────────────────
        best_label, best_conf, best_src = "unknown", 0.0, "no_image"
        if self._primary is not None:
            label18, conf18 = self._infer(self._primary, tensor)
            best_label, best_conf, best_src = label18, conf18, "resnet18"
            if conf18 >= self.primary_conf_thr:
                return ClassifierResult(label18, conf18, "resnet18")

        # ── Plan B: ResNet-50 ─────────────────────────────────────────────────
        if self._fallback is not None:
            label50, conf50 = self._infer(self._fallback, tensor)
            if conf50 >= best_conf:
                best_label, best_conf, best_src = label50, conf50, "resnet50"

        cnn_result = ClassifierResult(best_label, best_conf, best_src)

        # ── Plan C: morfología (solo si la CNN duda y no dijo "other") ────────
        plan_c_enabled = (
            self.morphology_conf_thr > 0
            and self.has_morphology
            and best_conf < self.morphology_conf_thr
            and best_label != "other"
        )
        if plan_c_enabled:
            plan_c = self._run_plan_c(image_bgr)
            if plan_c is not None:
                # adjunta el detalle del Plan C al resultado de la CNN
                cnn_result.morphology = plan_c.morphology
                if plan_c.source == "morphology":
                    return plan_c   # el Plan C decidió con confianza suficiente

        return cnn_result

    def classify_image(self, image: Union[str, np.ndarray]) -> ClassifierResult:
        """
        Clasifica desde una ruta de archivo o un array BGR.

        `image` puede ser:
          - str  → ruta a una imagen en disco (se carga con cv2)
          - np.ndarray → imagen BGR ya cargada
        Retorna ClassifierResult con source="no_image" si no se puede leer.
        """
        if isinstance(image, str):
            if not os.path.isfile(image):
                return ClassifierResult("unknown", 0.0, "no_image")
            img = cv2.imread(image)
            if img is None:
                return ClassifierResult("unknown", 0.0, "no_image")
        else:
            img = image

        return self.classify_array(img)
