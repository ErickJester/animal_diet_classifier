"""
Extractor de embeddings visuales.

Usa un ResNet de torchvision preentrenado en ImageNet, sin la capa final de
clasificación, como extractor de características. Cada imagen se convierte en un
vector (2048-d para ResNet-50) que captura su contenido visual: imágenes
parecidas producen vectores cercanos, lo que permite detectar duplicados o
fotos casi idénticas.

Mismo esquema de preprocesamiento que classifier.preprocess (pad a cuadrado +
normalización ImageNet) para mantener coherencia en todo el proyecto.

Si torch/torchvision no están instalados, `is_available` es False.
"""

from __future__ import annotations

__version__ = "1.0.0"  # 1.0 extractor ResNet sin capa final (ImageNet)

import cv2
import numpy as np

from . import config

# ── importación opcional de torch ─────────────────────────────────────────────
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torchvision.models as tv_models
    _TORCH_AVAILABLE = True
except ImportError:
    pass


class EmbeddingModel:
    """ResNet preentrenado (sin cabeza fc) como extractor de embeddings L2-normalizados."""

    def __init__(self, device: str = "") -> None:
        self.is_available = False
        self.input_size   = config.INPUT_SIZE

        if not _TORCH_AVAILABLE:
            return

        # dispositivo (misma lógica de autodetección que classifier.py)
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # backbone preentrenado, sin cabeza de clasificación
        if config.EMBED_ARCH == "resnet50":
            model = tv_models.resnet50(weights="IMAGENET1K_V2")
        else:
            model = tv_models.resnet18(weights="IMAGENET1K_V1")
        model.fc = nn.Identity()          # la salida pasa a ser el vector de features
        model.to(self.device).eval()

        self._model       = model
        self.is_available = True

    # ── preprocesamiento (idéntico esquema que classifier.preprocess) ──────────
    def _preprocess(self, image_bgr: np.ndarray) -> "torch.Tensor":
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        h, w = rgb.shape[:2]
        side       = max(h, w)
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

        arr  = resized.astype(np.float32) / 255.0
        arr  = (arr - np.array(config.IMAGENET_MEAN, np.float32)) \
               / np.array(config.IMAGENET_STD, np.float32)

        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device)

    # ── inferencia ─────────────────────────────────────────────────────────────
    def embed(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        BGR numpy → vector de features normalizado L2 (np.float32, EMBED_DIM).

        Normalizar a norma 1 permite usar el producto punto como similitud coseno.
        """
        tensor = self._preprocess(image_bgr)
        with torch.no_grad():
            feat = self._model(tensor)[0]
        vec  = feat.cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec
