"""
Servidor web para clasificar la dieta de un animal desde el navegador.

Levanta una página donde el usuario sube una foto, la ve en pantalla y, al
pulsar "Analizar", se ejecuta el DietClassifier (cascada ResNet-18 → ResNet-50)
y se muestra la clase resultante — carnívoro, herbívoro u omnívoro — en grande.

Uso:
    pip install -r webapp/requirements.txt      # flask (torch/opencv/numpy ya están)
    python webapp/app.py
    # abrir http://127.0.0.1:5000

El modelo se carga una sola vez al arrancar. Si faltan los pesos o torch,
la página sigue cargando pero /api/predict responde con un error claro
(clf.is_available == False), sin tumbar el servidor.
"""

from __future__ import annotations

__version__ = "1.0.0"  # 1.0 página de subida + preview + animación + resultado

import os
import sys

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

# Permitir importar classifier.py desde la raíz del proyecto (un nivel arriba).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from classifier import DietClassifier  # noqa: E402  (tras ajustar sys.path)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB por imagen

# El clasificador es pesado: se construye una sola vez al importar el módulo.
clf = DietClassifier()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    """Recibe una imagen (multipart 'image'), la clasifica y devuelve JSON."""
    if not clf.is_available:
        return jsonify(
            ok=False,
            error="El clasificador no está disponible. Verifica que torch y "
                  "torchvision estén instalados y que existan los pesos en "
                  "weights/diet_resnet18.pt y/o weights/diet_resnet50.pt.",
        ), 503

    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify(ok=False, error="No se recibió ninguna imagen."), 400

    raw = np.frombuffer(file.read(), dtype=np.uint8)
    img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify(ok=False, error="No se pudo leer la imagen (formato no válido)."), 400

    res = clf.classify_array(img_bgr)
    return jsonify(
        ok=True,
        label=res.label,                       # carnivore | herbivore | omnivore
        confidence=round(res.confidence, 4),   # 0..1
        source=res.source,                     # resnet18 | resnet50
    )


def main() -> None:
    print(f"webapp v{__version__}")
    print(f"Clasificador disponible: {clf.is_available}")
    if clf.is_available:
        print(f"  Dispositivo: {clf.device}")
    else:
        print("  AVISO: sin modelo cargado; la página abre pero /api/predict dará 503.")
    print("Servidor en http://127.0.0.1:5000  (Ctrl+C para detener)")
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
