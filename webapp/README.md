# Web — Clasificador de Dieta Animal

Página para usar el modelo desde el navegador: subes una foto (arrastrando,
eligiendo archivo o pegándola con **Ctrl+V**), la ves en pantalla, pulsas
**Analizar**, aparece una animación de escaneo y el resultado
sale en grande — **carnívoro**, **herbívoro** u **omnívoro** — con su confianza
y qué modelo de la cascada lo resolvió (ResNet-18 o ResNet-50).

## Requisitos

- El proyecto raíz ya instalado (`pip install -r requirements.txt`): torch,
  torchvision, opencv-python, numpy.
- Los pesos entrenados en `weights/diet_resnet18.pt` y/o `weights/diet_resnet50.pt`.
- Flask:

```bash
pip install -r webapp/requirements.txt
```

## Ejecutar

Desde la raíz del proyecto:

```bash
python webapp/app.py
```

Abre **http://127.0.0.1:5000** en el navegador.

> El modelo se carga una sola vez al arrancar. Si falta torch o los pesos, la
> página abre igual pero al analizar responde con un error claro (no se cae el
> servidor).

## Cómo funciona

```
navegador  ──(imagen multipart)──▶  POST /api/predict
                                       │
                                       ▼
                            DietClassifier.classify_array()
                            (cascada ResNet-18 → ResNet-50)
                                       │
           ◀──── { label, confidence, source } (JSON) ────┘
```

- `app.py` — servidor Flask: sirve la página y expone `/api/predict`.
- `templates/index.html` — estructura de la página.
- `static/style.css` — tema oscuro, animación de escaneo, resultado grande.
- `static/script.js` — subida/preview, animación y pintado del resultado.

La animación de "análisis de rasgos" (ojos, dentadura, garras…) es cosmética:
evoca el módulo morfológico del proyecto mientras el backend hace la inferencia
real con la CNN.
