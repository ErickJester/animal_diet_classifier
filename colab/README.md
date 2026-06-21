# Entrenamiento en Google Colab Pro+

Notebook para entrenar los modelos de la cascada (ResNet-18 + ResNet-50) usando
la GPU de Colab Pro+ (idealmente A100), con precisión mixta (AMP) y carga de
datos en paralelo.

## Archivo

- [`train_diet_classifier.ipynb`](train_diet_classifier.ipynb) — notebook completo, de principio a fin.

## Cómo usarlo

1. **Sube el notebook a Colab**: <https://colab.research.google.com> → `Archivo → Subir notebook` → elige `train_diet_classifier.ipynb`.
2. **Activa la GPU**: `Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU` (elige A100 si está disponible).
3. **Prepara el dataset una sola vez** — en tu PC, dentro de `D:\animal_diet_classifier`:
   ```powershell
   Compress-Archive -Path dataset -DestinationPath dataset.zip
   ```
   Sube `dataset.zip` a tu Drive en `MyDrive/animal_diet_classifier/`.
4. **Ejecuta las celdas en orden.** El notebook:
   - verifica la GPU,
   - monta Drive,
   - clona este repo (si es privado, usa una URL con token — ver la celda 3),
   - descomprime el dataset al disco local de Colab,
   - entrena ResNet-18 y luego ResNet-50,
   - copia los pesos `.pt` de vuelta a tu Drive.

## Por qué es más rápido que en local

| Optimización | Qué hace |
|---|---|
| `--amp` | precisión mixta FP16/FP32 → hasta ~2× más rápido en Tensor Cores (A100/V100/L4) |
| `--num-workers 4` | carga imágenes en paralelo, la GPU no espera al disco |
| `--batch-size` grande | aprovecha la VRAM de la GPU (256–512 en A100) |
| dataset en SSD local | descomprimir a `/content` evita leer de Drive en cada época |

Estas opciones (`--amp`, `--num-workers`) se añadieron a `train_classifier.py` y
también sirven en local si tienes GPU CUDA.

## Salida

Los pesos quedan en `MyDrive/animal_diet_classifier/weights/`:
- `diet_resnet18.pt` (primario)
- `diet_resnet50.pt` (respaldo)

Descárgalos a tu `weights/` local para usar `predict.py` / `DietClassifier`.
