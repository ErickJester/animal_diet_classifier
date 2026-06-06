# Animal Diet Classifier

Clasificador de **dieta animal** a partir de una foto: dice si el animal es
**carnívoro**, **herbívoro** u **omnívoro**.

Basado en la misma arquitectura del clasificador FST: CNN ResNet con cascada
de confianza (ResNet-18 primario rápido → ResNet-50 respaldo más preciso).

> Solo código. Aún **no hay modelos entrenados** — primero hay que armar el
> dataset y entrenar (ver abajo).

---

## Estructura

```
animal_diet_classifier/
├── classifier.py          # DietClassifier (inferencia con cascada ResNet-18/50)
├── train_classifier.py    # entrenamiento
├── predict.py             # CLI para clasificar una o varias fotos
├── requirements.txt
├── dataset/               # (vacío) imágenes etiquetadas por carpeta
│   ├── train/{carnivore,herbivore,omnivore}/
│   └── val/{carnivore,herbivore,omnivore}/
└── weights/               # (vacío) aquí se guardan los modelos entrenados
```

## Clases

`carnivore` (0) · `herbivore` (1) · `omnivore` (2)

---

## 1) Instalar dependencias

```bash
pip install -r requirements.txt
```

## 2) Preparar el dataset

Coloca tus imágenes en carpetas por clase. El nombre de la carpeta es la etiqueta:

```
dataset/
  train/
    carnivore/   leon01.jpg, tigre02.jpg, ...
    herbivore/   vaca01.jpg, conejo02.jpg, ...
    omnivore/    oso01.jpg, cerdo02.jpg, ...
  val/
    carnivore/   ...
    herbivore/   ...
    omnivore/    ...
```

Recomendado: ~80% de las imágenes en `train/` y ~20% en `val/`.

## 3) Entrenar

```bash
# Modelo primario (rápido)
python train_classifier.py --arch resnet18 --epochs 30

# Modelo de respaldo (más preciso)
python train_classifier.py --arch resnet50 --epochs 30
```

Genera `weights/diet_resnet18.pt` y `weights/diet_resnet50.pt`.

Opciones útiles: `--batch-size`, `--lr`, `--freeze-backbone`, `--device cpu|cuda|mps`.

## 4) Clasificar fotos

```bash
python predict.py mi_foto.jpg
python predict.py carpeta_de_fotos/
```

---

## Uso como librería

```python
from classifier import DietClassifier

clf = DietClassifier()
if clf.is_available:
    res = clf.classify_image("animal.jpg")
    print(res.label, res.confidence, res.source)
    # p.ej.  carnivore 0.91 resnet18
```

## Cómo funciona la cascada

1. **ResNet-18** clasifica la foto. Si la confianza ≥ `0.65` → devuelve ese resultado.
2. Si la confianza es menor (y existe el modelo de respaldo), se ejecuta **ResNet-50**
   y se devuelve su resultado.
3. Si solo existe ResNet-18, se devuelve su resultado aunque la confianza sea baja.

Si torch/torchvision no están instalados o faltan los pesos, `clf.is_available`
es `False` y nada lanza excepción al importar.
