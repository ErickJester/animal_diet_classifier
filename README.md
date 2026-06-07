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
├── curate.py              # CLI del curador de dataset (index/curate/status/commit)
├── curator/               # paquete del curador (ver "Curador de dataset" abajo)
│   ├── config.py          #   umbrales, rutas y constantes
│   ├── embeddings.py      #   extractor de embeddings visuales (ResNet feature)
│   ├── index.py           #   índice de similitud (coseno, persistente)
│   ├── inaturalist.py     #   identificación de especie vía API (opcional)
│   ├── registry.py        #   registro de especies + etiquetas de dieta
│   ├── curator.py         #   orquestador de la cascada (opción C)
│   └── data/              #   diet_labels.json + índice/registro generados
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

---

## Curador de dataset

Herramienta para **armar y mantener el dataset** sin acumular duplicados ni
fotos de más de una misma especie. Pensado para alimentar el dataset con
imágenes nuevas (p. ej. descargadas de iNaturalist) filtrando automáticamente
lo que ya tenemos.

### Cómo decide (cascada "opción C")

Para cada foto candidata:

1. **Etapa 1 — similitud visual** (rápida, offline). Calcula un *embedding* con
   un ResNet preentrenado y lo compara con todo lo que ya está en el índice.
   - similitud ≥ `0.95` → **descarta** (casi idéntica a una que ya tenemos).
   - similitud < `0.80` → casi seguro especie nueva → pasa a la Etapa 2.
2. **Etapa 2 — identificación de especie** (vía iNaturalist, solo para las que
   sobreviven a la Etapa 1).
   - especie con cupo lleno (`60` fotos) → **descarta**.
   - especie con cupo libre → **acepta**.

Al aceptar, la imagen se **copia a `curator/staging/`** (no toca el dataset
directamente): por dieta si se conoce, o a `_pending_review/` (especie sin
dieta asignada) / `_unidentified/` (sin especie). Tú revisas y confirmas con
`commit`.

> La dieta nunca se adivina: sale de `curator/data/diet_labels.json` (editable a
> mano). Si una especie no está ahí, su foto queda en `_pending_review/` hasta
> que le asignes la dieta.

### Uso

```bash
# 1) Indexar lo que ya tienes en dataset/ (hazlo una vez al principio)
python curate.py index

# 2) Curar imágenes nuevas (archivos o carpetas; recursivo)
python curate.py curate descargas_inaturalist/

# 3) Ver estado (índice, especies, pendientes)
python curate.py status

# 4) Mover lo aceptado (con dieta) al dataset, repartido train/val
python curate.py commit
```

### Identificación de especie (opcional)

La Etapa 2 usa la **API de visión de iNaturalist**, que requiere un token
gratuito. Sin token, el curador degrada con elegancia y decide **solo por
similitud visual**.

1. Inicia sesión en iNaturalist y obtén tu token: <https://www.inaturalist.org/users/api_token>
2. Pásalo por variable de entorno o por flag:

```bash
# PowerShell
$env:INATURALIST_API_TOKEN = "tu_token"
python curate.py curate fotos/

# o directo
python curate.py curate fotos/ --inat-token "tu_token"
```

### Ajustes

Todos los umbrales (similitud de duplicado, cupo por especie, ratio train/val,
backbone de embeddings) están en un único sitio: `curator/config.py`.
