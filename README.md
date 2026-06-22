# Animal Diet Classifier

Clasificador de **dieta animal** a partir de una foto: dice si el animal es
**carnívoro**, **herbívoro** u **omnívoro** — y rechaza imágenes que no contienen un animal.

Arquitectura CNN ResNet con cascada de confianza:
ResNet-18 (primario, rápido) → ResNet-50 (respaldo, más preciso) → morfología (último recurso).

---

## Clases del modelo

| Clase | Significado |
|---|---|
| `carnivore` | Come principalmente carne |
| `herbivore` | Come principalmente vegetales |
| `omnivore` | Come de ambos tipos |
| `other` | La imagen **no** contiene un animal |

La clase `other` es la clase negativa: se entrena con imágenes de escenas, comida
y objetos (~10k imágenes, balanceada con las clases animales) para que el modelo
rechace entradas que no debería clasificar como dieta.

---

## Cómo funciona la cascada

1. **ResNet-18** clasifica la foto. Si la confianza ≥ `0.65` → devuelve ese resultado.
2. Si la confianza es menor y existe el modelo de respaldo → se ejecuta **ResNet-50**.
3. Si ambos modelos tienen baja confianza (o no están disponibles), el módulo de
   **morfología** analiza rasgos físicos (posición de ojos, dentadura, extremidades).

La webapp muestra la nota "llegamos a este resultado analizando las características
físicas del animal" solo cuando se activa el módulo de morfología.

---

## Secuencia completa (de cero a modelo listo)

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

### 2. Descargar imágenes

**Fase A — datasets masivos desde Kaggle** (requiere `~/.kaggle/kaggle.json`):

```bash
python download.py datasets
```

Descarga `animals90` y `mammals45` desde Kaggle (~20k imágenes animales) a `downloads/`.

Para descargar también las fuentes no-animales (`other`):

```bash
python download.py datasets --only other-scenes other-food
```

Las imágenes quedan en `downloads/other/` (ya etiquetadas, no pasan por el registro de especies).

Las fuentes `animals10`, `roboflow-ch` y `hf-big-animals` están deshabilitadas
por defecto en `curator/sources.py` (ver comentarios para activarlas).

**Fase B — suplemento verificado desde iNaturalist** (sin token, API pública):

```bash
python download.py more --per-species 300 --insecure
```

Baja fotos adicionales por especie con identificación verificada por la comunidad.
El flag `--insecure` solo es necesario en redes con proxy corporativo.

> Las imágenes animales quedan en `downloads/carnivore/`, `downloads/herbivore/`,
> `downloads/omnivore/` con formato `Genus_species_<photoid>.jpg`.
> Las imágenes no-animales quedan en `downloads/other/`.

---

### 3. Curar el dataset (deduplicación)

El curador elimina fotos casi idénticas usando similitud visual (embeddings ResNet).
**No borra archivos**: los originales en `downloads/` quedan intactos siempre.

```bash
# Indexar el dataset existente (la primera vez estará vacío, es normal)
python curate.py index

# Curar: copia las imágenes aceptadas a curator/staging/
python curate.py curate downloads/

# Mover lo aceptado a dataset/train/ y dataset/val/ (split 80/20)
python curate.py commit
```

**Umbrales relevantes en `curator/config.py`:**

| Parámetro | Valor | Descripción |
|---|---|---|
| `DUPLICATE_SIM` | 0.95 | Similitud >= descarta (casi idéntica) |
| `NEW_SPECIES_SIM` | 0.95 | Similitud < acepta (variedad legítima) |
| `TARGET_PER_SPECIES` | 175 | Máx. fotos por especie (solo con token iNaturalist) |

Con ambos umbrales en 0.95 el curador actúa como **quita-duplicados puro**:
solo descarta fotos prácticamente calcadas y conserva toda la variedad.

> **Sin token de iNaturalist:** el curador decide solo por similitud visual.
> Para obtener un token gratuito: https://www.inaturalist.org/users/api_token

**Alternativa sin curar:** si quieres saltar la deduplicación, usa `prepare.py`
directamente (ver paso 4).

---

### 4. Preparar el dataset (split train/val)

Si usaste `curate.py commit`, el dataset ya está listo en `dataset/`.

Si saltaste la curación, `prepare.py` organiza las imágenes de `downloads/`
directamente — infiere la dieta desde el nombre de archivo (`Genus_species_...`)
o desde la carpeta contenedora (para `downloads/other/`) y hace el split 80/20:

```bash
python prepare.py downloads/carnivore downloads/herbivore downloads/omnivore downloads/other
```

Es idempotente: re-ejecutarlo no duplica imágenes ya presentes en `dataset/`.

---

### 5. Entrenar

```bash
# Modelo primario (rápido)
python train_classifier.py --arch resnet18 --epochs 30

# Modelo de respaldo (más preciso)
python train_classifier.py --arch resnet50 --epochs 30
```

Genera `weights/diet_resnet18.pt` y `weights/diet_resnet50.pt`.

Opciones útiles: `--batch-size`, `--lr`, `--freeze-backbone`, `--device cpu|cuda|mps`.

**Para entrenar en Google Colab** (GPU gratis/Pro):
usa el notebook en `colab/train_diet_classifier.ipynb` — es autónomo, no requiere
clonar el repositorio; sube `dataset.zip` a tu Drive y listo.

---

### 6. Evaluar con cross-validation (opcional)

```bash
# 5 folds, 10 épocas por fold (default)
python crossval_classifier.py --arch resnet18

# Guardando el mejor fold
python crossval_classifier.py --arch resnet18 --save-best

# Cambiando folds y épocas
python crossval_classifier.py --arch resnet50 --folds 10 --epochs 15
```

Reporta accuracy por fold, media ± desviación estándar y matriz de confusión
agregada de todos los folds.

---

### 7. Clasificar fotos

**Interfaz web:**

```bash
cd webapp
python app.py
```

Abre `http://localhost:5000` — sube una foto con drag-drop, selector de archivo
o pegar desde el portapapeles (Ctrl+V) y obtén el resultado con animación de análisis.

**CLI:**

```bash
python predict.py mi_foto.jpg
python predict.py carpeta_de_fotos/
```

---

## Estructura

```
animal_diet_classifier/
├── classifier.py           # DietClassifier (inferencia con cascada ResNet-18/50)
├── train_classifier.py     # entrenamiento (fine-tuning desde ImageNet)
├── crossval_classifier.py  # evaluación k-fold desde pesos ya ajustados
├── predict.py              # CLI para clasificar una o varias fotos
├── curate.py               # CLI del curador (index/curate/status/commit)
├── prepare.py              # organiza downloads/ → dataset/train/ y dataset/val/
├── download.py             # descarga datasets masivos + suplemento iNaturalist
├── curator/                # paquete del curador
│   ├── config.py           #   umbrales, rutas y constantes (DIET_CLASSES aquí)
│   ├── embeddings.py       #   extractor de embeddings visuales (ResNet feature)
│   ├── index.py            #   índice de similitud (coseno, persistente)
│   ├── inaturalist.py      #   identificación de especie vía API
│   ├── registry.py         #   registro de especies + etiquetas de dieta
│   ├── sources.py          #   declaración de fuentes (Kaggle, HF, iNaturalist)
│   ├── curator.py          #   orquestador de la cascada
│   └── data/               #   diet_labels.json + índice/registro generados
├── colab/
│   └── train_diet_classifier.ipynb  # notebook autónomo para entrenar en Colab
├── webapp/
│   ├── app.py              # servidor Flask (API /api/predict + página web)
│   ├── templates/index.html
│   └── static/
│       ├── script.js       # lógica del cliente (upload, animación, resultado)
│       └── style.css       # tema oscuro con animación de escaneo
├── requirements.txt
├── dataset/                # imágenes etiquetadas por carpeta (generado)
│   ├── train/{carnivore,herbivore,omnivore,other}/
│   └── val/{carnivore,herbivore,omnivore,other}/
├── downloads/              # imágenes descargadas (generado)
│   ├── carnivore/
│   ├── herbivore/
│   ├── omnivore/
│   └── other/
└── weights/                # modelos entrenados (generado)
    ├── diet_resnet18.pt
    ├── diet_resnet50.pt
    └── diet_resnet18_cv.pt  # mejor fold (si se usa --save-best)
```

---

## Fuentes de datos

| Fuente | Tipo | Clase | Estado |
|---|---|---|---|
| `animals90` (Kaggle) | masiva, 90 especies | animales | habilitada |
| `mammals45` (Kaggle) | masiva, 45 mamíferos | animales | habilitada |
| iNaturalist | verificada por especie, API pública | animales | habilitada |
| `other-scenes` (Kaggle: Intel) | paisajes, ciudades, glaciares | other | habilitada |
| `other-food` (Kaggle: Food-101) | comida y platos | other | habilitada |
| `other-objects` | objetos/personas | other | deshabilitada (verifica slug) |
| `animals10` (Kaggle) | masiva, 28k imgs en italiano | animales | deshabilitada |
| `roboflow-ch` | ya etiquetada por dieta | animales | deshabilitada (falta API key) |
| `hf-big-animals` (HuggingFace) | dataset grande variado | animales | deshabilitada |

Para activar una fuente deshabilitada: editar `enabled=True` en `curator/sources.py`.

Las fuentes `other-*` tienen un tope `max_images` (5500 + 4500 = ~10k) para
mantener la clase `other` balanceada con las clases animales (~10k cada una).

---

## Uso como librería

```python
from classifier import DietClassifier

clf = DietClassifier()
if clf.is_available:
    res = clf.classify_image("animal.jpg")
    print(res.label, res.confidence, res.source)
    # p.ej.  carnivore 0.91 resnet18
    # p.ej.  other 0.97 resnet50   ← imagen sin animal
```

Si torch/torchvision no están instalados o faltan los pesos, `clf.is_available`
es `False` y nada lanza excepción al importar.
