# Animal Diet Classifier

Clasificador de **dieta animal** a partir de una foto: dice si el animal es
**carnívoro**, **herbívoro** u **omnívoro**.

Basado en la misma arquitectura del clasificador FST: CNN ResNet con cascada
de confianza (ResNet-18 primario rápido → ResNet-50 respaldo más preciso).

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

Descarga `animals90` y `mammals45` desde Kaggle (~20k imágenes) a `downloads/`.
Las fuentes `animals10`, `roboflow-ch` y `hf-big-animals` están deshabilitadas
por defecto en `curator/sources.py` (ver comentarios para activarlas).

**Fase B — suplemento verificado desde iNaturalist** (sin token, API pública):

```bash
python download.py more --per-species 300 --insecure
```

Baja fotos adicionales por especie con identificación verificada por la comunidad.
El flag `--insecure` solo es necesario en redes con proxy corporativo.

> Las imágenes quedan en `downloads/carnivore/`, `downloads/herbivore/`,
> `downloads/omnivore/` con formato `Genus_species_<photoid>.jpg`.

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
> La identificación de especie (Etapa 2) se desactiva automáticamente y
> `TARGET_PER_SPECIES` no tiene efecto. Para obtener un token gratuito:
> https://www.inaturalist.org/users/api_token

**Alternativa sin curar:** si quieres saltar la deduplicación, usa `prepare.py`
directamente (ver paso 4).

---

### 4. Preparar el dataset (split train/val)

Si usaste `curate.py commit`, el dataset ya está listo en `dataset/`.

Si saltaste la curación, `prepare.py` organiza las imágenes de `downloads/`
directamente — infiere la dieta desde el nombre de archivo (`Genus_species_...`)
y hace el split 80/20:

```bash
python prepare.py downloads/carnivore downloads/herbivore downloads/omnivore
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

---

### 6. Evaluar con cross-validation (opcional)

Una vez entrenado el modelo, evalúa su generalización con k-fold estratificado.
Cada fold arranca desde los pesos ya ajustados (no desde ImageNet), por lo que
solo necesita 10 épocas para converger:

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
│   ├── config.py           #   umbrales, rutas y constantes
│   ├── embeddings.py       #   extractor de embeddings visuales (ResNet feature)
│   ├── index.py            #   índice de similitud (coseno, persistente)
│   ├── inaturalist.py      #   identificación de especie vía API (requiere token)
│   ├── registry.py         #   registro de especies + etiquetas de dieta
│   ├── curator.py          #   orquestador de la cascada (opción C)
│   └── data/               #   diet_labels.json + índice/registro generados
├── requirements.txt
├── dataset/                # imágenes etiquetadas por carpeta (generado)
│   ├── train/{carnivore,herbivore,omnivore}/
│   └── val/{carnivore,herbivore,omnivore}/
├── downloads/              # imágenes descargadas por dieta (generado)
│   ├── carnivore/
│   ├── herbivore/
│   └── omnivore/
└── weights/                # modelos entrenados (generado)
    ├── diet_resnet18.pt
    ├── diet_resnet50.pt
    └── diet_resnet18_cv.pt  # mejor fold (si se usa --save-best)
```

---

## Clases

`carnivore` (0) · `herbivore` (1) · `omnivore` (2)

---

## Cómo funciona la cascada

1. **ResNet-18** clasifica la foto. Si la confianza ≥ `0.65` → devuelve ese resultado.
2. Si la confianza es menor (y existe el modelo de respaldo), se ejecuta **ResNet-50**
   y se devuelve su resultado.
3. Si solo existe ResNet-18, se devuelve su resultado aunque la confianza sea baja.

Si torch/torchvision no están instalados o faltan los pesos, `clf.is_available`
es `False` y nada lanza excepción al importar.

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

---

## Fuentes de datos

| Fuente | Tipo | Estado |
|---|---|---|
| `animals90` (Kaggle) | masiva, 90 especies | habilitada |
| `mammals45` (Kaggle) | masiva, 45 mamíferos | habilitada |
| iNaturalist | verificada por especie, API pública | habilitada |
| `animals10` (Kaggle) | masiva, 28k imgs en italiano | deshabilitada |
| `roboflow-ch` | ya etiquetada por dieta | deshabilitada (falta API key) |
| `hf-big-animals` (HuggingFace) | dataset grande variado | deshabilitada |

Para activar una fuente deshabilitada: editar `enabled=True` en `curator/sources.py`.
