"""
Predicción de dieta animal sobre una o varias fotos.

Uso:
    python predict.py foto1.jpg
    python predict.py carpeta_con_fotos/
    python predict.py foto1.jpg foto2.png --device cpu

Carga DietClassifier (ResNet-18 primario, ResNet-50 respaldo) y reporta
la clase predicha (carnivore | herbivore | omnivore | other) con su confianza.
"other" significa que la imagen no parece contener un animal.

Requiere pesos entrenados en weights/diet_resnet18.pt y/o weights/diet_resnet50.pt.
"""

__version__ = "1.0.0"  # 1.0 CLI de inferencia sobre fotos o carpetas

import argparse
import sys
from pathlib import Path

from classifier import DietClassifier

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(paths) -> list:
    """Expande rutas: archivos sueltos o carpetas (no recursivo)."""
    images = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            images.extend(sorted(
                f for f in path.iterdir() if f.suffix.lower() in IMG_EXTS
            ))
        elif path.is_file():
            images.append(path)
        else:
            print(f"ADVERTENCIA: no existe: {path}")
    return images


def parse_args():
    parser = argparse.ArgumentParser(description="Clasificador de dieta animal")
    parser.add_argument("inputs", nargs="+", help="Foto(s) o carpeta(s) a clasificar")
    parser.add_argument("--device", default="", help="cuda | cpu | mps | '' (autodetectar)")
    return parser.parse_args()


def main():
    print(f"predict.py v{__version__}")
    args = parse_args()

    clf = DietClassifier(device=args.device)
    if not clf.is_available:
        print("ERROR: clasificador no disponible.")
        print("  Verifica que torch/torchvision estén instalados y que existan los pesos:")
        print("    weights/diet_resnet18.pt  y/o  weights/diet_resnet50.pt")
        sys.exit(1)

    images = collect_images(args.inputs)
    if not images:
        print("No se encontraron imágenes para clasificar.")
        sys.exit(1)

    print(f"{'archivo':<40} {'clase':<12} {'conf':>6}  modelo")
    print("─" * 72)
    for img_path in images:
        res = clf.classify_image(str(img_path))
        print(f"{img_path.name:<40} {res.label:<12} {res.confidence*100:>5.1f}%  {res.source}")


if __name__ == "__main__":
    main()
