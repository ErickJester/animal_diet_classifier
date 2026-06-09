"""
Entrenamiento del clasificador CNN de dieta animal.

Uso básico:
    python train_classifier.py --arch resnet18 --epochs 30

Requiere dataset con imágenes etiquetadas por carpeta:
    dataset/
      train/
        carnivore/  *.jpg
        herbivore/  *.jpg
        omnivore/   *.jpg
      val/
        carnivore/  *.jpg
        herbivore/  *.jpg
        omnivore/   *.jpg

Referencia: classifier.py
"""

__version__ = "1.0.0"  # 1.0 fine-tuning ResNet-18/50 desde ImageNet

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── validación de dependencias ────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, WeightedRandomSampler
    import torchvision.models as tv_models
    import torchvision.transforms as T
    from torchvision.datasets import ImageFolder
except ImportError:
    print("ERROR: PyTorch/torchvision no instalados.")
    print("  pip install torch torchvision")
    sys.exit(1)

CLASSES     = ["carnivore", "herbivore", "omnivore"]
NUM_CLASSES = len(CLASSES)
INPUT_SIZE  = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── modelo ────────────────────────────────────────────────────────────────────

def build_model(arch: str, freeze_backbone: bool) -> nn.Module:
    if arch == "resnet18":
        model = tv_models.resnet18(weights="IMAGENET1K_V1")
        in_features = 512
    elif arch == "resnet50":
        model = tv_models.resnet50(weights="IMAGENET1K_V2")
        in_features = 2048
    else:
        raise ValueError(f"Arquitectura no soportada: {arch}")

    model.fc = nn.Linear(in_features, NUM_CLASSES)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False

    return model


# ── transforms ────────────────────────────────────────────────────────────────

def get_transforms():
    train_tf = T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_tf = T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.CenterCrop(INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, val_tf


# ── sampler balanceado ────────────────────────────────────────────────────────

def make_weighted_sampler(dataset: ImageFolder) -> WeightedRandomSampler:
    class_counts = Counter(dataset.targets)
    weights = [1.0 / class_counts[t] for t in dataset.targets]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ── entrenamiento ─────────────────────────────────────────────────────────────

def train(args):
    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir   = data_dir / "val"

    if not train_dir.exists():
        print(f"ERROR: No se encontró {train_dir}")
        print("  Crea las carpetas train/carnivore, train/herbivore, train/omnivore con imágenes.")
        sys.exit(1)

    train_tf, val_tf = get_transforms()

    train_ds = ImageFolder(str(train_dir), transform=train_tf)
    val_ds   = ImageFolder(str(val_dir),   transform=val_tf) if val_dir.exists() else None

    print(f"Clases detectadas: {train_ds.classes}")
    print(f"Train: {len(train_ds)} imgs  |  Val: {len(val_ds) if val_ds else 'N/A'} imgs")

    # verificar orden de clases (debe coincidir con CLASSES)
    expected = sorted(CLASSES)
    if train_ds.classes != expected:
        print(f"ADVERTENCIA: orden de clases {train_ds.classes} != esperado {expected}")
        print("  El modelo usará el orden del directorio — asegúrate que coincide con classifier.py")

    sampler    = make_weighted_sampler(train_ds)
    train_dl   = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                            num_workers=0, pin_memory=True)
    val_dl     = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0) if val_ds else None

    # dispositivo
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Dispositivo: {device}")

    model = build_model(args.arch, args.freeze_backbone).to(device)

    # pesos de clase inversos para pérdida (combate desbalance)
    class_counts = Counter(train_ds.targets)
    total = len(train_ds.targets)
    class_weights = torch.tensor(
        [total / (NUM_CLASSES * class_counts.get(i, 1)) for i in range(NUM_CLASSES)],
        dtype=torch.float32,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    best_epoch   = 0

    for epoch in range(1, args.epochs + 1):
        # entrenamiento
        model.train()
        train_loss = 0.0
        train_correct = 0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * imgs.size(0)
            train_correct += (out.argmax(1) == labels).sum().item()

        train_loss /= len(train_ds)
        train_acc   = train_correct / len(train_ds) * 100

        # validación
        val_acc = float("nan")
        if val_dl is not None:
            model.eval()
            val_correct = 0
            with torch.no_grad():
                for imgs, labels in val_dl:
                    imgs, labels = imgs.to(device), labels.to(device)
                    out = model(imgs)
                    val_correct += (out.argmax(1) == labels).sum().item()
            val_acc = val_correct / len(val_ds) * 100

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch   = epoch
                torch.save(model.state_dict(), str(output_path))
                print(f"  ✓ Nuevo mejor modelo guardado ({val_acc:.1f}%)")

        scheduler.step()
        print(f"Época {epoch:3d}/{args.epochs}  "
              f"loss={train_loss:.4f}  train_acc={train_acc:.1f}%  "
              f"val_acc={val_acc:.1f}%")

    if val_dl is None:
        # sin validación: guardar el modelo final
        torch.save(model.state_dict(), str(output_path))
        print(f"\nModelo guardado en: {output_path}")
    else:
        print(f"\nMejor época: {best_epoch}  val_acc={best_val_acc:.1f}%")
        print(f"Modelo guardado en: {output_path}")

    # matriz de confusión final (en val si existe, sino en train)
    _print_confusion(model, val_dl or train_dl, device, train_ds.classes)


def _print_confusion(model, dataloader, device, class_names):
    model.eval()
    conf = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            for t, p in zip(labels.cpu(), preds.cpu()):
                conf[t.item()][p.item()] += 1

    print("\nMatriz de confusión (filas=real, cols=predicho):")
    header = "           " + "  ".join(f"{c:>10}" for c in class_names)
    print(header)
    for i, row in enumerate(conf):
        total_row = sum(row)
        acc = row[i] / total_row * 100 if total_row > 0 else 0
        row_str = "  ".join(f"{v:>10}" for v in row)
        print(f"  {class_names[i]:>10}  {row_str}  ({acc:.0f}%)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento del clasificador CNN de dieta animal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--arch",            default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--data-dir",        default="dataset")
    parser.add_argument("--epochs",          type=int,   default=30)
    parser.add_argument("--batch-size",      type=int,   default=32)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Congelar backbone, solo entrenar la cabeza fc")
    parser.add_argument("--output",          default=None,
                        help="Ruta de salida del modelo (default: weights/diet_{arch}.pt)")
    parser.add_argument("--device",          default="",
                        help="cuda | cpu | mps | '' (autodetectar)")
    return parser.parse_args()


def main():
    print(f"train_classifier.py v{__version__}")
    args = parse_args()

    if args.output is None:
        args.output = f"weights/diet_{args.arch}.pt"

    print(f"Arquitectura : {args.arch}")
    print(f"Dataset      : {args.data_dir}")
    print(f"Épocas       : {args.epochs}")
    print(f"Batch size   : {args.batch_size}")
    print(f"LR           : {args.lr}")
    print(f"Freeze BB    : {args.freeze_backbone}")
    print(f"Salida       : {args.output}")
    print("─" * 50)
    train(args)


if __name__ == "__main__":
    main()
