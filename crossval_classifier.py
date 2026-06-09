"""
Cross-validation del clasificador de dieta animal.

Parte de pesos ya ajustados (fine-tuned) y evalúa su generalización con
k-fold estratificado sobre el dataset completo (train/ + val/ combinados).

Uso:
    python crossval_classifier.py --arch resnet18 --folds 5 --epochs 10
    python crossval_classifier.py --arch resnet50 --folds 5 --epochs 10 --weights weights/diet_resnet50.pt

Opciones:
    --arch        resnet18 | resnet50  (default: resnet18)
    --weights     ruta a los pesos ajustados (default: weights/diet_{arch}.pt)
    --folds       número de folds (default: 5)
    --epochs      épocas de fine-tuning por fold (default: 10)
    --batch-size  (default: 32)
    --lr          tasa de aprendizaje (default: 1e-4)
    --data-dir    directorio del dataset (default: dataset/)
    --save-best   guarda el mejor fold en weights/diet_{arch}_cv.pt
    --device      cuda | cpu | mps | '' (autodetectar)
    --seed        semilla para reproducibilidad (default: 42)
"""

from __future__ import annotations

__version__ = "1.0.0"  # 1.0 k-fold estratificado desde pesos fine-tuned

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    import torchvision.models as tv_models
    import torchvision.transforms as T
    from PIL import Image
except ImportError:
    print("ERROR: PyTorch/torchvision no instalados.")
    print("  pip install torch torchvision Pillow")
    sys.exit(1)

CLASSES       = ["carnivore", "herbivore", "omnivore"]
NUM_CLASSES   = len(CLASSES)
CLASS_TO_IDX  = {c: i for i, c in enumerate(CLASSES)}
INPUT_SIZE    = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_EXTS      = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── dataset ────────────────────────────────────────────────────────────────────

class ImageListDataset(Dataset):
    def __init__(self, samples: list, transform=None) -> None:
        self.samples   = samples    # lista de (Path, int)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def collect_all_images(data_dir: Path) -> list:
    """Recoge (path, label) de train/ y val/ combinados."""
    samples = []
    for split in ("train", "val"):
        for cls in CLASSES:
            folder = data_dir / split / cls
            if not folder.is_dir():
                continue
            label = CLASS_TO_IDX[cls]
            for f in sorted(folder.iterdir()):
                if f.suffix.lower() in IMG_EXTS:
                    samples.append((f, label))
    return samples


# ── k-fold estratificado ───────────────────────────────────────────────────────

def stratified_kfold(samples: list, k: int, seed: int = 42) -> list:
    """
    Devuelve k tuplas (train_indices, val_indices) con igual proporción de
    clases en cada fold.
    """
    rng = random.Random(seed)
    by_class: dict[int, list] = {i: [] for i in range(NUM_CLASSES)}
    for idx, (_, label) in enumerate(samples):
        by_class[label].append(idx)

    for indices in by_class.values():
        rng.shuffle(indices)

    # repartir cada clase en k cubetas
    buckets: list[list] = [[] for _ in range(k)]
    for indices in by_class.values():
        for i, idx in enumerate(indices):
            buckets[i % k].append(idx)

    splits = []
    for fold in range(k):
        val_set = set(buckets[fold])
        train_idx = [i for i in range(len(samples)) if i not in val_set]
        val_idx   = list(val_set)
        splits.append((train_idx, val_idx))

    return splits


# ── modelo ─────────────────────────────────────────────────────────────────────

def build_model(arch: str) -> nn.Module:
    if arch == "resnet18":
        model = tv_models.resnet18(weights=None)
        model.fc = nn.Linear(512, NUM_CLASSES)
    elif arch == "resnet50":
        model = tv_models.resnet50(weights=None)
        model.fc = nn.Linear(2048, NUM_CLASSES)
    else:
        raise ValueError(f"Arquitectura no soportada: {arch}")
    return model


def load_pretrained(arch: str, weights_path: Path, device: torch.device) -> nn.Module:
    if not weights_path.is_file():
        print(f"ERROR: no se encontraron pesos en {weights_path}")
        print("  Entrena primero:  python train_classifier.py --arch", arch)
        sys.exit(1)
    model = build_model(arch)
    state = torch.load(str(weights_path), map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    return model


# ── transforms ─────────────────────────────────────────────────────────────────

def get_transforms():
    train_tf = T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.CenterCrop(INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def make_weighted_sampler(labels: list) -> WeightedRandomSampler:
    counts  = Counter(labels)
    weights = [1.0 / counts[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ── un fold ────────────────────────────────────────────────────────────────────

def run_fold(
    fold_idx:      int,
    train_samples: list,
    val_samples:   list,
    args,
    device:        torch.device,
) -> tuple[float, list, dict]:
    """
    Entrena un fold partiendo de los pesos ajustados.
    Devuelve (mejor_val_acc, matriz_confusion, mejor_state_dict).
    """
    train_tf, val_tf = get_transforms()

    train_ds = ImageListDataset(train_samples, transform=train_tf)
    val_ds   = ImageListDataset(val_samples,   transform=val_tf)

    train_labels = [s[1] for s in train_samples]
    sampler      = make_weighted_sampler(train_labels)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          sampler=sampler, num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, num_workers=0)

    # cada fold parte desde los mismos pesos ajustados
    model = load_pretrained(args.arch, Path(args.weights), device).to(device)

    counts = Counter(train_labels)
    total  = len(train_labels)
    class_weights = torch.tensor(
        [total / (NUM_CLASSES * counts.get(i, 1)) for i in range(NUM_CLASSES)],
        dtype=torch.float32,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    best_state   = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                correct += (model(imgs).argmax(1) == labels).sum().item()
        val_acc = correct / len(val_ds) * 100

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            marker = "  *"
        print(f"  Fold {fold_idx+1} | Época {epoch:2d}/{args.epochs}  "
              f"val_acc={val_acc:.1f}%{marker}")

    # matriz de confusión con el mejor estado del fold
    model.load_state_dict(best_state)
    model.eval()
    conf = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    with torch.no_grad():
        for imgs, labels in val_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            for t, p in zip(labels.cpu(), preds.cpu()):
                conf[t.item()][p.item()] += 1

    return best_val_acc, conf, best_state


# ── reporte ────────────────────────────────────────────────────────────────────

def print_confusion(conf: list, title: str = "") -> None:
    if title:
        print(f"\n{title}")
    header = "           " + "  ".join(f"{c:>10}" for c in CLASSES)
    print(header)
    for i, row in enumerate(conf):
        total_row = sum(row)
        acc       = row[i] / total_row * 100 if total_row > 0 else 0.0
        row_str   = "  ".join(f"{v:>10}" for v in row)
        print(f"  {CLASSES[i]:>10}  {row_str}  ({acc:.0f}%)")


def aggregate_conf(confs: list) -> list:
    agg = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    for conf in confs:
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                agg[i][j] += conf[i][j]
    return agg


# ── entrenamiento principal ────────────────────────────────────────────────────

def crossval(args):
    data_dir = Path(args.data_dir)

    # dispositivo
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Arquitectura : {args.arch}")
    print(f"Pesos base   : {args.weights}")
    print(f"Dataset      : {data_dir}")
    print(f"Folds        : {args.folds}")
    print(f"Épocas/fold  : {args.epochs}")
    print(f"Batch size   : {args.batch_size}")
    print(f"LR           : {args.lr}")
    print(f"Dispositivo  : {device}")
    print("─" * 55)

    samples = collect_all_images(data_dir)
    if not samples:
        print(f"ERROR: no se encontraron imágenes en {data_dir}")
        print("  Corre primero prepare.py o curate.py commit")
        sys.exit(1)

    counts = Counter(label for _, label in samples)
    print(f"Total imágenes: {len(samples)}")
    for cls, idx in CLASS_TO_IDX.items():
        print(f"  {cls:<12} {counts[idx]:>6}")
    print()

    splits    = stratified_kfold(samples, k=args.folds, seed=args.seed)
    fold_accs = []
    fold_confs= []
    best_overall_acc   = 0.0
    best_overall_state = None

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_samples = [samples[i] for i in train_idx]
        val_samples   = [samples[i] for i in val_idx]

        print(f"\n{'─'*55}")
        print(f"Fold {fold_idx+1}/{args.folds}  "
              f"(train={len(train_samples)}  val={len(val_samples)})")
        print(f"{'─'*55}")

        val_acc, conf, state = run_fold(
            fold_idx, train_samples, val_samples, args, device
        )

        fold_accs.append(val_acc)
        fold_confs.append(conf)
        print(f"  -> Mejor val_acc fold {fold_idx+1}: {val_acc:.2f}%")

        if val_acc > best_overall_acc:
            best_overall_acc   = val_acc
            best_overall_state = state

    # ── resumen final ──────────────────────────────────────────────────────────
    mean_acc = sum(fold_accs) / len(fold_accs)
    variance = sum((a - mean_acc) ** 2 for a in fold_accs) / len(fold_accs)
    std_acc  = variance ** 0.5

    print(f"\n{'═'*55}")
    print("RESUMEN CROSS-VALIDATION")
    print(f"{'═'*55}")
    for i, acc in enumerate(fold_accs):
        print(f"  Fold {i+1}: {acc:.2f}%")
    print(f"  {'─'*30}")
    print(f"  Media : {mean_acc:.2f}%  ±  {std_acc:.2f}%")
    print(f"  Mejor : {max(fold_accs):.2f}%  (fold {fold_accs.index(max(fold_accs))+1})")
    print(f"  Peor  : {min(fold_accs):.2f}%  (fold {fold_accs.index(min(fold_accs))+1})")

    print_confusion(aggregate_conf(fold_confs),
                    title="Matriz de confusión agregada (suma de todos los folds):")

    if args.save_best and best_overall_state is not None:
        out = Path(f"weights/diet_{args.arch}_cv.pt")
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_overall_state, str(out))
        print(f"\nMejor modelo (fold {fold_accs.index(max(fold_accs))+1}) "
              f"guardado en: {out}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-validation del clasificador de dieta animal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--arch",       default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--weights",    default=None,
                        help="Pesos ajustados de partida (default: weights/diet_{arch}.pt)")
    parser.add_argument("--folds",      type=int,   default=5)
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--data-dir",   default="dataset")
    parser.add_argument("--save-best",  action="store_true",
                        help="Guarda el mejor fold en weights/diet_{arch}_cv.pt")
    parser.add_argument("--device",     default="",
                        help="cuda | cpu | mps | '' (autodetectar)")
    parser.add_argument("--seed",       type=int,   default=42)
    return parser.parse_args()


def main():
    print(f"crossval_classifier.py v{__version__}")
    args = parse_args()
    if args.weights is None:
        args.weights = f"weights/diet_{args.arch}.pt"
    crossval(args)


if __name__ == "__main__":
    main()
