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

__version__ = "1.2.0"  # 1.0 fine-tuning · 1.1 AMP+workers+UTF-8 · 1.2 Macro-F1, 2 fases, channels_last/TF32, tqdm

import argparse
import sys
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

# tqdm es opcional: en Colab viene preinstalado y da barras por época; en local
# (Windows) si no está, se degrada a un no-op sin romper nada.
try:
    from tqdm.auto import tqdm as _tqdm

    def _pbar(iterable, **kw):
        return _tqdm(iterable, **kw)

    def _set_postfix(bar, **kw):
        try:
            bar.set_postfix(**{k: f"{v:.4f}" for k, v in kw.items()})
        except Exception:
            pass
except ImportError:
    def _pbar(iterable, **kw):
        return iterable

    def _set_postfix(bar, **kw):
        pass

# La consola de Windows suele ser cp1252 y no traga caracteres de caja/acentos
# (el bucle imprime '✓' y '─'). Forzamos UTF-8 para no reventar al imprimir.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

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


def unfreeze_last_n(model: nn.Module, n: int) -> int:
    """
    Descongela la cabeza fc (siempre) más los últimos `n` bloques residuales
    del backbone (layer4, luego layer3, ...). Devuelve el nº de params
    entrenables resultante. Usado en la Fase 2 del entrenamiento en 2 fases.
    """
    for p in model.fc.parameters():
        p.requires_grad = True
    blocks = [model.layer4, model.layer3, model.layer2, model.layer1]
    for blk in blocks[: max(0, n)]:
        for p in blk.parameters():
            p.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── rendimiento (GPU) ───────────────────────────────────────────────────────────

def setup_perf(device: "torch.device") -> bool:
    """
    Activa TF32 + cudnn.benchmark en CUDA (acelera sin tocar exactitud) y
    devuelve si conviene usar memory_format=channels_last (también solo en CUDA).
    """
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return True
    return False


# ── métricas ────────────────────────────────────────────────────────────────────

def macro_f1(conf: list) -> float:
    """Macro F1 (%) a partir de la matriz de confusión (filas=real, cols=pred)."""
    f1s = []
    for i in range(NUM_CLASSES):
        tp = conf[i][i]
        fp = sum(conf[r][i] for r in range(NUM_CLASSES)) - tp
        fn = sum(conf[i]) - tp
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    return sum(f1s) / len(f1s) * 100


@torch.no_grad()
def evaluate(model, loader, device, autocast_ctx, channels_last):
    """Devuelve (acc%, macro_f1%, matriz_confusión) sobre el loader dado."""
    model.eval()
    conf = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        if channels_last:
            imgs = imgs.to(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        with autocast_ctx():
            preds = model(imgs).argmax(1)
        for t, p in zip(labels.cpu(), preds.cpu()):
            conf[t.item()][p.item()] += 1
    total = sum(sum(r) for r in conf)
    acc = sum(conf[i][i] for i in range(NUM_CLASSES)) / max(1, total) * 100
    return acc, macro_f1(conf), conf


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


# ── precisión mixta (AMP) ─────────────────────────────────────────────────────

def _make_amp(use_amp: bool):
    """
    Devuelve (scaler, autocast_ctx) compatibles con la API nueva (torch.amp,
    torch>=2.3) y la antigua (torch.cuda.amp). Si use_amp es False, ambos son
    no-ops, así el bucle de entrenamiento es idéntico con o sin AMP.
    """
    from contextlib import nullcontext
    try:                                   # API nueva (recomendada)
        from torch.amp import GradScaler, autocast
        scaler = GradScaler("cuda", enabled=use_amp)
        ctx = (lambda: autocast("cuda")) if use_amp else nullcontext
    except (ImportError, TypeError):       # API antigua (torch < 2.3)
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler(enabled=use_amp)
        ctx = (lambda: autocast()) if use_amp else nullcontext
    return scaler, ctx


# ── entrenamiento ─────────────────────────────────────────────────────────────

def _train_phase(*, phase, model, train_dl, val_dl, train_len, optimizer,
                 scheduler, plateau, criterion, device, scaler, autocast_ctx,
                 channels_last, epochs, output_path, best):
    """
    Corre `epochs` épocas de una fase. Selecciona el mejor modelo por Macro F1
    de validación (no por accuracy) y lo guarda en `output_path`. `best` es un
    dict mutable {'f1','acc','tag'} que se actualiza y devuelve.
    """
    for epoch in range(1, epochs + 1):
        model.train()
        run_loss, run_correct = 0.0, 0
        bar = _pbar(train_dl, desc=f"{phase} {epoch:2d}/{epochs}", unit="batch", leave=False)
        for imgs, labels in bar:
            imgs = imgs.to(device, non_blocking=True)
            if channels_last:
                imgs = imgs.to(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast_ctx():
                out  = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss    += loss.item() * imgs.size(0)
            run_correct += (out.argmax(1) == labels).sum().item()
            _set_postfix(bar, loss=loss.item())

        train_loss = run_loss / train_len
        train_acc  = run_correct / train_len * 100

        if val_dl is not None:
            val_acc, val_f1, _ = evaluate(model, val_dl, device, autocast_ctx, channels_last)
            if plateau:
                scheduler.step(val_f1)   # ReduceLROnPlateau sigue la métrica
            else:
                scheduler.step()
            star = ""
            if val_f1 > best["f1"]:
                best.update(f1=val_f1, acc=val_acc, tag=f"{phase} ép.{epoch}")
                torch.save(model.state_dict(), str(output_path))
                star = "  ✓ guardado"
            print(f"[{phase} {epoch:2d}/{epochs}] loss={train_loss:.4f}  "
                  f"train_acc={train_acc:.1f}%  val_acc={val_acc:.1f}%  "
                  f"val_f1={val_f1:.1f}%{star}")
        else:
            if not plateau:
                scheduler.step()
            print(f"[{phase} {epoch:2d}/{epochs}] loss={train_loss:.4f}  "
                  f"train_acc={train_acc:.1f}%  (sin val)")
    return best


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

    sampler   = make_weighted_sampler(train_ds)
    loader_kw = dict(num_workers=args.num_workers, pin_memory=True)
    if args.num_workers > 0:
        loader_kw["persistent_workers"] = True   # evita recrear workers cada época
    train_dl  = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, **loader_kw)
    val_dl    = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                           **loader_kw) if val_ds else None

    # dispositivo
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # TF32 + cudnn.benchmark en CUDA; channels_last solo si conviene (CUDA)
    channels_last = setup_perf(device)
    print(f"Dispositivo: {device}  |  channels_last: {channels_last}")

    # precisión mixta (AMP): acelera y reduce memoria en GPU CUDA. Solo activa ahí.
    use_amp = bool(args.amp and device.type == "cuda")
    scaler, autocast_ctx = _make_amp(use_amp)
    print(f"AMP (precisión mixta): {'activada' if use_amp else 'desactivada'}")

    # pesos de clase inversos para pérdida (combate desbalance)
    class_counts = Counter(train_ds.targets)
    total = len(train_ds.targets)
    class_weights = torch.tensor(
        [total / (NUM_CLASSES * class_counts.get(i, 1)) for i in range(NUM_CLASSES)],
        dtype=torch.float32,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best      = {"f1": -1.0, "acc": 0.0, "tag": "—"}
    train_len = len(train_ds)

    phase_kw = dict(model=None, train_dl=train_dl, val_dl=val_dl, train_len=train_len,
                    criterion=criterion, device=device, scaler=scaler,
                    autocast_ctx=autocast_ctx, channels_last=channels_last,
                    output_path=output_path)

    if args.two_phase:
        # ── FASE 1 — solo la cabeza fc (backbone congelado) ──────────────────
        p1_epochs = min(args.phase1_epochs, args.epochs)
        p2_epochs = max(0, args.epochs - p1_epochs)

        model = build_model(args.arch, freeze_backbone=True).to(device)
        if channels_last:
            model = model.to(memory_format=torch.channels_last)
        phase_kw["model"] = model

        opt1 = torch.optim.Adam(model.fc.parameters(), lr=args.phase1_lr, weight_decay=1e-4)
        sch1 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt1, mode="max", factor=0.5, patience=2)
        print(f"\n── FASE 1 — cabeza fc | {p1_epochs} épocas | lr={args.phase1_lr} | ReduceLROnPlateau")
        best = _train_phase(phase="P1", optimizer=opt1, scheduler=sch1, plateau=True,
                            epochs=p1_epochs, best=best, **phase_kw)

        # ── FASE 2 — fine-tuning de los últimos bloques ──────────────────────
        if p2_epochs > 0:
            n_train = unfreeze_last_n(model, args.unfreeze_n)
            print(f"\n── FASE 2 — fine-tuning {args.unfreeze_n} bloque(s) | {p2_epochs} épocas | "
                  f"lr={args.phase2_lr} | Cosine | {n_train:,} params entrenables")
            opt2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=args.phase2_lr, weight_decay=1e-5)
            sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=p2_epochs)
            best = _train_phase(phase="P2", optimizer=opt2, scheduler=sch2, plateau=False,
                                epochs=p2_epochs, best=best, **phase_kw)
    else:
        # ── modo clásico de una sola fase ────────────────────────────────────
        model = build_model(args.arch, args.freeze_backbone).to(device)
        if channels_last:
            model = model.to(memory_format=torch.channels_last)
        phase_kw["model"] = model

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        print(f"\n── Entrenamiento (1 fase) | {args.epochs} épocas | lr={args.lr} | Cosine")
        best = _train_phase(phase="EP", optimizer=optimizer, scheduler=scheduler, plateau=False,
                            epochs=args.epochs, best=best, **phase_kw)

    # ── cierre ───────────────────────────────────────────────────────────────
    if val_dl is None:
        # sin validación: guardar el modelo final tal cual
        torch.save(model.state_dict(), str(output_path))
        print(f"\nModelo guardado en: {output_path}")
        _print_confusion(model, train_dl, device, train_ds.classes, channels_last, autocast_ctx)
    else:
        # recargar el mejor guardado (puede no ser la última época). Es un
        # state_dict de tensores → weights_only=True es seguro; fallback para
        # torch antiguos que no soportan el argumento.
        try:
            state = torch.load(str(output_path), map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(str(output_path), map_location=device)
        model.load_state_dict(state)
        print(f"\nMejor modelo: {best['tag']}  |  val_acc={best['acc']:.1f}%  "
              f"val_f1={best['f1']:.1f}%")
        print(f"Modelo guardado en: {output_path}")
        _print_confusion(model, val_dl, device, train_ds.classes, channels_last, autocast_ctx)


def _print_confusion(model, dataloader, device, class_names, channels_last, autocast_ctx):
    acc, f1, conf = evaluate(model, dataloader, device, autocast_ctx, channels_last)

    print("\nMatriz de confusión (filas=real, cols=predicho):")
    header = "           " + "  ".join(f"{c:>10}" for c in class_names)
    print(header)
    for i, row in enumerate(conf):
        total_row = sum(row)
        cls_acc = row[i] / total_row * 100 if total_row > 0 else 0
        row_str = "  ".join(f"{v:>10}" for v in row)
        print(f"  {class_names[i]:>10}  {row_str}  ({cls_acc:.0f}%)")
    print(f"\n  Accuracy global : {acc:.1f}%")
    print(f"  Macro F1        : {f1:.1f}%")


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
    parser.add_argument("--num-workers",     type=int,   default=0,
                        help="procesos de carga de datos (GPU/Colab: 2-4; def. 0)")
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--amp",             action="store_true",
                        help="precisión mixta en GPU CUDA (más rápido, menos memoria)")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Congelar backbone, solo entrenar la cabeza fc (modo 1 fase)")
    # ── entrenamiento en 2 fases (cabeza congelada → fine-tuning) ──────────────
    parser.add_argument("--two-phase",       action="store_true",
                        help="Fase 1: solo la cabeza (backbone congelado). "
                             "Fase 2: fine-tuning de los últimos bloques. "
                             "Reparte --epochs entre ambas fases.")
    parser.add_argument("--phase1-epochs",   type=int,   default=10,
                        help="Épocas de la Fase 1 (cabeza); el resto van a Fase 2")
    parser.add_argument("--phase1-lr",       type=float, default=1e-3,
                        help="LR de la Fase 1 (cabeza fc sobre backbone congelado)")
    parser.add_argument("--phase2-lr",       type=float, default=1e-5,
                        help="LR de la Fase 2 (fine-tuning, más bajo)")
    parser.add_argument("--unfreeze-n",      type=int,   default=2,
                        help="Bloques residuales a descongelar en Fase 2 (layer4, layer3, ...)")
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
    print(f"Num workers  : {args.num_workers}")
    print(f"AMP          : {args.amp}")
    if args.two_phase:
        print(f"Modo         : 2 fases (cabeza → fine-tuning)")
        print(f"  Fase 1     : {min(args.phase1_epochs, args.epochs)} ép.  lr={args.phase1_lr}")
        print(f"  Fase 2     : {max(0, args.epochs - args.phase1_epochs)} ép.  "
              f"lr={args.phase2_lr}  unfreeze={args.unfreeze_n}")
    else:
        print(f"Modo         : 1 fase  lr={args.lr}  freeze_bb={args.freeze_backbone}")
    print(f"Salida       : {args.output}")
    print("─" * 50)
    train(args)


if __name__ == "__main__":
    main()
