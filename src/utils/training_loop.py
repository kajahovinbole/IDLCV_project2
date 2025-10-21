import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils.model_factory import select_model
from src.utils.transforms import get_single_frame_transform
from src.datasets.datasets import FrameImageDataset, FrameVideoDataset
from config import DATA_ROOT, NUM_CLASSES, LR, NUM_WORKERS, BATCH_SIZE, EPOCHS, SEED


# --- Seed ---
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --- Device helper (trengs om x kan være liste) ---
def _move_to_device(x, device):
    if isinstance(x, (list, tuple)):
        return [f.to(device, non_blocking=True) for f in x]
    return x.to(device, non_blocking=True)


# --- Train/Eval ---
def train_one_epoch(model, loader, opt, device):
    model.train()
    total = correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x = _move_to_device(x, device)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        pred = logits.argmax(1)
        total += y.numel()
        correct += (pred == y).sum().item()
        loss_sum += loss.item() * y.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate_frames(model, loader, device):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x = _move_to_device(x, device)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        pred = logits.argmax(1)
        total += y.numel()
        correct += (pred == y).sum().item()
        loss_sum += loss.item() * y.size(0)
    return loss_sum / total, correct / total


# --- Dataloaders valgt ut fra modellnavn ---
def make_datasets_and_loaders(model_name, tf, device):
    name = model_name.lower()
    pin = device.type == "cuda"

    if name in ("late_fusion", "late-fusion"):
        # Video-datasett (bruk stack_frames=True for enkelhets skyld)
        train_ds = FrameVideoDataset(
            DATA_ROOT, split="train", transform=tf, stack_frames=True
        )
        val_ds = FrameVideoDataset(
            DATA_ROOT, split="val", transform=tf, stack_frames=True
        )
        bs = max(1, BATCH_SIZE // 4)  # video er tyngre
    else:
        # Single frame
        train_ds = FrameImageDataset(DATA_ROOT, split="train", transform=tf)
        val_ds = FrameImageDataset(DATA_ROOT, split="val", transform=tf)
        bs = BATCH_SIZE

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin
    )
    return train_loader, val_loader


# --- Main ---
def main(model_name):
    set_seed(SEED)  # viktig: før datasett/modell/opt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    tf = get_single_frame_transform()  # ResNet-normalisering per frame
    train_loader, val_loader = make_datasets_and_loaders(model_name, tf, device)

    model = select_model(model_name, num_classes=NUM_CLASSES).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

    best_va = 0.0
    for e in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, device)
        va_loss, va_acc = evaluate_frames(model, val_loader, device)
        print(
            f"Epoch {e:02d} | train {tr_acc:.3f} ({tr_loss:.4f}) | "
            f"val {va_acc:.3f} ({va_loss:.4f})"
        )
        best_va = max(best_va, va_acc)

    print(f"[INFO] Best val acc ({model_name}): {best_va:.3f}")
