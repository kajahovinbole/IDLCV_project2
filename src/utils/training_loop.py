import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils.model_factory import select_model
from src.utils.transforms import get_single_frame_transform
from src.datasets.datasets import FrameImageDataset  # sørg for riktig path
from config import DATA_ROOT, NUM_CLASSES, LR, NUM_WORKERS, BATCH_SIZE, EPOCHS


def train_one_epoch(model, loader, opt, device):
    model.train()
    total = correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
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
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        pred = logits.argmax(1)
        total += y.numel()
        correct += (pred == y).sum().item()
        loss_sum += loss.item() * y.size(0)
    return loss_sum / total, correct / total


def main(model_name="per_frame_agg"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Data og transforms
    tf = get_single_frame_transform()
    train_ds = FrameImageDataset(root_dir=DATA_ROOT, split="train", transform=tf)
    val_ds = FrameImageDataset(root_dir=DATA_ROOT, split="val", transform=tf)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
    )

    # Modell fra factory
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
