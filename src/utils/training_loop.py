# src/train_video_mean.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.models.models import SingleFrameModel
from src.utils.transforms import get_single_frame_transform
from video_utils import logits_mean_over_time
import os


frames_root = os.path.expanduser("~/prosjekt_2/ufc10/frames")
train_dir = os.path.join(frames_root, "train")
val_dir   = os.path.join(frames_root, "val")
test_dir  = os.path.join(frames_root, "test")

BATCH_SIZE = 64
EPOCHS = 1
LR = 1e-4


# ---------- Aggregator (for senere video-evaluering) ----------
@torch.no_grad()
def aggregate_logits_mean(logits_list):
    """Tar en liste/stack av logits [T, num_classes] og returnerer snittet [num_classes]."""
    return torch.stack(logits_list, dim=0).mean(dim=0)

# ---------- Forward-adapter ----------
def forward_batch(model, x, mode="per_frame"):
    """
    Ett felles inngangspunkt til modellen.
    Nå: mode='per_frame' (x: [B,C,H,W]) -> model(x).
    Senere: mode='sequence' (x: [C,T,H,W] eller liste med T bilder) -> kall sekvens-agg.
    """
    if mode == "per_frame":
        return model(x)  # [B, K]
    else:
        raise NotImplementedError("Legg til sekvensvei her når du går til early/late fusion.")

# ---------- Train / Eval ----------
def train_one_epoch(model, loader, opt, device):
    model.train()
    total = correct = 0
    loss_sum = 0.0

    for x, y in loader:                 # x: [B,C,H,W] (per-frame)
        x, y = x.to(device), y.to(device)

        logits = forward_batch(model, x, mode="per_frame")
        loss = F.cross_entropy(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        pred = logits.argmax(1)
        total   += y.numel()
        correct += (pred == y).sum().item()
        loss_sum += loss.item() * y.size(0)

    return loss_sum/total, correct/total

@torch.no_grad()
def evaluate_frames(model, loader, device):
    model.eval()
    total = correct = 0
    loss_sum = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = forward_batch(model, x, mode="per_frame")
        loss = F.cross_entropy(logits, y)

        pred = logits.argmax(1)
        total   += y.numel()
        correct += (pred == y).sum().item()
        loss_sum += loss.item() * y.size(0)

    return loss_sum/total, correct/total

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tf = get_single_frame_transform()

    # Per-frame datasett (ImageFolder går rekursivt under klasse-mapper)
    train_ds = ImageFolder(root=train_dir, transform=tf)
    val_ds   = ImageFolder(root=val_dir,   transform=tf)

    print("Classes:", train_ds.classes)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=pin)

    model = SingleFrameModel(num_classes=len(train_ds.classes), trainable_blocks=0).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

    best_va = 0.0
    for e in range(1, EPOCHS+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, device)
        va_loss, va_acc = evaluate_frames(model, val_loader, device)
        print(f"Epoch {e:02d} | train {tr_acc:.3f} ({tr_loss:.4f}) | val {va_acc:.3f} ({va_loss:.4f})")
        best_va = max(best_va, va_acc)

    print(f"Best val acc (per-frame): {best_va:.3f}")

if __name__ == "__main__":
    main()
