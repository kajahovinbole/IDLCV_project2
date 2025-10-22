import math
import time
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# prosjekt-importer
from config import DATA_ROOT, NUM_CLASSES, LR, NUM_WORKERS, BATCH_SIZE, EPOCHS, SEED
from src.utils.model_factory import select_model
from src.utils.transforms import get_single_frame_transform
from src.datasets.datasets import FrameImageDataset, FrameVideoDataset
from src.utils.eval_utils import evaluate_loader  # <-- felles eval


# -------- Early stopping --------
class EarlyStopper:
    def __init__(self, patience=7, min_delta=1e-4, mode="max"):
        """
        Overvåk en val-metrikk (f.eks. F1). Stopper når ingen forbedring.
        """
        assert mode in ("min", "max")
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best = -math.inf if mode == "max" else math.inf
        self.num_bad = 0

    def step(self, value: float):
        """
        value: ny måleverdi
        return: (should_stop, is_best)
        """
        improved = (
            (value > self.best + self.min_delta)
            if self.mode == "max"
            else (value < self.best - self.min_delta)
        )
        if improved:
            self.best = value
            self.num_bad = 0
            return False, True
        else:
            self.num_bad += 1
            return self.num_bad > self.patience, False


# -------- Seeds --------
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------- Device helper (for list av frames) --------
def _move_to_device(x, device):
    if isinstance(x, (list, tuple)):
        return [f.to(device, non_blocking=True) for f in x]
    return x.to(device, non_blocking=True)


# -------- Train ett epoch --------
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        pred = logits.argmax(1)
        total += y.numel()
        correct += (pred == y).sum().item()
        loss_sum += loss.item() * y.size(0)

    return loss_sum / total, correct / total


# -------- Lag dataloaders for valgt modell --------
def make_datasets_and_loaders(model_name, tf, device):
    name = model_name.lower()
    pin = device.type == "cuda"
    if name in ("late_fusion", "early_fusion", "3d_cnn"):
        train_ds = FrameVideoDataset(
            DATA_ROOT, split="train", transform=tf, stack_frames=True
        )
        val_ds = FrameVideoDataset(
            DATA_ROOT, split="val", transform=tf, stack_frames=True
        )
        bs = max(1, BATCH_SIZE // 8)  # tyngre modeller → mindre batch
    else:
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


# -------- main --------
def main(model_name: str):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    tf = get_single_frame_transform()
    train_loader, val_loader = make_datasets_and_loaders(model_name, tf, device)

    model = select_model(model_name, num_classes=NUM_CLASSES).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=3
    )

    hist = {"model": model_name, "epochs": []}
    best_va = 0.0
    best_f1 = 0.0
    t0_all = time.time()

    # Early stopping på val-F1
    early = EarlyStopper(patience=7, min_delta=1e-4, mode="max")

    # Checkpoint for beste modell
    ckpt_dir = Path("results/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{model_name}_best.pth"

    print(f"[INFO] Starter trening for modell: {model_name}\n")
    for e in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, device)

        # Rask validering med felles evaluate_loader (ingen CM her)
        va = evaluate_loader(model, val_loader, device, compute_confusion=False)
        va_loss, va_acc, va_f1 = va["loss"], va["acc"], va["f1_macro"]
        sched.step(va_f1)  # plateauscheduler mot F1

        dt = time.time() - t0
        print(
            f"Epoch {e:02d} | train {tr_acc:.3f} ({tr_loss:.4f}) | val {va_acc:.3f} ({va_loss:.4f}) | F1 {va_f1:.3f}"
        )

        # logg
        hist["epochs"].append(
            {
                "epoch": e,
                "train_acc": tr_acc,
                "train_loss": tr_loss,
                "val_acc": va_acc,
                "val_loss": va_loss,
                "val_f1": va_f1,
                "time_s": dt,
            }
        )

        # Early stopping på val-F1
        should_stop, is_best = early.step(va_f1)
        if is_best:
            best_va = va_acc
            best_f1 = va_f1
            torch.save(
                {
                    "epoch": e,
                    "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "val_acc": va_acc,
                    "val_loss": va_loss,
                    "val_f1": va_f1,
                },
                ckpt_path,
            )

        if should_stop:
            print(
                f"[EARLY STOP] Ingen F1-forbedring på {early.patience} epoker. Stopper ved epoch {e}."
            )
            break

    hist["best_val_acc"] = best_va
    hist["best_val_f1"] = best_f1
    hist["total_time_s"] = time.time() - t0_all

    # lagre historikk per modell til results/history/<model>.json
    outdir = Path("results/history")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / f"{model_name}.json", "w") as f:
        json.dump(hist, f, indent=2)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)
    args = ap.parse_args()
    main(args.model_name)
