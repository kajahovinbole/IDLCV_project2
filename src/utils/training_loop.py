# src/train_video_mean.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.models.models import SingleFrameModel
from src.utils.transforms import get_single_frame_transform
from video_utils import logits_mean_over_time
# import dataset!

datasets = ["placeholder for datasets module"]  # replace with actual import


num_epochs = 1


def train_one_epoch(model, loader, opt, device):
    model.train()
    total = correct = 0
    for x, y in loader:  # x: [B,C,T,H,W] ELLER list[T x [B,C,H,W]]
        y = y.to(device)
        if isinstance(x, (list, tuple)):
            x = [f.to(device) for f in x]
        else:
            x = x.to(device)

        logits = logits_mean_over_time(model, x)  # [B,K]
        loss = F.cross_entropy(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        pred = logits.argmax(1)
        total += y.numel()
        correct += (pred == y).sum().item()
    return correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = correct = 0
    for x, y in loader:
        y = y.to(device)
        if isinstance(x, (list, tuple)):
            x = [f.to(device) for f in x]
        else:
            x = x.to(device)
        logits = logits_mean_over_time(model, x)
        pred = logits.argmax(1)
        total += y.numel()
        correct += (pred == y).sum().item()
    return correct / total


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tf = get_single_frame_transform()

    # tilpass stier/filnavn til kurspakken (csv og root)
    train_ds = datasets.VideosDataset(
        csv_path="data/train_videos.csv",
        root="data",
        stack_frames=True,
        frame_transform=tf,
    )
    val_ds = datasets.VideosDataset(
        csv_path="data/val_videos.csv",
        root="data",
        stack_frames=True,
        frame_transform=tf,
    )

    train_loader = DataLoader(
        train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True
    )

    model = SingleFrameModel(num_classes=10, trainable_blocks=0).to(
        device
    )  # frys backbone
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    for epoch in range(5):
        acc_tr = train_one_epoch(model, train_loader, opt, device)
        acc_va = evaluate(model, val_loader, device)
        print(f"epoch {epoch}: train {acc_tr:.3f}  val {acc_va:.3f}")


if __name__ == "__main__":
    main()
