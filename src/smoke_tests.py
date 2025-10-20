
import torch
from src.datasets.datasets import FrameVideoDataset, FrameImageDataset
from src.models.models import SingleFrameModel
from src.utils.video_utils import logits_mean_over_time
from src.utils.transforms import get_single_frame_transform
from config import DATA_ROOT, NUM_CLASSES, N_FRAMES

def main():
    print(f"[INFO] Tester dataset fra {DATA_ROOT}")
    transform = get_single_frame_transform()
    ds = FrameImageDataset(root_dir=DATA_ROOT, split="train", transform=transform)
    #ds = FrameVideoDataset(root_dir=DATA_ROOT, split="train", transform=transform, stack_frames=True)
    print(f"Antall videoer i train: {len(ds)}")

    # hent én video
    x, y = ds[0]
    print(f"Video shape: {x.shape if isinstance(x, torch.Tensor) else len(x)} frames")
    print(f"Label: {y}")

    # initier modell
    model = SingleFrameModel(num_classes=NUM_CLASSES)
    model.eval()

    # kjør forward-pass (enten med stackede frames eller liste)
    with torch.no_grad():
        logits = logits_mean_over_time(model, x)
    print(f"Output logits shape: {logits.shape}")

    print("[OK] Smoke-test fullført uten feil ✅")

if __name__ == "__main__":
    main()
