import os
import sys
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# sørg for at prosjektroten er på sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# prosjekt-importer
from config import DATA_ROOT, NUM_CLASSES, BATCH_SIZE, NUM_WORKERS
from src.utils.transforms import get_single_frame_transform
from src.datasets.datasets import FrameImageDataset, FrameVideoDataset
from src.utils.model_factory import select_model
from src.utils.eval_utils import evaluate_loader


def get_class_names(split="test"):
    tf = get_single_frame_transform()
    ds = FrameVideoDataset(DATA_ROOT, split=split, transform=tf, stack_frames=True)
    return getattr(ds, "classes", [str(i) for i in range(NUM_CLASSES)])


def build_loader(model_name, split):
    tf = get_single_frame_transform()
    name = model_name.lower()
    if name in (
        "late_fusion",
        "late-fusion",
        "early_fusion",
        "early-fusion",
        "c3d",
        "3dcnn",
        "3d_cnn",
    ):
        ds = FrameVideoDataset(DATA_ROOT, split=split, transform=tf, stack_frames=True)
        bs = max(1, BATCH_SIZE // 4)
    else:
        ds = FrameImageDataset(DATA_ROOT, split=split, transform=tf)
        bs = BATCH_SIZE
    loader = DataLoader(
        ds, batch_size=bs, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )
    return loader, getattr(ds, "classes", [str(i) for i in range(NUM_CLASSES)])


def maybe_load_best_checkpoint(model, model_name):
    project_root = Path(__file__).resolve().parent.parent
    ckpt_path = project_root / "results" / "checkpoints" / f"{model_name}_best.pth"

    if not ckpt_path.exists():
        print(
            f"[WARN] ⚠️  Fant ikke checkpoint for {model_name} (forventet {ckpt_path})"
        )
        return model

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 🔹 Spesialhåndtering for EarlyFusionModel
    if "early" in model_name.lower():
        print("[INFO] Justerer EarlyFusionModel for lagrede vekter...")
        state_dict = ckpt["model_state"]
        conv1_weight = state_dict.get("backbone.conv1.weight", None)
        if conv1_weight is not None:
            in_ch = conv1_weight.shape[1]
            # bruk samme device og dtype som modellens parametere
            first_param = next(model.parameters())
            device = first_param.device
            dtype = first_param.dtype
            model._ensure_conv1_in_channels(T=in_ch // 3, device=device, dtype=dtype)

    model.load_state_dict(ckpt["model_state"])
    print(
        f"[INFO] ✅ Lastet checkpoint: {ckpt_path.name} "
        f"(epoch {ckpt.get('epoch', '?')}, val_f1={ckpt.get('val_f1', 'n/a')})"
    )
    return model


def main(models):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(__file__).resolve().parent.parent / "results" / "eval"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for mname in models:
        print(f"\n[INFO] --- Evaluerer modell: {mname} ---")
        loader, class_names = build_loader(mname, "test")
        model = select_model(mname, num_classes=NUM_CLASSES).to(device)
        model = maybe_load_best_checkpoint(model, mname)

        metrics = evaluate_loader(
            model, loader, device, compute_confusion=True, class_names=class_names
        )

        with open(outdir / f"{mname}_test.json", "w") as f:
            json.dump({"model": mname, **metrics}, f, indent=2)

        rows.append([mname, metrics["acc"], metrics["f1_macro"], metrics["loss"]])

    # lag oppsummeringsfil for alle modeller
    with open(outdir / "summary.csv", "w") as f:
        f.write("model,acc,f1_macro,loss\n")
        for r in rows:
            f.write(",".join([r[0]] + [f"{x:.6f}" for x in r[1:]]) + "\n")

    print(f"\n[INFO] Ferdig! Resultater lagret i {outdir}")


# if __name__ == "__main__":
#     main(MODEL_NAMES)

if __name__ == "__main__":
    main(["per_frame"])
