from glob import glob
from pathlib import Path
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T
from config import DATA_ROOT, N_FRAMES


class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir=DATA_ROOT, split="train", transform=None):
        self.root = Path(root_dir)
        self.frame_paths = sorted(glob(f"{self.root}/frames/{split}/*/*/*.jpg"))
        self.df = pd.read_csv(f"{self.root}/metadata/{split}.csv")
        self.split = split
        self.transform = transform or T.ToTensor()

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        video_name = Path(frame_path).parent.name
        label = self.df.loc[self.df["video_name"] == video_name, "label"].item()
        img = Image.open(frame_path).convert("RGB")
        return self.transform(img), label


class FrameVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self, root_dir=DATA_ROOT, split="train", transform=None, stack_frames=True
    ):
        self.root = Path(root_dir)
        self.video_paths = sorted(glob(f"{self.root}/videos/{split}/*/*.avi"))
        self.df = pd.read_csv(f"{self.root}/metadata/{split}.csv")
        self.split = split
        self.transform = transform or T.ToTensor()
        self.stack_frames = stack_frames
        self.n_sampled_frames = N_FRAMES

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = Path(video_path).stem
        label = self.df.loc[self.df["video_name"] == video_name, "label"].item()
        frames_dir = video_path.replace("/videos/", "/frames/").replace(".avi", "")
        frames = [
            self.transform(
                Image.open(os.path.join(frames_dir, f"frame_{i}.jpg")).convert("RGB")
            )
            for i in range(1, self.n_sampled_frames + 1)
        ]
        if self.stack_frames:
            # [T,C,H,W] -> [C,T,H,W]
            return torch.stack(frames).permute(1, 0, 2, 3), label
        return frames, label
