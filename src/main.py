import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import MODEL_NAMES
from src.utils.training_loop import main as train_one_model

if __name__ == "__main__":
    # train_one_model("3d_cnn")
    for name in MODEL_NAMES:
        print(f"\n[INFO] Starter trening for modell: {name}\n")
        train_one_model(name)
