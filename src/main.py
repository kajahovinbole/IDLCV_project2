import os
import sys

# legg til prosjektroten i sys.path slik at src.* import fungerer
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import MODEL_NAME
from src.utils.training_loop import main as train_frames

# når du får early/late/3d kan du legge til flere:
# from src.utils.video_training import main as train_videos

if __name__ == "__main__":
    model_name = MODEL_NAME if "MODEL_NAME" in locals() else "per_frame_agg"
    print(f"[INFO] Starter trening med modell: {model_name}")

    if model_name == "per_frame_agg":
        train_frames(model_name=model_name)
    else:
        raise NotImplementedError(f"Treningsløype for {model_name} er ikke laget ennå.")
