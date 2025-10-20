# to initialize the run
# ◦ Parser kommandolinjeargumenter (eller laster inn config.yaml).
# ◦ Velger riktig modell (f.eks. basert på et argument som model_name='LateFusion').
# ◦ Initialiserer datasett og dataloadere ved hjelp av datasets.py.
# ◦ Definerer tapfunksjon (f.eks. nn.CrossEntropyLoss) og optimalisator (f.eks. Adam eller SGD+Momentum).
# ◦ Kaller treningsfunksjonen fra utils/training_loop.py.

import sys
import os
import torch.nn as nn
import torch.nn as nn
from config import NUM_CLASSES
from src.utils.model_factory import select_model

# legg til prosjektroten i sys.path slik at config.py kan importeres
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# #change model name here to test other models
model = select_model("per_frame_agg", num_classes=NUM_CLASSES)


class VideoModel(nn.Module):
    def __init__(self):
        super(VideoModel, self).__init__()
        # Modellarkitekturdefinisjon her

    def forward(self, x):
        # Fremoverpasseringslogikk her
        return x
