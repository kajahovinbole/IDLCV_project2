SEED = 42

# datasett path
# DATA_ROOT = "/dtu/datasets1/02516/ufc10"
DATA_ROOT = "/dtu/datasets1/02516/ucf101_noleakage" 

# models
MODEL_NAMES = ["per_frame", "late_fusion", "early_fusion", "3d_cnn"]

# antall klasser i datasettet
NUM_CLASSES = 10


# Hvis True: frys backbone og tren kun klassifiseringshodet
FREEZE_BACKBONE = True
# Hvor mange blokker i backbone som skal trenes (0 = frys alt) --> når FREEZE_BACKBONE=True
TRAINABLE_BLOCKS = 0

# treningsparametre
N_FRAMES = 10
BATCH_SIZE = 64
NUM_WORKERS = 4
LR = 1e-4
EPOCHS = 50
DROPOUT = 0.2
