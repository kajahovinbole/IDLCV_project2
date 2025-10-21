SEED = 42

# datasett path
DATA_ROOT = "/dtu/datasets1/02516/ufc10"
# DATA_ROOT = "/dtu/datasets1/02516/ucf101_noleakage" #til senere oppagve

# models
MODEL_NAMES = ["per_frame", "late_fusion", "early_fusion"]
# MODEL_NAMES = ["per_frame_agg", "early_fusion", "late_fusion", "c3d"]

# antall klasser i datasettet
NUM_CLASSES = 10

# Hvor mange blokker i backbone som skal trenes (0 = frys alt)
TRAINABLE_BLOCKS = 0

# Hvis True: frys backbone og tren kun klassifiseringshodet
FREEZE_BACKBONE = True

# treningsparametre
N_FRAMES = 10
BATCH_SIZE = 64
NUM_WORKERS = 4
LR = 1e-4
EPOCHS = 1
