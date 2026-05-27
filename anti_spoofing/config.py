from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent

# paths
DATA_ROOT       = _PROJECT_ROOT / "Data_anti_spoofing"
FRAMES_ROOT     = _PROJECT_ROOT / "Data_anti_spoofing_frames"
MODEL_SAVE_PATH = _HERE / "liveness_model.keras"

# which video folders count as real vs fake
REAL_DIRS = ["live_selfie", "live_video"]
FAKE_DIRS = ["cut-out printouts", "printouts", "replay"]

# frames to extract per video
FRAMES_PER_VIDEO = 50

# train / val / test split
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
SEED       = 42

# training settings
IMG_SIZE      = 224
BATCH_SIZE    = 32
NUM_EPOCHS    = 30
LR            = 3e-4
LR_PATIENCE   = 5
EARLY_STOP    = 10
WARMUP_EPOCHS = 5

# probability cutoff — above this = live, below = spoof
LIVENESS_THRESHOLD = 0.4
