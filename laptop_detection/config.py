from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE         = Path(__file__).resolve().parent   # .../laptop_detection/
_PROJECT_ROOT = _HERE.parent                      # .../Porject/

# Raw laptop images (positive examples — all 17 angle classes = laptop present)
LAPTOP_DATA_ROOT = _PROJECT_ROOT / "laptop_data" / "images"

# Face recognition images used as negative examples (no laptop in these images)
NEGATIVE_DATA_ROOT = _PROJECT_ROOT / "Data" / "classification_data" / "train_data"

# Where the trained model is saved
MODEL_SAVE_PATH = _HERE / "laptop_model.keras"

# ── Dataset ───────────────────────────────────────────────────────────────────
# How many negative (no-laptop) images to sample.
# Matched to the number of positive images (~980) to keep classes balanced.
NUM_NEGATIVES = 980

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
SEED       = 42

# ── Model / training ──────────────────────────────────────────────────────────
IMG_SIZE     = 224
BATCH_SIZE   = 32
NUM_EPOCHS   = 20
LR           = 3e-4
LR_PATIENCE  = 4
EARLY_STOP   = 8
WARMUP_EPOCHS = 5

# ── Inference ─────────────────────────────────────────────────────────────────
# P(laptop) >= threshold → "LAPTOP DETECTED"
DETECTION_THRESHOLD = 0.9

# Only run detection every N webcam frames (saves compute)
DETECT_EVERY_N_FRAMES = 15
