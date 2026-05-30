import os

# -------------------------------
# PATH SETUP
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DB_PATH = os.path.join(PROJECT_ROOT, "Facial_Detection", "faces_db")
LOG_FILE = os.path.join(PROJECT_ROOT, "Facial_Detection", "log.csv")

# -------------------------------
# TIMING CONFIG
# -------------------------------
EXIT_DELAY = 3.0
DISPLAY_DELAY = 3.0

# -------------------------------
# REGISTRATION CONFIG
# -------------------------------
MAX_IMAGES = 5