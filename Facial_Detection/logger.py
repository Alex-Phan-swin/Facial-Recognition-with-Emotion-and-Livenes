import os
import pandas as pd
from datetime import datetime
from .config import LOG_FILE

# -------------------------------
# INITIALIZE LOG FILE
# -------------------------------
def init_log():
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=["Name", "Event", "Time"])
        df.to_csv(LOG_FILE, index=False)


# -------------------------------
# WRITE LOG
# -------------------------------
def log_event(name, event):
    log = pd.DataFrame([{
        "Name": name,
        "Event": event,
        "Time": datetime.now()
    }])

    log.to_csv(LOG_FILE, mode='a', header=False, index=False)