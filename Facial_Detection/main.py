import cv2
import os
import pandas as pd
import time
from datetime import datetime
from deepface import DeepFace  # type: ignore

# -------------------------------
# CONFIG
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(BASE_DIR)  # goes one level up
DB_PATH = os.path.join(PROJECT_ROOT, "Facial_Detection", "faces_db")
LOG_FILE = os.path.join(PROJECT_ROOT, "Facial_Detection", "log.csv")

os.makedirs(DB_PATH, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

EXIT_DELAY = 3.0  # seconds before confirming exit
DISPLAY_DELAY = 3.0  # seconds to keep name displayed after last detection

#Registering settings
REGISTER_MODE = False
REGISTER_NAME = ""
SAVE_COUNT = 0
MAX_IMAGES = 5

# -------------------------------
# STATE TRACKING
# -------------------------------
last_seen = {}     # {name: last_time_seen}
inside = set()     # confirmed people inside
visible = set()    # people currently displayed
last_visible = {}  # {name: last_time_visible}
frame_count = 0

# -------------------------------
# CREATE LOG FILE
# -------------------------------
if not os.path.exists(LOG_FILE):
    df = pd.DataFrame(columns=["Name", "Event", "Time"])
    df.to_csv(LOG_FILE, index=False)

# -------------------------------
# CAMERA
# -------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

print("Press 'q' to quit")

# -------------------------------
# MAIN LOOP
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    detected_people = set()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )
    face_present = len(faces) > 0

    #Key Controls
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('r') and not REGISTER_MODE:
        new_person_name = input("Enter employee name: ")
        person_path = os.path.join(DB_PATH, new_person_name)
        os.makedirs(person_path, exist_ok=True)

        REGISTER_MODE = True
        SAVE_COUNT = 0
        print(f"Registering {new_person_name}. Please look at the camera...")


    # -------------------------------
    # Saving New faces logic
    # -------------------------------
    if REGISTER_MODE and face_present:
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]

            face = cv2.resize(face, (224, 224))

            save_path = os.path.join(
                DB_PATH,
                new_person_name,
                f"{new_person_name}_{SAVE_COUNT}.jpg"
            )

            cv2.imwrite(save_path, face)
            SAVE_COUNT += 1
            print(f"Saved {save_path}")

            time.sleep(0.5)  # small delay between captures


            if SAVE_COUNT >= MAX_IMAGES:
                print(f"Finished registering {new_person_name}")
                REGISTER_MODE = False
                break

    # -------------------------------
    # RUN DEEPFACE EVERY 10 FRAMES WHEN A FACE IS DETECTED
    # -------------------------------
    if frame_count % 10 == 0 and face_present:
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = DeepFace.find(
                img_path=frame_rgb,
                db_path=DB_PATH,
                enforce_detection=True
            )

            if len(results) > 0 and len(results[0]) > 0:
                for i in range(len(results[0])):
                    identity_path = results[0].iloc[i]['identity']
                    name = identity_path.split(os.sep)[-2]
                    detected_people.add(name)
                    visible.add(name)
                    last_visible[name] = time.time()

        except Exception as e:
            print("DeepFace error:", e)

    

    # -------------------------------
    # DISPLAY NAMES
    # -------------------------------
    for i, name in enumerate(visible):
        cv2.putText(frame, name, (50, 50 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        

    # -------------------------------
    # ENTRY LOGIC
    # -------------------------------
    for person in detected_people:
        last_seen[person] = time.time()

        if person not in inside:
            inside.add(person)

            log = pd.DataFrame([{
                "Name": person,
                "Event": "ENTER",
                "Time": datetime.now()
            }])
            log.to_csv(LOG_FILE, mode='a', header=False, index=False)

            print(person, "ENTER")

    # -------------------------------
    # EXIT LOGIC
    # -------------------------------
    for person in list(inside):
        if person not in last_seen:
            continue

        if time.time() - last_seen[person] > EXIT_DELAY:
            inside.remove(person)

            log = pd.DataFrame([{
                "Name": person,
                "Event": "EXIT",
                "Time": datetime.now()
            }])
            log.to_csv(LOG_FILE, mode='a', header=False, index=False)

            print(person, "EXIT")

    # -------------------------------
    # CLEAN OLD MEMORY
    # -------------------------------
    for person in list(last_seen.keys()):
        if time.time() - last_seen[person] > EXIT_DELAY * 2:
            del last_seen[person]

    # -------------------------------
    # REMOVE FROM DISPLAY
    # -------------------------------
    for person in list(visible):
        if person not in last_visible:
            continue
        if time.time() - last_visible[person] > DISPLAY_DELAY:
            visible.remove(person)
            del last_visible[person]

    # -------------------------------
    # SHOW FRAME
    # -------------------------------
    cv2.imshow("Face Recognition", frame)

    #Detecting key press 
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# -------------------------------
# CLEANUP
# -------------------------------
cap.release()
cv2.destroyAllWindows()