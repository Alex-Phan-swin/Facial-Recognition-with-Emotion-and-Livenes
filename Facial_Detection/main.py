import cv2
import os
import time
import numpy as np
from tensorflow.keras.models import load_model

import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from laptop_detection import LaptopDetector

from config import DB_PATH, EXIT_DELAY, DISPLAY_DELAY, MAX_IMAGES
from logger import init_log, log_event

# -------------------------------
# SETUP
# -------------------------------
# -------------------------------
# DATABASE PATH
# -------------------------------
DB_PATH = r"C:\Users\minhp\Music\Year 3\COS30082\project\Facial-Recognition-with-Emotion-and-Livenes\Facial_Detection\faces_db"

os.makedirs(DB_PATH, exist_ok=True)
init_log()

# -------------------------------
# REGISTER STATE
# -------------------------------
REGISTER_MODE = False
SAVE_COUNT = 0
new_person_name = ""

# -------------------------------
# STATE TRACKING
# -------------------------------
last_seen = {}
inside = set()
visible = set()
last_visible = {}

# -------------------------------
# CAMERA
# -------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# -------------------------------
# FACE DETECTOR
# -------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------------------------------
# LOAD CLASSIFIER
# -------------------------------
classifier = load_model("face_classifier.keras")

# ONLY LOAD FOLDERS
class_names = sorted(
    [name for name in os.listdir(DB_PATH) if os.path.isdir(os.path.join(DB_PATH, name))]
)

print("Loaded classes:", class_names)

print("Press 'q' to quit | Press 'r' to register")

# -------------------------------
# LAPTOP DETECTOR
# -------------------------------
laptop_detector = LaptopDetector()
laptop_result = None
frame_count = 0

# -------------------------------
# MAIN LOOP
# -------------------------------
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Run laptop check every 15 frames — saves CPU without missing detections
    if frame_count % 15 == 0:
        laptop_result = laptop_detector.check(frame)

    detected_people = set()

    # -------------------------------
    # FACE DETECTION
    # -------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )

    face_present = len(faces) > 0

    # -------------------------------
    # KEY INPUT
    # -------------------------------
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    # -------------------------------
    # REGISTER MODE
    # -------------------------------
    if key == ord("r") and not REGISTER_MODE:
        new_person_name = input("Enter employee name: ")

        person_path = os.path.join(DB_PATH, new_person_name)

        os.makedirs(person_path, exist_ok=True)

        REGISTER_MODE = True
        SAVE_COUNT = 0

        print(f"Registering {new_person_name}...")

    # -------------------------------
    # SAVE FACES
    # -------------------------------
    if REGISTER_MODE and face_present:
        for x, y, w, h in faces:
            face = frame[y : y + h, x : x + w]

            # MUST MATCH TRAINING SIZE
            face = cv2.resize(face, (80, 80))

            save_path = os.path.join(
                DB_PATH, new_person_name, f"{new_person_name}_{SAVE_COUNT}.jpg"
            )

            cv2.imwrite(save_path, face)

            SAVE_COUNT += 1

            print(f"Saved {save_path}")

            time.sleep(0.3)

            if SAVE_COUNT >= MAX_IMAGES:
                print(f"Finished registering {new_person_name}")

                REGISTER_MODE = False

                # reload class names
                class_names = sorted(
                    [
                        name
                        for name in os.listdir(DB_PATH)
                        if os.path.isdir(os.path.join(DB_PATH, name))
                    ]
                )

                print("Updated classes:", class_names)

                break

    # -------------------------------
    # FACE RECOGNITION
    # -------------------------------
    for x, y, w, h in faces:
        face = frame[y : y + h, x : x + w]

        try:
            # preprocess
            face_resized = cv2.resize(face, (80, 80))
            face_resized = face_resized.astype("float32") / 255.0
            face_resized = np.expand_dims(face_resized, axis=0)

            # prediction
            preds = classifier.predict(face_resized, verbose=0)[0]

            confidence = float(np.max(preds))
            label_index = int(np.argmax(preds))

            # safety check
            if label_index >= len(class_names):
                label = "Unknown"

            elif confidence < 0.70:
                label = "Unknown"

            else:
                label = class_names[label_index]

                detected_people.add(label)
                visible.add(label)
                last_visible[label] = time.time()

            # -------------------------------
            # DRAW BOX
            # -------------------------------
            color = (0, 0, 255) if label == "Unknown" else (0, 255, 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            cv2.putText(
                frame,
                f"{label} ({confidence:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

        except Exception as e:
            print("Prediction error:", e)

    # -------------------------------
    # DISPLAY LIST
    # -------------------------------
    for i, name in enumerate(visible):
        cv2.putText(
            frame, name, (50, 50 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

    # -------------------------------
    # ENTRY LOGIC
    # -------------------------------
    for person in detected_people:
        last_seen[person] = time.time()

        if person not in inside:
            inside.add(person)

            log_event(person, "ENTER")

            print(person, "ENTER")

    # -------------------------------
    # EXIT LOGIC
    # -------------------------------
    for person in list(inside):
        if person not in last_seen:
            continue

        if time.time() - last_seen[person] > EXIT_DELAY:
            inside.remove(person)

            log_event(person, "EXIT")

            print(person, "EXIT")

    # -------------------------------
    # CLEAN MEMORY
    # -------------------------------
    for person in list(last_seen.keys()):
        if time.time() - last_seen[person] > EXIT_DELAY * 2:
            del last_seen[person]

    # -------------------------------
    # DISPLAY CLEANUP
    # -------------------------------
    for person in list(visible):
        if person not in last_visible:
            continue

        if time.time() - last_visible[person] > DISPLAY_DELAY:
            visible.remove(person)

            del last_visible[person]

    # -------------------------------
    # LAPTOP DETECTION BANNER
    # -------------------------------
    if laptop_result and laptop_result.detected:
        cv2.putText(
            frame,
            f"LAPTOP DETECTED ({laptop_result.confidence:.2f})",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),  # yellow
            2,
        )

    # -------------------------------
    # SHOW FRAME
    # -------------------------------
    cv2.imshow("Face Recognition", frame)

# -------------------------------
# CLEANUP
# -------------------------------
cap.release()
cv2.destroyAllWindows()
