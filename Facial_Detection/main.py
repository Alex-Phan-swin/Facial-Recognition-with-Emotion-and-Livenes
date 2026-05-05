import cv2
import os
import time
from deepface import DeepFace  # type: ignore

from config import DB_PATH, EXIT_DELAY, DISPLAY_DELAY, MAX_IMAGES
from logger import init_log, log_event

# -------------------------------
# SETUP
# -------------------------------
os.makedirs(DB_PATH, exist_ok=True)
init_log()

# Register state (dynamic → stays here)
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
frame_count = 0

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

print("Press 'q' to quit | Press 'r' to register")

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

    key = cv2.waitKey(1) & 0xFF

    # -------------------------------
    # EXIT KEY
    # -------------------------------
    if key == ord('q'):
        break

    # -------------------------------
    # REGISTER MODE
    # -------------------------------
    if key == ord('r') and not REGISTER_MODE:
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

            time.sleep(0.5)

            if SAVE_COUNT >= MAX_IMAGES:
                print(f"Finished registering {new_person_name}")
                REGISTER_MODE = False
                break

    # -------------------------------
    # FACE RECOGNITION
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
    # DISPLAY
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
    # SHOW FRAME
    # -------------------------------
    cv2.imshow("Face Recognition", frame)

# -------------------------------
# CLEANUP
# -------------------------------
cap.release()
cv2.destroyAllWindows()