import cv2
import tkinter as tk
import tkinter.simpledialog as sd
import numpy as np
from PIL import Image, ImageTk
import os
import time
from collections import deque

from Facial_Detection import main
from Facial_Detection.logger import log_event
from Facial_Detection.config import DB_PATH, MAX_IMAGES
from emotion_prediction import predict_emotion
from laptop_detection import LaptopDetector
from anti_spoofing import LivenessChecker

WIN_W = 960
WIN_H = 640


FACE_BOX_TIMEOUT = 1.5


class FaceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition")
        self.root.configure(bg="#1a1a1a")
        self.root.resizable(False, False)

        # Models
        self.liveness_checker = LivenessChecker()
        self.laptop_detector  = LaptopDetector()

        main.build_face_database()

        # Layout
        self.video_label = tk.Label(root, bg="black", borderwidth=0)
        self.video_label.pack()

        bar = tk.Frame(root, bg="#1a1a1a", height=60)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        btn_style = dict(font=("Helvetica", 13, "bold"), bd=0,
                         padx=24, pady=8, cursor="hand2", relief="flat")

        self.quit_btn = tk.Button(
            bar, text="Quit",
            bg="#c0392b", fg="white", activebackground="#e74c3c",
            command=self.quit, **btn_style
        )
        self.quit_btn.pack(side="left", padx=20, pady=10)

        self.reg_btn = tk.Button(
            bar, text="Register Person",
            bg="#2471a3", fg="white", activebackground="#2e86c1",
            command=self.start_registration, **btn_style
        )
        self.reg_btn.pack(side="right", padx=20, pady=10)

        # App state
        self.name          = "Unknown"
        self.emotion       = "-"
        self.liveness      = "-"
        self.laptop_result = None
        self.register_mode = False
        self.save_count    = 0
        self.last_save_time = 0
        self.new_person    = ""
        self.frame_count   = 0
        self.status_msg    = ""
        self.status_until  = 0

        # Stable face box
        self.last_face      = None   # (x, y, w, h)
        self.last_face_time = 0

        # OpenCV
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open camera")
            return

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.update_frame()

    # Drawing helpers

    def draw_rounded_rect(self, img, x1, y1, x2, y2, r, color, thickness):
        cv2.line(img, (x1+r, y1),  (x2-r, y1),  color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1+r, y2),  (x2-r, y2),  color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1,  y1+r), (x1,  y2-r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2,  y1+r), (x2,  y2-r), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1+r, y1+r), (r,r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2-r, y1+r), (r,r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1+r, y2-r), (r,r), 90,  0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2-r, y2-r), (r,r), 0,   0, 90, color, thickness, cv2.LINE_AA)

    def draw_pill(self, img, text, cx, cy, bg_bgr, text_bgr=(30, 30, 30)):
        if not text or text == "-":
            return
        font      = cv2.FONT_HERSHEY_SIMPLEX
        scale     = 0.52
        thickness = 1
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        px, py = 16, 7
        bx1 = cx - tw // 2 - px
        by1 = cy - th // 2 - py
        bx2 = cx + tw // 2 + px
        by2 = cy + th // 2 + py
        r   = (by2 - by1) // 2
        cv2.rectangle(img, (bx1+r, by1), (bx2-r, by2), bg_bgr, -1)
        cv2.circle(img, (bx1+r, cy), r, bg_bgr, -1)
        cv2.circle(img, (bx2-r, cy), r, bg_bgr, -1)
        cv2.putText(img, text, (bx1+px, cy + th//2),
                    font, scale, text_bgr, thickness, cv2.LINE_AA)

    def draw_scan_dots(self, img, x1, y1, x2, y2):
        mx      = (x1 + x2) // 2
        spacing = 14
        overlay = img.copy()
        for dy in range(y1 + spacing, y2, spacing):
            for dx in range(mx + spacing, x2, spacing):
                if 0 <= dy < img.shape[0] and 0 <= dx < img.shape[1]:
                    cv2.circle(overlay, (dx, dy), 1, (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)

    # Frame loop

    def update_frame(self):
        try:
            self._process_frame()
        except Exception as e:
            print("Frame error:", e)
        finally:
            self.root.after(15, self.update_frame)

    def _process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        self.frame_count += 1
        if self.name != "Unknown":
            main.last_seen[self.name] = time.time()
            if self.name not in main.inside:
                main.inside.add(self.name)
                log_event(self.name, "ENTER")
                print(self.name, "ENTER")

        # Exit logic
        for person in list(main.inside):
            if person not in main.last_seen:
                continue
            if time.time() - main.last_seen[person] > 3.0:
                main.inside.remove(person)
                log_event(person, "EXIT")
                print(person, "EXIT")
        # Clean memory
        for person in list(main.last_seen.keys()):
            if time.time() - main.last_seen[person] > 6.0:
                del main.last_seen[person]

        # Resize and mirror
        frame = cv2.resize(frame, (WIN_W, WIN_H))
        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]

        # Keep a clean copy for all model inference
        raw = frame.copy()

        # ── Liveness
        liveness_result = self.liveness_checker.check(raw)
        self.liveness   = liveness_result.label

        # ── Laptop detection
        if self.frame_count % 15 == 0:
            self.laptop_result = self.laptop_detector.check(raw)

        # ── Vignette
        mask = np.zeros((fh, fw), dtype=np.float32)
        cv2.ellipse(mask, (fw//2, fh//2),
                    (int(fw*0.65), int(fh*0.65)), 0, 0, 360, 1.0, -1)
        mask    = cv2.GaussianBlur(mask, (201, 201), 0)
        vignette = np.stack([mask]*3, axis=-1)
        display  = (frame * (0.55 + 0.45 * vignette)).clip(0, 255).astype(np.uint8)

        # ── Face detection
        gray  = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        # Update last known face position
        if len(faces) > 0:
            self.last_face      = faces[0]
            self.last_face_time = time.time()

        # Use last known face if current frame has no detection (stable box)
        active_face = None
        if len(faces) > 0:
            active_face = faces[0]
        elif self.last_face is not None:
            if time.time() - self.last_face_time < FACE_BOX_TIMEOUT:
                active_face = self.last_face   # keep box visible briefly

        # Per-face processing
        if active_face is not None:
            x, y, w, h = active_face
            pad = int(w * 0.25)
            x1, y1 = max(0, x - pad),    max(0, y - pad)
            x2, y2 = min(fw, x+w + pad), min(fh, y+h + pad)

            self.draw_scan_dots(display, x1, y1, x2, y2)
            self.draw_rounded_rect(display, x1, y1, x2, y2,
                                   r=20, color=(220, 220, 220), thickness=2)

            # Recognition + emotion every 10 frames
            if self.frame_count % 10 == 0 and not self.register_mode:
                face_crop = raw[y:y+h, x:x+w]   # crop from raw, not display

                self.name, conf = main.predict_face(face_crop)
                print(f"[PREDICT] {self.name}  conf={conf:.3f}")

                self.emotion, emo_conf = predict_emotion(face_crop)
                print(f"[EMOTION] {self.emotion}  conf={emo_conf:.3f}")

            # Registration capture — 0.5s apart, from raw frame
            if self.register_mode:
                now = time.time()
                if now - self.last_save_time >= 0.5:
                    face_crop = raw[y:y+h, x:x+w]
                    face_crop = cv2.resize(face_crop, (224, 224))
                    save_path = os.path.join(
                        DB_PATH, self.new_person,
                        f"{self.new_person}_{self.save_count}.jpg"
                    )
                    ok = cv2.imwrite(save_path, face_crop)
                    print(f"[REGISTER] {self.save_count+1}/{MAX_IMAGES} — {save_path} ok={ok}")
                    self.save_count    += 1
                    self.last_save_time = now
                    self.status_msg    = f"Capturing {self.new_person}... {self.save_count}/{MAX_IMAGES}"
                    self.status_until  = time.time() + 2

                    if self.save_count >= MAX_IMAGES:
                        self.register_mode = False
                        self.reg_btn.config(bg="#2471a3", text="Register Person")
                        self.status_msg   = f"Registered: {self.new_person} ✓"
                        self.status_until = time.time() + 3
                        main.build_face_database()

            # Pills
            # Name — top left
            self.draw_pill(display, self.name,
                           cx=x1 + 60, cy=y1,
                           bg_bgr=(175, 195, 200),
                           text_bgr=(30, 30, 30))

            # Emotion — top right
            emotion_label = self.emotion if self.emotion not in ("-", "") else "Emotion"
            self.draw_pill(display, emotion_label,
                           cx=x2 - 60, cy=y1,
                           bg_bgr=(30, 165, 225),
                           text_bgr=(255, 255, 255))

            # Liveness — bottom centre
            liveness_color = (50, 200, 50) if self.liveness == "LIVE" else (30, 30, 180)
            self.draw_pill(display, self.liveness,
                           cx=(x1 + x2) // 2, cy=y2,
                           bg_bgr=liveness_color,
                           text_bgr=(255, 255, 255))

        # Laptop banner
        if self.laptop_result and self.laptop_result.detected:
            cv2.putText(display,
                        f"LAPTOP DETECTED ({self.laptop_result.confidence:.2f})",
                        (10, fh - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # Register pulsing border
        if self.register_mode:
            alpha = 0.5 + 0.5 * np.sin(time.time() * 6)
            col   = (int(80*alpha), int(180*alpha), int(255*alpha))
            cv2.rectangle(display, (4, 4), (fw-4, fh-4), col, 3)

        # Status message
        if self.status_msg and time.time() < self.status_until:
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, _), _ = cv2.getTextSize(self.status_msg, font, 0.75, 2)
            cv2.putText(display, self.status_msg,
                        ((fw - tw)//2, 40), font, 0.75,
                        (100, 230, 100), 2, cv2.LINE_AA)

        # Push to Tkinter
        img = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(img))
        self.video_label.imgtk = img
        self.video_label.configure(image=img)

    # Interaction

    def start_registration(self):
        name = sd.askstring("Register", "Enter person's name:")
        if name and name.strip():
            self.new_person     = name.strip()
            self.save_count     = 0
            self.last_save_time = 0
            self.register_mode  = True
            self.status_msg     = f"Look at the camera — capturing {self.new_person}..."
            self.status_until   = time.time() + 60
            self.reg_btn.config(bg="#e67e22", text="Registering...")
            os.makedirs(os.path.join(DB_PATH, self.new_person), exist_ok=True)

    def quit(self):
        for person in list(main.inside):
            log_event(person, "EXIT")
            print(person, "EXIT")
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app  = FaceGUI(root)
    root.mainloop()