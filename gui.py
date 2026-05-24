import cv2
import tkinter as tk
import tkinter.simpledialog as sd
import numpy as np
from PIL import Image, ImageTk
import os
import time

from Facial_Detection import main
from Facial_Detection.logger import log_event
from Facial_Detection.config import DB_PATH, MAX_IMAGES
from emotion_prediction import predict_emotion

WIN_W = 960
WIN_H = 640


class FaceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition")
        self.root.configure(bg="#1a1a1a")
        self.root.resizable(False, False)

        main.build_face_database()

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
        self.name         = "Unknown"
        self.emotion      = "-"
        self.liveness     = "-"
        self.register_mode = False
        self.save_count    = 0
        self.new_person    = ""
        self.frame_count   = 0
        self.status_msg    = ""
        self.status_until  = 0

        # OpenCV
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open camera"); return

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.update_frame()



    def draw_rounded_rect(self, img, x1, y1, x2, y2, r, color, thickness):
        cv2.line(img,  (x1+r, y1),  (x2-r, y1),  color, thickness, cv2.LINE_AA)
        cv2.line(img,  (x1+r, y2),  (x2-r, y2),  color, thickness, cv2.LINE_AA)
        cv2.line(img,  (x1,  y1+r), (x1,  y2-r), color, thickness, cv2.LINE_AA)
        cv2.line(img,  (x2,  y1+r), (x2,  y2-r), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1+r, y1+r), (r,r), 180,  0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2-r, y1+r), (r,r), 270,  0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1+r, y2-r), (r,r), 90,   0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2-r, y2-r), (r,r), 0,    0, 90, color, thickness, cv2.LINE_AA)

    def draw_pill(self, img, text, cx, cy, bg_bgr, text_bgr=(30, 30, 30)):
        font      = cv2.FONT_HERSHEY_SIMPLEX
        scale     = 0.52
        thickness = 1
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        px, py = 16, 7
        x1 = cx - tw // 2 - px
        y1 = cy - th // 2 - py
        x2 = cx + tw // 2 + px
        y2 = cy + th // 2 + py
        r  = (y2 - y1) // 2
        cv2.rectangle(img, (x1+r, y1), (x2-r, y2), bg_bgr, -1)
        cv2.circle(img, (x1+r, cy), r, bg_bgr, -1)
        cv2.circle(img, (x2-r, cy), r, bg_bgr, -1)
        cv2.putText(img, text, (x1+px, cy + th//2),
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


    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self.update_frame)
            return

        self.frame_count += 1

        frame = cv2.resize(frame, (WIN_W, WIN_H))
        frame = cv2.flip(frame, 1)
        clean_frame = frame.copy()
        fh, fw = frame.shape[:2]

        # Vignette
        mask = np.zeros((fh, fw), dtype=np.float32)
        cv2.ellipse(mask, (fw//2, fh//2),
                    (int(fw*0.65), int(fh*0.65)), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (201, 201), 0)
        vignette = np.stack([mask]*3, axis=-1)
        frame = (frame * (0.55 + 0.45 * vignette)).clip(0, 255).astype(np.uint8)

        # ── Face detection ───────────────────────────────────────────────────
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        for (x, y, w, h) in faces[:1]:
            pad = int(w * 0.25)
            x1, y1 = max(0, x - pad),    max(0, y - pad)
            x2, y2 = min(fw, x+w + pad), min(fh, y+h + pad)

            self.draw_scan_dots(frame, x1, y1, x2, y2)
            self.draw_rounded_rect(frame, x1, y1, x2, y2,
                                   r=20, color=(220, 220, 220), thickness=2)

            # Recognition every 10 frames
            if self.frame_count % 10 == 0 and not self.register_mode:
                face_crop = frame[y:y+h, x:x+w]
                self.name, conf = main.predict_face(face_crop)
                print("Prediction:", self.name, round(float(conf), 3))
                self.emotion, emo_conf  = predict_emotion(face_crop)
                print("Emotion:", self.emotion, round(float(emo_conf), 3))
                # self.liveness = your_liveness_fn(face_crop)

            # Registration capture
            if self.register_mode:
                face_crop = cv2.resize(frame[y:y+h, x:x+w], (224, 224))
                save_path = os.path.join(DB_PATH, self.new_person,
                            f"{self.new_person}_{self.save_count}.jpg")
                cv2.imwrite(save_path, face_crop)
                print(f"Saved: {save_path}")
                self.save_count += 1
                if self.save_count >= MAX_IMAGES:
                    self.register_mode = False
                    self.reg_btn.config(bg="#2471a3", text="Register")
                    self.status_msg   = f"Registered: {self.new_person}"
                    self.status_until = time.time() + 3
                    main.build_face_database()

            # Name
            self.draw_pill(frame, self.name,
                           cx=x1 + 55, cy=y1,
                           bg_bgr=(175, 195, 200),
                           text_bgr=(30, 30, 30))

            # Emotion
            emotion_label = str(self.emotion) if self.emotion != "-" else "Emotion"
            self.draw_pill(frame, emotion_label,
                           cx=x2 - 55, cy=y1,
                           bg_bgr=(30, 165, 225),
                           text_bgr=(255, 255, 255))

        if self.register_mode:
            alpha = 0.5 + 0.5 * np.sin(time.time() * 6)
            col   = (int(80*alpha), int(180*alpha), int(255*alpha))
            cv2.rectangle(frame, (4, 4), (fw-4, fh-4), col, 3)

        if self.status_msg and time.time() < self.status_until:
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, _), _ = cv2.getTextSize(self.status_msg, font, 0.75, 2)
            cv2.putText(frame, self.status_msg,
                        ((fw - tw)//2, 40), font, 0.75,
                        (100, 230, 100), 2, cv2.LINE_AA)

        # Push to Tkinter
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(img))
        self.video_label.imgtk = img
        self.video_label.configure(image=img)

        self.root.after(15, self.update_frame)


    def start_registration(self):
        name = sd.askstring("Register", "Enter person's name:")
        if name and name.strip():
            self.new_person    = name.strip()
            self.save_count    = 0
            self.register_mode = True
            self.status_msg    = f"Registering {self.new_person}..."
            self.status_until  = time.time() + 60
            self.reg_btn.config(bg="#e67e22", text="Registering...")
            os.makedirs(os.path.join(DB_PATH, self.new_person), exist_ok=True)

    def quit(self):
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app  = FaceGUI(root)
    root.mainloop()