from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anti_spoofing.config import MODEL_SAVE_PATH, IMG_SIZE, LIVENESS_THRESHOLD


@dataclass
class LivenessResult:
    is_live:    bool
    confidence: float

    @property
    def label(self) -> str:
        return "LIVE" if self.is_live else "SPOOF"

    def __str__(self) -> str:
        return f"{self.label}  (confidence={self.confidence:.3f})"


class LivenessChecker:
    def __init__(
        self,
        model_path: Path | str | None = None,
        threshold: float = LIVENESS_THRESHOLD,
    ):
        self.threshold = threshold
        self._model = self._load_model(Path(model_path or MODEL_SAVE_PATH))

    def check(self, face_bgr: np.ndarray) -> LivenessResult:
        # run one face through the model
        tensor = self._preprocess(face_bgr)
        prob   = float(self._model.predict(tensor, verbose=0)[0, 0])
        return LivenessResult(is_live=prob >= self.threshold, confidence=prob)

    def check_batch(self, faces_bgr: list[np.ndarray]) -> list[LivenessResult]:
        # run multiple faces at once (faster than calling check() in a loop)
        if not faces_bgr:
            return []
        batch = np.stack([self._preprocess_array(f) for f in faces_bgr])
        probs = self._model.predict(batch, verbose=0).flatten().tolist()
        return [
            LivenessResult(is_live=p >= self.threshold, confidence=p) for p in probs
        ]

    @staticmethod
    def _load_model(path: Path) -> keras.Model:
        if not path.exists():
            raise FileNotFoundError(
                f"Model weights not found at: {path}\n"
                "Train the model first:  python -m anti_spoofing.train"
            )
        return keras.models.load_model(str(path))

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        return np.expand_dims(self._preprocess_array(face_bgr), axis=0)

    @staticmethod
    def _preprocess_array(face_bgr: np.ndarray) -> np.ndarray:
        # BGR to RGB, resize, float32
        face_rgb     = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
        return face_resized.astype(np.float32)


# runs only when called directly: python anti_spoofing/inference.py
if __name__ == "__main__":
    checker = LivenessChecker()

    if len(sys.argv) > 1:
        # test on an image file passed as an argument
        frame = cv2.imread(sys.argv[1])
        if frame is None:
            print(f"Could not read image: {sys.argv[1]}")
            sys.exit(1)
        result = checker.check(frame)
        print(f"Image : {sys.argv[1]}")
        print(f"Result: {result}")
    else:
        # open the webcam for a live test
        print("Starting webcam demo — press Q to quit")
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            result = checker.check(frame)
            color  = (0, 255, 0) if result.is_live else (0, 0, 255)
            cv2.putText(frame, f"{result.label} ({result.confidence:.2f})",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            cv2.imshow("Liveness Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
