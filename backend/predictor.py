import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os

class SignPredictor:
    def __init__(self):
        try:
            # Load ASL Model
            self.asl_model = load_model("backend/models/ASL/asl_model.h5")
            self.asl_labels = np.load("backend/models/ASL/labels.npy")
            print("DEBUG: ASL Model loaded successfully")

            # Load ISL Model
            self.isl_model = load_model("backend/models/ISL/isl_alphabet_model.h5")
            self.isl_mean = np.load("backend/models/ISL/mean.npy")
            self.isl_std = np.load("backend/models/ISL/std.npy")
            self.isl_labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
            print("DEBUG: ISL Model loaded successfully")
        except Exception as e:
            print(f"DEBUG: FATAL - Error loading models: {e}")
            raise e

        # MediaPipe Setup
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def predict(self, frame, mode="ASL"):
        # Original projects flip before processing
        frame = cv2.flip(frame, 1)
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        letter, confidence = None, 0.0

        if results.multi_hand_landmarks:
            # Draw landmarks exactly as in original projects
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style()
                )

            if mode == "ASL":
                letter, confidence = self._predict_asl(results.multi_hand_landmarks[0])
            else:
                letter, confidence = self._predict_isl(results)

        return letter, confidence, frame

    def _predict_asl(self, hand_landmarks):
        landmarks = []
        base = hand_landmarks.landmark[0] # The base landmark

        for lm in hand_landmarks.landmark:
            landmarks.extend([
                lm.x - base.x,
                lm.y - base.y,
                lm.z - base.z
            ])
        
        data = np.array([landmarks], dtype=np.float32)
        preds = self.asl_model.predict(data, verbose=0)
        idx = np.argmax(preds)
        return self.asl_labels[idx], float(preds[0][idx])

    def _predict_isl(self, results):
        left, right = [], []

        for idx, hl in enumerate(results.multi_hand_landmarks):
            handed = results.multi_handedness[idx].classification[0].label
            coords = []
            for lm in hl.landmark:
                coords.extend([lm.x, lm.y, lm.z])

            if handed == "Left":
                left = coords
            else:
                right = coords

        # Original ISL logic pads missing hands with zeros
        if not left:
            left = [0.0] * 63
        if not right:
            right = [0.0] * 63

        feature = np.array(left + right, dtype=np.float32)
        # Normalization exactly as in original
        feature = (feature - self.isl_mean) / self.isl_std
        
        preds = self.isl_model.predict(feature.reshape(1, -1), verbose=0)
        idx = np.argmax(preds)
        return self.isl_labels[idx], float(preds[0][idx])
