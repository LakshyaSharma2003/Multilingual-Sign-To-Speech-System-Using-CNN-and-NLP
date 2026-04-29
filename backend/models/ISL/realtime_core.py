import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# ===== LOAD MODEL =====
model = load_model("models/ISL/isl_alphabet_model.h5")
mean = np.load("models/ISL/mean.npy")   # shape (126,)
std = np.load("models/ISL/std.npy")     # shape (126,)

LABELS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# ===== MEDIAPIPE (MATCH TRAINING) =====
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # IMPORTANT
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

def predict(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None, 0.0, frame

    features = []

    # TAKE UP TO 2 HANDS (ORDER MATTERS)
    for hand_landmarks in results.multi_hand_landmarks[:2]:
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        for lm in hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])

    # PAD IF ONLY ONE HAND DETECTED
    while len(features) < 126:
        features.append(0.0)

    data = np.array(features, dtype=np.float32)

    # EXACT SAME NORMALIZATION AS TRAINING
    data = (data - mean) / (std + 1e-6)
    data = np.expand_dims(data, axis=0)

    preds = model.predict(data, verbose=0)
    idx = np.argmax(preds)

    return LABELS[idx], float(preds[0][idx]), frame
