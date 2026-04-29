import cv2
from collections import deque
from common.camera import get_camera
from models.ASL import realtime_core as asl
from models.ISL import realtime_core as isl

# ================= CONFIG =================
CONF_THRESHOLD = 0.75
BUFFER_SIZE = 15
STABLE_FRAMES = 7

# ================= STATE ==================
prediction_buffer = deque(maxlen=BUFFER_SIZE)
sentence = ""
last_added = ""

mode = "ASL"
predictor = asl

cap = get_camera()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # ===== PREDICTION =====
    letter, conf, frame = predictor.predict(frame)

    # ===== CONFIDENCE FILTER =====
    if letter is not None and conf >= CONF_THRESHOLD:
        prediction_buffer.append(letter)

    # ===== TEMPORAL STABILITY =====
    stable_letter = None
    if prediction_buffer:
        most_common = max(set(prediction_buffer), key=prediction_buffer.count)
        if prediction_buffer.count(most_common) >= STABLE_FRAMES:
            stable_letter = most_common

    # ===== SENTENCE FORMATION =====
    if stable_letter and stable_letter != last_added:
        sentence += stable_letter
        last_added = stable_letter

    # ===== UI =====
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (20, 20, 20), -1)
    cv2.putText(
        frame, f"MODE: {mode}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    if stable_letter:
        cv2.putText(
            frame, stable_letter,
            (frame.shape[1] // 2 - 60, frame.shape[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            4,
            (0, 255, 0),
            6
        )

    cv2.putText(
        frame, sentence[-30:],
        (20, frame.shape[0] - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2
    )

    cv2.imshow("ASL + ISL Combined (Final)", frame)

    # ===== KEY CONTROLS =====
    key = cv2.waitKey(1) & 0xFF

    if key == ord('a'):
        mode = "ASL"
        predictor = asl
        prediction_buffer.clear()
        last_added = ""

    elif key == ord('i'):
        mode = "ISL"
        predictor = isl
        prediction_buffer.clear()
        last_added = ""

    elif key == ord(' '):  # SPACE
        sentence += " "
        last_added = ""

    elif key == ord('b'):  # BACKSPACE
        sentence = sentence[:-1]
        last_added = ""

    elif key == ord('c'):  # CLEAR
        sentence = ""
        last_added = ""

    elif key == ord('q'):  # QUIT
        break

cap.release()
cv2.destroyAllWindows()
