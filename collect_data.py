
import cv2
import mediapipe as mp
import csv
import os
import numpy as np

CSV_FILE = "gestures.csv"
GESTURES = ["FIST", "OPEN_PALM", "THUMBS_UP", "PEACE", "ONE_FINGER"]
SAMPLES_PER_GESTURE = 100

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def extract_features(hand_landmarks):
    points = np.array(
        [[p.x, p.y, p.z] for p in hand_landmarks.landmark],
        dtype=np.float32
    )

    # Wrist becomes origin
    points -= points[0].copy()

    # Normalize by hand size
    scale = np.max(np.abs(points[:, :2]))
    if scale < 1e-6:
        return None

    points /= scale
    return points.flatten().tolist()


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

columns = ["label"] + [f"f{i}" for i in range(63)]

try:
    # Start fresh: the new model needs consistently normalized features
    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)

        with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        ) as hands:

            stop = False

            for gesture in GESTURES:
                count = 0
                started = False

                print(f"\nPrepare {gesture}")
                print("Make the gesture and press S in the camera window.")

                while count < SAMPLES_PER_GESTURE:
                    ok, frame = cap.read()
                    if not ok:
                        stop = True
                        break

                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = hands.process(rgb)

                    if result.multi_hand_landmarks:
                        hand = result.multi_hand_landmarks[0]
                        mp_draw.draw_landmarks(
                            frame, hand, mp_hands.HAND_CONNECTIONS
                        )

                        if started:
                            features = extract_features(hand)
                            if features is not None:
                                writer.writerow([gesture] + features)
                                count += 1

                                if count % 10 == 0:
                                    file.flush()

                    cv2.putText(
                        frame, f"{gesture}: {count}/{SAMPLES_PER_GESTURE}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame, "S = start | Q = quit",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2
                    )

                    cv2.imshow("SignSense AI - Collect Data", frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("s"):
                        started = True
                    elif key == ord("q"):
                        stop = True
                        break

                print(f"{gesture}: {count} samples collected")

                if stop:
                    break

finally:
    cap.release()
    cv2.destroyAllWindows()

print(f"Dataset saved: {os.path.abspath(CSV_FILE)}")
