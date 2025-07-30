import cv2
import mediapipe as mp
import numpy as np
import json
import time

# ==== CONFIG ====
MODE = 1  # 1 = Pose + Hands (LSTM), 0 = Hands only (MLP)
SAVE_PATH = "landmarks.json"
CAPTURE_TIME = 10  # seconds

# ==== SETUP ====
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
data = []

start_time = time.time()
print("Recording started. Press 'q' to quit early...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    # Draw landmarks for visualization
    annotated_image = frame.copy()
    mp_drawing.draw_landmarks(annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if MODE == 1:
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    # Show annotated frame
    cv2.imshow('Webcam Feed', annotated_image)

    # Extract and save landmarks
    def extract_landmarks(results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, lh, rh]) if MODE == 1 else np.concatenate([lh, rh])

    landmarks = extract_landmarks(results)
    data.append(landmarks.tolist())

    # Stop conditions
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopped by user.")
        break
    if time.time() - start_time > CAPTURE_TIME:
        print("Finished recording.")
        break

# ==== SAVE TO JSON ====
with open(SAVE_PATH, 'w') as f:
    json.dump(data, f)

# Cleanup
cap.release()
cv2.destroyAllWindows()
holistic.close()
print(f"Saved {len(data)} frames to {SAVE_PATH}")
