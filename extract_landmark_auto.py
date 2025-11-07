# extract_landmarks_both_hands_v2.py
import cv2
import mediapipe as mp
import pandas as pd
import os

# -----------------------------
# Configurations
# -----------------------------
LABEL = input("Enter the label for this sign: ").upper()
MAX_SAMPLES = int(input("Enter number of samples to capture: "))
OUTPUT_DIR = "/Users/pavitraa/Desktop/SIGNLANG_V2/csv_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, f"{LABEL}.csv")  # one CSV per label

# -----------------------------
# MediaPipe init
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# -----------------------------
# Data storage
# -----------------------------
all_data = []
frame_count = 0

# -----------------------------
# Video capture
# -----------------------------
cap = cv2.VideoCapture(0)
print("Starting data capture. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    row = []

    if results.multi_hand_landmarks:
        # Process each detected hand
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # If only one hand, pad second hand with zeros
        if len(results.multi_hand_landmarks) == 1:
            row.extend([0.0]*21*3)

        # Add label at the end
        row.append(LABEL)
        all_data.append(row)
        frame_count += 1

    # Display capture info
    cv2.putText(frame, f"Captured: {frame_count}/{MAX_SAMPLES}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("Hand Landmark Capture", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or frame_count >= MAX_SAMPLES:
        break

cap.release()
cv2.destroyAllWindows()

# -----------------------------
# Save CSV for this label only
# -----------------------------
if all_data:
    df = pd.DataFrame(all_data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Data saved to {OUTPUT_CSV}")
else:
    print("No landmarks captured! CSV not created.")
