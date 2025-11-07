import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque, Counter

# -----------------------------
# Load trained model and labels
# -----------------------------
MODEL_PATH = "/Users/pavitraa/Desktop/SIGNLANG_V2/signlang_v2_model.keras"
LABELS_PATH = "/Users/pavitraa/Desktop/SIGNLANG_V2/label_classes.npy"

print("📦 Loading model...")
model = load_model(MODEL_PATH)
labels = np.load(LABELS_PATH, allow_pickle=True)
print(f"✅ Model and labels loaded. Total classes: {len(labels)}")

# -----------------------------
# Initialize Mediapipe Hands
# -----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -----------------------------
# Helper: Normalize landmarks
# -----------------------------
def extract_hand_features(hand_landmarks):
    """Extracts (x, y, z) flattened list from mediapipe landmarks"""
    features = []
    for lm in hand_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])
    return np.array(features, dtype=np.float32)

def combine_hands_features(results):
    """Handles both left/right hands consistently"""
    right_hand = np.zeros(63, dtype=np.float32)
    left_hand = np.zeros(63, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            features = extract_hand_features(hand_landmarks)
            if handedness.classification[0].label == "Right":
                right_hand = features
            else:
                left_hand = features

    # Combine both hands into one flat vector (63*2 = 126)
    return np.concatenate([right_hand, left_hand])

# -----------------------------
# Video Capture Initialization
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not detected.")
    exit()

print("✅ Starting real-time detection... Press 'q' to quit.")

# Smooth prediction window
WINDOW_SIZE = 8
pred_window = deque(maxlen=WINDOW_SIZE)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    EXPECTED_FEATURES = 190
    X_input = np.zeros((1, EXPECTED_FEATURES), dtype=np.float32)

    if results.multi_hand_landmarks:
        features = combine_hands_features(results)
        X_input[0, :len(features)] = features

        # Draw landmarks on frame
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # -----------------------------
        # Predict gesture
        # -----------------------------
        preds = model.predict(X_input, verbose=0)
        pred_idx = np.argmax(preds)
        pred_label = labels[pred_idx]
        pred_conf = np.max(preds)

        # Add to prediction smoothing window
        pred_window.append(pred_label)

        # Determine most frequent prediction in window
        most_common_pred = Counter(pred_window).most_common(1)[0][0]

        # Display results
        cv2.putText(
            frame,
            f"{most_common_pred} ({pred_conf:.2f})",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

    else:
        # If no hands are detected, clear prediction history
        pred_window.clear()

    cv2.imshow("Sign Language Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
