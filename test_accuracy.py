# test_accuracy.py
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "/Users/pavitraa/Desktop/SIGNLANG_V2/signlang_v2_model.keras"
DATA_PATH = "/Users/pavitraa/Desktop/SIGNLANG_V2/csv_data/ALL_DATA.csv"

print("📂 Loading model and dataset...")
model = load_model(MODEL_PATH)
data = pd.read_csv(DATA_PATH, low_memory=False)

# -----------------------------
# Preprocess dataset
# -----------------------------
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
label_col = data.columns[-1]

# Convert features to float
for col in data.columns[:-1]:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data = data.fillna(0)

X = data.drop(label_col, axis=1).astype('float32')
y = data[label_col]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded)
num_classes = len(label_encoder.classes_)

print(f"✅ Dataset shape: {X.shape}, Total classes: {num_classes}")

# -----------------------------
# Evaluate Model
# -----------------------------
print("🧠 Evaluating model...")
loss, acc = model.evaluate(X, y_onehot, verbose=1)
print(f"\n✅ Model Accuracy: {acc * 100:.2f}%")

# -----------------------------
# Detailed Metrics
# -----------------------------
y_pred_probs = model.predict(X)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n📊 Classification Report:")
print(classification_report(y_encoded, y_pred, target_names=label_encoder.classes_))

# -----------------------------
# Confusion Matrix Visualization
# -----------------------------
cm = confusion_matrix(y_encoded, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
