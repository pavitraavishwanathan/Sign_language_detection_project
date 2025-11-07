# training_isl_final.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ----------------------------
# Load and clean dataset
# ----------------------------
data_path = "/Users/pavitraa/Desktop/SIGNLANG_V2/csv_data/ALL_DATA.csv"
print(f"📂 Loading data from: {data_path}")

data = pd.read_csv(data_path, low_memory=False)
print(f"✅ CSV loaded. Shape: {data.shape}")

# Remove unnamed or index columns if present
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
print(f"Remaining columns: {data.shape[1]}")

# Convert all feature columns to numeric safely
for col in data.columns[:-1]:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Replace invalid or missing values with 0 (instead of dropping them)
data = data.fillna(0)

# Separate features (X) and labels (y)
label_col = data.columns[-1]
X = data.drop(label_col, axis=1).astype('float32')
y = data[label_col]

print(f"🧩 Features shape: {X.shape}")
print(f"🎯 Sample labels: {y.unique()[:10]}")
print(f"Total classes detected: {len(y.unique())}")

# ----------------------------
# Encode labels
# ----------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)
y_onehot = to_categorical(y_encoded, num_classes)

# Save class labels for realtime detection
np.save("/Users/pavitraa/Desktop/SIGNLANG_V2/label_classes.npy", label_encoder.classes_)
print("✅ Label classes saved as 'label_classes.npy'")

# ----------------------------
# Split data
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"📊 Training set: {X_train.shape}, Testing set: {X_test.shape}")

# ----------------------------
# Build and compile the model
# ----------------------------
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# ----------------------------
# Train the model
# ----------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    verbose=1
)

# ----------------------------
# Save model
# ----------------------------
model.save("/Users/pavitraa/Desktop/SIGNLANG_V2/signlang_v2_model.keras")
print("✅ Model saved as 'signlang_v2_model.keras'")
