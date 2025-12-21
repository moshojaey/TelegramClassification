import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Config
ROOT = Path(r"D:\New_ITC_Reformatted\OnlyLength")
RESULTS = Path(r"D:\Codes\LabProj\Telegram_New\Result\OnlyLength")
RESULTS.mkdir(parents=True, exist_ok=True)

TRAIN_DS, TEST_DS = "1", "2"
BATCH, EPOCHS = 128, 100

# Load Data
def load_data(ds_id):
    xs, ys = [], []
    for cat in ["Telegram", "Non-Telegram"]:
        path = ROOT / ds_id / f"{cat}.json"
        j = json.load(open(path))
        xs.extend(j["features"])
        ys.extend([cat.lower()] * len(j["features"]))
    return np.array(xs, dtype=np.float32), np.array(ys)

print("Loading Data...")
X_train, y_train = load_data(TRAIN_DS)
X_test, y_test   = load_data(TEST_DS)

# Encode Labels
le = LabelEncoder().fit(y_train)
y_train_oh = tf.keras.utils.to_categorical(le.transform(y_train))
y_test_oh  = tf.keras.utils.to_categorical(le.transform(y_test))

# Simple Model (MLP) for scalar input
model = Sequential([
    Dense(64, activation="relu", input_shape=(1,)), # Input is 1 feature (Length)
    BatchNormalization(),
    Dense(32, activation="relu"),
    BatchNormalization(),
    Dense(16, activation="relu"),
    Dense(len(le.classes_), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train
print(f"Training on Length ONLY (Train={TRAIN_DS}, Test={TEST_DS})...")
model.fit(
    X_train, y_train_oh,
    validation_data=(X_test, y_test_oh),
    epochs=EPOCHS, batch_size=BATCH, verbose=1,
    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
)

# Evaluate
y_pred = np.argmax(model.predict(X_test), axis=1)
acc = accuracy_score(np.argmax(y_test_oh, axis=1), y_pred)

print(f"\nResults for 'Length Only':")
print(f"Accuracy: {acc:.4f}")
print(f"Check saved report at: {RESULTS}")

# Save CSV
df = pd.DataFrame({"Actual": le.transform(y_test), "Predicted": y_pred})
df.to_csv(RESULTS / "length_only_results.csv", index=False)