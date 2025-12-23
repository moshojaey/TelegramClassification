import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

# ==================================================================
# CONFIGURATION
# ==================================================================
ROOT = Path(r"D:\New_ITC_Reformatted\Length_ID_TTL_Flags")
RESULTS = Path(r"D:\Codes\LabProj\Telegram_New\Result\Length_ID_TTL_Flags")
RESULTS.mkdir(parents=True, exist_ok=True)

TRAIN_DS, TEST_DS = "1", "2"
BATCH, EPOCHS = 128, 100

# ==================================================================
# DATA LOADING
# ==================================================================
def load_data(ds_id):
    xs, ys = [], []
    for cat in ["Telegram", "Non-Telegram"]:
        path = ROOT / ds_id / f"{cat}.json"
        if not path.exists():
            print(f"[!] Warning: Missing file {path}")
            continue
        j = json.load(open(path))
        xs.extend(j["features"])
        ys.extend([cat.lower()] * len(j["features"]))
    return np.array(xs, dtype=np.float32), np.array(ys)

print("Loading Data...")
X_train, y_train = load_data(TRAIN_DS)
X_test, y_test   = load_data(TEST_DS)

# Encode Labels (Non-Telegram=0, Telegram=1 usually due to alphabetical sort)
le = LabelEncoder().fit(y_train)
y_train_oh = tf.keras.utils.to_categorical(le.transform(y_train))
y_test_oh  = tf.keras.utils.to_categorical(le.transform(y_test))

print(f"Classes: {le.classes_}") # Verify order (likely ['non-telegram', 'telegram'])

# Reshape for CNN: (Samples, 4 Features, 1 Channel)
X_train_cnn = X_train[..., np.newaxis]
X_test_cnn  = X_test[..., np.newaxis]

# ==================================================================
# CNN MODEL
# ==================================================================
model = Sequential([
    Conv1D(32, kernel_size=2, activation="relu", input_shape=(4, 1)),
    Conv1D(64, kernel_size=2, activation="relu"),
    Flatten(),
    Dense(64, activation="relu"), BatchNormalization(),
    Dense(32, activation="relu"),
    Dense(len(le.classes_), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print(f"Training CNN on 4 Features (Len, ID, TTL, Flags)...")
model.fit(
    X_train_cnn, y_train_oh,
    validation_data=(X_test_cnn, y_test_oh),
    epochs=EPOCHS, batch_size=BATCH, verbose=1,
    callbacks=[EarlyStopping(patience=15, restore_best_weights=True)]
)

# ==================================================================
# EVALUATION & REPORTING
# ==================================================================
print("\nCalculating Metrics...")
# Get predictions
y_pred_probs = model.predict(X_test_cnn)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test_oh, axis=1)

# Calculate Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# Extract TP, TN, FP, FN
# Note: confusion_matrix output is [[TN, FP], [FN, TP]] for binary 0,1
tn, fp, fn, tp = cm.ravel()

acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='binary')
recall = recall_score(y_true, y_pred, average='binary')

print("-" * 30)
print(f"CNN Test Accuracy: {acc:.4f}")
print("-" * 30)
print(f"True Positives  (TP): {tp}")
print(f"True Negatives  (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print("-" * 30)

# Save Raw Predictions
pd.DataFrame({"Actual": le.transform(y_test), "Predicted": y_pred}).to_csv(RESULTS / "cnn_predictions.csv", index=False)

# Save Metrics Report
metrics_df = pd.DataFrame([{
    "TP": tp,
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "Accuracy": acc,
    "Precision": precision,
    "Recall": recall
}])
metrics_csv = RESULTS / "cnn_metrics_report.csv"
metrics_df.to_csv(metrics_csv, index=False)
print(f"Detailed metrics saved to: {metrics_csv}")