import json
import random
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------------------------------------------------ #
#  hardware setup                                                    #
# ------------------------------------------------------------------ #
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    print(f"{len(gpus)} GPU(s) enabled.")
else:
    print("No GPU detected; running on CPU.")

# ------------------------------------------------------------------ #
#  configuration                                                     #
# ------------------------------------------------------------------ #
ROOT      = Path(r"D:\New_ITC_Reformatted\SubSampled")  # subsampled folder path
RESULTS   = Path(r"D:\Codes\LabProj\Telegram_New\Result\CNN")
RESULTS.mkdir(parents=True, exist_ok=True)

TRAIN_DS  = "1"
TEST_DS   = "2"
VAL_FRAC  = 0.20
BATCH     = 128
EPOCHS    = 300

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ------------------------------------------------------------------ #
#  data loader                                                       #
# ------------------------------------------------------------------ #
def load_subsampled_dataset(root_path, ds_id):
    xs, ys = [], []
    for cat in ["Telegram", "Non-Telegram"]:
        json_path = root_path / ds_id / f"{cat}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Missing file: {json_path}")
        with open(json_path, "r") as f:
            j = json.load(f)
        # Take only first 20 features per sample
        xs.extend([feat[:20] for feat in j["features"]])
        ys.extend([cat.lower()] * len(j["features"]))
    return np.asarray(xs, np.float32), np.asarray(ys)

print("Loading train dataset …")
X_train_full, y_train_full = load_subsampled_dataset(ROOT, TRAIN_DS)
print(f"Train dataset shape: {X_train_full.shape}")

print("Loading test dataset …")
X_test, y_test = load_subsampled_dataset(ROOT, TEST_DS)
print(f"Test dataset shape: {X_test.shape}")

# Encode labels
le = LabelEncoder().fit(np.concatenate([y_train_full, y_test]))
NUM_CLASSES = len(le.classes_)

# Split train into train+val
idx = np.random.permutation(len(X_train_full))
X_train_full, y_train_full = X_train_full[idx], y_train_full[idx]
split = int((1 - VAL_FRAC) * len(X_train_full))
X_train, y_train = X_train_full[:split], y_train_full[:split]
X_val, y_val = X_train_full[split:], y_train_full[split:]
print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

# Prepare data for Conv1D (add channel dim)
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

y_train_oh = tf.keras.utils.to_categorical(le.transform(y_train), NUM_CLASSES)
y_val_oh = tf.keras.utils.to_categorical(le.transform(y_val), NUM_CLASSES)
y_test_oh = tf.keras.utils.to_categorical(le.transform(y_test), NUM_CLASSES)

INPUT_LEN = X_train.shape[1]

# ------------------------------------------------------------------ #
#  CNN model definition                                              #
# ------------------------------------------------------------------ #
def build_cnn():
    model = Sequential([
        Conv1D(32, 7, activation="relu", input_shape=(INPUT_LEN, 1)),
        Conv1D(64, 7, activation="relu"),
        MaxPooling1D(2),
        Flatten(),
        Dense(256, activation="relu"), BatchNormalization(), Dropout(.05),
        Dense(128, activation="relu"), BatchNormalization(), Dropout(.05),
        Dense(64,  activation="relu"), BatchNormalization(), Dropout(.05),
        Dense(NUM_CLASSES, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ------------------------------------------------------------------ #
#  training                                                           #
# ------------------------------------------------------------------ #
model = build_cnn()
es = EarlyStopping(monitor="val_loss", patience=30, min_delta=1e-4,
                   restore_best_weights=True, verbose=2)

print("Training CNN model …")
history = model.fit(
    X_train, y_train_oh,
    epochs=EPOCHS,
    batch_size=BATCH,
    validation_data=(X_val, y_val_oh),
    callbacks=[es],
    verbose=1
)

# ------------------------------------------------------------------ #
#  evaluation                                                        #
# ------------------------------------------------------------------ #
print("\nEvaluating on test dataset …")
y_pred = np.argmax(model.predict(X_test, batch_size=BATCH), axis=1)
y_true = np.argmax(y_test_oh, axis=1)
acc = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {acc:.4f}")

cm = confusion_matrix(y_true, y_pred)
total = cm.sum()
rows = []
for i, cname in enumerate(le.classes_):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = total - TP - FP - FN
    rows.append({
        "Class": cname, "TP": TP, "FP": FP,
        "TN": TN, "FN": FN,
        "Accuracy": (TP + TN) / total
    })
rows.append({
    "Class": "overall", "TP": "", "FP": "",
    "TN": "", "FN": "", "Accuracy": acc
})
df = pd.DataFrame(rows)
out_csv = RESULTS / f"cnn_report_train{TRAIN_DS}_test{TEST_DS}.csv"
df.to_csv(out_csv, index=False)
print(f"Saved report to {out_csv}")

