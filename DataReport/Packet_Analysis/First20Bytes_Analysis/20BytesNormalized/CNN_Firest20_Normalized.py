import json, random, numpy as np, pandas as pd, tensorflow as tf
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
ROOT    = Path(r"D:\New_ITC_Reformatted\first20bytesNormalized")
RESULTS = Path(r"D:\Codes\LabProj\Telegram_New\Result\CNN")
RESULTS.mkdir(parents=True, exist_ok=True)

TRAIN_DS = "1"
TEST_DS  = "2"

VAL_FRAC = 0.20
BATCH    = 128
EPOCHS  = 300

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ------------------------------------------------------------------ #
#  data loader                                                       #
# ------------------------------------------------------------------ #
def load_dataset(ds_id):
    xs, ys = [], []
    for cat in ["Telegram", "Non-Telegram"]:
        path = ROOT / ds_id / f"{cat}.json"
        if not path.exists():
            raise FileNotFoundError(path)

        j = json.load(open(path))
        xs.extend(j["features"])
        ys.extend([cat.lower()] * len(j["features"]))

    return np.asarray(xs, np.float32), np.asarray(ys)

print("Loading datasets …")
X_train_full, y_train_full = load_dataset(TRAIN_DS)
X_test,       y_test       = load_dataset(TEST_DS)

print(f"Train dataset {TRAIN_DS}: {X_train_full.shape}")
print(f"Test  dataset {TEST_DS}: {X_test.shape}")

# ------------------------------------------------------------------ #
#  label encoding                                                    #
# ------------------------------------------------------------------ #
le = LabelEncoder().fit(np.concatenate([y_train_full, y_test]))
NUM_CLASSES = len(le.classes_)
INPUT_LEN   = X_train_full.shape[1]

# ------------------------------------------------------------------ #
#  train/val split                                                   #
# ------------------------------------------------------------------ #
idx = np.random.permutation(len(X_train_full))
X_train_full, y_train_full = X_train_full[idx], y_train_full[idx]

split = int((1 - VAL_FRAC) * len(X_train_full))
X_train, y_train = X_train_full[:split], y_train_full[:split]
X_val,   y_val   = X_train_full[split:], y_train_full[split:]

print(f"Train split: {X_train.shape}")
print(f"Val   split: {X_val.shape}")

# one-hot + reshape for Conv1D
y_train_oh = tf.keras.utils.to_categorical(le.transform(y_train), NUM_CLASSES)
y_val_oh   = tf.keras.utils.to_categorical(le.transform(y_val),   NUM_CLASSES)
y_test_oh  = tf.keras.utils.to_categorical(le.transform(y_test),  NUM_CLASSES)

X_train = X_train[..., np.newaxis]
X_val   = X_val[...,   np.newaxis]
X_test  = X_test[...,  np.newaxis]

# ------------------------------------------------------------------ #
#  CNN model (UNCHANGED)                                             #
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
#  training                                                         #
# ------------------------------------------------------------------ #
model = build_cnn()
es = EarlyStopping(
    monitor="val_loss",
    patience=30,
    min_delta=1e-4,
    restore_best_weights=True,
    verbose=2
)

print("\n→ Training CNN on Dataset 1 …")
model.fit(
    X_train, y_train_oh,
    epochs=EPOCHS,
    batch_size=BATCH,
    validation_data=(X_val, y_val_oh),
    callbacks=[es],
    verbose=1
)

# ------------------------------------------------------------------ #
#  evaluation on Dataset 2                                           #
# ------------------------------------------------------------------ #
print("\n→ Evaluating on Dataset 2 …")
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
        "Class": cname,
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "Accuracy": (TP + TN) / total
    })

rows.append({
    "Class": "overall",
    "TP": "", "FP": "", "TN": "", "FN": "",
    "Accuracy": acc
})

df = pd.DataFrame(rows)
out_csv = RESULTS / "cnn_train1_test2_first20bytesNormalized.csv"
df.to_csv(out_csv, index=False)

print(f"\n✔ Report saved to {out_csv}")
