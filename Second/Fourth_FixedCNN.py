import json, random, os, numpy as np, pandas as pd, tensorflow as tf
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
ROOT      = Path(r"D:\New_ITC_Reformatted\SubSampled")
RESULTS   = Path(r"D:\Codes\LabProj\Telegram_New\Result\CNN")
RESULTS.mkdir(parents=True, exist_ok=True)

DATASETS  = ["1", "2", "3"]
VAL_FRAC  = 0.20
BATCH     = 128
EPOCHS    = 300

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ------------------------------------------------------------------ #
#  data loader                                                       #
# ------------------------------------------------------------------ #
def load_dataset(ds_id):
    xs, ys = [], []
    for cat in ["Telegram", "Non-Telegram"]:
        json_path = ROOT / ds_id / f"{cat}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Missing file: {json_path}")
        j = json.load(open(json_path))
        xs.extend(j["features"])
        ys.extend([cat.lower()] * len(j["features"]))
    return np.asarray(xs, np.float32), np.asarray(ys)

print("Loading datasets …")
per_ds = {d: load_dataset(d) for d in DATASETS}
for d in DATASETS:
    print(f"Dataset {d}: {per_ds[d][0].shape[0]} samples")

le           = LabelEncoder().fit(np.concatenate([per_ds[d][1] for d in DATASETS]))
NUM_CLASSES  = len(le.classes_)
INPUT_LEN    = per_ds[DATASETS[0]][0].shape[1]

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
#  training + evaluation                                             #
# ------------------------------------------------------------------ #
reports = []

for train_ds in DATASETS:
    print(f"\n############################")
    print(f"### Training on dataset {train_ds}")
    X_full, y_full = per_ds[train_ds]

    idx   = np.random.permutation(len(X_full))
    X_full, y_full = X_full[idx], y_full[idx]
    split = int((1 - VAL_FRAC) * len(X_full))
    X_train, y_train = X_full[:split], y_full[:split]
    X_val,   y_val   = X_full[split:], y_full[split:]
    print(f"Train {X_train.shape}, Val {X_val.shape}")

    # one-hot + reshape for Conv1D
    y_train_oh = tf.keras.utils.to_categorical(le.transform(y_train), NUM_CLASSES)
    y_val_oh   = tf.keras.utils.to_categorical(le.transform(y_val),   NUM_CLASSES)
    X_train    = X_train[..., np.newaxis]
    X_val      = X_val[...,   np.newaxis]

    model = build_cnn()
    es = EarlyStopping(monitor="val_loss", patience=30, min_delta=1e-4,
                       restore_best_weights=True, verbose=2)

    print("→ Training CNN …")
    model.fit(
        X_train, y_train_oh,
        epochs=EPOCHS,
        batch_size=BATCH,
        validation_data=(X_val, y_val_oh),
        callbacks=[es],
        verbose=1
    )

    # ---------------- test on remaining datasets ------------------ #
    for test_ds in DATASETS:
        if test_ds == train_ds:
            continue
        print(f"Evaluating on dataset {test_ds} …")
        X_test, y_test = per_ds[test_ds]
        y_test_oh = tf.keras.utils.to_categorical(le.transform(y_test), NUM_CLASSES)
        X_test    = X_test[..., np.newaxis]

        y_pred = np.argmax(model.predict(X_test, batch_size=BATCH), axis=1)
        y_true = np.argmax(y_test_oh, axis=1)
        acc    = accuracy_score(y_true, y_pred)
        print(f"   Test accuracy = {acc:.4f}")

        cm   = confusion_matrix(y_true, y_pred)
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
        out_csv = RESULTS / f"cnn_report_train{train_ds}_test{test_ds}.csv"
        df.to_csv(out_csv, index=False)
        print(f"   Saved {out_csv}")
        reports.append((f"{train_ds}->{test_ds}", acc))

# ------------------------------------------------------------------ #
#  summary                                                           #
# ------------------------------------------------------------------ #
print("\n====== SUMMARY ======")
for lbl, acc in reports:
    print(f"{lbl}: {acc:.4f}")

