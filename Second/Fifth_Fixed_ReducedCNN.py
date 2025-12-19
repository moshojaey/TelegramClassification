import json, random, os, numpy as np, pandas as pd, tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# ---------------- hardware -------------------------------------------------- #
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    print(f"{len(gpus)} GPU(s) available.")
else:
    print("Running on CPU.")

# ---------------- configuration -------------------------------------------- #
ROOT       = Path(r"D:\New_ITC_Reformatted\SubSampled")
MODEL_DIR  = Path(r"D:\Codes\LabProj\Telegram_New\Result\ReducedCNN")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DS   = "1"
TEST_DS    = "2"
VAL_FRAC   = 0.20
BATCH      = 128
EPOCHS     = 300
PATIENCE   = 10
random.seed(42); np.random.seed(42); tf.random.set_seed(42)

# ---------------- prefix schedule ------------------------------------------ #
prefixes = []
n = 250
while n > 128:
    prefixes.append(n); n -= 10
prefixes.append(128)
n = 123
while n > 64:
    prefixes.append(n); n -= 5
prefixes.append(64)
n = 62
while n >= 2:
    prefixes.append(n); n -= 2
prefixes.append(1)
print("Prefix lengths:", prefixes)

# ---------------- helpers --------------------------------------------------- #
def load_dataset(ds, L):
    xs, ys = [], []
    for cat in ["Telegram", "Non-Telegram"]:
        data_path = ROOT / ds / f"{cat}.json"
        if not data_path.exists():
            print(f"[!] Missing file: {data_path}")
            continue

        data = json.load(open(data_path))
        for pkt in data["features"]:
            sliced = pkt[:L]
            assert len(sliced) == L
            xs.append(sliced)
        ys.extend([cat.lower()] * len(data["features"]))

    X, y = np.asarray(xs, np.float32), np.asarray(ys)
    print(f"Loaded DS {ds} @ {L} bytes → {X.shape[0]} samples")
    return X, y


def build_cnn(input_len, n_class):
    k = min(7, input_len)
    model = Sequential([
        Conv1D(32, k, padding="same", activation="relu", input_shape=(input_len, 1)),
        Conv1D(64, k, padding="same", activation="relu"),
        MaxPooling1D(2),
        Flatten(),
        Dense(256, activation="relu"), BatchNormalization(), Dropout(.05),
        Dense(128, activation="relu"), BatchNormalization(), Dropout(.05),
        Dense(64,  activation="relu"), BatchNormalization(), Dropout(.05),
        Dense(n_class, activation="softmax")
    ])
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    return model

# ---------------- label encoder (fit once) ---------------------------------- #
_, y_train = load_dataset(TRAIN_DS, 10)
_, y_test  = load_dataset(TEST_DS,  10)
le          = LabelEncoder().fit(np.concatenate([y_train, y_test]))
NUM_CLASSES = len(le.classes_)
POS_IDX     = np.where(le.classes_ == "telegram")[0][0]

results = []
out_csv = MODEL_DIR / "cnn_prefix_scan_report.csv"

def save_results():
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print("Progress saved to", out_csv)

# ---------------- main loop ------------------------------------------------- #
for L in prefixes:
    try:
        print(f"\n================  {L} bytes  ================")
        X, y = load_dataset(TRAIN_DS, L)
        p = np.random.permutation(len(X)); X, y = X[p], y[p]
        split = int((1 - VAL_FRAC) * len(X))
        X_tr, y_tr   = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]

        y_tr_oh  = tf.keras.utils.to_categorical(le.transform(y_tr),  NUM_CLASSES)
        y_val_oh = tf.keras.utils.to_categorical(le.transform(y_val), NUM_CLASSES)
        X_tr, X_val = X_tr[..., None], X_val[..., None]

        model = build_cnn(L, NUM_CLASSES)
        es = EarlyStopping("val_loss", patience=PATIENCE, min_delta=1e-4,
                           restore_best_weights=True, verbose=2)

        print("→ Training")
        hist = model.fit(
            X_tr, y_tr_oh,
            epochs=EPOCHS, batch_size=BATCH,
            validation_data=(X_val, y_val_oh),
            callbacks=[es], verbose=1
        )

        epochs_done   = len(hist.history["loss"])
        best_val_loss = min(hist.history["val_loss"])
        best_val_acc  = max(hist.history["val_accuracy"])
        print(f"Finished in {epochs_done} epochs | "
              f"best val_loss={best_val_loss:.4f} | best val_acc={best_val_acc:.4f}")

        # ------ save the trained model ------------------------------------ #
        model_path = MODEL_DIR / f"cnn_prefix{L}.keras"
        model.save(model_path)
        print("Saved model →", model_path)

        # ---------------- test on dataset 2 ------------------------------- #
        X_test, y_test_lbl = load_dataset(TEST_DS, L)
        y_true = le.transform(y_test_lbl)
        y_pred = np.argmax(model.predict(X_test[..., None], batch_size=BATCH), 1)

        cm = confusion_matrix(y_true, y_pred, labels=[POS_IDX, 1 - POS_IDX])
        TP, FN, FP, TN = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        acc = accuracy_score(y_true, y_pred)
        f1  = 2*TP / (2*TP + FP + FN) if (2*TP + FP + FN) else 0.0
        print(f"Test → TP={TP} FP={FP} TN={TN} FN={FN} | acc={acc:.4f} f1={f1:.4f}")

        results.append({
            "PrefixBytes": L, "Epochs": epochs_done,
            "TP": TP, "FP": FP, "TN": TN, "FN": FN,
            "Accuracy": acc, "F1": f1,
            "ModelPath": str(model_path)
        })
        save_results()

    except Exception as e:
        print(f"[ERROR] prefix {L} bytes → {e}")
        results.append({
            "PrefixBytes": L, "Epochs": 0,
            "TP": "", "FP": "", "TN": "", "FN": "",
            "Accuracy": "", "F1": "",
            "ModelPath": "", "Error": str(e)
        })
        save_results()
        continue

print("\n=== Scan complete. Final results in", out_csv, "===")
print(pd.DataFrame(results).to_string(index=False))
