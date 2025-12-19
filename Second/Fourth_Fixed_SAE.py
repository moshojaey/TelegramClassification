import json, random, numpy as np, pandas as pd, tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------------------------------------------------ #
#  configuration                                                     #
# ------------------------------------------------------------------ #
ROOT       = Path(r"D:\New_ITC_Reformatted\SubSampled")          # 1\Telegram.json, etc.
RESULT_DIR = Path(r"D:\Codes\LabProj\Telegram_New\Result\SAE")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS  = ["1", "2", "3"]
VAL_FRAC  = 0.20
BATCH     = 128
EPOCHS_AE = 200
EPOCHS_CL = 200

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ------------------------------------------------------------------ #
#  data loading                                                      #
# ------------------------------------------------------------------ #
def load_dataset(ds_id):
    xs, ys = [], []
    for cat in ["Telegram", "Non-Telegram"]:
        fp = ROOT / ds_id / f"{cat}.json"
        if not fp.exists():
            raise FileNotFoundError(f"Missing file: {fp}")
        j = json.load(open(fp, "r"))
        xs.extend(j["features"])
        ys.extend([cat.lower()] * len(j["features"]))
    return np.asarray(xs, np.float32), np.asarray(ys)

print("Loading datasets …")
per_ds = {d: load_dataset(d) for d in DATASETS}
for d in DATASETS:
    print(f"Dataset {d}: {per_ds[d][0].shape[0]} samples")

le           = LabelEncoder().fit(np.concatenate([per_ds[d][1] for d in DATASETS]))
NUM_CLASSES  = len(le.classes_)
INPUT_DIM    = per_ds[DATASETS[0]][0].shape[1]

# ------------------------------------------------------------------ #
#  model factory                                                     #
# ------------------------------------------------------------------ #
def build_models(input_dim):
    # ---------- Autoencoder --------------------------------------- #
    inp = Input(shape=(input_dim,))
    x = Dense(400, activation="relu")(inp); x = BatchNormalization()(x); x = Dropout(.05)(x)
    x = Dense(300, activation="relu")(x); x = BatchNormalization()(x); x = Dropout(.05)(x)
    x = Dense(200, activation="relu")(x); x = BatchNormalization()(x); x = Dropout(.05)(x)
    x = Dense(100, activation="relu")(x); x = BatchNormalization()(x); x = Dropout(.05)(x)
    latent = Dense(50, activation="relu")(x); latent = BatchNormalization()(latent); latent = Dropout(.05)(latent)

    d = Dense(100, activation="relu")(latent); d = BatchNormalization()(d)
    d = Dense(200, activation="relu")(d); d = BatchNormalization()(d)
    d = Dense(300, activation="relu")(d); d = BatchNormalization()(d)
    d = Dense(400, activation="relu")(d); d = BatchNormalization()(d)
    out_ae = Dense(input_dim, activation="sigmoid")(d)

    ae = Model(inp, out_ae)
    ae.compile(optimizer="adam", loss="mse")

    # ---------- Classifier ---------------------------------------- #
    enc = Model(inp, latent)
    cin = Input(shape=(50,))
    c = Dense(17, activation="relu")(cin)
    c = Dense(12, activation="relu")(c)
    cout = Dense(NUM_CLASSES, activation="softmax")(c)
    clf_head = Model(cin, cout)

    comb_out = clf_head(enc(inp))
    comb = Model(inp, comb_out)
    comb.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return ae, comb

# ------------------------------------------------------------------ #
#  training + evaluation loop                                        #
# ------------------------------------------------------------------ #
reports = []

for train_ds in DATASETS:
    print(f"\n############################")
    print(f"### Training on dataset {train_ds}")
    X_full, y_full = per_ds[train_ds]

    idx = np.random.permutation(len(X_full))
    X_full, y_full = X_full[idx], y_full[idx]
    split = int((1 - VAL_FRAC) * len(X_full))
    X_train, y_train = X_full[:split], y_full[:split]
    X_val,   y_val   = X_full[split:], y_full[split:]
    print(f"Train {X_train.shape}, Val {X_val.shape}")

    y_train_oh = tf.keras.utils.to_categorical(le.transform(y_train), NUM_CLASSES)
    y_val_oh   = tf.keras.utils.to_categorical(le.transform(y_val),   NUM_CLASSES)

    ae, comb = build_models(INPUT_DIM)

    print("→ Pretraining autoencoder …")
    ae.fit(
        X_train, X_train,
        epochs=EPOCHS_AE,
        batch_size=BATCH,
        validation_data=(X_val, X_val),
        callbacks=[EarlyStopping("val_loss", patience=10, restore_best_weights=True)],
        verbose=1
    )

    print("→ Fine-tuning classifier …")
    comb.fit(
        X_train, y_train_oh,
        epochs=EPOCHS_CL,
        batch_size=BATCH,
        validation_data=(X_val, y_val_oh),
        callbacks=[EarlyStopping("val_loss", patience=10, restore_best_weights=True)],
        verbose=1
    )

    # ------------------ test on each other dataset ---------------- #
    for test_ds in DATASETS:
        if test_ds == train_ds:
            continue
        print(f"Evaluating on dataset {test_ds} …")
        X_test, y_test = per_ds[test_ds]
        y_test_oh = tf.keras.utils.to_categorical(le.transform(y_test), NUM_CLASSES)

        y_pred = np.argmax(comb.predict(X_test, batch_size=BATCH), axis=1)
        y_true = np.argmax(y_test_oh, axis=1)
        acc = accuracy_score(y_true, y_pred)
        print(f"   Test accuracy = {acc:.4f}")

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
        out_file = RESULT_DIR / f"sae_report_train{train_ds}_test{test_ds}.csv"
        df.to_csv(out_file, index=False)
        print(f"   Saved {out_file}")
        reports.append((f"{train_ds}->{test_ds}", acc))

# ------------------------------------------------------------------ #
#  summary                                                           #
# ------------------------------------------------------------------ #
print("\n====== SUMMARY ======")
for label, acc in reports:
    print(f"{label}: {acc:.4f}")
