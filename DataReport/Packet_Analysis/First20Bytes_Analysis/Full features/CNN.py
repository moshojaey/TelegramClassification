import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Conv1D, Flatten, Dropout, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping

# Config
ROOT = Path(r"D:\New_ITC_Reformatted\Features20Full")
RESULTS = Path(r"D:\Codes\LabProj\Telegram_New\Result\Features20Full")
RESULTS.mkdir(parents=True, exist_ok=True)

TRAIN_DS, TEST_DS = "1", "2"
BATCH, EPOCHS = 128, 200

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

le = LabelEncoder().fit(y_train)
y_train_oh = tf.keras.utils.to_categorical(le.transform(y_train))
y_test_oh  = tf.keras.utils.to_categorical(le.transform(y_test))

# Reshape for CNN: (Samples, 10 Features, 1 Channel)
X_train_cnn = X_train[..., np.newaxis]
X_test_cnn  = X_test[..., np.newaxis]

# CNN Architecture
model = Sequential([
    # Input is 10 features. Kernel size 3 allows it to slide and find correlations.
    Conv1D(32, kernel_size=3, activation="relu", input_shape=(10, 1)),
    Conv1D(64, kernel_size=3, activation="relu"),
    Flatten(), # No pooling needed for such small input
    Dense(128, activation="relu"), BatchNormalization(), Dropout(0.1),
    Dense(64, activation="relu"), BatchNormalization(), Dropout(0.1),
    Dense(len(le.classes_), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print(f"Training CNN on 10 Full IP Features...")
model.fit(
    X_train_cnn, y_train_oh,
    validation_data=(X_test_cnn, y_test_oh),
    epochs=EPOCHS, batch_size=BATCH, verbose=1,
    callbacks=[EarlyStopping(patience=20, restore_best_weights=True)]
)

y_pred = np.argmax(model.predict(X_test_cnn), axis=1)
acc = accuracy_score(np.argmax(y_test_oh, axis=1), y_pred)
print(f"\nCNN (10 Feat) Test Accuracy: {acc:.4f}")

pd.DataFrame({"Actual": le.transform(y_test), "Predicted": y_pred}).to_csv(RESULTS / "cnn_results.csv", index=False)