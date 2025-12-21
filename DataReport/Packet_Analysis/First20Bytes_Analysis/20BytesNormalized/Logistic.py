import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras import layers, models

# Paths to your feature datasets (extracted from first 20 bytes)
DATASET_BASE = r"D:\New_ITC_Reformatted\first20bytes"
DATASET_1_PATH = os.path.join(DATASET_BASE, "1")
DATASET_2_PATH = os.path.join(DATASET_BASE, "2")

def load_features_and_label(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    features = data["features"]
    label_str = data["label"].lower()
    label = 1 if label_str == "telegram" else 0
    return features, label

def load_dataset_features_labels(dataset_path):
    """
    Loads Telegram and Non-Telegram JSON files from dataset_path,
    combines features and labels, shuffles and returns DataFrame X and numpy array y.
    """
    telegram_file = os.path.join(dataset_path, "Telegram.json")
    non_telegram_file = os.path.join(dataset_path, "Non-Telegram.json")

    telegram_feats, telegram_label = load_features_and_label(telegram_file)
    non_telegram_feats, non_telegram_label = load_features_and_label(non_telegram_file)

    # Convert lists of dicts to DataFrames
    df_telegram = pd.DataFrame(telegram_feats)
    df_non_telegram = pd.DataFrame(non_telegram_feats)

    df_telegram["label"] = telegram_label
    df_non_telegram["label"] = non_telegram_label

    # Concatenate and shuffle
    df_all = pd.concat([df_telegram, df_non_telegram], ignore_index=True)
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

    y = df_all["label"].values
    X = df_all.drop(columns=["label"])

    return X, y

# --- Models ---

def build_autoencoder(input_dim, encoding_dim=8):
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(32, activation='relu')(input_layer)
    encoded = layers.Dense(encoding_dim, activation='relu')(encoded)

    decoded = layers.Dense(32, activation='relu')(encoded)
    decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = models.Model(input_layer, decoded)
    encoder = models.Model(input_layer, encoded)

    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

def build_mlp_classifier(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_evaluate():
    print("Loading Dataset 1 (TRAIN)...")
    X_train, y_train = load_dataset_features_labels(DATASET_1_PATH)
    print("Loading Dataset 2 (TEST)...")
    X_test, y_test = load_dataset_features_labels(DATASET_2_PATH)

    # Zero out IP fields
    ip_fields = ["ip_version", "ip_ihl", "ip_tos", "ip_total_length", "ip_identification"]
    for field in ip_fields:
        if field in X_train.columns:
            X_train[field] = 0
        if field in X_test.columns:
            X_test[field] = 0

    print(f"Train samples: {len(y_train)}")
    print(f"Test samples : {len(y_test)}")
    print(f"Feature count: {X_train.shape[1]}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    input_dim = X_train_scaled.shape[1]
    encoding_dim = min(8, input_dim // 2)

    print("\nTraining Autoencoder...")
    autoencoder, encoder = build_autoencoder(input_dim, encoding_dim)
    autoencoder.fit(X_train_scaled, X_train_scaled,
                    epochs=30,
                    batch_size=256,
                    shuffle=True,
                    validation_split=0.1,
                    verbose=2)

    print("\nEncoding train and test features...")
    X_train_encoded = encoder.predict(X_train_scaled)
    X_test_encoded = encoder.predict(X_test_scaled)

    print("\nTraining MLP Classifier on encoded features...")
    clf = build_mlp_classifier(encoding_dim)
    clf.fit(X_train_encoded, y_train,
            epochs=30,
            batch_size=256,
            validation_split=0.1,
            verbose=2)

    print("\nPredicting on test data...")
    y_pred_prob = clf.predict(X_test_encoded).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n=== Evaluation Results ===")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"TN: {tn}")
    print(f"FN: {fn}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

if __name__ == "__main__":
    train_and_evaluate()
