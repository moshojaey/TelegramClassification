import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_PATH = Path(r"D:\New_ITC_Reformatted\first20bytes")

TRAIN_DS = "1"
TEST_DS  = "2"

CLASSES = ["Telegram", "Non-Telegram"]

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
def load_dataset(ds_id):
    X, y = [], []

    for label, cls in enumerate(CLASSES):
        path = BASE_PATH / ds_id / f"{cls}.json"

        with open(path, "r") as f:
            data = json.load(f)

        for pkt in data["features"]:
            X.append(list(pkt.values()))
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    print("Loading Dataset 1 (TRAIN)...")
    X_train, y_train = load_dataset(TRAIN_DS)

    print("Loading Dataset 2 (TEST)...")
    X_test, y_test = load_dataset(TEST_DS)

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples : {len(X_test)}")
    print(f"Feature count: {X_train.shape[1]}")

    # --------------------------------------------------
    # Normalize (VERY IMPORTANT for Logistic Regression)
    # --------------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    )

    clf.fit(X_train, y_train)

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    y_pred = clf.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print("\n=== Evaluation Results ===")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"TN: {tn}")
    print(f"FN: {fn}")

    print("\nMetrics:")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_test, y_pred):.4f}")

    # --------------------------------------------------
    # Feature importance (coefficients)
    # --------------------------------------------------
    print("\nTop 10 most influential features:")
    coef = np.abs(clf.coef_[0])
    top_idx = np.argsort(coef)[::-1][:10]

    for i in top_idx:
        print(f"Feature {i:02d} -> {coef[i]:.4f}")

if __name__ == "__main__":
    main()
